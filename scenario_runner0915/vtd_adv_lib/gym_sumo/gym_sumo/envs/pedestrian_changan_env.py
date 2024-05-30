import math
from typing import Optional, TypeVar, Dict, Text

import numpy as np

from gym_sumo import utils
from gym_sumo.envs.common.abstract import AbstractEnv
from gym_sumo.envs.common.action import Action
from gym_sumo.road.lane import LineType, StraightLane, PolyLaneFixedWidth
from gym_sumo.road.road import Road, RoadNetwork
from gym_sumo.vehicle.controller import ControlledVehicle, MDPVehicle
from gym_sumo.vehicle.behavior import IDMVehicle
import random
from gym_sumo.algo.global_route_planner_vtd_xodr import GlobalRoutePlanner, RoadOption

Observation = TypeVar("Observation")
from shapely.geometry import Polygon
from shapely.geometry import LineString, Point
import math
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

def check_lane_intersection(lane1_center, lane1_width, lane2_center, lane2_width):
    # 将中心线坐标转换为多边形
    lane1_polygon = LineString([(x, y) for x, y in lane1_center])
    lane2_polygon = LineString([(x, y) for x, y in lane2_center])

    # 使用buffer方法增加车道宽度
    lane1_polygon_buffered = lane1_polygon.buffer(lane1_width / 2, cap_style=2)
    lane2_polygon_buffered = lane2_polygon.buffer(lane2_width / 2, cap_style=2)
    intersection = False
    intersection_center = None
    if not (lane1_polygon_buffered.is_valid and lane2_polygon_buffered.is_valid):
        pass
    else:
        if lane1_polygon_buffered.intersects(lane2_polygon_buffered):
            # 计算相交区域
            intersection_area = lane1_polygon_buffered.intersection(lane2_polygon_buffered)
            if not intersection_area.is_empty:
                intersection = True
                intersection_center = list(intersection_area.centroid.coords)

    return intersection, intersection_center


def calculate_perpendicular_points(x0, y0, theta, distance=1, num_points=11):
    # 检查是否为垂直或水平情况
    if math.isclose(theta % (math.pi / 2), 0):  # 垂直或水平
        if math.isclose(theta % math.pi, 0):  # 水平方向
            direction_vector = sp.Matrix([0, 1])  # 垂直直线向上
        else:  # 垂直方向
            direction_vector = sp.Matrix([1, 0])  # 垂直直线向右
    else:  # 一般情况
        cot_theta_value = math.cos(theta) / math.sin(theta)
        direction_vector = sp.Matrix([1, -cot_theta_value]).normalized()  # 标准化
    # 计算点
    points = [sp.Matrix([x0, y0]) + i * distance * direction_vector for i in range(-num_points // 2, num_points // 2 + 1)]
    # 提取坐标
    coordinates_vertical = [(float(point.evalf()[0]), float(point.evalf()[1])) for point in points]
    return coordinates_vertical


def generate_bezier_points(P0, P1, random_offset=[5, 5]):
    t = np.linspace(0, 1,11)
    # 假设我们选择的垂直偏移量为1，向上偏移
    offset_x = random.choice(range(-random_offset[0], random_offset[0]+1))
    offset_y = random.choice(range(-random_offset[1], random_offset[1]+1))
    offset = np.array([offset_x, offset_y])  # 向上偏移
    P0 = np.array(P0)
    P1 = np.array(P1)
    # 计算两端点的中点
    M = (P0 + P1) / 2

    Pc = M + offset  # 控制点
    # 重新生成贝塞尔曲线的点
    points = np.array([(1 - t_) ** 2 * P0 + 2 * (1 - t_) * t_ * Pc + t_ ** 2 * P1 for t_ in t])
    return points.tolist()

class PedestrianEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "ChangAnAdv",
                "vehicles_count": 6,
                "features": ["x", "y", "vx", "vy", 'heading', 'lat_off', 'lane_heading',
                             'left_change_lane', 'right_lane_change'],
                "features_range": {
                    "x": [-60, 60],
                    "y": [-60, 60],
                    "vx": [-30, 30],
                    "vy": [-30, 30],
                },
                "absolute": False,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "centering_position": [0.5, 0.5],
            "duration": 50,
            'max_ttc_value': 5,
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "screen_width": 1920,  # [px]
            "screen_height": 1080,  # [px]
            'make_road': False,
            'map_file_path': '/home/sx/wjc/wjc/datasets/ca_vtd_train_data/有交通灯的十字路口/stopandgo+.xodr',
            'ACTIONS': {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',
                        5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"},
            'AtomicScene': 'ita_mode',  # lon lat ita
            'vehicles_num': 10,
            'cut_in_min_lon_dis': 5,
            'speed_coef': 0,
            'ped_lon_dis_range': (20, 25),
            'obs_v2': False,
            'obs_v1': False,
        })
        return cfg

    def make_pre_data(self):
        pass

    def _reward(self, action: int) -> float:
        """
        对抗奖励设计的核心是主车与后车TTC越小奖励越大,但如果主车与前方车辆即将发生碰撞则奖励只考虑主车与前车TTC以避免放生碰撞

        :param action: the action performed
        :return: the reward of the state-action transition
        返回状态-动作转换的奖励
        """
        # 定义距离阈值

        reward = 0.0

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        ego_lon, _ = self.vehicle.lane.local_coordinates(self.meeting_points)
        ego_current_lon = self.vehicle.lane_offset[0]
        bv_lon, _ = self.ego_vehicle.lane.local_coordinates(self.meeting_points)
        bv_current_lon = self.ego_vehicle.lane_offset[0]
        d1 = (ego_lon - ego_current_lon)
        d2 = (bv_lon - bv_current_lon)
        ego_t = d1 / utils.not_zero(self.vehicle.speed) + 0.2
        bv_t = d2 / utils.not_zero(self.ego_vehicle.speed)

        if d1 > 5:
            if ego_t > bv_t:
                reward += -(1 - math.exp(- abs(ego_t - bv_t)))
            else:
                reward += math.exp(- abs(ego_t - bv_t))
        elif d1 < -5:
            reward += -(1 - math.exp(- np.linalg.norm(self.vehicle.position - self.ego_vehicle.position) / 10))
        else:
            if d2 >= (3 + d1):
                reward += math.exp(- abs(d1))
                # reward += (1 - math.exp(- max(self.ego_vehicle.speed - self.vehicle.speed, 0))) * self.config['speed_coef']
            else:
                reward += -(1 - math.exp(- abs(d1 - d2)))

        # last_action = self.pre_data.get('last_action')
        # if last_action is not None:
        #     if (self.config['ACTIONS'][int(action)] != 'IDLE' and self.config['ACTIONS'][
        #         int(last_action)] != 'IDLE' and
        #             self.config['ACTIONS'][int(action)] != self.config['ACTIONS'][last_action]):
        #         reward = -1

        # 如果被测主车前方车为对抗车
        if front_vehicle is not None and front_vehicle is self.ego_vehicle:
            if self.vehicle.crashed:  # 如果车辆碰撞或不在道路上，奖励为-1
                reward = -1

        # self.pre_data['last_action'] = int(action)

        return reward

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""

        lon, _ = self.vehicle.lane.local_coordinates(self.vehicle.position)

        return (self.vehicle.crashed or self.run_steps >= (self.config['duration'] - 2) or lon >= (self.vehicle.lane.length - self.vehicle.LENGTH/2))

    def _is_truncated(self) -> bool:
        if self.ego_vehicle is None:
            return False
        targe_lane_index = self.road.network.next_lane(self.ego_vehicle.lane_index,
                                                       route=self.ego_vehicle.route,
                                                       position=self.ego_vehicle.position,
                                                       np_random=self.road.np_random,
                                                       vehciel_length=self.ego_vehicle.LENGTH,
                                                       pop=False)
        lane = None
        lon = None
        if targe_lane_index[0] is not None:
            lane = self.road.network.get_lane(targe_lane_index)
            lon, _ = lane.local_coordinates(self.ego_vehicle.position)

        return (self.ego_vehicle.crashed or self.run_steps >= (self.config['duration'] - 2) or lon is None or
                (targe_lane_index[0] is None or (self.ego_vehicle.route is not None and len(self.ego_vehicle.route) <= 2))
                and lon >= (lane.length - self.ego_vehicle.LENGTH))

    def _reset(self) -> None:
        self.run_steps = 0
        # for rewards
        self.last_step_have_bv = False
        self.last_step_min_ttc = 50
        self.generate_ped_lane_ids = set()

        if self.config['make_road']:
            self.name = 'ped'
            self.map = GlobalRoutePlanner(self.config['map_file_path'])
            self._make_road()
            self._make_vehicles()
            self.pre_data = dict()
        else:
            self.config['make_road'] = True

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork(self.map)
        self.lane_id = set()
        for edge_id in self.map.graph.edges():
            edge = self.map.graph.get_edge_data(edge_id[0], edge_id[1])
            length, start_wpt, end_wpt, edge_type = edge.get('length'), edge.get('entry_waypoint'), edge.get(
                'exit_waypoint'), edge.get('type')
            if length > 1:
                center_points = [start_wpt]
                center_points.extend(edge.get('path'))
                center_points.append(end_wpt)
                center_lines = []
                for wpt in center_points:
                    center_lines.append([wpt.x, wpt.y])
                dis = np.linalg.norm([center_lines[-1][0] - center_lines[-2][0], center_lines[-1][1] - center_lines[-2][1]])
                if dis < 1e-4:
                    print(edge_id)
                self.lane_id.add(edge_id)
                net.add_lane(edge_id[0], edge_id[1],
                             PolyLaneFixedWidth(center_lines,
                                                start_wpt.width,
                                                lane_index=edge_id,
                                                line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        if len(self.generate_ped_lane_ids) == 0:
            for edge_id in self.map.graph.edges:
                edge = self.map.graph.get_edge_data(edge_id[0], edge_id[1])
                if edge.get('length') > 30:
                    self.generate_ped_lane_ids.add(edge_id)

        av_test_lane_id = random.choice(list(self.generate_ped_lane_ids))
        av_test_lane = self.road.network.get_lane(av_test_lane_id + (0,))
        av_test_random_lon = random.choice(range(1, int(av_test_lane.length - self.config['ped_lon_dis_range'][1])))
        ped_random_lon = random.choice(range(*self.config['ped_lon_dis_range']))

        self.meeting_points = av_test_lane.position(av_test_random_lon+ped_random_lon, 0)
        lane_heading = av_test_lane.heading_at(av_test_random_lon+ped_random_lon)
        lane_points = calculate_perpendicular_points(self.meeting_points[0], self.meeting_points[1], lane_heading)
        bezier_points = generate_bezier_points(lane_points[0], lane_points[-1])
        if random.random() < 0.5:
           bezier_points = bezier_points[::-1]
        self.road.network.add_lane('p0', 'p1', PolyLaneFixedWidth(bezier_points, 4, lane_index=('p0', 'p1'),
                                                                  line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)))

        self.vehicle = MDPVehicle.make_on_lane(self.road, self.name, ('p0', 'p1', 0), 0, speed=random.choice(np.linspace(1, 2, 10).tolist()))
        self.vehicle.target_speeds = np.linspace(-2, 2, 5)
        self.vehicle.LENGTH = 1
        self.vehicle.WIDTH = 1
        av_test = IDMVehicle.make_on_lane(self.road, 'av_test', av_test_lane_id + (0,), av_test_random_lon, speed=random.choice(range(5, 10)))
        av_test.plan_route_to(av_test_lane_id)
        self.controlled_vehicles.append(av_test)
        self.road.vehicles.append(self.vehicle)
        self.road.vehicles.append(av_test)
        self.road.set_ego_vehicles(self.controlled_vehicles)

    def pred_future_traj(self, obs:np.ndarray, ped_center, degree=1, future_num=100, show=False):

        # 已知的十个轨迹点
        x_known = obs[:, 0]  # 假设的x坐标
        y_known = obs[:, 1]  # 假设的y坐标，例如基于正弦函数生成

        vertical = False
        if math.isclose(abs(x_known[0] - x_known[1]), 0):
            vertical = True

        if vertical:
            x_known = np.linspace(0, 9, 10)

        # 使用多项式拟合轨迹曲线
        poly_y = np.polyfit(x_known[-5:], y_known[-5:], degree)  # 对于y轴
        # 使用拟合的曲线预测未来二十个轨迹点
        x_future = np.linspace(x_known[-1], (2 * x_known[-1] - x_known[0]), future_num)  # 未来的x坐标
        if vertical:
            x_future = np.linspace(10, future_num+9, future_num)

        poly_y = np.poly1d(poly_y)  # 生成多项式函数
        y_future = poly_y(x_future)  # 使用多项式函数预测未来的y坐标

        if vertical:
            x_future = np.linspace(x_known[-1], (2 * x_known[-1] - x_known[0]), future_num)  # 未来的x坐标

        if show:
            ped_center = np.array(ped_center)
            # 绘制结果
            plt.figure(figsize=[10, 5])
            plt.scatter(x_known, y_known, 2, 'blue', label='Known Trajectory')
            plt.scatter(x_future, y_future, 2,'red', label='Predicted Future Trajectory')
            plt.scatter(ped_center[:, 0], ped_center[:, 1], 2,'green', label='Pedestrian Trajectory')
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Trajectory Prediction Using Polynomial Fit')
            plt.grid(True)
            plt.show()

        return np.array([x_future, y_future]).T

    def post_process(self):
        if self.run_steps <= 1:
            self.his_state = []

        self.his_state.append([self.ego_vehicle.position[0], self.ego_vehicle.position[1]])

        if self.run_steps >= 10:
            ped_center = np.array(self.vehicle.lane.map_vector)[:, :2].tolist()
            av_test_center = self.pred_future_traj(np.array(self.his_state), ped_center)
            intersection, intersection_center = check_lane_intersection(ped_center, self.vehicle.WIDTH, av_test_center, self.ego_vehicle.WIDTH)

            if not intersection:
                av_test_center = np.array(self.ego_vehicle.lane.map_vector)[:, :2].tolist()
                intersection, intersection_center = check_lane_intersection(ped_center, self.vehicle.WIDTH, av_test_center, self.ego_vehicle.WIDTH)

            if intersection_center is not None:
                self.meeting_points = intersection_center[len(intersection_center) // 2]


    def compute_action(self, action: np.ndarray):
        # 使用numpy中的clip函数将action数组中的值限制在-1和1之间
        action = np.clip(action, -1, 1)
        # 使用utils模块中的lmap函数，将action数组中的第一个元素从[-1, 1]的范围映射至self.steering_range范围
        # 将action数组中的第二个元素从[-1, 1]的范围映射至self.acceleration_range范围
        return {"steering": utils.lmap(action[0], [-1, 1], self.config['action']['steering_range']),
                "acceleration": utils.lmap(action[1], [-1, 1], self.config['action']['acceleration_range'])}

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            self.make_pre_data()
            # Forward action to the vehicle
            self.road.act()

            if self.config['action']['type'] == 'ContinuousAction':  # 如果动作类型是连续的
                self.vehicle.action.update(self.compute_action(action))  # 更新车辆动作

            elif self.config['action']['type'] == 'DiscreteMetaAction':
                action = int(action[0]) if isinstance(action, np.ndarray) else action
                action = self.config['ACTIONS'][action]
                self.vehicle.act(action)
                if self.config['AtomicScene'] == 'lat_mode':
                    self.vehicle.speed = self.ego_vehicle.speed if self.ego_vehicle.speed > 1 else 1
                    self.vehicle.action['acceleration'] = self.ego_vehicle.action['acceleration']

            self.road.step(1 / self.config["simulation_frequency"])
            self.run_steps += 1
            self._clear_vehicles()
            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False
        self.post_process()

    def _clear_vehicles(self) -> None:
        """
        该函数用于清除道路上的车辆。
        """
        vehicles = []  # 创建一个空列表存储车辆
        remove_vehicle = []  # 创建一个空列表存储要删除的车辆
        for veh in self.road.vehicles:  # 遍历道路上的车辆
            if veh.get_name.startswith('ped'):
                vehicles.append(veh)
                continue

            targe_lane_index = self.road.network.next_lane(veh.lane_index,
                                                           route=veh.route,
                                                           position=veh.position,
                                                           np_random=self.road.np_random,
                                                           vehciel_length=veh.LENGTH,
                                                           pop=False)
            lane = None
            lon = None
            if targe_lane_index[0] is not None:
                lane = self.road.network.get_lane(targe_lane_index)
                lon, _ = lane.local_coordinates(veh.position)

            flag = (lon is None or (targe_lane_index[0] is None or (veh.route is not None and len(veh.route) <= 2)) and
                    lon >= (lane.length - veh.LENGTH))

            if veh not in self.controlled_vehicles and flag:
                remove_vehicle.append(veh)  # 将车辆添加到删除列表中
            else:
                vehicles.append(veh)

        self.road.vehicles = vehicles  # 更新道路上的车辆列表