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


class InterSectionEnv(AbstractEnv):
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
            "duration": 100,
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
            'obs_v2': False,
            'obs_v1': False,
        })
        return cfg

    def make_pre_data(self):
        change_lane_left, change_lane_right = self.road.network.can_change_lane(self.vehicle.lane_index)
        self.pre_data['change_lane_left'] = change_lane_left
        self.pre_data['change_lane_right'] = change_lane_right
        self.pre_data['bv_lane_index'] = self.ego_vehicle.lane_index
        self.pre_data['ego_lane_index'] = self.vehicle.lane_index
        self.pre_data['bv_road_info'] = self.road.network.get_road_info(self.ego_vehicle.lane_index)

    def _reward(self, action: int) -> float:
        """
        对抗奖励设计的核心是主车与后车TTC越小奖励越大,但如果主车与前方车辆即将发生碰撞则奖励只考虑主车与前车TTC以避免放生碰撞

        :param action: the action performed
        :return: the reward of the state-action transition
        返回状态-动作转换的奖励
        """
        # 定义距离阈值

        reward = 0.0
        if self.config['action']['type'] == 'DiscreteMetaAction':
            if self.config['AtomicScene'] == 'lon_mode':
                diff_pos = self.vehicle.position - self.ego_vehicle.position
                lon_offset = np.dot(diff_pos, [math.cos(self.ego_vehicle.heading), math.sin(self.ego_vehicle.heading)])
                if self.vehicle.lane_index != self.ego_vehicle.lane_index:
                    flag = False if lon_offset < 1 or lon_offset > 6 else True
                    if flag:
                        reward += math.exp(-lon_offset / 5)
                    else:
                        reward += -(1 - math.exp(
                            - np.linalg.norm(self.vehicle.position - self.ego_vehicle.position) / 10))
                else:
                    reward += -(1 - math.exp(- np.linalg.norm(self.vehicle.position - self.ego_vehicle.position) / 10))

                front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
                # 如果被测主车前方车为对抗车
                if front_vehicle is not None:
                    front_ttc = self.compute_ttc(front_vehicle, self.vehicle, self.config['max_ttc_value'])
                    if front_ttc < 3:
                        reward = -math.exp(-front_ttc / 5)

                last_action = self.pre_data.get('last_action')
                if last_action is not None:
                    if (self.config['ACTIONS'][int(action)] != 'IDLE' and self.config['ACTIONS'][
                        int(last_action)] != 'IDLE' and
                            self.config['ACTIONS'][int(action)] != self.config['ACTIONS'][last_action]):
                        reward = -1

                if self.vehicle.crashed and not self.ego_vehicle.crashed:  # 如果车辆碰撞或不在道路上，奖励为-1
                    reward = -1

                if self.ego_vehicle.crashed and not self.vehicle.crashed:
                    reward = 2

                self.pre_data['last_action'] = int(action)

            elif self.config['AtomicScene'] == 'lat_mode':

                # 求被测主车前方车辆
                front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
                # 如果被测主车前方车为对抗车

                diff_pos = self.vehicle.position - self.ego_vehicle.position
                lon_offset = np.dot(diff_pos, [math.cos(self.ego_vehicle.heading), math.sin(self.ego_vehicle.heading)])
                lat_offset = np.dot(diff_pos, [math.sin(self.ego_vehicle.heading), -math.cos(self.ego_vehicle.heading)])
                if lon_offset >= 0:
                    reward = math.exp(-abs(lat_offset) / 5)

                if self.config['ACTIONS'][int(action)] in ["LANE_LEFT", "LANE_RIGHT"]:
                    cut_in_min_lon_dis = self.ego_vehicle.speed * 0.1 + 4.0
                    if (lon_offset >= (cut_in_min_lon_dis + (self.ego_vehicle.LENGTH + self.vehicle.LENGTH) / 2) and
                            self.vehicle.lane_index != self.vehicle.target_lane_index and self.vehicle.target_lane_index == self.ego_vehicle.lane_index):
                        reward = math.exp(-abs(lat_offset) / 5) * 1.1
                    else:
                        reward = -math.exp(-abs(lat_offset) / 5)

                if ((self.config['ACTIONS'][int(action)] == "LANE_LEFT" and not self.pre_data["change_lane_left"]) or
                        (self.config['ACTIONS'][int(action)] == "LANE_RIGHT" and not self.pre_data[
                            "change_lane_right"])):
                    reward = -0.5

                if self.vehicle.crashed:
                    reward = -1.0

                if (self.ego_vehicle.crashed and any([not self.vehicle.crashed, rear_vehicle is self.ego_vehicle])
                ):
                    reward = 2

            elif self.config['AtomicScene'] == 'ita_mode':
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
                        reward += (1 - math.exp(- max(self.ego_vehicle.speed - self.vehicle.speed, 0))) * self.config[
                            'speed_coef']
                    else:
                        reward += -(1 - math.exp(- abs(d1 - d2)))

                last_action = self.pre_data.get('last_action')
                if last_action is not None:
                    if (self.config['ACTIONS'][int(action)] != 'IDLE' and self.config['ACTIONS'][
                        int(last_action)] != 'IDLE' and
                            self.config['ACTIONS'][int(action)] != self.config['ACTIONS'][last_action]):
                        reward = -1

                # 如果被测主车前方车为对抗车
                if front_vehicle is not None and front_vehicle is self.ego_vehicle:
                    if self.vehicle.crashed:  # 如果车辆碰撞或不在道路上，奖励为-1
                        reward = -1

                self.pre_data['last_action'] = int(action)

        return reward

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""

        targe_lane_index = self.road.network.next_lane(self.vehicle.lane_index,
                                                       route=self.vehicle.route,
                                                       position=self.vehicle.position,
                                                       np_random=self.road.np_random,
                                                       vehciel_length=self.vehicle.LENGTH,
                                                       pop=False)
        lane = None
        lon = None
        if targe_lane_index[0] is not None:
            lane = self.road.network.get_lane(targe_lane_index)
            lon, _ = lane.local_coordinates(self.vehicle.position)

        return (self.vehicle.crashed or self.run_steps >= (self.config['duration'] - 2) or lon is None or
                (targe_lane_index[0] is None or (self.vehicle.route is not None and len(self.vehicle.route) <= 2))
                and lon >= (lane.length - self.vehicle.LENGTH/2))

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
        self.junction_edge_list = set()
        self.junction_road_info = dict()
        self.interaction_lane_id = dict()
        self.interaction_point_id = dict()
        self.have_sides_lanes = dict()
        self.have_sides_lanes_info = dict()

        if self.config['make_road']:
            self.name = 'ego'
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
                dis = np.linalg.norm(
                    [center_lines[-1][0] - center_lines[-2][0], center_lines[-1][1] - center_lines[-2][1]])
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

    def make_scenario(self, ego_road_info, test_av_road_info, ego_obj_road_info, test_av_obj_road_info):

        ego_vehicle_type = MDPVehicle if self.config['action']['type'] == 'DiscreteMetaAction' else IDMVehicle

        ego_lane_index = self.road.network.get_lane_index_by_road(ego_road_info[0], ego_road_info[1], ego_road_info[2])

        ego_obj_lane_index = self.road.network.get_lane_index_by_road(ego_obj_road_info[0], ego_obj_road_info[1],
                                                                      ego_obj_road_info[2])
        test_av_obj_lane_index = self.road.network.get_lane_index_by_road(test_av_obj_road_info[0],
                                                                          test_av_obj_road_info[1],
                                                                          test_av_obj_road_info[2])
        bv_lane_index = self.road.network.get_lane_index_by_road(test_av_road_info[0], test_av_road_info[1],
                                                                 test_av_road_info[2])
        ego_lon, bv_lon = 0.0, 0.0
        ego_speed, bv_speed = random.choice(range(5, 10)), random.choice(range(5, 10))
        if self.config['AtomicScene'] in ['lon_mode', 'lat_mode']:

            ego_lane = self.road.network.get_lane(ego_lane_index + (0,))
            if ego_lane.length >= 50:
                start_lon = random.choice(range(int(ego_lane.length) // 3, int(ego_lane.length) // 3 * 2))
                ego_lon = ego_lane.length * 0.95 - start_lon
                bv_lon = ego_lane.length * 0.95 - random.choice(range(start_lon + 8, start_lon + 15))
            else:
                start_lon = random.choice(range(1, int(ego_lane.length) // 3))
                ego_lon = min(random.choice(range(start_lon + 8, start_lon + 12)), int(ego_lane.length) - 5)
                bv_lon = start_lon

            follow_lanes = self.road.network.get_all_next_lanes(ego_lane_index)
            if len(follow_lanes) > 0:
                ego_obj_lane_index = random.choice(follow_lanes)
                test_av_obj_lane_index = ego_obj_lane_index

        elif self.config['AtomicScene'] in ['ita_mode']:
            self.meeting_points = np.array(self.interaction_point_id[ego_lane_index][bv_lane_index])
            lane = self.road.network.get_lane(ego_lane_index + (0,))
            bv_lane = self.road.network.get_lane(bv_lane_index + (0,))
            ego_lon, _ = lane.local_coordinates(self.meeting_points)
            bv_lon, _ = bv_lane.local_coordinates(self.meeting_points)
            if ego_lon > bv_lon:
                ego_lon = (ego_lon - bv_lon)
                bv_lon = 0
            else:
                bv_lon = (bv_lon - ego_lon)
                ego_lon = 0

            ego_speed = bv_speed + random.randint(-2, 2)

        self.vehicle = ego_vehicle_type.make_on_lane(self.road, 'ego', (ego_lane_index[0], ego_lane_index[1], 0),
                                                     ego_lon,
                                                     speed=ego_speed)
        self.vehicle.color = (0, 255, 0)
        self.vehicle.plan_route_to(ego_obj_lane_index)
        self.road.vehicles.append(self.vehicle)

        vehicle = IDMVehicle.make_on_lane(self.road, 'bv', (bv_lane_index[0], bv_lane_index[1], 0), bv_lon,
                                          speed=bv_speed)

        vehicle.plan_route_to(test_av_obj_lane_index)
        vehicle.color = (0, 0, 255)
        self.road.vehicles.append(vehicle)
        self.controlled_vehicles.append(vehicle)
        self.road.set_ego_vehicles(self.controlled_vehicles)

        lane_indexs = [self.vehicle.lane_index]
        ego_left_lane = self.road.network.get_left_lane(lane_indexs[0])
        if ego_left_lane is not None:
            lane_indexs.append(ego_left_lane+(0,))

        ego_right_lane = self.road.network.get_right_lane(lane_indexs[0])
        if ego_right_lane is not None:
            lane_indexs.append(ego_right_lane+(0,))

        lane_indexs.append(self.ego_vehicle.lane_index)
        bv_left_lane = self.road.network.get_left_lane(self.ego_vehicle.lane_index)
        if bv_left_lane is not None:
            lane_indexs.append(bv_left_lane+(0,))

        bv_right_lane = self.road.network.get_right_lane(self.ego_vehicle.lane_index)
        if bv_right_lane is not None:
            lane_indexs.append(bv_right_lane+(0,))

        if self.config['obs_v1']:
            for v in range(self.config['vehicles_num']):
                lane_idx = random.choice(lane_indexs)
                lane = self.road.network.get_lane(lane_idx)
                veh = IDMVehicle.make_on_lane(
                    self.road, f'bv_{v}', lane_idx, random.choice(range(int(lane.length))),
                    speed=random.choice(range(8, 15))
                )
                if np.linalg.norm(self.vehicle.position - veh.position) > 60:
                    continue

                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - veh.position) < 10:
                        break
                else:
                    self.road.vehicles.append(veh)

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        if len(self.junction_edge_list) == 0:
            for edge_id in self.map.graph.edges:
                edge = self.map.graph.get_edge_data(*edge_id)
                if len(edge['path']) == 0:
                    continue
                sides_lanes = self.road.network.side_lanes(edge_id + (0,))
                if len(sides_lanes) > 0:
                    if len(edge['path']) < 10:
                        continue

                    self.have_sides_lanes[edge_id] = sides_lanes
                    self.have_sides_lanes_info[edge_id] = (edge['entry_waypoint'].road_id, edge['entry_waypoint'].section_id,
                                                           edge['entry_waypoint'].lane_id)

                if edge['entry_waypoint'].junction_id is not None:
                    self.junction_edge_list.add(edge_id)
                    self.junction_road_info[edge_id] = (
                    edge['entry_waypoint'].road_id, edge['entry_waypoint'].section_id, edge['entry_waypoint'].lane_id)

        for random_edge in list(self.junction_edge_list):
            if random_edge not in self.interaction_lane_id.keys():
                self.interaction_lane_id[random_edge] = []
                self.interaction_point_id[random_edge] = dict()
                ego_lane = self.road.network.get_lane(random_edge + (0,))
                ego_center = np.array(ego_lane.map_vector)[:, :2].tolist()
                ego_lane_width = ego_lane.map_vector[0][2] - 0.1
                for bv_edge_id in list(self.junction_edge_list):
                    if random_edge == bv_edge_id:
                        continue
                    bv_lane = self.road.network.get_lane(bv_edge_id + (0,))
                    bv_center = np.array(bv_lane.map_vector)[:, :2].tolist()
                    bv_lane_width = bv_lane.map_vector[0][2] - 0.1
                    if np.linalg.norm(np.array(ego_center[0]) - np.array(bv_center[0])) <= 1:
                        continue
                    intersection, intersection_center = check_lane_intersection(ego_center, ego_lane_width, bv_center, bv_lane_width)
                    if intersection:
                        self.interaction_lane_id[random_edge].append(bv_edge_id)
                        self.interaction_point_id[random_edge][bv_edge_id] = intersection_center[len(intersection_center) // 2]

        if self.config['AtomicScene'] == 'lon_mode':

            random_edge = random.choice(list(self.have_sides_lanes.keys()))
            bv_edge = random.choice(self.have_sides_lanes[random_edge])
            scenarios = {'ita_scene': {'ego_road_info': self.have_sides_lanes_info[random_edge],
                                       'test_av_road_info': self.have_sides_lanes_info[bv_edge[:-1]],
                                       'ego_obj_road_info': self.have_sides_lanes_info[random_edge],
                                       'test_av_obj_road_info': self.have_sides_lanes_info[random_edge]}}

        elif self.config['AtomicScene'] == 'lat_mode':
            random_edge = random.choice(list(self.have_sides_lanes.keys()))
            bv_edge = random.choice(self.have_sides_lanes[random_edge])
            scenarios = {'ita_scene': {'ego_road_info': self.have_sides_lanes_info[random_edge],
                                       'test_av_road_info': self.have_sides_lanes_info[bv_edge[:-1]],
                                       'ego_obj_road_info': self.have_sides_lanes_info[random_edge],
                                       'test_av_obj_road_info': self.have_sides_lanes_info[random_edge]}}
            if random.random() > 0.7:
                scenarios['ita_scene']['test_av_road_info'] = self.have_sides_lanes_info[random_edge]

        elif self.config['AtomicScene'] == 'ita_mode':
            if len(self.junction_edge_list) == 0:
                for edge_id in self.map.graph.edges:
                    edge = self.map.graph.get_edge_data(*edge_id)
                    if len(edge['path']) == 0:
                        continue

                    if edge['entry_waypoint'].junction_id is not None:
                        self.junction_edge_list.add(edge_id)
                        self.junction_road_info[edge_id] = (edge['entry_waypoint'].road_id, edge['entry_waypoint'].section_id, edge['entry_waypoint'].lane_id)

            random_edge = None
            while True:
                random_edge = random.choice(list(self.junction_edge_list))

                if len(self.interaction_lane_id[random_edge]) == 0:
                    self.interaction_lane_id.pop(random_edge)
                    self.interaction_point_id.pop(random_edge)
                    self.junction_edge_list.remove(random_edge)
                    continue
                break

            bv_random_edge = random.choice(self.interaction_lane_id[random_edge])
            self.meeting_points = np.array(self.interaction_point_id[random_edge][bv_random_edge])
            scenarios = {'ita_scene': {'ego_road_info': self.junction_road_info[random_edge], 'test_av_road_info': self.junction_road_info[bv_random_edge],
                              'ego_obj_road_info': self.junction_road_info[random_edge], 'test_av_obj_road_info': self.junction_road_info[bv_random_edge]}}
        scene_index = random.choice(list(scenarios.keys()))
        self.make_scenario(**scenarios[scene_index])

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

    def _clear_vehicles(self) -> None:
        """
        该函数用于清除道路上的车辆。
        """
        vehicles = []  # 创建一个空列表存储车辆
        remove_vehicle = []  # 创建一个空列表存储要删除的车辆
        for veh in self.road.vehicles:  # 遍历道路上的车辆
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