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
# import traci
from gym_sumo.envs.simulate_VTD import VTD_Manager
from math import atan2, degrees, pi
import os
Observation = TypeVar("Observation")


class RampEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    def __init__(self, config: dict = None, render_mode: Optional[str] = 'rgb_array' , open_vtd = False) -> None:
        super().__init__(config=config, render_mode=render_mode)

        self.vtd_manager = VTD_Manager(open_vtd)

        self.vtd_close_objects = {}

    @classmethod
    def default_config(cls) -> dict:
        ca = True
        xodr_path = None
        if ca:
            data_path = '/data/'
            for i in os.listdir(data_path):
                if i[-4:] == 'xodr':
                    # print(i)
                    xodr_path = os.path.join(data_path, i)
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
            'map_file_path':xodr_path,
            'ACTIONS': {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',
                        5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"},
            'AtomicScene': 'lon_mode',  # lon lat
            'vehicles_num': 20,
            'cut_in_min_lon_dis': 5,
        })
        return cfg

    def make_pre_data(self):
        change_lane_left, change_lane_right = self.road.network.can_change_lane(self.vehicle.lane_index)
        self.pre_data = dict()
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

        if self.config['AtomicScene'] == 'lat_mode':
            diff_pos = self.vehicle.position - self.ego_vehicle.position
            lon_offset = np.dot(diff_pos, [math.cos(self.ego_vehicle.heading), math.sin(self.ego_vehicle.heading)])
            flag = False if lon_offset < 1 or lon_offset > 6 else True
            if flag:
                reward += math.exp(-lon_offset / 5)
            else:
                reward += -(1 - math.exp(- np.linalg.norm(self.vehicle.position - self.ego_vehicle.position) / 20))

        elif self.config['AtomicScene'] == 'lon_mode':

            if ((self.config['ACTIONS'][int(action)] == "LANE_LEFT" and not self.pre_data["change_lane_left"]) or
                    (self.config['ACTIONS'][int(action)] == "LANE_RIGHT" and not self.pre_data[
                        "change_lane_right"])):
                reward += -0.5
            # 求被测主车前方车辆
            bv_front_vehicle, _ = self.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.target_lane_index)
            # 如果被测主车前方车为对抗车
            if bv_front_vehicle is self.vehicle:  # 计算与前车的ttc
                front_ttc = self.compute_ttc(self.vehicle, self.ego_vehicle, self.config['max_ttc_value'])
                if front_ttc < self.config['max_ttc_value']:
                    reward += math.exp(-front_ttc / 5)

            diff_pos = self.vehicle.position - self.ego_vehicle.position
            lon_offset = np.dot(diff_pos, [math.cos(self.ego_vehicle.heading), math.sin(self.ego_vehicle.heading)])
            coef = 0.7 if lon_offset <= 0 else 0.3
            reward += -(1 - math.exp(- np.linalg.norm(self.vehicle.position - self.ego_vehicle.position) / 20)) * coef

            if (action in [0, 2] and self.vehicle.target_lane_index == self.ego_vehicle.lane_index):
                cut_in_min_lon_dis = self.ego_vehicle.speed * 0.1 + 4.0
                if (0 < lon_offset < (cut_in_min_lon_dis +(self.ego_vehicle.LENGTH + self.vehicle.LENGTH) / 2)):
                    reward = -0.5
                else:
                    reward += (0.2 + math.exp(- abs(self.vehicle.speed - self.ego_vehicle.speed) / 5) * 0.3)

        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        # 如果被测主车前方车为对抗车
        if front_vehicle is not None:
            front_ttc = self.compute_ttc(front_vehicle, self.vehicle, self.config['max_ttc_value'])
            if front_ttc < 3:
                reward = -math.exp(-front_ttc / 5)

        if self.vehicle.crashed or not self.vehicle.on_road:  # 如果车辆碰撞或不在道路上，奖励为-1
            reward = -1

        if (any([self.ego_vehicle.crashed, not self.ego_vehicle.on_road]) and
                any([not self.vehicle.crashed,
                     all([8 >= action >= 5, bv_front_vehicle is self.vehicle])
                     ])
        ):
            reward = 2

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

        if self.config['make_road']:
            self.name = 'ego'
            self.map = GlobalRoutePlanner(self.config['map_file_path'])
            self._make_road()
            if self.vtd_manager.open_vtd:
                self._make_vehicles_VTD()
            else:
                self._make_vehicles()
            if self.config['AtomicScene'] == 'lat_mode':
                self.config['ACTIONS'] = {0: 'IDLE', 1: 'FASTER', 2: 'SLOWER'}
            else:
                self.config['duration'] = 300
        else:
            self.config['make_road'] = True
    def _make_vehicles_VTD(self ):
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        scenarios = {
            'block_on_ramp': {'ego_road_info': (2, 0, -4), 'test_av_road_info': (2, 0, -3), 'obj_road_info': (7, 0, -3)},
                     'block_on_ramp1': {'ego_road_info': (2, 0, -3), 'test_av_road_info': (2, 0, -2), 'obj_road_info': (7, 0, -3)},
                     'block_on_ramp2': {'ego_road_info': (2, 0, -2), 'test_av_road_info': (2, 0, -1), 'obj_road_info': (7, 0, -3)},
                     'block_off_ramp': {'ego_road_info': (67, 0, -3), 'test_av_road_info': (72, 0, -1), 'obj_road_info': (7, 0, -3)},
                     'lane_reduction': {'ego_road_info': (4, 1, -1), 'test_av_road_info': (4, 1, -2), 'obj_road_info': (7, 0, -3)},
                     }

        if self.config['AtomicScene'] == 'lon_mode':
            scenarios.update(
                {
                    'block_on_ramp_same_lane': {'ego_road_info': (2, 0, -3), 'test_av_road_info': (2, 0, -3), 'obj_road_info': (7, 0, -3)},
                    'block_on_ramp1_same_lane': {'ego_road_info': (2, 0, -2), 'test_av_road_info': (2, 0, -2), 'obj_road_info': (7, 0, -3)},
                    'block_on_ramp2_same_lane': {'ego_road_info': (2, 0, -1), 'test_av_road_info': (2, 0, -1), 'obj_road_info': (7, 0, -3)},
                    'block_off_ramp_same_lane': {'ego_road_info': (72, 0, -1), 'test_av_road_info': (72, 0, -1), 'obj_road_info': (7, 0, -3)},
                    'lane_reduction_same_lane': {'ego_road_info': (4, 1, -2), 'test_av_road_info': (4, 1, -2), 'obj_road_info': (7, 0, -3)},
                }
            )

        scene_index = random.choice(list(scenarios.keys()))

        obj_road_info =(7,0,-3)
        print(obj_road_info[0], obj_road_info[1], obj_road_info[2])

        ego_obj_lane_index = self.road.network.get_lane_index_by_road(obj_road_info[0], obj_road_info[1], obj_road_info[2])
        # adv
        if self.vtd_manager.gl.adv is not None:
            i = self.vtd_manager.gl.adv 
            self.vehicle = MDPVehicle(road=self.road,name=i.name,position=[i.pos_x, i.pos_y ],heading=i.pos_h , speed=math.sqrt(math.pow(i.vx,2) + math.pow( i.vy,2)), adv_id=i.id )
            self.vehicle.plan_route_to(ego_obj_lane_index)
            self.road.vehicles.append(self.vehicle)
        # ego
        # 探测到主车
        if self.vtd_manager.gl.ego is not None:
            if self.vtd_manager.gl.ego.name == 'Ego':
                vehicle = IDMVehicle(road=self.road,name=self.vtd_manager.gl.ego.name,position=[self.vtd_manager.gl.ego.pos_x, self.vtd_manager.gl.ego.pos_y],heading=self.vtd_manager.gl.ego.pos_h, speed=math.sqrt(math.pow(self.vtd_manager.gl.ego.vx,2) + math.pow( self.vtd_manager.gl.ego.vy, 2)))
        vehicle.plan_route_to(ego_obj_lane_index)
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
        # print("len(self.controlled_vehicles):",len(self.controlled_vehicles))
        # print(self.controlled_vehicles)
        # for v in range(self.config['vehicles_num']):
        #     lane_idx = random.choice(lane_indexs)
        #     lane = self.road.network.get_lane(lane_idx)
        #     veh = IDMVehicle.make_on_lane(
        #         self.road, f'bv_{v}', lane_idx, random.choice(range(int(lane.length))),
        #         speed=random.choice(range(8, 15))
        #     )

        #     for v in self.road.vehicles:
        #         if np.linalg.norm(v.position - veh.position) < 10:
        #             break
        #     else:
        #         self.road.vehicles.append(veh)


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

    def make_scenario(self, ego_road_info, test_av_road_info, obj_road_info):
        start_lon = random.choice(range(5, 10))
        ego_lon, bv_lon = random.choice(range(start_lon+5, start_lon+10)), start_lon

        ego_lane_index = self.road.network.get_lane_index_by_road(ego_road_info[0], ego_road_info[1], ego_road_info[2])
        ego_obj_lane_index = self.road.network.get_lane_index_by_road(obj_road_info[0], obj_road_info[1], obj_road_info[2])
        # adv
        self.vehicle = MDPVehicle.make_on_lane(self.road, 'ego', (ego_lane_index[0], ego_lane_index[1], 0), ego_lon,
                                               speed=random.choice(range(8, 15)))
        self.vehicle.plan_route_to(ego_obj_lane_index)
        self.road.vehicles.append(self.vehicle)
        # ego
        bv_lane_index = self.road.network.get_lane_index_by_road(test_av_road_info[0], test_av_road_info[1], test_av_road_info[2])
        vehicle = IDMVehicle.make_on_lane(self.road, 'bv', (bv_lane_index[0], bv_lane_index[1], 0), bv_lon,
                                          speed=random.choice(range(8, 15)))
        vehicle.plan_route_to(ego_obj_lane_index)
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

        for v in range(self.config['vehicles_num']):
            lane_idx = random.choice(lane_indexs)
            lane = self.road.network.get_lane(lane_idx)
            veh = IDMVehicle.make_on_lane(
                self.road, f'bv_{v}', lane_idx, random.choice(range(int(lane.length))),
                speed=random.choice(range(8, 15))
            )

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
        scenarios = {
            'block_on_ramp': {'ego_road_info': (2, 0, -4), 'test_av_road_info': (2, 0, -3), 'obj_road_info': (7, 0, -3)},
                     'block_on_ramp1': {'ego_road_info': (2, 0, -3), 'test_av_road_info': (2, 0, -2), 'obj_road_info': (7, 0, -3)},
                     'block_on_ramp2': {'ego_road_info': (2, 0, -2), 'test_av_road_info': (2, 0, -1), 'obj_road_info': (7, 0, -3)},
                     'block_off_ramp': {'ego_road_info': (67, 0, -3), 'test_av_road_info': (72, 0, -1), 'obj_road_info': (7, 0, -3)},
                     'lane_reduction': {'ego_road_info': (4, 1, -1), 'test_av_road_info': (4, 1, -2), 'obj_road_info': (7, 0, -3)},
                     }

        if self.config['AtomicScene'] == 'lon_mode':
            scenarios.update(
                {
                    'block_on_ramp_same_lane': {'ego_road_info': (2, 0, -3), 'test_av_road_info': (2, 0, -3), 'obj_road_info': (7, 0, -3)},
                    'block_on_ramp1_same_lane': {'ego_road_info': (2, 0, -2), 'test_av_road_info': (2, 0, -2), 'obj_road_info': (7, 0, -3)},
                    'block_on_ramp2_same_lane': {'ego_road_info': (2, 0, -1), 'test_av_road_info': (2, 0, -1), 'obj_road_info': (7, 0, -3)},
                    'block_off_ramp_same_lane': {'ego_road_info': (72, 0, -1), 'test_av_road_info': (72, 0, -1), 'obj_road_info': (7, 0, -3)},
                    'lane_reduction_same_lane': {'ego_road_info': (4, 1, -2), 'test_av_road_info': (4, 1, -2), 'obj_road_info': (7, 0, -3)},
                }
            )

        scene_index = random.choice(list(scenarios.keys()))
        self.make_scenario(**scenarios[scene_index])
    def run_vtd(self,action):
        self.vtd_manager.run(action)
    def _simulate_vtd(self,action):
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            action = int(action[0]) if isinstance(action, np.ndarray) else action
            action = self.config['ACTIONS'][action]
            self.make_pre_data()
            # Forward action to the vehicle

            self.road.act()
            self.vehicle.act(action)

            # vtd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # vtd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # vtd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # result = self.vehicle.action
            # # print("result:",result)
            # # if abs(result['acceleration']) > 2:
            # if result['acceleration'] > 3:
            #     result['acceleration'] = 3
            # elif  result['acceleration'] < -3:
            #     result['acceleration'] < -3
            # result.update({"adv_id":self.vehicle.adv_id})
            # print("result:",result)
            self.run_vtd( action )
            ego = {"position":np.array([self.vtd_manager.gl.ego.pos_x, self.vtd_manager.gl.ego.pos_y]), "speed":math.sqrt(math.pow(self.vtd_manager.gl.ego.vx,2) + math.pow( self.vtd_manager.gl.ego.vy,2)) ,"heading":self.vtd_manager.gl.ego.pos_h}

            if self.vtd_manager.gl.adv is not None:
                # self.vtd_manager.gl.adv.show()
                adv = {"id":self.vtd_manager.gl.adv.id,"position":np.array([self.vtd_manager.gl.adv.pos_x , self.vtd_manager.gl.adv.pos_y ] ), "speed":math.sqrt(math.pow(self.vtd_manager.gl.adv.vx,2) + math.pow( self.vtd_manager.gl.adv.vy,2)) ,"heading":self.vtd_manager.gl.adv.pos_h}
                # print("dqn ego :",ego)
                #                 # print("dqn adv:",adv)
                self.road.step_vtd(1 / self.config["simulation_frequency"],ego, adv) #TODO vtd info send to highway

            # self.road.step(1 / self.config["simulation_frequency"],ego,adv)

            # vtd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # vtd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # vtd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                self.run_steps += 1
                self._clear_vehicles()
                # Automatically render intermediate simulation steps if a viewer has been launched
                # Ignored if the rendering is done offscreen
                if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                    self._automatic_rendering()
                return True
            return False

        self.enable_auto_render = False



    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            action = int(action[0]) if isinstance(action, np.ndarray) else action
            action = self.config['ACTIONS'][action]
            self.make_pre_data()
            # Forward action to the vehicle

            self.road.act()
            self.vehicle.act(action)
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