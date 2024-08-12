import os.path

import carla
import time
import py_trees
import random
import numpy as np
# from scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
#     AtomicBehavior,
# )
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior,
)
# from scenario_runner.srunner.tools.scenario_helper import (
#     get_same_dir_lanes,
#     get_opposite_dir_lanes,
# )
from srunner.tools.scenario_helper import (
    get_same_dir_lanes,
    get_opposite_dir_lanes,
)
from srunner.scenariomanager.carla_data_provider import (
    CarlaDataProvider,
)
import logging
from collections import defaultdict
from srunner.scenariomanager.actorcontrols.npc_vehicle_control import NpcVehicleControl
path = '/home/zjx/work/BehaviorTree1/scenario_runner0915/vtd_adv_lib/gym_sumo'

import sys
from lxml import etree
sys.path.insert(0,path)
from gym_sumo.opendrive_parse.parser import parse_opendrive
logger = logging.getLogger(__name__)

from enum import IntEnum


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def draw_waypoints(world, waypoints, vertical_shift, deplay_color=1):
    """
    Draw a list of waypoints at a certain height given in vertical_shift.
    """
    for w in waypoints:
        wp = w.transform.location + carla.Location(z=vertical_shift)
        if deplay_color == 1:
            color = carla.Color(0, 255, 0)  # Green
        if deplay_color == 2:
            color = carla.Color(255, 255, 0)
        # if w[1] == RoadOption.LEFT:  # Yellow
        #     color = carla.Color(255, 255, 0)
        # elif w[1] == RoadOption.RIGHT:  # Cyan
        #     color = carla.Color(0, 255, 255)
        # elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
        #     color = carla.Color(255, 64, 0)
        # elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
        #     color = carla.Color(0, 64, 255)
        # elif w[1] == RoadOption.STRAIGHT:  # Gray
        #     color = carla.Color(128, 128, 128)
        # else:  # LANEFOLLOW
        #     color = carla.Color(0, 255, 0)  # Green

        world.debug.draw_point(wp, size=0.1, color=color, life_time=100)

    # world.debug.draw_point(waypoints[0][0].transform.location + carla.Location(z=vertical_shift), size=0.5,
    #                        color=carla.Color(0, 0, 255), life_time=10)
    #
    # world.debug.draw_point(waypoints[-1][0].transform.location + carla.Location(z=vertical_shift), size=0.5,
    #                        color=carla.Color(255, 0, 0), life_time=10)


CARLA_TYPE_TO_WALKER = {
    "pedestrian": [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.0004",
        "walker.pedestrian.0005",
        "walker.pedestrian.0006",
        "walker.pedestrian.0007",
        "walker.pedestrian.0008",
        "walker.pedestrian.0009",
        "walker.pedestrian.0010",

    ]
}

# "vehicle.audi.a2",
# "vehicle.audi.tt",
# "vehicle.jeep.wrangler_rubicon",
# "vehicle.chevrolet.impala",
# "vehicle.bmw.grandtourer",
# "vehicle.citroen.c3",
# "vehicle.seat.leon",
# "vehicle.nissan.patrol",
# "vehicle.nissan.micra",
# "vehicle.audi.etron",
# "vehicle.toyota.prius",
# "vehicle.tesla.model3",
# "vehicle.tesla.cybertruck",

# EGO_ROAD = 'road'
CARLA_TYPE_TO_VEHICLE = {
    "car": [
        "vehicle.audi.a2",
        "vehicle.audi.tt",
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.chevrolet.impala",
        "vehicle.bmw.grandtourer",
        "vehicle.citroen.c3",
        "vehicle.seat.leon",
        "vehicle.nissan.patrol",
        "vehicle.nissan.micra",
        "vehicle.audi.etron",
        "vehicle.toyota.prius",
        "vehicle.tesla.model3",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mini.cooper_s"

    ],
    "van": ["vehicle.volkswagen.t2_2021", "vehicle.volkswagen.t2_2021"],
    "truck": ["vehicle.tesla.cybertruck", "vehicle.carlamotors.carlacola", "vehicle.synkrotron.box_truck",
              "vehicle.mercedes.sprinter", ],
    'trailer': [],
    'semitrailer': [],
    'bus': [],
    "motorbike": [
        "vehicle.toyota.prius",
        "vehicle.tesla.model3",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mini.cooper_s"
    ],
    "bicycle": [
        "vehicle.toyota.prius",
        "vehicle.tesla.model3",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mini.cooper_s"
    ],
    'special_vehicles': [
        "vehicle.ford.ambulance"
    ],
}


# 摩托车 控制会有滑移等现象 暂时屏蔽，替换为car类型
# "vehicle.harley-davidson.low_rider",
# "vehicle.kawasaki.ninja",
# "vehicle.yamaha.yzf",
# "vehicle.bh.crossbike",
# "vehicle.diamondback.century",
# "vehicle.gazelle.omafiets",

# "vehicle.harley-davidson.low_rider",
# "vehicle.kawasaki.ninja",
# "vehicle.yamaha.yzf",


class OasisTrafficflow(AtomicBehavior):
    """
    Handles the background activity
    """

    def __init__(self, ego_actor, tf_param=None, debug=False, name="OasisTrafficflow"):
        """
        Setup class members
        """
        super(OasisTrafficflow, self).__init__(name)
        self.debug = debug
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        blueprint_library = self._world.get_blueprint_library()
        self._tm_port = CarlaDataProvider.get_traffic_manager_port()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(self._tm_port)
        self.client = CarlaDataProvider.get_client()
        # 预期速度与当前限制速度之间的百分比差。
        self._rng = CarlaDataProvider.get_random_seed()
        self._attribute_filter = None

        # Global variables
        self._ego_actor = ego_actor
        self._actors_speed_perc = {}  # Dictionary actor - percentage
        self._lane_width_threshold = (
            2.25  # Used to stop some behaviors at narrow lanes to avoid problems [m]
        )
        self._spawn_vertical_shift = 0.2
        self._fake_junction_ids = []
        self._road_front_vehicles = 2  # Amount of vehicles in front of the ego
        self._road_back_vehicles = 2  # Amount of vehicles behind the ego
        self._road_spawn_dist = 15  # Distance between spawned vehicles [m]
        self.frame = 1
        self._vehicle_list = []
        self._destroy_list = []
        self.centralObject = tf_param['centralObject']
        self.semiMajorAxis = int(tf_param['semiMajorAxis'])
        # self.semiMinorAxis = tf_param['semiMinorAxis']
        self.innerRadius = int(tf_param['innerRadius'])
        self.numberOfVehicles = int(tf_param['numberOfVehicles'])
        self.numberOfPedestrian = int(tf_param['numberOfPedestrian'])
        self.trafficDistribution = tf_param['trafficDistribution']
        self.directionOfTravelDistribution = tf_param['directionOfTravelDistribution']
        self.same = self.directionOfTravelDistribution['same'] * 0.01
        self.opposite = self.directionOfTravelDistribution['opposite'] * 0.01
        self.drivingModel = tf_param['drivingModel']
        # self.controllerType = tf_param['drivingModel']
        # self.controllerDistribution = tf_param['controllerDistribution']
        # Initialisation values
        if self.drivingModel['controllerType'] == 'Cooperative':
            self._vehicle_lane_change = False
            self._vehicle_lights = False
            self._vehicle_leading_distance = 20
            self._vehicle_offset = 0.1
        else:
            self._vehicle_lane_change = True
            self._vehicle_lights = False
            self._vehicle_leading_distance = 10
            self._vehicle_offset = 0.5
        # 车辆与生成半径约束关系
        self.max_vecs = (
            int(self.semiMajorAxis * 0.2)
            if self.numberOfVehicles > int(self.semiMajorAxis * 0.15)
            else self.numberOfVehicles
        )
        self.vehicles_ratio = [
            int(tf_param["trafficDistribution"][t])
            for t in list(CARLA_TYPE_TO_VEHICLE.keys())
        ]
        self.vehicles_ratio = [
            ratio / sum(self.vehicles_ratio) for ratio in self.vehicles_ratio
        ]
        self.vehicle_models_list = []
        if self.debug:
            logger.info(f"vehicles_ratio:{self.vehicles_ratio}")
            logger.info(f"vehicle_models:{self.vehicle_models}")
            logger.info(f"tf_param:{tf_param}")
        # 前边界
        self.front_traffic_bound = 0
        # 反向车道前边界
        self.front_traffic_bound_opp = 0
        # 后边界
        self.back_traffic_bound = 0
        # 反向车道后边界
        self.back_traffic_bound_opp = 0
        self.apll_spawn_points = self._world.get_map().get_spawn_points()
        # tm预期速度
        self._tm.global_percentage_speed_difference(-20)
        # 断头路销毁车辆距离
        self.dist2endway = 150
        # self.initialise()
        # 交通流driver模式，现有两种，1. tm控制，只对速度，异常情况做一些干预，2. default driver 还在开发中
        self.tm_autopilot = True
        # 默认carla内置地图用tm控制，处理速度快
        self.istown = False
        if self._map.name.split('/')[-1].find('Town') != -1:
            self.istown = True
            self.tm_autopilot = True
        # if self.tm_autopilot:
        #     self._tm.set_hybrid_physics_mode(True)
        #     self._tm.set_hybrid_physics_radius(100)
        # 对抗代码，对抗默认用tm
        self.adversarialModelEnable = False
        if tf_param['adversarialModel']['adversarialModelEnable']:
            self.tm_autopilot = True
            self.adversarialModelEnable = True
        # default driver
        self.traffic_flow_vecs_control = {}
        # 同车道车辆数量，用于平衡车辆分布
        self.same_road_vecs_num = 0
        # weather = carla.WeatherParameters(
        #     cloudiness=80.0,
        #     precipitation=30.0,
        #     sun_altitude_angle=60.0)
        #
        # self._world.set_weather(weather)
        self.sun_altitude_angle = self._world.get_weather().sun_altitude_angle
        if self.sun_altitude_angle < 20:
            self._vehicle_lights = True
        self.traffic_flow_nums = 0

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        # calculate fake junctions
        self._calculate_fake_junctions(self.debug)
        # 获取主车初始位置
        # ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
        # same_dir_wps = get_same_dir_lanes(ego_wp)
        # opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
        # # # 初始化辆车
        # self._initialise_road_behavior(ego_wp, same_dir_wps + opposite_dir_wps)

    # 计算waypoint与车辆location的差距

    # 车辆测试，无交通流生成逻辑
    def update1(self):
        logger.info("--------------------------------------start111222")
        # return py_trees.common.Status.RUNNING
        self.max_vecs = 20
        self._destroy_list = []
        destroy_indexs = []
        for vec in self._vehicle_list:
            # 断头路处理
            vec_wp = self._map.get_waypoint(
                vec.get_location()
            )
            ahead_wp = vec_wp.next(self.dist2endway)
            if len(ahead_wp) == 0:
                logger.info(f"---------------vec{vec.id} has no way!!!")
                destroy_indexs.append(vec.id)
                self._destroy_list.append(
                    carla.command.DestroyActor(vec)
                )
                continue
            # 离路处理
            wp2loc_dist = self.cal_dis_wp2loc(vec)
            if wp2loc_dist > 5:
                destroy_indexs.append(vec.id)
                self._destroy_list.append(
                    carla.command.DestroyActor(vec)
                )
                continue
            logger.info(f"===============vec location:{vec.get_location()}")
        logger.info("--------------------------------------end")
        if len(self._destroy_list) > 0:
            self.client.apply_batch(self._destroy_list)
            self._vehicle_list = list(
                filter(lambda x: x.id not in destroy_indexs, self._vehicle_list)
            )
        autopilot = True
        self.frame += 1
        logger.info(f"---------------------len(self._vehicle_list):{len(self._vehicle_list)}")

        ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
        if len(self._vehicle_list) < self.max_vecs:
            same_dir_wps = get_same_dir_lanes(ego_wp)
            logger.info(f"---------------------len(same_dir_wps):{len(same_dir_wps)}")
            # same_dir_wps = []
            opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
            logger.info(f"---------------------len(opposite_dir_wps):{len(opposite_dir_wps)}")
            opposite_dir_wps = []
            self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, autopilot)
            for i in self._vehicle_list:
                self._tm.set_desired_speed(i, 50)
        if not autopilot:
            if self.frame > 30:
                for vec in self._vehicle_list:
                    # if np.sqrt(
                    #         vec.get_velocity().x * vec.get_velocity().x + vec.get_velocity().y * vec.get_velocity().y) < 1:
                    if vec not in self.traffic_flow_vecs_control:
                        args = {"desired_velocity": 10, "desired_acceleration": 5, "emergency_param": 0.4,
                                "desired_deceleration": 5, "safety_time": 5,
                                "lane_changing_dynamic": True, "urge_to_overtake": True, "obey_traffic_lights": True,
                                "identify_object": True, "obey_speed_limit": True
                                }
                        self.traffic_flow_vecs_control[vec] = NpcVehicleControl(vec, args)
                        self.traffic_flow_vecs_control[vec].run_step()
                    else:
                        self.traffic_flow_vecs_control[vec].run_step()
        return py_trees.common.Status.RUNNING

    # 交通流主程序
    def update(self):
        # print("---------------------")
        # try:
        if True:
            flag = True
            logger.info(f"start traffic flow update!!!")
            ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
            ahead_ego_wp = ego_wp.next(self.dist2endway + 20)

            self.frame += 1
            destroy_indexs = []
            # 临时变量
            front_tmpmax = 0
            opp_front_tmpmax = 0
            bake_tmpmax = 0
            opp_bake_tmpmax = 0
            self.same_road_vecs_num = 0

            for i in range(len(self._vehicle_list)):
                # ego车与目标车的距离
                dist = (self._vehicle_list[i].get_location().distance(ego_wp.transform.location))
                wp2loc_dist = self.cal_dis_wp2loc(self._vehicle_list[i])

                vec_wp = self._map.get_waypoint(
                    self._vehicle_list[i].get_location()
                )
                # 断头路处理
                ahead_wp = vec_wp.next(self.dist2endway)
                if len(ahead_wp) == 0:
                    destroy_indexs.append(self._vehicle_list[i].id)
                    self._destroy_list.append(
                        carla.command.DestroyActor(self._vehicle_list[i])
                    )
                    continue
                if self.same == 0:
                    if ego_wp.road_id == vec_wp.road_id and ego_wp.lane_id * vec_wp.lane_id > 0:
                        destroy_indexs.append(self._vehicle_list[i].id)
                        self._destroy_list.append(
                            carla.command.DestroyActor(self._vehicle_list[i])
                        )
                        continue

                # 如果车辆使出道路，删除处理
                if wp2loc_dist > 5:
                    destroy_indexs.append(self._vehicle_list[i].id)
                    self._destroy_list.append(
                        carla.command.DestroyActor(self._vehicle_list[i])
                    )
                    continue
                # 侧翻车辆，删除处理
                if abs(self._vehicle_list[i].get_transform().rotation.roll) > 40:
                    destroy_indexs.append(self._vehicle_list[i].id)
                    self._destroy_list.append(
                        carla.command.DestroyActor(self._vehicle_list[i])
                    )
                    continue
                # 如果车辆与给定坐标的距离大于半径
                relative_dis = self.get_local_location(
                    self._ego_actor, self._vehicle_list[i].get_location()
                )
                # 对抗模式下，主车后方车辆直接删除
                if self.adversarialModelEnable:
                    if relative_dis.x < -50 or relative_dis.x > 120:
                        destroy_indexs.append(self._vehicle_list[i].id)
                        self._destroy_list.append(
                            carla.command.DestroyActor(self._vehicle_list[i])
                        )
                        continue

                destroy_range = self.semiMajorAxis * 1.1
                if relative_dis.x < 0:
                    destroy_range *= 1.5
                if relative_dis.x > 0:
                    destroy_range += self.get_speed() * 2

                if (
                        dist > destroy_range
                ):
                    destroy_indexs.append(self._vehicle_list[i].id)
                    self._destroy_list.append(
                        carla.command.DestroyActor(self._vehicle_list[i])
                    )

                # 更新前后距离
                if (
                        relative_dis.x
                        > 0
                        and dist > front_tmpmax
                        and ego_wp.road_id == vec_wp.road_id
                        and ego_wp.lane_id * vec_wp.lane_id > 0
                ):
                    front_tmpmax = dist
                elif (
                        relative_dis.x
                        > 0
                        and dist > opp_front_tmpmax
                        and ego_wp.road_id == vec_wp.road_id
                        and ego_wp.lane_id * vec_wp.lane_id < 0
                ):
                    opp_front_tmpmax = dist
                elif (
                        relative_dis.x
                        < 0
                        and dist > bake_tmpmax
                        and ego_wp.road_id == vec_wp.road_id
                        and ego_wp.lane_id * vec_wp.lane_id > 0
                ):
                    bake_tmpmax = dist
                elif (
                        relative_dis.x
                        < 0
                        and dist > opp_bake_tmpmax
                        and ego_wp.road_id == vec_wp.road_id
                        and ego_wp.lane_id * vec_wp.lane_id < 0
                ):
                    opp_bake_tmpmax = dist
                if ego_wp.road_id == vec_wp.road_id and ego_wp.lane_id * vec_wp.lane_id > 0:
                    self.same_road_vecs_num += 1
            self.front_traffic_bound = self.semiMajorAxis if front_tmpmax > self.semiMajorAxis else front_tmpmax
            self.front_traffic_bound_opp = self.semiMajorAxis if opp_front_tmpmax > self.semiMajorAxis else opp_front_tmpmax
            self.back_traffic_bound = self.semiMajorAxis if bake_tmpmax > self.semiMajorAxis else bake_tmpmax
            self.back_traffic_bound_opp = self.semiMajorAxis if opp_bake_tmpmax > self.semiMajorAxis else opp_bake_tmpmax
            if len(destroy_indexs) > 0:
                self.client.apply_batch(self._destroy_list)
                self._vehicle_list = list(
                    filter(lambda x: x.id not in destroy_indexs, self._vehicle_list)
                )
                if not self.tm_autopilot:
                    self.traffic_flow_vecs_control = {key: val for key, val in self.traffic_flow_vecs_control.items() if
                                                      key not in destroy_indexs}

            logger.info(f"front_traffic_bound:{self.front_traffic_bound}")
            logger.info(f"front_traffic_bound_opp:{self.front_traffic_bound_opp}")
            logger.info(f"back_traffic_bound:{self.back_traffic_bound}")
            logger.info(f"back_traffic_bound_opp:{self.back_traffic_bound_opp}")

            # 补充车辆
            logger.info(
                f"len vehicle list:{len(self._vehicle_list)}, self.same_road_vecs_num:{self.same_road_vecs_num}")

            if self.frame % 20 == 0:
                same_dir_wps = []
                if self.adversarialModelEnable:
                    if self.same_road_vecs_num < 1:
                        adv_same_dir_wps = get_same_dir_lanes(ego_wp)
                        self._add_adv_vecs(ego_wp, adv_same_dir_wps, self.tm_autopilot)
                elif len(ahead_ego_wp) != 0 and len(self._vehicle_list) < self.max_vecs:
                    if self.same_road_vecs_num < self.same * self.numberOfVehicles:
                        same_dir_wps = get_same_dir_lanes(ego_wp)
                    else:
                        same_dir_wps = []
                opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
                self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, self.tm_autopilot)
            # driver 模式
            if not self.tm_autopilot:
                for vec in self._vehicle_list:
                    if vec.is_alive:
                        if vec.id not in self.traffic_flow_vecs_control:
                            args = {"desired_velocity": 10, "desired_acceleration": 5, "emergency_param": 0.4,
                                    "desired_deceleration": 5, "safety_time": 5,
                                    "lane_changing_dynamic": True, "urge_to_overtake": True,
                                    "obey_traffic_lights": True,
                                    "identify_object": True, "obey_speed_limit": True
                                    }
                            self.traffic_flow_vecs_control[vec.id] = NpcVehicleControl(vec, args)
                            self.traffic_flow_vecs_control[vec.id].run_step()
                        else:
                            self.traffic_flow_vecs_control[vec.id].run_step()

            logger.info(f"end of traffic flow update!!!")
            # 车辆速度规划
            if self.tm_autopilot:
                for i in range(len(self._vehicle_list)):
                    wp2loc_dist = self.cal_dis_wp2loc(self._vehicle_list[i])
                    vec_wp = self._map.get_waypoint(
                        self._vehicle_list[i].get_location()
                    )
                    self.set_speed(ego_wp, vec_wp, i, wp2loc_dist)
            return py_trees.common.Status.RUNNING
        # except Exception:
        #     logger.error(
        #         f"===============================An error occurred while update Oasis Traffic Flow!!!!=================================="
        #     )
        #     return py_trees.common.Status.FAILURE

    def get_filter_points(self, ego_vec_road_id):
        spawn_points_filtered = []
        num = 0
        for i, around_spawn_point in enumerate(
                self.apll_spawn_points
        ):  # 遍历所有出生点
            tmp_wpt = self._map.get_waypoint(around_spawn_point.location)
            diff_road = around_spawn_point.location.distance(
                self._ego_actor.get_location()
            )
            # 如果出生点与给定坐标的距离小于半径
            if (
                    diff_road < self.semiMajorAxis
                    and diff_road > self.innerRadius * 3
                    and self._map.get_waypoint(around_spawn_point.location).road_id
                    != ego_vec_road_id
            ):
                if num < abs(self.max_vecs - len(self._vehicle_list)):
                    num += 1
                    spawn_points_filtered.append(
                        tmp_wpt
                    )  # 将出生点添加到过滤后的列表中
                else:
                    break
        return spawn_points_filtered

    def _add_adv_vecs(self, ego_wp, same_dir_wps, tm_autopilot=True):
        adv_spawn_wps = []
        for wp in same_dir_wps:
            if ego_wp.lane_id == wp.lane_id:
                continue
            # left 100
            for dis in [30, 50, 60, 80, 100]:
                next = wp.next(dis)
                if len(next) == 1:
                    dist = next[0].transform.location.distance(
                        self._ego_actor.get_location()
                    )
                    if dist > self.semiMajorAxis:
                        continue
                    if not self._check_junction_spawnable(next[0]):
                        continue  # Stop when there's no next or found a junction

                    adv_spawn_wps.append(wp.next(dis))
        random.shuffle(adv_spawn_wps)
        self._spawn_actors(adv_spawn_wps[0], tm_autopilot=tm_autopilot, ego_dist=self.innerRadius)

    def _add_road_vecs(self, ego_wp, same_dir_wps, opposite_dir_wps, tm_autopilot=True):
        '''筛选出生点并生成车辆'''
        spawn_wps = []
        adv_spawn_wps = []
        # offset_var = self.semiMajorAxis * 0.1 if self.semiMajorAxis * 0.1 > 15 else 15
        offset_var = 0
        # 同向车道出生点筛选
        speed_dist = self.get_speed()

        for wp in same_dir_wps:
            if self.numberOfVehicles * self.same <= 0:
                break
            same_num = int(self.numberOfVehicles * self.same / len(same_dir_wps))
            if same_num < 1 and self.numberOfVehicles * self.same > 0:
                same_num = 1

            innerboundarywp = wp.next(
                self.innerRadius + 1 if self.innerRadius > self.front_traffic_bound * 0.7 else self.front_traffic_bound * 0.7)
            if len(innerboundarywp) == 0:
                continue
            temp_next_wps = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            # 控制生成车辆车距
            offset = 0
            for i in range(same_num):
                # logger.info(f"-------i----------:{i}")
                if i != 0:
                    temp_next_wps = temp_next_wps.next(
                        self._road_spawn_dist * 2
                        + random.randint(-3, 3) * 2
                        + speed_dist * 3
                        + offset
                    )
                # self._road_spawn_dist = 15
                offset += offset_var
                if len(temp_next_wps) <= 0:
                    break
                temp_next_wps = temp_next_wps[random.randint(0, len(temp_next_wps) - 1)]

                dist = temp_next_wps.transform.location.distance(
                    self._ego_actor.get_location()
                )

                if dist > self.semiMajorAxis + speed_dist * 2:
                    continue
                if not self._check_junction_spawnable(temp_next_wps):
                    continue  # Stop when there's no next or found a junction
                if i != 0:
                    spawn_wps.insert(0, temp_next_wps)

            # self.back_traffic_bound
            innerboundarywp = wp.previous(
                self.innerRadius + 1 if self.innerRadius > self.back_traffic_bound / 2 else self.back_traffic_bound / 2)
            # wp.next(self.innerRadius + 1   if self.innerRadius > self.front_traffic_bound else self.front_traffic_bound)
            if len(innerboundarywp) <= 0:
                continue
            temp_prev_wps = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            # spawn_wps.insert(0, prev_wp_queue[0])
            offset = 0
            for i in range(same_num):
                if i != 0:
                    temp_prev_wps = temp_prev_wps.previous(
                        self._road_spawn_dist
                        + random.randint(-3, 3) * 3
                        + speed_dist * 1
                    )
                offset += offset_var
                if len(temp_prev_wps) <= 0:
                    break
                temp_prev_wps = temp_prev_wps[random.randint(0, len(temp_prev_wps) - 1)]

                dist = temp_prev_wps.transform.location.distance(
                    self._ego_actor.get_location()
                )
                if dist > self.semiMajorAxis:
                    continue
                if not self._check_junction_spawnable(temp_prev_wps):
                    continue  # Stop when there's no next or found a junction
                if i != 0:
                    spawn_wps.append(temp_prev_wps)

        # 反向车道出生点筛选
        opp_spawn_wps = []
        for wp in opposite_dir_wps:
            opposite_num = int(self.numberOfVehicles * self.opposite / len(opposite_dir_wps))
            if opposite_num < 1 and self.numberOfVehicles * self.opposite > 0:
                opposite_num = 1
            elif self.numberOfVehicles * self.opposite <= 0:
                break
            innerboundarywp = wp.previous(
                self.innerRadius + 1 if self.innerRadius > self.front_traffic_bound_opp else self.front_traffic_bound_opp)
            if len(innerboundarywp) <= 0:
                continue

            temp_prev_wps = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            # opp_spawn_wps.insert(0, prev_wp_queue[0])
            # for _ in range(self._road_back_vehicles):
            offset = 0
            for i in range(opposite_num):
                if i != 0:
                    temp_prev_wps = temp_prev_wps.previous(
                        self._road_spawn_dist * 4
                        + random.randint(-4, 4) * 12
                    )
                offset += offset_var
                if len(temp_prev_wps) <= 0:
                    break
                temp_prev_wps = temp_prev_wps[random.randint(0, len(temp_prev_wps) - 1)]

                dist = temp_prev_wps.transform.location.distance(
                    self._ego_actor.get_location()
                )
                if dist > self.semiMajorAxis + speed_dist * 2:
                    continue
                if not self._check_junction_spawnable(temp_prev_wps):
                    continue  # Stop when there's no next or found a junction
                if i != 0:
                    opp_spawn_wps.append(temp_prev_wps)

        # if len(spawn_wps) >0:
        #     print("===================len of spawn_wps:====", len(spawn_wps) )
        if len(spawn_wps) > 0 or len(opp_spawn_wps) > 0:
            random.shuffle(spawn_wps)
            random.shuffle(opp_spawn_wps)

            gl_spawn_wps = []
            gap_nums = self.max_vecs - len(self._vehicle_list)
            if gap_nums > 10:
                gap_nums = 10
            if self.adversarialModelEnable:
                spawn_wps = spawn_wps[:1]
            elif len(spawn_wps) > int(gap_nums * self.same * 1.1):
                spawn_wps = spawn_wps[:int(gap_nums * self.same * 1.1) + 1 if int(gap_nums * self.same * 1.1) + 1 < len(
                    spawn_wps) else -1]

            if len(opp_spawn_wps) > int(gap_nums * self.opposite):
                opp_spawn_wps = opp_spawn_wps[
                                :int(gap_nums * self.opposite) + 1 if int(gap_nums * self.opposite) + 1 < len(
                                    opp_spawn_wps) else -1]

            gl_spawn_wps += spawn_wps + opp_spawn_wps
            # 同方向与反方向补充出生点小于最大车辆数
            if len(self._vehicle_list) + len(gl_spawn_wps) <= self.max_vecs:
                gap = self.max_vecs - (len(self._vehicle_list) + len(gl_spawn_wps))
                if gap > 10:
                    gap = 10
                spawn_points_filtered = self.get_filter_points(ego_wp.road_id)
                # 补充出生点大于0
                if len(spawn_points_filtered) > 0:
                    gl_spawn_wps += spawn_points_filtered[
                                    : len(spawn_points_filtered) - 1 if gap > len(spawn_points_filtered) else gap - 1
                                    ]
            gl_able_spawn_wps = []
            for index in range(len(gl_spawn_wps)):
                # 检查出生点前方是否为断头路
                if len(gl_spawn_wps[index].next(self.dist2endway)) > 0:
                    gl_able_spawn_wps.append(gl_spawn_wps[index])
            show = False
            if show:
                draw_waypoints(self._world, gl_able_spawn_wps, vertical_shift=5,
                               )
            self._spawn_actors(gl_able_spawn_wps, tm_autopilot=tm_autopilot, ego_dist=self.innerRadius)

    # def _initialise_road_behavior(self, ego_wp, road_wps, rdm=False):
    #     """
    #     Initialises the road behavior, consisting on several vehicle in front of the ego,
    #     and several on the back and are only spawned outside junctions.
    #     If there aren't enough actors behind, road sources will be created that will do so later on
    #     """
    #     # Vehicles in front
    #     spawn_wps = []
    #     for wp in road_wps:
    #         # Front spawn points
    #         innerboundarywp = wp.next(self.innerRadius + 1)
    #         if len(innerboundarywp) <= 0:
    #             continue
    #         next_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
    #         spawn_wps.insert(0, next_wp_queue[0])
    #         for _ in range(self._road_front_vehicles):
    #             temp_next_wp_queue = []
    #             for temp_wp in next_wp_queue:
    #                 # 获取 wp
    #                 temp_next_wps = temp_wp.next(self._road_spawn_dist + random.randint(0, 10))
    #                 num_wps = len(temp_next_wps)
    #                 if num_wps <= 0:
    #                     continue
    #                 # 前方发现多个waypoint 随机抽取一个
    #                 elif num_wps > 1:
    #                     temp_next_wp = temp_next_wps[random.randint(0, num_wps - 1)]
    #                 else:
    #                     temp_next_wp = temp_next_wps[0]
    #                 # 超出限定范围丢弃
    #                 dist = temp_next_wp.transform.location.distance(wp.transform.location)
    #                 if dist > self.semiMajorAxis:
    #                     continue
    #                 if not self._check_junction_spawnable(temp_next_wp):
    #                     continue  # Stop when there's no next or found a junction
    #                 temp_next_wp_queue.append(temp_next_wp)
    #                 spawn_wps.insert(0, temp_next_wp)
    #             next_wp_queue = temp_next_wp_queue

    #         innerboundarywp = wp.previous(self.innerRadius + 1)

    #         prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
    #         spawn_wps.insert(0, prev_wp_queue[0])
    #         for _ in range(self._road_back_vehicles):
    #             temp_prev_wp_queue = []
    #             for temp_wp in prev_wp_queue:
    #                 if ego_wp.lane_id == temp_wp.lane_id:
    #                     continue
    #                 temp_prev_wps = temp_wp.previous(
    #                     self._road_spawn_dist + random.randint(0, 10)
    #                 )

    #                 num_wps = len(temp_prev_wps)
    #                 if num_wps <= 0:
    #                     continue
    #                 # 前方发现多个waypoint 随机抽取一个
    #                 elif num_wps > 1:
    #                     temp_prev_wp = temp_prev_wps[random.randint(0, num_wps - 1)]
    #                 else:
    #                     temp_prev_wp = temp_prev_wps[0]

    #                 if not self._check_junction_spawnable(temp_prev_wp):
    #                     continue  # Stop when there's no next or found a junction
    #                 temp_prev_wp_queue.append(temp_prev_wp)
    #                 spawn_wps.append(temp_prev_wp)
    #             prev_wp_queue = temp_prev_wp_queue

    #     random.shuffle(spawn_wps)
    #     spawn_wps = spawn_wps[0: self.max_vecs if len(spawn_wps) > self.max_vecs else len(spawn_wps)]
    #     # spawn_wps = spawn_wps[:int(len(spawn_wps)/2)]
    #     start = time.time()
    #     self._vehicle_list = list(
    #         set(self._vehicle_list).union(set(self._spawn_actors(spawn_wps, ego_dist=self.innerRadius)))
    #     )
    #     for i in self._vehicle_list:
    #         # self._tm.set_desired_speed(i, float(random.randint(int(self.max_speed*0.5), int(self.max_speed))))
    #         self._tm.vehicle_percentage_speed_difference(i, -10)
    #     dur_time = time.time() - start

    def _spawn_actors(self, spawn_wps, tm_autopilot=True, ego_dist=0):
        """Spawns several actors in batch"""
        spawn_transforms = []
        ego_location = self._ego_actor.get_location()
        for wp in spawn_wps:
            if len(wp.next(self.dist2endway)) == 0:
                continue
            if ego_location.distance(wp.transform.location) < ego_dist:
                continue
            spawn_transforms.append(
                carla.Transform(
                    wp.transform.location
                    + carla.Location(z=self._spawn_vertical_shift),
                    wp.transform.rotation,
                )
            )
        ego_speed = self.get_speed()

        chosen_vehicle_class = np.random.choice(
            # [x for x in range(len(self.vehicles_ratio))], p=self.vehicles_ratio
            [x for x in range(len(self.vehicles_ratio))], size=len(spawn_transforms), p=self.vehicles_ratio
        )
        # self.vehicle_models = [CARLA_TYPE_TO_VEHICLE[t] for t in chosen_vehicle_class ]
        vehicle_model_list = [list(CARLA_TYPE_TO_VEHICLE.keys())[t] for t in chosen_vehicle_class]
        self.vehicle_models_list = []
        for obj_type in vehicle_model_list:
            if CARLA_TYPE_TO_VEHICLE[obj_type]:
                self.vehicle_models_list.append(random.choice(CARLA_TYPE_TO_VEHICLE[obj_type]))
            else:
                self.vehicle_models_list.append(random.choice(CARLA_TYPE_TO_VEHICLE['car']))
        actors = CarlaDataProvider.request_new_batch_actors_with_specified_model_sets(
            self.vehicle_models_list,
            len(spawn_transforms),
            spawn_transforms,
            tm_autopilot,
            False,
            "traffic_flow_",
            attribute_filter=self._attribute_filter,
            tick=False,
            veloc=3,
            ego_actor=self._ego_actor,
            traffic_flow_nums=self.traffic_flow_nums + 1
        )

        if not actors:
            return actors

        for actor in actors:
            self.traffic_flow_nums += 1
            self._initialise_actor(actor)
        self._vehicle_list = list(set(self._vehicle_list).union(set(actors)))
        return

    def _is_junction(self, waypoint):
        if not waypoint.is_junction or waypoint.junction_id in self._fake_junction_ids:
            return False
        return True

    def get_speed(self, actor=None):
        if actor == None:
            return np.sqrt(
                np.square(self._ego_actor.get_velocity().x)
                + np.square(self._ego_actor.get_velocity().y)
            )
        return np.sqrt(
            np.square(actor.get_velocity().x) + np.square(actor.get_velocity().y)
        )

    def _initialise_actor(self, actor):
        """
        Save the actor into the needed structures, disable its lane changes and set the leading distance.
        """
        self._tm.auto_lane_change(actor, self._vehicle_lane_change)
        self._tm.update_vehicle_lights(actor, self._vehicle_lights)
        self._tm.distance_to_leading_vehicle(
            actor, self._vehicle_leading_distance + random.randint(2, 4) * 3
        )
        self._tm.vehicle_lane_offset(actor, 0.2)

    def get_local_location(self, vehicle, location) -> carla.Location:
        """将全局坐标系下的坐标转到局部坐标系下

        Args:
            location (Location): 待变换的全局坐标系坐标
        """
        res = np.array(vehicle.get_transform().get_inverse_matrix()).dot(
            np.array([location.x, location.y, location.z, 1])
        )
        return carla.Location(x=res[0], y=res[1], z=res[2])

    def terminate(self, new_status):
        """Destroy all actors"""

        destroy_list = []
        for i in self._vehicle_list:
            if i:
                destroy_list.append(carla.command.DestroyActor(i))
        self.client.apply_batch(destroy_list)

        all_actors = list(self._actors_speed_perc)
        for actor in list(all_actors):
            self._destroy_actor(actor)
        super(OasisTrafficflow, self).terminate(new_status)

    def _calculate_fake_junctions(self, debug=False):
        """Calculate the fake junctions"""
        self._fake_junction_ids = []
        self._junction_data = (
            {}
        )  # junction_id -> road_id -> lane_id -> start_wp, end_wp
        self._fake_junction_roads = {}  # junction_id -> road_id
        self._fake_junction_lanes = {}  # junction_id -> road_id -> lane_id
        topology = self._map.get_topology()
        junction_lanes = []
        junction_connection_data = []
        for lane_start, lane_end in topology:
            if lane_start.is_junction:
                if lane_start.junction_id not in self._junction_data:
                    self._junction_data[lane_start.junction_id] = {}
                if (
                        lane_start.road_id
                        not in self._junction_data[lane_start.junction_id]
                ):
                    self._junction_data[lane_start.junction_id][lane_start.road_id] = {}
                self._junction_data[lane_start.junction_id][lane_start.road_id][
                    lane_start.lane_id
                ] = [lane_start, lane_end]
                junction_lanes.append([lane_start, lane_end])
                junction_connection_data.append([1, 1])
                if debug:
                    self._world.debug.draw_arrow(
                        lane_start.transform.location,
                        lane_end.transform.location,
                        thickness=0.1,
                        color=carla.Color(255, 0, 0),
                        life_time=100,
                    )

        for i in range(len(junction_lanes)):
            s1, e1 = junction_lanes[i]
            for j in range(i + 1, len(junction_lanes)):
                s2, e2 = junction_lanes[j]
                if s1.transform.location.distance(s2.transform.location) < 0.1:
                    junction_connection_data[i][0] += 1
                    junction_connection_data[j][0] += 1
                if s1.transform.location.distance(e2.transform.location) < 0.1:
                    junction_connection_data[i][0] += 1
                    junction_connection_data[j][1] += 1
                if e1.transform.location.distance(s2.transform.location) < 0.1:
                    junction_connection_data[i][1] += 1
                    junction_connection_data[j][0] += 1
                if e1.transform.location.distance(e2.transform.location) < 0.1:
                    junction_connection_data[i][1] += 1
                    junction_connection_data[j][1] += 1

        for i in range(len(junction_lanes)):
            s, e = junction_lanes[i]
            cnt = junction_connection_data[i]
            self._junction_data[s.junction_id][s.road_id][s.lane_id] = [
                cnt[0] > 1 or cnt[1] > 1,
                s,
                e,
            ]
            if cnt[0] > 1 or cnt[1] > 1:
                if debug:
                    self._world.debug.draw_arrow(
                        s.transform.location,
                        e.transform.location,
                        thickness=0.1,
                        color=carla.Color(0, 255, 0),
                        life_time=10,
                    )

        for j in self._junction_data:
            self._fake_junction_roads[j] = []
            self._fake_junction_lanes[j] = {}
            fake_junction = True
            for r in self._junction_data[j]:
                self._fake_junction_lanes[j][r] = []
                fake_road = True
                for l in self._junction_data[j][r]:
                    if self._junction_data[j][r][l][0]:
                        fake_road = False
                    else:
                        self._fake_junction_lanes[j][r].append(l)
                if fake_road:
                    self._fake_junction_roads[j].append(r)
                else:
                    fake_junction = False
            if fake_junction:
                self._fake_junction_ids.append(j)

        # if debug:
        #     print("Fake junction lanes: ", self._fake_junction_lanes)
        #     print("Fake junction roads: ", self._fake_junction_roads)
        #     print("Fake junction ids: ", self._fake_junction_ids)

    def _check_junction_spawnable(self, wp):
        if wp.is_junction:
            if wp.junction_id in self._fake_junction_ids:
                return True
            elif wp.road_id in self._fake_junction_roads[wp.junction_id]:
                return True
            # elif wp.lane_id in self._fake_junction_lanes[wp.junction_id][wp.road_id]:
            #     return True
            else:
                return False
        return True

    def get_arc_curve(self, pts):
        '''
        获取弧度值
        :param pts:
        :return:
        '''

        # 计算弦长
        start = np.array(pts[0])
        end = np.array(pts[len(pts) - 1])
        l_arc = np.sqrt(np.sum(np.power(end - start, 2)))

        # 计算弧上的点到直线的最大距离
        # 计算公式：\frac{1}{2a}\sqrt{(a+b+c)(a+b-c)(a+c-b)(b+c-a)}
        a = l_arc
        b = np.sqrt(np.sum(np.power(pts - start, 2), axis=1))
        c = np.sqrt(np.sum(np.power(pts - end, 2), axis=1))
        dist = np.sqrt((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) / (2 * a)
        h = dist.max()

        # 计算曲率
        r = ((a * a) / 4 + h * h) / (2 * h)

        return r

    def get_prediction_curve(self, vec, wp):
        x = [0]
        y = [0]
        resolution = 4
        for i in range(1, 8):
            if wp.next(resolution * i) and len(wp.next(resolution * i)) == 1:
                location = self.get_local_location(vec, wp.next(resolution * i)[0].transform.location)
                x.append(location.x)
                y.append(location.y)
        prediction_path = list(zip(x, y))
        a = len(prediction_path)
        # print("len prediction path",a)
        if a >= 6:
            close_r = self.get_arc_curve(prediction_path[:3])
            Farther_r = self.get_arc_curve(prediction_path[3:])
            return [close_r if close_r is not None else 10000, Farther_r if Farther_r is not None else 10000]
        else:
            return None

    def cal_dis_wp2loc(self, actor=None):
        if actor == None:
            return None
        close_wp = self._map.get_waypoint(actor.get_location())
        location = actor.get_location()
        return close_wp.transform.location.distance(location)

    def set_speed(self, ego_wp, vec_wp, i, wp2loc_dist):
        # 在换道过程中不进行操作
        if self.frame > 100 and self._vehicle_list[i].is_alive and wp2loc_dist < 2:
            logger.info(f"====={self._vehicle_list[i].is_alive},,,{wp2loc_dist}")
            try:
                next_action = self._tm.get_next_action(self._vehicle_list[i])
                if next_action and next_action[0] in ['ChangeLaneRight', 'ChangeLaneLeft']:
                    return
            except Exception:
                logger.info("get_next_action failure")
                return
                    #         # except Exception:
                    #         #     logger.error(
                    #         #         f"===============================An error occurred while update Oasis Traffic Flow!!!!=================================="
                    #         #     )
                    #         #     return py_trees.common.Status.FAILURE

        # 如果在十字路口中或者前方是十字路口，则车辆可能会打滑，减速处理
        # 此处获取车辆前方五米wp，如果为空，返回None
        ahead_wp = vec_wp.next(40) if len(vec_wp.next(40)) > 0 else None
        relative_dis = self.get_local_location(
            self._ego_actor, self._vehicle_list[i].get_location()
        )

        curve = self.get_prediction_curve(self._vehicle_list[i], vec_wp)

        # 车辆是否偏移车道中心线，如果偏移过多，减速处理

        if wp2loc_dist > 4 or abs(self._vehicle_list[i].get_transform().rotation.roll) > 1:
            tmp_speed = self.get_speed(self._vehicle_list[i]) * 0.9 if self.get_speed(
                self._vehicle_list[i]) * 0.9 > 20 else 20
            self._tm.set_desired_speed(self._vehicle_list[i], tmp_speed)
        # 处理车辆静止
        elif self.get_speed(self._vehicle_list[i]) <= 1:
            self._tm.set_desired_speed(self._vehicle_list[i], 40)
        # 路口减速
        elif vec_wp.is_junction or (ahead_wp and ahead_wp[0].is_junction):
            self._tm.set_desired_speed(self._vehicle_list[i], 40)
            # print("self._vehicle_list[i].get_transform():",self._vehicle_list[i].get_transform())
        # 对向车道匀速行驶
        elif ego_wp.lane_id * vec_wp.lane_id < 0:
            self._tm.set_desired_speed(self._vehicle_list[i], 40)
            # return
        # 道路曲率较大，减速处理
        elif curve is not None and min(curve) < 100:
            tmp_speed = self.get_speed(self._vehicle_list[i]) * 0.95 if self.get_speed(
                self._vehicle_list[i]) * 0.95 > 60 else 60
            if tmp_speed:
                self._tm.set_desired_speed(self._vehicle_list[i], tmp_speed)
        # elif self.frame % 20 == 0:
        else:
            # print("-----------------------------")
            # 巡航速度，为主车的1-2倍
            if self.get_speed() * 3.6 > 6:
                # speed = self.get_speed() * 3.6 * 1.5
                speed = self.get_speed() * 3.6 * random.randint(2, 3) * 1.2
                # if self.istown and speed > 50:
                #     speed = 50
                if speed > 80:
                    speed = 80


                # if relative_dis.x < 0:
                #     speed = self.get_speed() * 3.6 * random.randint(2, 4) * 1.2
                    # 最低巡航速度30
            else:
                speed = 30
            self._tm.set_desired_speed(self._vehicle_list[i], speed)

#
#
#
#
#
# class RoadOption(IntEnum):
#     """
#     RoadOption represents the possible topological configurations when moving from a segment of lane to other.
#
#     """
#     VOID = -1
#     LEFT = 1
#     RIGHT = 2
#     STRAIGHT = 3
#     LANEFOLLOW = 4
#     CHANGELANELEFT = 5
#     CHANGELANERIGHT = 6
#
# def draw_waypoints(world, waypoints, vertical_shift, deplay_color=1):
#     """
#     Draw a list of waypoints at a certain height given in vertical_shift.
#     """
#     for w in waypoints:
#         wp = w.transform.location + carla.Location(z=vertical_shift)
#         if deplay_color == 1:
#             color = carla.Color(0, 255, 0)  # Green
#         if deplay_color == 2:
#             color = carla.Color(255, 255, 0)
#         # if w[1] == RoadOption.LEFT:  # Yellow
#         #     color = carla.Color(255, 255, 0)
#         # elif w[1] == RoadOption.RIGHT:  # Cyan
#         #     color = carla.Color(0, 255, 255)
#         # elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
#         #     color = carla.Color(255, 64, 0)
#         # elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
#         #     color = carla.Color(0, 64, 255)
#         # elif w[1] == RoadOption.STRAIGHT:  # Gray
#         #     color = carla.Color(128, 128, 128)
#         # else:  # LANEFOLLOW
#         #     color = carla.Color(0, 255, 0)  # Green
#
#         world.debug.draw_point(wp, size=0.1, color=color, life_time=1000)
#
#     # world.debug.draw_point(waypoints[0][0].transform.location + carla.Location(z=vertical_shift), size=0.5,
#     #                        color=carla.Color(0, 0, 255), life_time=10)
#     #
#     # world.debug.draw_point(waypoints[-1][0].transform.location + carla.Location(z=vertical_shift), size=0.5,
#     #                        color=carla.Color(255, 0, 0), life_time=10)
#
# CARLA_TYPE_TO_WALKER = {
#     "pedestrian":[
#         "walker.pedestrian.0001",
#         "walker.pedestrian.0002",
#         "walker.pedestrian.0003",
#         "walker.pedestrian.0004",
#         "walker.pedestrian.0005",
#         "walker.pedestrian.0006",
#         "walker.pedestrian.0007",
#         "walker.pedestrian.0008",
#         "walker.pedestrian.0009",
#         "walker.pedestrian.0010",
#
#     ]
# }
#
# # "vehicle.audi.a2",
# # "vehicle.audi.tt",
# # "vehicle.jeep.wrangler_rubicon",
# # "vehicle.chevrolet.impala",
# # "vehicle.bmw.grandtourer",
# # "vehicle.citroen.c3",
# # "vehicle.seat.leon",
# # "vehicle.nissan.patrol",
# # "vehicle.nissan.micra",
# # "vehicle.audi.etron",
# # "vehicle.toyota.prius",
# # "vehicle.tesla.model3",
# # "vehicle.tesla.cybertruck",
#
# # EGO_ROAD = 'road'
# CARLA_TYPE_TO_VEHICLE = {
#     "car": [
#         "vehicle.audi.a2",
#         "vehicle.audi.tt",
#         "vehicle.jeep.wrangler_rubicon",
#         "vehicle.chevrolet.impala",
#         "vehicle.bmw.grandtourer",
#         "vehicle.citroen.c3",
#         "vehicle.seat.leon",
#         "vehicle.nissan.patrol",
#         "vehicle.nissan.micra",
#         "vehicle.audi.etron",
#         "vehicle.toyota.prius",
#         "vehicle.tesla.model3",
#         "vehicle.mercedes.coupe_2020",
#         "vehicle.mini.cooper_s"
#
#     ],
#     "van": ["vehicle.volkswagen.t2"],
#     "truck": ["vehicle.tesla.cybertruck","vehicle.carlamotors.carlacola", "vehicle.synkrotron.box_truck", "vehicle.mercedes.sprinter",],
#     'trailer': [],
#     'semitrailer': [],
#     'bus': [],
#     "motorbike": [
#         "vehicle.toyota.prius",
#         "vehicle.tesla.model3",
#         "vehicle.mercedes.coupe_2020",
#         "vehicle.mini.cooper_s"
#     ],
#     "bicycle": [
#         "vehicle.harley-davidson.low_rider",
#         "vehicle.kawasaki.ninja",
#         "vehicle.yamaha.yzf",
#     ],
#     'special_vehicles':[
#         "vehicle.ford.ambulance"
#     ],
# }
# # 摩托车 控制会有滑移等现象 暂时屏蔽，替换为car类型
# # "vehicle.harley-davidson.low_rider",
# # "vehicle.kawasaki.ninja",
# # "vehicle.yamaha.yzf",
# # "vehicle.bh.crossbike",
# # "vehicle.diamondback.century",
# # "vehicle.gazelle.omafiets",
#
# class OasisTrafficflow(AtomicBehavior):
#     """
#     Handles the background activity
#     """
#     def __init__(self, ego_actor, tf_param=None, debug=False, name="OasisTrafficflow"):
#         """
#         Setup class members
#         """
#         super(OasisTrafficflow, self).__init__(name)
#         self.debug = debug
#         self._map = CarlaDataProvider.get_map()
#         self._world = CarlaDataProvider.get_world()
#         blueprint_library = self._world.get_blueprint_library()
#         self._tm_port = CarlaDataProvider.get_traffic_manager_port()
#         self._tm = CarlaDataProvider.get_client().get_trafficmanager(self._tm_port)
#         self.client = CarlaDataProvider.get_client()
#         # 预期速度与当前限制速度之间的百分比差。
#         self._rng = CarlaDataProvider.get_random_seed()
#         self._attribute_filter = None
#
#         # Global variables
#         self._ego_actor = ego_actor
#         self._actors_speed_perc = {}  # Dictionary actor - percentage
#         self._lane_width_threshold = (
#             2.25  # Used to stop some behaviors at narrow lanes to avoid problems [m]
#         )
#         self._spawn_vertical_shift = 0.2
#         self._fake_junction_ids = []
#         self._road_front_vehicles = 2  # Amount of vehicles in front of the ego
#         self._road_back_vehicles = 2  # Amount of vehicles behind the ego
#         self._road_spawn_dist = 15  # Distance between spawned vehicles [m]
#         self.frame = 1
#         self._vehicle_list = []
#         self._destroy_list = []
#         self.centralObject = tf_param['centralObject']
#         self.semiMajorAxis = int(tf_param['semiMajorAxis'])
#         # self.semiMinorAxis = tf_param['semiMinorAxis']
#         self.innerRadius = int(tf_param['innerRadius'])
#         self.numberOfVehicles = int(tf_param['numberOfVehicles'])
#         self.numberOfPedestrian = int(tf_param['numberOfPedestrian'])
#         self.trafficDistribution = tf_param['trafficDistribution']
#         self.directionOfTravelDistribution = tf_param['directionOfTravelDistribution']
#         self.same = self.directionOfTravelDistribution['same']*0.01
#         self.opposite = self.directionOfTravelDistribution['opposite']*0.01
#         self.drivingModel = tf_param['drivingModel']
#         # self.controllerType = tf_param['drivingModel']
#         # self.controllerDistribution = tf_param['controllerDistribution']
#         # Initialisation values
#         if self.drivingModel['controllerType'] == 'Cooperative':
#             self._vehicle_lane_change = False
#             self._vehicle_lights = False
#             self._vehicle_leading_distance = 20
#             self._vehicle_offset = 0.1
#         else:
#             self._vehicle_lane_change = True
#             self._vehicle_lights = False
#             self._vehicle_leading_distance = 10
#             self._vehicle_offset = 0.5
#         # 车辆与生成半径约束关系
#         self.max_vecs = (
#             int(self.semiMajorAxis  * 0.15)
#             if self.numberOfVehicles > int(self.semiMajorAxis  * 0.15)
#             else self.numberOfVehicles
#         )
#         self.vehicles_ratio = [
#             int(tf_param["trafficDistribution"][t])
#             for t in list(CARLA_TYPE_TO_VEHICLE.keys())
#         ]
#         self.vehicles_ratio = [
#             ratio / sum(self.vehicles_ratio) for ratio in self.vehicles_ratio
#         ]
#         self.vehicle_models_list = []
#         if self.debug:
#             logger.info(f"vehicles_ratio:{self.vehicles_ratio}")
#             logger.info(f"vehicle_models:{self.vehicle_models}")
#             logger.info(f"tf_param:{tf_param}")
#         # 前边界
#         self.front_traffic_bound = 0
#         # 反向车道前边界
#         self.front_traffic_bound_opp = 0
#         # 后边界
#         self.back_traffic_bound = 0
#         # 反向车道后边界
#         self.back_traffic_bound_opp = 0
#         self.apll_spawn_points = self._world.get_map().get_spawn_points()
#         # tm预期速度
#         self._tm.global_percentage_speed_difference(-20)
#         # 断头路销毁车辆距离
#         self.dist2endway = 150
#         # self.initialise()
#         # 交通流driver模式，现有两种，1. tm控制，只对速度，异常情况做一些干预，2. default driver 还在开发中
#         self.tm_autopilot = True
#         # 默认carla内置地图用tm控制，处理速度快
#         if self._map.name.split('/')[-1].find('Town') != -1:
#             self.tm_autopilot = True
#         # 对抗代码，对抗默认用tm
#         self.adversarialModelEnable = False
#         if tf_param['adversarialModel']['adversarialModelEnable']:
#             self.tm_autopilot = True
#             self.adversarialModelEnable = True
#         # default driver
#         self.traffic_flow_vecs_control = {}
#         # 同车道车辆数量，用于平衡车辆分布
#         self.same_road_vecs_num = 0
#
#         weather = carla.WeatherParameters(
#             cloudiness=100.0,
#             precipitation=100.0,
#             sun_altitude_angle=20.0)
#
#         self._world.set_weather(weather)
#         self.sun_altitude_angle = self._world.get_weather().sun_altitude_angle
#         if self.sun_altitude_angle < 20:
#             self._vehicle_lights = True
#
#         self.traffic_flow_nums = 0
#
#     def is_in_poly(self,p, poly):
#         """
#         :param p: [x, y]
#         :param poly: [[], [], [], [], ...]
#         :return:
#         """
#         px, py = p
#         is_in = False
#         for i, corner in enumerate(poly):
#             next_i = i + 1 if i + 1 < len(poly) else 0
#             x1, y1 = corner
#             x2, y2 = poly[next_i]
#             if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
#                 is_in = True
#                 break
#             if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
#                 x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
#                 if x == px:  # if point is on edge
#                     is_in = True
#                     break
#                 elif x > px:  # if point is on left-side of line
#                     is_in = not is_in
#         return is_in
#     def initialise(self):
#         """Creates the background activity actors. Pressuposes that the ego is at a road"""
#         # calculate fake junctions
#         poly = [ [-73,111],[75,97],[113,-8],[-52,-7]     ]
#         self._calculate_fake_junctions(self.debug)
#         center_point = carla.Location(x=0, y=46, z=5)
#         self._world.debug.draw_point(center_point, size=0.1, color=carla.Color(255, 0, 0), life_time=1000)
#         center_wp = self._map.get_waypoint(center_point)
#         draw_waypoints(self._world,[center_wp],5)
#         same_dir_wps = get_same_dir_lanes(center_wp)
#         opposite_dir_wps = get_opposite_dir_lanes(center_wp)
#         origin_wps = same_dir_wps + opposite_dir_wps
#         for wp in origin_wps:
#             for i in range(1,100):
#                 temp_prev_wps = wp.previous(5*i)
#                 print("i,len :",i,len(temp_prev_wps))
#                 for point in temp_prev_wps:
#                     if  not self.is_in_poly([ point.transform.location.x, point.transform.location.y ],poly) :
#                         if self.is_in_poly([point.next(10)[0].transform.location.x, point.next(10)[0].transform.location.y],poly):
#                             if not self._is_junction(point):
#                                 draw_waypoints(self._world, [point], 5,deplay_color=2)
#             self.frame += 1
#
#         # 获取主车初始位置
#         # ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
#         # same_dir_wps = get_same_dir_lanes(ego_wp)
#         # opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
#         # # # 初始化辆车
#         # self._initialise_road_behavior(ego_wp, same_dir_wps + opposite_dir_wps)
#
#     # 计算waypoint与车辆location的差距
#     def cal_dis_wp2loc(self,actor = None):
#         if actor == None:
#             return None
#         close_wp = self._map.get_waypoint(actor.get_location())
#         location = actor.get_location()
#         return  close_wp.transform.location.distance(location)
#
#     def set_speed(self,vec_wp,i,wp2loc_dist):
#
#         if self.frame > 100 and self._vehicle_list[i].is_alive and wp2loc_dist < 2 :
#             logger.info(f"====={self._vehicle_list[i].is_alive},,,{wp2loc_dist}")
#             print("=========",self._vehicle_list[i].is_alive,wp2loc_dist)
#             next_action = self._tm.get_next_action(self._vehicle_list[i])
#             if next_action and next_action[0] in ['ChangeLaneRight','ChangeLaneLeft']:
#
#                 if True:
#                     color = carla.Color(0, 255, 0)
#                     self._world.debug.draw_point(next_action[1].transform.location, size=0.1, color=color, life_time=0.1)
#                 print("hhhh")
#                 return
#         # 如果在十字路口中或者前方是十字路口，则车辆可能会打滑，减速处理
#         # 此处获取车辆前方五米wp，如果为空，返回None
#         # path = self.get_prediction_path(self._vehicle_list[i],vec_wp)
#         # curve = self.get_prediction_curve(self._vehicle_list[i],vec_wp)
#         ahead_wp =vec_wp.next(40) if  len(vec_wp.next(40)) > 0 else None
#         # 处理车辆静止
#         if self.get_speed(self._vehicle_list[i]) <= 1:
#             self._tm.set_desired_speed(self._vehicle_list[i], 30)
#         # 路口减速
#         elif vec_wp.is_junction or (ahead_wp and  ahead_wp[0].is_junction):
#             self._tm.set_desired_speed(self._vehicle_list[i], 18)
#         # 道路曲率较大，减速处理
#         # elif curve is not None and  min(curve) < 100:
#         #     tmp_speed = self.get_speed(self._vehicle_list[i]) * 0.9 if self.get_speed(
#         #         self._vehicle_list[i]) * 0.9 > 50 else None
#         #     if tmp_speed:
#         #         self._tm.set_desired_speed(self._vehicle_list[i], tmp_speed)
#         # 车辆是否偏移车道中心线，如果偏移过多，减速处理
#         elif wp2loc_dist > 1.5:
#             tmp_speed =self.get_speed(self._vehicle_list[i])*0.9 if  self.get_speed(self._vehicle_list[i])*0.9 > 30 else  30
#             self._tm.set_desired_speed(self._vehicle_list[i],tmp_speed )
#         elif self.frame%30 == 0:
#             # 巡航速度，为主车的1-2倍
#             if self.get_speed() * 3.6 > 6:
#                 speed = self.get_speed()*3.6*random.randint(1,2) + 10
#             # 最低巡航速度30
#             else :
#                 speed = 30
#             self._tm.set_desired_speed(self._vehicle_list[i], speed)
#     # 车辆测试，无交通流生成逻辑
#     def update1(self):
#         logger.info("--------------------------------------start111222")
#         # return py_trees.common.Status.RUNNING
#         self.max_vecs = 20
#         self._destroy_list = []
#         destroy_indexs = []
#         for vec in self._vehicle_list:
#             # 断头路处理
#             vec_wp = self._map.get_waypoint(
#                 vec.get_location()
#             )
#             ahead_wp = vec_wp.next(self.dist2endway)
#             if len(ahead_wp) == 0:
#                 logger.info(f"---------------vec{vec.id} has no way!!!")
#                 destroy_indexs.append(vec.id)
#                 self._destroy_list.append(
#                     carla.command.DestroyActor(vec)
#                 )
#                 continue
#             # 离路处理
#             wp2loc_dist = self.cal_dis_wp2loc(vec)
#             if wp2loc_dist > 5:
#                 destroy_indexs.append(vec.id)
#                 self._destroy_list.append(
#                     carla.command.DestroyActor(vec)
#                 )
#                 continue
#             logger.info(f"===============vec location:{vec.get_location()}")
#         logger.info("--------------------------------------end")
#         if len(self._destroy_list) > 0:
#             self.client.apply_batch(self._destroy_list)
#             self._vehicle_list = list(
#                 filter(lambda x: x.id not in destroy_indexs, self._vehicle_list)
#             )
#         autopilot = True
#         self.frame += 1
#         logger.info(f"---------------------len(self._vehicle_list):{len(self._vehicle_list)}")
#
#         ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
#         if len(self._vehicle_list) < self.max_vecs:
#             same_dir_wps = get_same_dir_lanes(ego_wp)
#             logger.info(f"---------------------len(same_dir_wps):{len(same_dir_wps)}")
#             # same_dir_wps = []
#             opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
#             logger.info(f"---------------------len(opposite_dir_wps):{len(opposite_dir_wps)}")
#             opposite_dir_wps = []
#             self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, autopilot)
#             for i in self._vehicle_list:
#                 self._tm.set_desired_speed(i, 50)
#         if not autopilot:
#             if self.frame > 30:
#                 for vec in self._vehicle_list:
#                     if np.sqrt(
#                             vec.get_velocity().x * vec.get_velocity().x + vec.get_velocity().y * vec.get_velocity().y) < 1:
#                         print("1111")
#                     if vec not in self.traffic_flow_vecs_control:
#                         args = {"desired_velocity": 10, "desired_acceleration": 5, "emergency_param": 0.4,
#                                 "desired_deceleration": 5, "safety_time": 5,
#                                 "lane_changing_dynamic": True, "urge_to_overtake": True, "obey_traffic_lights": True,
#                                 "identify_object": True, "obey_speed_limit": True
#                                 }
#                         self.traffic_flow_vecs_control[vec] = NpcVehicleControl(vec, args)
#                         self.traffic_flow_vecs_control[vec].run_step()
#                     else:
#                         self.traffic_flow_vecs_control[vec].run_step()
#         return py_trees.common.Status.RUNNING
#
#     def get_arc_curve(self, pts):
#         '''
#         获取弧度值
#         :param pts:
#         :return:
#         '''
#
#         # 计算弦长
#         start = np.array(pts[0])
#         end = np.array(pts[len(pts) - 1])
#         l_arc = np.sqrt(np.sum(np.power(end - start, 2)))
#
#         # 计算弧上的点到直线的最大距离
#         # 计算公式：\frac{1}{2a}\sqrt{(a+b+c)(a+b-c)(a+c-b)(b+c-a)}
#         a = l_arc
#         b = np.sqrt(np.sum(np.power(pts - start, 2), axis=1))
#         c = np.sqrt(np.sum(np.power(pts - end, 2), axis=1))
#         dist = np.sqrt((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) / (2 * a)
#         h = dist.max()
#
#         # 计算曲率
#         r = ((a * a) / 4 + h * h) / (2 * h)
#
#         return r
#     def get_prediction_curve(self,vec,wp):
#         x = [0]
#         y = [0]
#         resolution = 4
#         for i in range(1,8):
#             if wp.next(resolution*i) and len(wp.next(resolution*i)) == 1:
#                 location = self.get_local_location(vec, wp.next(resolution*i)[0].transform.location)
#                 x.append(location.x)
#                 y.append(location.y)
#         prediction_path  = list(zip(x, y))
#         a = len(prediction_path)
#         # print("len prediction path",a)
#         if a >=6:
#             close_r = self.get_arc_curve(prediction_path[:3])
#             Farther_r =  self.get_arc_curve(prediction_path[3:])
#             return [close_r if close_r is not None else 10000,Farther_r if Farther_r is not None else 10000]
#         else:
#             return None
#     # 区域交通流 test
#     def update2(self):
#         # self._world.debug.draw_point(wp, size=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
#
#         # center_point = carla.Location(x=0, y=46, z=5)
#         # self._world.debug.draw_point(center_point, size=0.1, color=carla.Color(255, 0, 0), life_time=1000)
#         # center_wp = self._map.get_waypoint(center_point)
#         # draw_waypoints(self._world,[center_wp],5)
#         # if self.frame < 1000:
#         #     temp_prev_wps = center_wp.previous(5*self.frame)
#         #     draw_waypoints(self._world, temp_prev_wps, 5,deplay_color=2)
#         # self.frame += 1
#         return py_trees.common.Status.RUNNING
#
#
#         # draw_waypoints(self._world, gl_able_spawn_wps, vertical_shift=5,
#         #                )
#
#     #交通流主程序
#     def update(self):
#         # print("---------------------")
#         # try:
#         if True:
#             flag = True
#             logger.info(f"start traffic flow update!!!")
#             ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
#             ahead_ego_wp = ego_wp.next(self.dist2endway + 20)
#             curve = self.get_prediction_curve(self._ego_actor,ego_wp)
#             self.frame += 1
#             destroy_indexs = []
#             # 临时变量
#             front_tmpmax = 0
#             opp_front_tmpmax = 0
#             bake_tmpmax = 0
#             opp_bake_tmpmax = 0
#             self.same_road_vecs_num = 0
#
#             for i in range(len(self._vehicle_list)):
#                 # ego车与目标车的距离
#                 dist = (self._vehicle_list[i].get_location().distance(ego_wp.transform.location))
#                 wp2loc_dist = self.cal_dis_wp2loc(self._vehicle_list[i])
#
#                 vec_wp = self._map.get_waypoint(
#                     self._vehicle_list[i].get_location()
#                 )
#                 # 断头路处理
#                 ahead_wp = vec_wp.next(self.dist2endway)
#                 if len(ahead_wp) == 0:
#                     destroy_indexs.append(self._vehicle_list[i].id)
#                     self._destroy_list.append(
#                         carla.command.DestroyActor(self._vehicle_list[i])
#                     )
#                     continue
#                 if self.same == 0:
#                     if ego_wp.road_id == vec_wp.road_id and  ego_wp.lane_id * vec_wp.lane_id > 0:
#                         destroy_indexs.append(self._vehicle_list[i].id)
#                         self._destroy_list.append(
#                             carla.command.DestroyActor(self._vehicle_list[i])
#                         )
#                         continue
#
#                 # 如果车辆使出道路，删除处理
#                 if wp2loc_dist > 5:
#                     destroy_indexs.append(self._vehicle_list[i].id)
#                     self._destroy_list.append(
#                         carla.command.DestroyActor(self._vehicle_list[i])
#                     )
#                     continue
#
#                 # 如果车辆与给定坐标的距离大于半径
#                 if (
#                     dist
#                     > (
#                         self.semiMajorAxis
#                     )
#                     + self.get_speed() * 2
#                 ):
#                     destroy_indexs.append(self._vehicle_list[i].id)
#                     self._destroy_list.append(
#                         carla.command.DestroyActor(self._vehicle_list[i])
#                     )
#
#                 # 更新前后距离
#                 if (
#                     self.get_local_location(
#                         self._ego_actor, self._vehicle_list[i].get_location()
#                     ).x
#                     > 0
#                     and dist > front_tmpmax
#                     and ego_wp.road_id == vec_wp.road_id
#                     and ego_wp.lane_id * vec_wp.lane_id > 0
#                 ):
#                     front_tmpmax = dist
#                 elif (
#                     self.get_local_location(
#                         self._ego_actor, self._vehicle_list[i].get_location()
#                     ).x
#                     > 0
#                     and dist > opp_front_tmpmax
#                     and ego_wp.road_id == vec_wp.road_id
#                     and ego_wp.lane_id * vec_wp.lane_id < 0
#                 ):
#                     opp_front_tmpmax = dist
#                 elif (
#                     self.get_local_location(
#                         self._ego_actor, self._vehicle_list[i].get_location()
#                     ).x
#                     < 0
#                     and dist > bake_tmpmax
#                     and ego_wp.road_id == vec_wp.road_id
#                     and ego_wp.lane_id * vec_wp.lane_id > 0
#                 ):
#                     bake_tmpmax = dist
#                 elif (
#                     self.get_local_location(
#                         self._ego_actor, self._vehicle_list[i].get_location()
#                     ).x
#                     < 0
#                     and dist > opp_bake_tmpmax
#                     and ego_wp.road_id == vec_wp.road_id
#                     and ego_wp.lane_id * vec_wp.lane_id < 0
#                 ):
#                     opp_bake_tmpmax = dist
#                 if ego_wp.road_id == vec_wp.road_id:
#                     self.same_road_vecs_num += 1
#             self.front_traffic_bound = front_tmpmax + self.semiMajorAxis * 0.1
#             self.front_traffic_bound_opp = opp_front_tmpmax + self.semiMajorAxis * 0.1
#             self.back_traffic_bound = bake_tmpmax + self.semiMajorAxis * 0.1
#             self.back_traffic_bound_opp = opp_bake_tmpmax + self.semiMajorAxis * 0.1
#             if len(destroy_indexs) > 0:
#                 self.client.apply_batch(self._destroy_list)
#                 self._vehicle_list = list(
#                     filter(lambda x: x.id not in destroy_indexs, self._vehicle_list)
#                 )
#                 if not self.tm_autopilot:
#                     self.traffic_flow_vecs_control = {key: val for key, val in self.traffic_flow_vecs_control.items() if key not in destroy_indexs}
#
#             logger.info(f"front_traffic_bound:{self.front_traffic_bound}")
#             logger.info(f"front_traffic_bound_opp:{self.front_traffic_bound_opp}")
#             logger.info(f"back_traffic_bound:{self.back_traffic_bound}")
#             logger.info(f"back_traffic_bound_opp:{self.back_traffic_bound_opp}")
#
#             # 补充车辆
#             logger.info(f"len vehicle list:{len(self._vehicle_list)}")
#
#             if len(ahead_ego_wp) != 0  and  len(self._vehicle_list) < self.max_vecs:
#                 if self.adversarialModelEnable:
#                     if self.same_road_vecs_num < 1:
#                         same_dir_wps = get_same_dir_lanes(ego_wp)
#                     else:
#                         same_dir_wps = []
#                 elif self.same_road_vecs_num < self.same * self.numberOfVehicles:
#                     same_dir_wps = get_same_dir_lanes(ego_wp)
#                 else:
#                     same_dir_wps = []
#                 opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
#                 self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, self.tm_autopilot)
#
#
#             # driver 模式
#             if not self.tm_autopilot:
#                 for vec in self._vehicle_list:
#                     if vec.is_alive:
#                         if vec.id not in self.traffic_flow_vecs_control:
#                             args = {"desired_velocity": 10, "desired_acceleration": 5, "emergency_param": 0.4,
#                                     "desired_deceleration": 5, "safety_time": 5,
#                                     "lane_changing_dynamic": True, "urge_to_overtake": True,
#                                     "obey_traffic_lights": True,
#                                     "identify_object": True, "obey_speed_limit": True
#                                     }
#                             self.traffic_flow_vecs_control[vec.id] = NpcVehicleControl(vec, args)
#                             self.traffic_flow_vecs_control[vec.id].run_step()
#                         else:
#                             self.traffic_flow_vecs_control[vec.id].run_step()
#
#             logger.info(f"end of traffic flow update!!!")
#             # 车辆速度规划
#             if self.tm_autopilot:
#
#                 for i in range(len(self._vehicle_list)):
#                     print("=========vecs transform==============: ", self._vehicle_list[i].get_transform())
#                     wp2loc_dist = self.cal_dis_wp2loc(self._vehicle_list[i])
#                     vec_wp = self._map.get_waypoint(
#                         self._vehicle_list[i].get_location()
#                     )
#                     self.set_speed(vec_wp, i,wp2loc_dist)
#             print("len of self._vehicle_list:",len(self._vehicle_list))
#             return py_trees.common.Status.RUNNING
#         # except Exception:
#         #     logger.error(
#         #         f"===============================An error occurred while update Oasis Traffic Flow!!!!=================================="
#         #     )
#         #     return py_trees.common.Status.FAILURE
#
#     def get_filter_points(self,ego_vec_road_id):
#         spawn_points_filtered = []
#         num = 0
#         for i, around_spawn_point in enumerate(
#             self.apll_spawn_points
#         ):  # 遍历所有出生点
#             tmp_wpt = self._map.get_waypoint(around_spawn_point.location)
#             diff_road = around_spawn_point.location.distance(
#                 self._ego_actor.get_location()
#             )
#             if (
#                 diff_road < self.semiMajorAxis
#                 and diff_road > self.innerRadius*2
#                 and self._map.get_waypoint(around_spawn_point.location).road_id
#                 != ego_vec_road_id
#             ):  # 如果出生点与给定坐标的距离小于半径
#                 if num < abs(self.max_vecs - len(self._vehicle_list)):
#                     num += 1
#                     spawn_points_filtered.append(
#                         tmp_wpt
#                     )  # 将出生点添加到过滤后的列表中
#                 else:
#                     break
#         return spawn_points_filtered
#
#     def _add_road_vecs(self, ego_wp, same_dir_wps, opposite_dir_wps, tm_autopilot=True):
#         '''筛选出生点并生成车辆'''
#         spawn_wps = []
#         # offset_var = self.semiMajorAxis * 0.1 if self.semiMajorAxis * 0.1 > 15 else 15
#         offset_var = 0
#         # 同向车道出生点筛选
#         speed_dist = self.get_speed()
#         for wp in same_dir_wps:
#             if self.numberOfVehicles * self.same <= 0:
#                 break
#             # same_num = int(self.numberOfVehicles * self.same / len(same_dir_wps))
#             same_num = int(self.numberOfVehicles * self.same )
#             if same_num <1 and self.numberOfVehicles * self.same > 0:
#                 same_num = 1
#
#             innerboundarywp = wp.next(self.innerRadius + 1)
#             if len(innerboundarywp) == 0:
#                 continue
#             next_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
#             # spawn_wps.insert(0, next_wp_queue[0])
#             # 控制生成车辆车距
#             offset = 0
#             for _ in range(same_num):
#                 # self._road_spawn_dist = 15
#                 temp_next_wp_queue = []
#                 offset += offset_var
#                 for temp_wp in next_wp_queue:
#                     temp_next_wps = temp_wp.next(
#                         self.front_traffic_bound
#                         + self._road_spawn_dist
#                         + random.randint(-3, 3) * 2
#                         + speed_dist * 3
#                         + offset
#                     )
#                     num_wps = len(temp_next_wps)
#                     if num_wps <= 0:
#                         continue
#                     # 前方发现多个waypoint 随机抽取一个
#                     elif num_wps > 1:
#                         temp_next_wp = temp_next_wps[random.randint(0, num_wps - 1)]
#                     else:
#                         temp_next_wp = temp_next_wps[0]
#
#                     dist = temp_next_wp.transform.location.distance(
#                         self._ego_actor.get_location()
#                     )
#
#                     if dist > self.semiMajorAxis + speed_dist * 2:
#                         continue
#                     if not self._check_junction_spawnable(temp_next_wp):
#                         continue # Stop when there's no next or found a junction
#                     temp_next_wp_queue.append(temp_next_wp)
#                     spawn_wps.insert(0, temp_next_wp)
#                 next_wp_queue = temp_next_wp_queue
#
#             innerboundarywp = wp.previous(self.innerRadius + 1)
#             if len(innerboundarywp) <= 0:
#                 continue
#             prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
#             # spawn_wps.insert(0, prev_wp_queue[0])
#             offset = 0
#             for _ in range(same_num):
#                 temp_prev_wp_queue = []
#                 offset += offset_var
#                 for temp_wp in prev_wp_queue:
#                     if ego_wp.lane_id == temp_wp.lane_id:
#                         continue
#                     temp_prev_wps = temp_wp.previous(
#                         self.back_traffic_bound / 2 + self.innerRadius
#                         + self._road_spawn_dist
#                         + random.randint(0, 3)
#                         + speed_dist * 2
#                     )
#                     num_wps = len(temp_prev_wps)
#                     if num_wps <= 0:
#                         continue
#                     # 前方发现多个waypoint 随机抽取一个
#                     elif num_wps > 1:
#                         temp_prev_wp = temp_prev_wps[random.randint(0, num_wps - 1)]
#                     else:
#                         temp_prev_wp = temp_prev_wps[0]
#                     dist = temp_prev_wp.transform.location.distance(
#                         self._ego_actor.get_location()
#                     )
#                     if dist > self.semiMajorAxis + speed_dist * 2:
#                         continue
#                     if not self._check_junction_spawnable(temp_prev_wp):
#                         continue  # Stop when there's no next or found a junction
#                     temp_prev_wp_queue.append(temp_prev_wp)
#                     spawn_wps.append(temp_prev_wp)
#                 prev_wp_queue = temp_prev_wp_queue
#         # 反向车道出生点筛选
#         opp_spawn_wps = []
#         for wp in opposite_dir_wps:
#             opposite_num = int(self.numberOfVehicles * self.opposite  )
#             if opposite_num < 1 and self.numberOfVehicles * self.opposite > 0:
#                 opposite_num = 1
#             elif self.numberOfVehicles * self.opposite <= 0:
#                 break
#             innerboundarywp = wp.previous(self.innerRadius + 1)
#             if len(innerboundarywp) <= 0:
#                 continue
#             prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
#             # opp_spawn_wps.insert(0, prev_wp_queue[0])
#             # for _ in range(self._road_back_vehicles):
#             offset = 0
#             for _ in range(opposite_num):
#                 temp_prev_wp_queue = []
#                 offset += offset_var
#                 for temp_wp in prev_wp_queue:
#                     temp_prev_wps = temp_wp.previous(
#                         self.front_traffic_bound_opp
#                         + self._road_spawn_dist + self.innerRadius*1.2
#                         + speed_dist*2
#                     )
#                     num_wps = len(temp_prev_wps)
#                     if num_wps <= 0:
#                         continue
#                     # 前方发现多个waypoint 随机抽取一个
#                     elif num_wps > 1:
#                         temp_prev_wp = temp_prev_wps[random.randint(0, num_wps - 1)]
#                     else:
#                         temp_prev_wp = temp_prev_wps[0]
#
#                     dist = temp_prev_wp.transform.location.distance(
#                         self._ego_actor.get_location()
#                     )
#                     if dist > self.semiMajorAxis + speed_dist * 2:
#                         continue
#                     if not self._check_junction_spawnable(temp_prev_wp):
#                         continue  # Stop when there's no next or found a junction
#                     temp_prev_wp_queue.append(temp_prev_wp)
#                     opp_spawn_wps.append(temp_prev_wp)
#                 prev_wp_queue = temp_prev_wp_queue
#
#         # if len(spawn_wps) >0:
#         #     print("===================len of spawn_wps:====", len(spawn_wps) )
#         if len(spawn_wps) > 0 or len(opp_spawn_wps) > 0:
#             random.shuffle(spawn_wps)
#             random.shuffle(opp_spawn_wps)
#
#             gl_spawn_wps = []
#             gap_nums =  self.max_vecs - len(self._vehicle_list) + 1
#             if self.adversarialModelEnable:
#                 spawn_wps = spawn_wps[:1]
#             elif len(spawn_wps) > int(gap_nums*self.same*1.1):
#                 spawn_wps  = spawn_wps[:int(gap_nums * self.same*1.1) + 1]
#
#             if len(opp_spawn_wps) > int(gap_nums*self.opposite):
#                 opp_spawn_wps = opp_spawn_wps[:int(gap_nums*self.opposite) + 1]
#
#             gl_spawn_wps += spawn_wps + opp_spawn_wps
#             # 同方向与反方向补充出生点小于最大车辆数
#             if len(self._vehicle_list) + len(gl_spawn_wps) <= self.max_vecs:
#                 gap = self.max_vecs - (len(self._vehicle_list) + len(gl_spawn_wps))
#                 if gap > 10:
#                     gap = 10
#                 spawn_points_filtered = self.get_filter_points(ego_wp.road_id)
#                 # 补充出生点大于0
#                 if len(spawn_points_filtered) > 0:
#                     gl_spawn_wps += spawn_points_filtered[
#                         : len(spawn_points_filtered) - 1 if  gap > len(spawn_points_filtered) else gap - 1
#                         ]
#             gl_able_spawn_wps = []
#             for index in range(len(gl_spawn_wps)):
#                 # 检查出生点前方是否为断头路
#                 if len(gl_spawn_wps[index].next(self.dist2endway)) > 0 and not self._is_junction(gl_spawn_wps[index]):
#                     gl_able_spawn_wps.append(gl_spawn_wps[index])
#
#             show = False
#             if show:
#                 draw_waypoints(self._world, gl_able_spawn_wps,vertical_shift=5,
#                                  )
#             tmp_vecs = self._spawn_actors(gl_able_spawn_wps,tm_autopilot=tm_autopilot)
#             self._vehicle_list = list(set(self._vehicle_list).union(set(tmp_vecs)))
#
#     def _initialise_road_behavior(self, ego_wp, road_wps, rdm=False):
#         """
#         Initialises the road behavior, consisting on several vehicle in front of the ego,
#         and several on the back and are only spawned outside junctions.
#         If there aren't enough actors behind, road sources will be created that will do so later on
#         """
#         # Vehicles in front
#         spawn_wps = []
#         for wp in road_wps:
#             # Front spawn points
#             innerboundarywp = wp.next( self.innerRadius + 1 )
#             if len(innerboundarywp) <= 0:
#                 continue
#             next_wp_queue = [ innerboundarywp[ random.randint(0,len(innerboundarywp) - 1 ) ] ]
#             spawn_wps.insert(0, next_wp_queue[0])
#             for _ in range(self._road_front_vehicles):
#                 temp_next_wp_queue = []
#                 for temp_wp in next_wp_queue:
#                     # 获取 wp
#                     temp_next_wps = temp_wp.next(self._road_spawn_dist + random.randint(0, 10))
#                     num_wps = len(temp_next_wps)
#                     if num_wps <= 0:
#                         continue
#                     # 前方发现多个waypoint 随机抽取一个
#                     elif num_wps > 1:
#                         temp_next_wp = temp_next_wps[ random.randint(0,num_wps - 1) ]
#                     else:
#                         temp_next_wp = temp_next_wps[0]
#                     # 超出限定范围丢弃
#                     dist = temp_next_wp.transform.location.distance(wp.transform.location)
#                     if dist > self.semiMajorAxis:
#                         continue
#                     if not self._check_junction_spawnable(temp_next_wp):
#                         continue  # Stop when there's no next or found a junction
#                     temp_next_wp_queue.append(temp_next_wp)
#                     spawn_wps.insert(0, temp_next_wp)
#                 next_wp_queue = temp_next_wp_queue
#
#             innerboundarywp = wp.previous( self.innerRadius + 1 )
#
#             prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
#             spawn_wps.insert(0, prev_wp_queue[0])
#             for _ in range(self._road_back_vehicles):
#                 temp_prev_wp_queue = []
#                 for temp_wp in prev_wp_queue:
#                     if ego_wp.lane_id == temp_wp.lane_id:
#                         continue
#                     temp_prev_wps = temp_wp.previous(
#                         self._road_spawn_dist + random.randint(0, 10)
#                     )
#
#                     num_wps = len(temp_prev_wps)
#                     if num_wps <= 0:
#                         continue
#                     # 前方发现多个waypoint 随机抽取一个
#                     elif num_wps > 1:
#                         temp_prev_wp = temp_prev_wps[ random.randint(0,num_wps - 1) ]
#                     else:
#                         temp_prev_wp = temp_prev_wps[0]
#
#                     if not self._check_junction_spawnable(temp_prev_wp):
#                         continue  # Stop when there's no next or found a junction
#                     temp_prev_wp_queue.append(temp_prev_wp)
#                     spawn_wps.append(temp_prev_wp)
#                 prev_wp_queue = temp_prev_wp_queue
#
#         random.shuffle(spawn_wps)
#         spawn_wps = spawn_wps[0 : self.max_vecs if len(spawn_wps) > self.max_vecs else len(spawn_wps)]
#         # spawn_wps = spawn_wps[:int(len(spawn_wps)/2)]
#         start = time.time()
#         self._vehicle_list = list(
#             set(self._vehicle_list).union(set(self._spawn_actors(spawn_wps)))
#         )
#         for i in self._vehicle_list:
#             # self._tm.set_desired_speed(i, float(random.randint(int(self.max_speed*0.5), int(self.max_speed))))
#             self._tm.vehicle_percentage_speed_difference(i, -10)
#         dur_time = time.time() - start
#
#
#     def _spawn_actors(self, spawn_wps, tm_autopilot= True, ego_dist=0):
#         """Spawns several actors in batch"""
#         spawn_transforms = []
#         ego_location = self._ego_actor.get_location()
#         for wp in spawn_wps:
#             if ego_location.distance(wp.transform.location) < ego_dist:
#
#                 continue
#             spawn_transforms.append(
#                 carla.Transform(
#                     wp.transform.location
#                     + carla.Location(z=self._spawn_vertical_shift),
#                     wp.transform.rotation,
#                 )
#             )
#         ego_speed = self.get_speed()
#
#         chosen_vehicle_class = np.random.choice(
#             # [x for x in range(len(self.vehicles_ratio))], p=self.vehicles_ratio
#             [x for x in range(len(self.vehicles_ratio))],size=len(spawn_transforms), p=self.vehicles_ratio
#         )
#         # self.vehicle_models = [CARLA_TYPE_TO_VEHICLE[t] for t in chosen_vehicle_class ]
#         vehicle_model_list = [list(CARLA_TYPE_TO_VEHICLE.keys())[t] for t in chosen_vehicle_class]
#         self.vehicle_models_list = []
#         for obj_type in vehicle_model_list:
#             if CARLA_TYPE_TO_VEHICLE[obj_type]:
#                 self.vehicle_models_list.append(random.choice( CARLA_TYPE_TO_VEHICLE[obj_type] ))
#             else:
#                 self.vehicle_models_list.append(random.choice(CARLA_TYPE_TO_VEHICLE['car']))
#         actors = CarlaDataProvider.request_new_batch_actors_with_specified_model_sets(
#             self.vehicle_models_list,
#             len(spawn_transforms),
#             spawn_transforms,
#             tm_autopilot,
#             False,
#             "traffic_flow_",
#             attribute_filter=self._attribute_filter,
#             tick=False,
#             veloc=5,
#             traffic_flow_nums = self.traffic_flow_nums + 1
#         )
#
#         if not actors:
#             return actors
#
#         for actor in actors:
#             self.traffic_flow_nums += 1
#             self._initialise_actor(actor)
#
#         return actors
#
#     def _is_junction(self, waypoint):
#         if not waypoint.is_junction or waypoint.junction_id in self._fake_junction_ids:
#             return False
#         return True
#
#     def get_speed(self, actor=None):
#         if actor == None:
#             return np.sqrt(
#                 np.square(self._ego_actor.get_velocity().x)
#                 + np.square(self._ego_actor.get_velocity().y)
#             )
#         return np.sqrt(
#             np.square(actor.get_velocity().x) + np.square(actor.get_velocity().y)
#         )
#
#     def _initialise_actor(self, actor):
#         """
#         Save the actor into the needed structures, disable its lane changes and set the leading distance.
#         """
#         self._tm.auto_lane_change(actor, self._vehicle_lane_change)
#         self._tm.update_vehicle_lights(actor, self._vehicle_lights)
#         self._tm.distance_to_leading_vehicle(
#             actor, self._vehicle_leading_distance + random.randint(2, 4) * 3
#         )
#         self._tm.vehicle_lane_offset(actor, self._vehicle_offset)
#
#     def get_local_location(self, vehicle, location) -> carla.Location:
#         """将全局坐标系下的坐标转到局部坐标系下
#
#         Args:
#             location (Location): 待变换的全局坐标系坐标
#         """
#         res = np.array(vehicle.get_transform().get_inverse_matrix()).dot(
#             np.array([location.x, location.y, location.z, 1])
#         )
#         return carla.Location(x=res[0], y=res[1], z=res[2])
#
#     def terminate(self, new_status):
#         """Destroy all actors"""
#
#         destroy_list = []
#         for i in self._vehicle_list:
#             if i:
#                 destroy_list.append( carla.command.DestroyActor(i) )
#         self.client.apply_batch(destroy_list)
#
#         all_actors = list(self._actors_speed_perc)
#         for actor in list(all_actors):
#             self._destroy_actor(actor)
#         super(OasisTrafficflow, self).terminate(new_status)
#
#     def _calculate_fake_junctions(self, debug=False):
#         """Calculate the fake junctions"""
#         self._fake_junction_ids = []
#         self._junction_data = (
#             {}
#         )  # junction_id -> road_id -> lane_id -> start_wp, end_wp
#         self._fake_junction_roads = {}  # junction_id -> road_id
#         self._fake_junction_lanes = {}  # junction_id -> road_id -> lane_id
#         topology = self._map.get_topology()
#         junction_lanes = []
#         junction_connection_data = []
#         for lane_start, lane_end in topology:
#             if lane_start.is_junction:
#                 if lane_start.junction_id not in self._junction_data:
#                     self._junction_data[lane_start.junction_id] = {}
#                 if (
#                     lane_start.road_id
#                     not in self._junction_data[lane_start.junction_id]
#                 ):
#                     self._junction_data[lane_start.junction_id][lane_start.road_id] = {}
#                 self._junction_data[lane_start.junction_id][lane_start.road_id][
#                     lane_start.lane_id
#                 ] = [lane_start, lane_end]
#                 junction_lanes.append([lane_start, lane_end])
#                 junction_connection_data.append([1, 1])
#                 if debug:
#                     self._world.debug.draw_arrow(
#                         lane_start.transform.location,
#                         lane_end.transform.location,
#                         thickness=0.1,
#                         color=carla.Color(255, 0, 0),
#                         life_time=100,
#                     )
#
#         for i in range(len(junction_lanes)):
#             s1, e1 = junction_lanes[i]
#             for j in range(i + 1, len(junction_lanes)):
#                 s2, e2 = junction_lanes[j]
#                 if s1.transform.location.distance(s2.transform.location) < 0.1:
#                     junction_connection_data[i][0] += 1
#                     junction_connection_data[j][0] += 1
#                 if s1.transform.location.distance(e2.transform.location) < 0.1:
#                     junction_connection_data[i][0] += 1
#                     junction_connection_data[j][1] += 1
#                 if e1.transform.location.distance(s2.transform.location) < 0.1:
#                     junction_connection_data[i][1] += 1
#                     junction_connection_data[j][0] += 1
#                 if e1.transform.location.distance(e2.transform.location) < 0.1:
#                     junction_connection_data[i][1] += 1
#                     junction_connection_data[j][1] += 1
#
#         for i in range(len(junction_lanes)):
#             s, e = junction_lanes[i]
#             cnt = junction_connection_data[i]
#             self._junction_data[s.junction_id][s.road_id][s.lane_id] = [
#                 cnt[0] > 1 or cnt[1] > 1,
#                 s,
#                 e,
#             ]
#             if cnt[0] > 1 or cnt[1] > 1:
#                 if debug:
#                     self._world.debug.draw_arrow(
#                         s.transform.location,
#                         e.transform.location,
#                         thickness=0.1,
#                         color=carla.Color(0, 255, 0),
#                         life_time=10,
#                     )
#
#         for j in self._junction_data:
#             self._fake_junction_roads[j] = []
#             self._fake_junction_lanes[j] = {}
#             fake_junction = True
#             for r in self._junction_data[j]:
#                 self._fake_junction_lanes[j][r] = []
#                 fake_road = True
#                 for l in self._junction_data[j][r]:
#                     if self._junction_data[j][r][l][0]:
#                         fake_road = False
#                     else:
#                         self._fake_junction_lanes[j][r].append(l)
#                 if fake_road:
#                     self._fake_junction_roads[j].append(r)
#                 else:
#                     fake_junction = False
#             if fake_junction:
#                 self._fake_junction_ids.append(j)
#
#         # if debug:
#         #     print("Fake junction lanes: ", self._fake_junction_lanes)
#         #     print("Fake junction roads: ", self._fake_junction_roads)
#         #     print("Fake junction ids: ", self._fake_junction_ids)
#
#     def _check_junction_spawnable(self, wp):
#         if wp.is_junction:
#             if wp.junction_id in self._fake_junction_ids:
#                 return True
#             elif wp.road_id in self._fake_junction_roads[wp.junction_id]:
#                 return True
#             # elif wp.lane_id in self._fake_junction_lanes[wp.junction_id][wp.road_id]:
#             #     return True
#             else:
#                 return False
#         return True
#

