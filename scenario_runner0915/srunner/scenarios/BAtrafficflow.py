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

CARLA_TYPE_TO_WALKER = {
    "pedestrian":[
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
        "vehicle.mini.cooper_s",


    ],
    "van": ["vehicle.volkswagen.t2"],
    "truck": ["vehicle.tesla.cybertruck","vehicle.carlamotors.carlacola", "vehicle.synkrotron.box_truck", "vehicle.mercedes.sprinter",],
    'trailer': [],
    'semitrailer': [],
    'bus': [],
    "motorbike": [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
    ],
    "bicycle": [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
    ],
    'special_vehicles':[
        "vehicle.ford.ambulance"
    ],
}
# "vehicle.bh.crossbike",
# "vehicle.diamondback.century",
# "vehicle.gazelle.omafiets",
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
        # self.allroads = self._map.get_all_roads()
        blueprint_library = self._world.get_blueprint_library()
        self._tm_port = CarlaDataProvider.get_traffic_manager_port()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(self._tm_port)
        self.client = CarlaDataProvider.get_client()
        # 预期速度与当前限制速度之间的百分比差。
        # self._tm.global_percentage_speed_difference(0.0)
        self._rng = CarlaDataProvider.get_random_seed()
        self._attribute_filter = None
        # self._attribute_filter = {
        #     "base_type": "car",
        #     "special_type": "",
        #     "has_lights": True,
        # }
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

        self._vehicle_list_same = []
        self._vehicle_list_opp = []

        self._destroy_list = []
        self.centralObject = tf_param['centralObject']
        self.semiMajorAxis = int(tf_param['semiMajorAxis'])
        # self.semiMinorAxis = tf_param['semiMinorAxis']
        self.innerRadius = int(tf_param['innerRadius'])
        self.numberOfVehicles = int(tf_param['numberOfVehicles'])
        self.numberOfPedestrian = int(tf_param['numberOfPedestrian'])
        self.trafficDistribution = tf_param['trafficDistribution']
        self.directionOfTravelDistribution = tf_param['directionOfTravelDistribution']
        self.same = self.directionOfTravelDistribution['same']*0.01
        self.opposite = self.directionOfTravelDistribution['opposite']*0.01
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
            self._vehicle_leading_distance = 15
            self._vehicle_offset = 0.5
        # 车辆与生成半径约束关系
        self.max_vecs = (
            int(self.semiMajorAxis  * 0.15)
            if self.numberOfVehicles > int(self.semiMajorAxis  * 0.15)
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
        #
        self.same_road_vecs_num = 0
        self.tm_autopilot = False
        if tf_param['adversarialModel']['adversarialModelEnable']:
            self.tm_autopilot = True
        self.npcveccontrol = {}
        # self.npcveccontrol = {}
        # self.initialise()
        # self.road_graph  = None
        # self.openDriveXml = None
        # self.roads = {}
        # self.get_xodr_topology()

    # def get_xodr_topology(self,content='/home/zjx/software/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive'):
    #     map_name = self._map.name.split('/')[-1] + '.xodr'
    #     map_path = os.path.join(content,map_name)
    #     fh = open(map_path, "r")
    #     self.openDriveXml =  parse_opendrive( etree.parse(fh).getroot())
    #     for i in self.openDriveXml.roads:
    #         self.roads[i.id] =  i.link.successor.element_id
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

    def cal_dis_wp2loc(self,actor = None):
        if actor == None:
            return None
        close_wp = self._map.get_waypoint(actor.get_location())
        location = actor.get_location()
        return close_wp.transform.location.distance(location)

    def set_speed(self,vec_wp,i):
        # 如果在十字路口中或者前方是十字路口，则车辆可能会打滑，减速处理
        # 此处获取车辆前方五米wp，如果为空，返回None
        ahead_wp =vec_wp.next(60) if  len(vec_wp.next(60)) > 0 else None
        if vec_wp.is_junction or (ahead_wp and  ahead_wp[0].is_junction):
            self._tm.set_desired_speed(self._vehicle_list[i], 18)
        elif self.frame%30 == 0:
            if self.get_speed() * 3.6 > 6:
                speed = self.get_speed()*3.6*random.randint(2,3) + 5
            else :
                speed = 30
            self._tm.set_desired_speed(self._vehicle_list[i], speed)
        elif self.get_speed(self._vehicle_list[i]) <= 1:
            self._tm.set_desired_speed(self._vehicle_list[i], 30)

        # 此处判断保护，防止i溢出
        # elif i < len(self._vehicle_list) :
        #     if self.get_speed() > 5:
        #         self._tm.set_desired_speed(
        #             self._vehicle_list[i],
        #             float(
        #                 random.randint(
        #                     int(self.get_speed() * 0.8 * 3.6),
        #                     int(self.get_speed() * 1.5 * 3.6),
        #                 )
        #             ),
        #         )
        #     else:
        #         self._tm.set_desired_speed(self._vehicle_list[i], 30)
    def update1(self):
        self.frame += 1
        print("---------------------len(self._vehicle_list):",len(self._vehicle_list))
        self.max_vecs = 1
        ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
        if len(self._vehicle_list) < self.max_vecs:
            same_dir_wps = get_same_dir_lanes(ego_wp)
            opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
            self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, False)
        if self.frame > 30:
            for vec in self._vehicle_list:
                if np.sqrt(vec.get_velocity().x*vec.get_velocity().x+ vec.get_velocity().y * vec.get_velocity().y) < 1:
                    print("1111")
                if vec not in self.npcveccontrol:
                    args = {"desired_velocity":20 ,"desired_acceleration":5,"emergency_param":0.4, "desired_deceleration":5,"safety_time":5,
                            "lane_changing_dynamic":True,"urge_to_overtake":True,"obey_traffic_lights":True,
                            "identify_object":True,"obey_speed_limit":True
                            }
                    self.npcveccontrol[vec] = NpcVehicleControl(vec, args)
                    self.npcveccontrol[vec].run_step()
                else:
                    self.npcveccontrol[vec].run_step()
        return py_trees.common.Status.RUNNING

    def update(self):
        # print("---------------------")
        # try :
        # 显示车辆动力学参数
        # self._ego_actor.show_debug_telemetry()
        if True :
            flag = True
            logger.info(f"start traffic flow update!!!")
            ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
            self.frame += 1

            destroy_indexs = []
            # 临时变量
            front_tmpmax = 0
            opp_front_tmpmax = 0
            bake_tmpmax = 0
            opp_bake_tmpmax = 0
            self.same_road_vecs_num  = 0
            for i in range(len(self._vehicle_list)):
                # ego车与目标车的距离

                dist = (self._vehicle_list[i].get_location().distance(ego_wp.transform.location))
                wp2loc_dist = self.cal_dis_wp2loc(self._vehicle_list[i])

                vec_wp = self._map.get_waypoint(
                    self._vehicle_list[i].get_location()
                )
                if self.same == 0:
                    if ego_wp.road_id == vec_wp.road_id and  ego_wp.lane_id * vec_wp.lane_id > 0:
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

                # 每过一千帧，删除车流车头慢车
                if (
                    flag
                    and self.frame % 1200 == 0
                    and dist
                    > (80 if self.semiMajorAxis < 80 else self.semiMajorAxis * 0.6)
                ):
                    destroy_indexs.append(self._vehicle_list[i].id)
                    self._destroy_list.append(
                        carla.command.DestroyActor(self._vehicle_list[i])
                    )
                    continue

                # 断头路处理
                # ahead_wp = vec_wp.next(6)
                # if len(ahead_wp) == 0:
                #     destroy_indexs.append(self._vehicle_list[i].id)
                #     self._destroy_list.append(
                #         carla.command.DestroyActor(self._vehicle_list[i])
                #     )
                #     continue

                if (
                    dist
                    > (
                        self.semiMajorAxis
                    )
                    + self.get_speed() * 2
                ):  # 如果车辆与给定坐标的距离大于半径
                    destroy_indexs.append(self._vehicle_list[i].id)
                    self._destroy_list.append(
                        carla.command.DestroyActor(self._vehicle_list[i])
                    )
                # 更新前后距离
                elif (
                    self.get_local_location(
                        self._ego_actor, self._vehicle_list[i].get_location()
                    ).x
                    > 0
                    and dist > front_tmpmax
                    and ego_wp.road_id == vec_wp.road_id
                    and ego_wp.lane_id * vec_wp.lane_id > 0
                ):
                    front_tmpmax = dist
                elif (
                    self.get_local_location(
                        self._ego_actor, self._vehicle_list[i].get_location()
                    ).x
                    > 0
                    and dist > opp_front_tmpmax
                    and ego_wp.lane_id * vec_wp.lane_id < 0
                ):
                    opp_front_tmpmax = dist
                elif (
                    self.get_local_location(
                        self._ego_actor, self._vehicle_list[i].get_location()
                    ).x
                    < 0
                    and dist > bake_tmpmax
                    and ego_wp.lane_id * vec_wp.lane_id > 0
                ):
                    bake_tmpmax = dist
                elif (
                    self.get_local_location(
                        self._ego_actor, self._vehicle_list[i].get_location()
                    ).x
                    < 0
                    and dist > opp_bake_tmpmax
                    and ego_wp.lane_id * vec_wp.lane_id < 0
                ):
                    opp_bake_tmpmax = dist
                if ego_wp.road_id == vec_wp.road_id:
                    self.same_road_vecs_num += 1
            self.front_traffic_bound = front_tmpmax + self.semiMajorAxis * 0.1
            self.front_traffic_bound_opp = opp_front_tmpmax + self.semiMajorAxis * 0.1
            self.back_traffic_bound = bake_tmpmax + self.semiMajorAxis * 0.1
            self.back_traffic_bound_opp = opp_bake_tmpmax + self.semiMajorAxis * 0.1
            if len(destroy_indexs) > 0:
                self.client.apply_batch(self._destroy_list)
                self._vehicle_list = list(
                    filter(lambda x: x.id not in destroy_indexs, self._vehicle_list)
                )
                if not self.tm_autopilot:
                    self.npcveccontrol = {key: val for key, val in self.npcveccontrol.items() if key not in destroy_indexs}
            # 补充车辆
            logger.info(f"len vehicle list:{len(self._vehicle_list)}")
            logger.info(f"front_traffic_bound:{self.front_traffic_bound}")
            logger.info(f"front_traffic_bound_opp:{self.front_traffic_bound_opp}")
            logger.info(f"back_traffic_bound:{self.back_traffic_bound}")
            logger.info(f"back_traffic_bound_opp:{self.back_traffic_bound_opp}")

            if len(self._vehicle_list) < self.max_vecs:
                if self.same_road_vecs_num < self.same * self.numberOfVehicles:
                    same_dir_wps = get_same_dir_lanes(ego_wp)
                else:
                    same_dir_wps = []

                opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
                # self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, True)
                self._add_road_vecs(ego_wp, same_dir_wps, opposite_dir_wps, self.tm_autopilot)
            if not self.tm_autopilot:
                for vec in self._vehicle_list:
                    # print("vec speed: ",vec.id,  np.sqrt(np.square(vec.get_velocity().x)+np.square(vec.get_velocity().y) ) )
                    # print("vec control: ",vec.get_control())
                    # print("vec acceleration:",vec.get_acceleration())
                    # print("vec location:",vec.get_location() )
                    # print("vec is alive:",vec.is_alive)
                    if vec.is_alive:
                        if vec.id not in self.npcveccontrol:
                            args = {"desired_velocity": 10, "desired_acceleration": 5, "emergency_param": 0.4,
                                    "desired_deceleration": 5, "safety_time": 10,
                                    "lane_changing_dynamic": True, "urge_to_overtake": True,
                                    "obey_traffic_lights": True,
                                    "identify_object": True, "obey_speed_limit": False
                                    }
                            self.npcveccontrol[vec.id] = NpcVehicleControl(vec, args)
                            self.npcveccontrol[vec.id].set_init_speed()
                            # control = self.npcveccontrol[vec.id].run_step()
                            # if control:
                            #     vec.apply_control(control)
                        else:
                            # self.npcveccontrol[vec.id].set_target_speed(10*3.6)
                            control = self.npcveccontrol[vec.id].run_step()
                            if control:
                                vec.apply_control(control)
            logger.info(f"end of traffic flow update!!!")
            # 车辆规划
            if self.tm_autopilot:
                for i in range(len(self._vehicle_list)):
                    vec_wp = self._map.get_waypoint(
                        self._vehicle_list[i].get_location()
                    )
                    self.set_speed(vec_wp, i)
            # print("len of self._vehicle_list:",len(self._vehicle_list))
            return py_trees.common.Status.RUNNING
        # except Exception:
        #     logger.error(
        #         f"===============================An error occurred while update Oasis Traffic Flow!!!!=================================="
        #     )

    def get_filter_points(self,ego_vec_road_id):
        spawn_points_filtered = []
        num = 0
        for i, around_spawn_point in enumerate(
            self.apll_spawn_points
        ):  # 遍历所有出生点
            tmp_wpt = self._map.get_waypoint(around_spawn_point.location)
            diff_road = around_spawn_point.location.distance(
                self._ego_actor.get_location()
            )
            if (
                diff_road < self.semiMajorAxis
                and diff_road > self.innerRadius*2
                and self._map.get_waypoint(around_spawn_point.location).road_id
                != ego_vec_road_id
            ):  # 如果出生点与给定坐标的距离小于半径
                if num < abs(self.max_vecs - len(self._vehicle_list)):
                    num += 1
                    spawn_points_filtered.append(
                        tmp_wpt
                    )  # 将出生点添加到过滤后的列表中
                else:
                    break
        return spawn_points_filtered

    def _add_road_vecs(self, ego_wp, same_dir_wps, opposite_dir_wps, tm_autopilot=False):
        spawn_wps = []
        # offset_var = self.semiMajorAxis * 0.1 if self.semiMajorAxis * 0.1 > 15 else 15
        offset_var = 0

        speed_dist = self.get_speed()
        for wp in same_dir_wps:
            if self.numberOfVehicles * self.same <= 0:
                break
            same_num = int(self.numberOfVehicles * self.same / len(same_dir_wps))
            if same_num <1 and self.numberOfVehicles * self.same > 0:
                same_num = 1

            innerboundarywp = wp.next(self.innerRadius + 1)
            if len(innerboundarywp) == 0:
                continue
            next_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            # spawn_wps.insert(0, next_wp_queue[0])
            # 控制生成车辆车距
            offset = 0
            for _ in range(same_num):
                # self._road_spawn_dist = 15
                temp_next_wp_queue = []
                offset += offset_var
                for temp_wp in next_wp_queue:
                    temp_next_wps = temp_wp.next(
                        self.front_traffic_bound/2
                        + self._road_spawn_dist
                        + random.randint(0, 4) * 2
                        + speed_dist * 2
                        + offset
                    )
                    num_wps = len(temp_next_wps)
                    if num_wps <= 0:
                        continue
                    # 前方发现多个waypoint 随机抽取一个
                    elif num_wps > 1:
                        temp_next_wp = temp_next_wps[random.randint(0, num_wps - 1)]
                    else:
                        temp_next_wp = temp_next_wps[0]

                    dist = temp_next_wp.transform.location.distance(
                        self._ego_actor.get_location()
                    )

                    if dist > self.semiMajorAxis + speed_dist * 2:
                        continue
                    if not self._check_junction_spawnable(temp_next_wp):
                        continue # Stop when there's no next or found a junction
                    temp_next_wp_queue.append(temp_next_wp)
                    spawn_wps.insert(0, temp_next_wp)
                next_wp_queue = temp_next_wp_queue

            innerboundarywp = wp.previous(self.innerRadius + 1)
            if len(innerboundarywp) <= 0:
                continue
            prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            # spawn_wps.insert(0, prev_wp_queue[0])
            offset = 0
            for _ in range(same_num):
                temp_prev_wp_queue = []
                offset += offset_var
                for temp_wp in prev_wp_queue:
                    if ego_wp.lane_id == temp_wp.lane_id:
                        continue
                    temp_prev_wps = temp_wp.previous(
                        self.back_traffic_bound/2
                        + self._road_spawn_dist
                        + random.randint(0, 3)
                        + speed_dist * 2
                    )
                    num_wps = len(temp_prev_wps)
                    if num_wps <= 0:
                        continue
                    # 前方发现多个waypoint 随机抽取一个
                    elif num_wps > 1:
                        temp_prev_wp = temp_prev_wps[random.randint(0, num_wps - 1)]
                    else:
                        temp_prev_wp = temp_prev_wps[0]
                    dist = temp_prev_wp.transform.location.distance(
                        self._ego_actor.get_location()
                    )
                    if dist > self.semiMajorAxis + speed_dist * 2:
                        continue
                    if not self._check_junction_spawnable(temp_prev_wp):
                        continue  # Stop when there's no next or found a junction
                    temp_prev_wp_queue.append(temp_prev_wp)
                    spawn_wps.append(temp_prev_wp)
                prev_wp_queue = temp_prev_wp_queue

        opp_spawn_wps = []
        for wp in opposite_dir_wps:
            opposite_num = int(self.numberOfVehicles * self.opposite / len(opposite_dir_wps))
            if opposite_num < 1 and self.numberOfVehicles * self.opposite > 0:
                opposite_num = 1
            elif self.numberOfVehicles *     self.opposite <= 0:
                break
            innerboundarywp = wp.previous(self.innerRadius + 1)
            prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            # opp_spawn_wps.insert(0, prev_wp_queue[0])
            # for _ in range(self._road_back_vehicles):
            offset = 0
            for _ in range(opposite_num):
                temp_prev_wp_queue = []
                offset += offset_var
                for temp_wp in prev_wp_queue:
                    temp_prev_wps = temp_wp.previous(
                        self.front_traffic_bound_opp
                        + self._road_spawn_dist
                    )
                    num_wps = len(temp_prev_wps)
                    if num_wps <= 0:
                        continue
                    # 前方发现多个waypoint 随机抽取一个
                    elif num_wps > 1:
                        temp_prev_wp = temp_prev_wps[random.randint(0, num_wps - 1)]
                    else:
                        temp_prev_wp = temp_prev_wps[0]

                    dist = temp_prev_wp.transform.location.distance(
                        self._ego_actor.get_location()
                    )
                    if dist > self.semiMajorAxis + speed_dist * 2:
                        continue
                    if not self._check_junction_spawnable(temp_prev_wp):
                        continue  # Stop when there's no next or found a junction
                    temp_prev_wp_queue.append(temp_prev_wp)
                    opp_spawn_wps.append(temp_prev_wp)
                prev_wp_queue = temp_prev_wp_queue
        # if len(spawn_wps) >0:
            # print("===================len of spawn_wps:====", len(spawn_wps) )
        if len(spawn_wps) > 0 or len(opp_spawn_wps) > 0:
            random.shuffle(spawn_wps)
            random.shuffle(opp_spawn_wps)

            # spawn_wps = spawn_wps[:int(len(spawn_wps)*0.8)]
            loop = 0
            gl_spawn_wps = []
            spawn_wps_index_first = 0
            spawn_wps_index_second = 100
            opp_spawn_wps_index_first = 0
            opp_spawn_wps_index_second = 100
            while loop < 4:
                spawn_wps_index_second = (
                    int(np.ceil(0.6 * (self.max_vecs - len(self._vehicle_list))))
                    if len(spawn_wps)
                    > np.ceil(0.6 * (self.max_vecs - len(self._vehicle_list)))
                    else int((len(spawn_wps) - 1))
                )
                opp_spawn_wps_index_second = (
                    int(np.floor((self.max_vecs - len(self._vehicle_list)) * 0.5))
                    if len(opp_spawn_wps)
                    > np.floor((self.max_vecs - len(self._vehicle_list)) * 0.5)
                    else int((len(opp_spawn_wps) - 1))
                )
                if spawn_wps_index_second < len(spawn_wps):
                    gl_spawn_wps += spawn_wps[
                        spawn_wps_index_first:spawn_wps_index_second
                    ]
                    spawn_wps_index_first = spawn_wps_index_second
                if opp_spawn_wps_index_second < len(opp_spawn_wps):
                    gl_spawn_wps += opp_spawn_wps[
                        opp_spawn_wps_index_first:opp_spawn_wps_index_second
                    ]
                    opp_spawn_wps_index_first = opp_spawn_wps_index_second
                if (
                    len(gl_spawn_wps) > self.max_vecs - len(self._vehicle_list) + 1

                ):
                    break
                loop += 1

            # 同方向与反方向补充出生点小于最大车辆数
            if len(self._vehicle_list) + len(gl_spawn_wps) < self.max_vecs:
                gap = self.max_vecs - (len(self._vehicle_list) + len(gl_spawn_wps))
                if gap > 10:
                    gap = 10
                spawn_points_filtered = self.get_filter_points(ego_wp.road_id)
                # 补充出生点大于0
                if len(spawn_points_filtered) > 0:
                    gl_spawn_wps += spawn_points_filtered[
                        : len(spawn_points_filtered) - 1 if  gap > len(spawn_points_filtered) else gap - 1
                        ]
            tmp_vecs = self._spawn_actors(gl_spawn_wps,tm_autopilot=tm_autopilot)
            for i in tmp_vecs:
                self._tm.set_desired_speed(i, 10)
                # self._tm.vehicle_percentage_speed_difference(i, -10)
                # if self.get_speed() > 5:
                #     self._tm.set_desired_speed(
                #         i,
                #         float(
                #             random.randint(
                #                 int(self.get_speed() * 0.8 * 3.6),
                #                 int(self.get_speed() * 2 * 3.6),
                #             )
                #         ),
                #     )
                # else:
                #     # self._tm.set_desired_speed(i, float(random.randint(int(self.max_speed*0.5), int(self.max_speed))))

            self._vehicle_list = list(set(self._vehicle_list).union(set(tmp_vecs)))

    def _initialise_road_behavior(self, ego_wp, road_wps, rdm=False):
        """
        Initialises the road behavior, consisting on several vehicle in front of the ego,
        and several on the back and are only spawned outside junctions.
        If there aren't enough actors behind, road sources will be created that will do so later on
        """
        # Vehicles in front
        spawn_wps = []
        for wp in road_wps:
            # Front spawn points
            innerboundarywp = wp.next( self.innerRadius + 1 )
            if len(innerboundarywp) <= 0:
                continue
            next_wp_queue = [ innerboundarywp[ random.randint(0,len(innerboundarywp) - 1 ) ] ]
            spawn_wps.insert(0, next_wp_queue[0])
            for _ in range(self._road_front_vehicles):
                temp_next_wp_queue = []
                for temp_wp in next_wp_queue:
                    # 获取 wp
                    temp_next_wps = temp_wp.next(self._road_spawn_dist + random.randint(0, 10))
                    num_wps = len(temp_next_wps)
                    if num_wps <= 0:
                        continue
                    # 前方发现多个waypoint 随机抽取一个
                    elif num_wps > 1:
                        temp_next_wp = temp_next_wps[ random.randint(0,num_wps - 1) ]
                    else:
                        temp_next_wp = temp_next_wps[0]
                    # 超出限定范围丢弃
                    dist = temp_next_wp.transform.location.distance(wp.transform.location)
                    if dist > self.semiMajorAxis:
                        continue
                    if not self._check_junction_spawnable(temp_next_wp):
                        continue  # Stop when there's no next or found a junction
                    temp_next_wp_queue.append(temp_next_wp)
                    spawn_wps.insert(0, temp_next_wp)
                next_wp_queue = temp_next_wp_queue

            innerboundarywp = wp.previous( self.innerRadius + 1 )

            prev_wp_queue = [innerboundarywp[random.randint(0, len(innerboundarywp) - 1)]]
            spawn_wps.insert(0, prev_wp_queue[0])
            for _ in range(self._road_back_vehicles):
                temp_prev_wp_queue = []
                for temp_wp in prev_wp_queue:
                    if ego_wp.lane_id == temp_wp.lane_id:
                        continue
                    temp_prev_wps = temp_wp.previous(
                        self._road_spawn_dist + random.randint(0, 10)
                    )

                    num_wps = len(temp_prev_wps)
                    if num_wps <= 0:
                        continue
                    # 前方发现多个waypoint 随机抽取一个
                    elif num_wps > 1:
                        temp_prev_wp = temp_prev_wps[ random.randint(0,num_wps - 1) ]
                    else:
                        temp_prev_wp = temp_prev_wps[0]

                    if not self._check_junction_spawnable(temp_prev_wp):
                        continue  # Stop when there's no next or found a junction
                    temp_prev_wp_queue.append(temp_prev_wp)
                    spawn_wps.append(temp_prev_wp)
                prev_wp_queue = temp_prev_wp_queue

        random.shuffle(spawn_wps)
        spawn_wps = spawn_wps[0 : self.max_vecs if len(spawn_wps) > self.max_vecs else len(spawn_wps)]
        # spawn_wps = spawn_wps[:int(len(spawn_wps)/2)]
        start = time.time()
        self._vehicle_list = list(
            set(self._vehicle_list).union(set(self._spawn_actors(spawn_wps)))
        )
        for i in self._vehicle_list:
            # self._tm.set_desired_speed(i, float(random.randint(int(self.max_speed*0.5), int(self.max_speed))))
            self._tm.vehicle_percentage_speed_difference(i, -10)
        dur_time = time.time() - start


    def _spawn_actors(self, spawn_wps, ego_dist=0,tm_autopilot = True):
        """Spawns several actors in batch"""
        spawn_transforms = []
        ego_location = self._ego_actor.get_location()
        for wp in spawn_wps:
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
            [x for x in range(len(self.vehicles_ratio))],size=len(spawn_transforms), p=self.vehicles_ratio
        )
        random.shuffle(chosen_vehicle_class)
        # self.vehicle_models = [CARLA_TYPE_TO_VEHICLE[t] for t in chosen_vehicle_class ]
        vehicle_model_list = [list(CARLA_TYPE_TO_VEHICLE.keys())[t] for t in chosen_vehicle_class]
        self.vehicle_models_list = []
        for obj_type in vehicle_model_list:
            if CARLA_TYPE_TO_VEHICLE[obj_type]:
                self.vehicle_models_list.append(random.choice( CARLA_TYPE_TO_VEHICLE[obj_type] ))
            else:
                self.vehicle_models_list.append(random.choice(CARLA_TYPE_TO_VEHICLE['car']))
        actors = CarlaDataProvider.request_new_batch_actors_with_specified_model_sets(
            self.vehicle_models_list,
            len(spawn_transforms),
            spawn_transforms,
            tm_autopilot,
            False,
            "background",
            attribute_filter=self._attribute_filter,
            tick=False,
            veloc=ego_speed if ego_speed > 2 else 2,
        )

        if not actors:
            return actors

        for actor in actors:
            self._initialise_actor(actor)

        return actors

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
        self._tm.vehicle_lane_offset(actor, self._vehicle_offset)

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
                destroy_list.append( carla.command.DestroyActor(i) )
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


