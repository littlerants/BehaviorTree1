import  carla
import  time
import  py_trees
import  random
import  numpy as np
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.tools.scenario_helper import get_same_dir_lanes, get_opposite_dir_lanes
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# EGO_ROAD = 'road'

class BAtrafficflow(AtomicBehavior):
    """
    Handles the background activity
    """
    def __init__(self, ego_actor, tf_param = None, debug=False, name="BAtrafficflow"):
        """
        Setup class members
        """

        super(BAtrafficflow, self).__init__(name)
        self.debug = debug
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        self._tm_port = CarlaDataProvider.get_traffic_manager_port()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(self._tm_port)
        self.client = CarlaDataProvider.get_client()
        # 预期速度与当前限制速度之间的百分比差。
        self._tm.global_percentage_speed_difference(0.0)
        self._rng = CarlaDataProvider.get_random_seed()

        self._attribute_filter = {'base_type': 'car', 'special_type': '', 'has_lights': True, }

        # Global variables
        self._ego_actor = ego_actor
        # self._ego_state = EGO_ROAD
        self._ego_wp = None
        self._ego_key = ""
        self._route_index = 0
        # 将route信息进行解析，初始化若干参数
        # self._get_route_data(route)
        self._actors_speed_perc = {}  # Dictionary actor - percentage
        self._all_actors = []
        self._lane_width_threshold = 2.25  # Used to stop some behaviors at narrow lanes to avoid problems [m]

        self._spawn_vertical_shift = 0.2
        self._reuse_dist = 10  # When spawning actors, might reuse actors closer to this distance
        self._spawn_free_radius = 20  # Sources closer to the ego will not spawn actors
        self._fake_junction_ids = []
        self._fake_lane_pair_keys = []

        # Initialisation values
        self._vehicle_lane_change = True
        self._vehicle_lights = True
        self._vehicle_leading_distance = 10
        self._vehicle_offset = 0.1

        # Road variables
        self._road_dict = {}  # Dictionary lane key -> actor source
        self._road_checker_index = 0

        self._road_front_vehicles = 2  # Amount of vehicles in front of the ego
        self._road_back_vehicles = 2 # Amount of vehicles behind the ego
        self._radius_increase_ratio = 1.7  # Meters the radius increases per m/s of the ego

        self._base_junction_detection = 30
        self._detection_ratio = 1.5  # Meters the radius increases per m/s of the ego

        self._road_extra_front_actors = 0  # For cases where we want more space but not more vehicles

        self._road_extra_space = 0  # Extra space for the road vehicles

        self._active_road_sources = True

        self._base_min_radius = 0
        self._base_max_radius = 0
        self._min_radius = 0
        self._max_radius = 0
        self._detection_dist = 0
        # self._get_road_radius()

        # Junction variables
        self._junctions = []  # List with all the junctions part of the route, in order of appearance
        self._active_junctions = []  # List of all the active junctions

        self._junction_sources_dist = 40  # Distance from the entry sources to the junction [m]
        self._junction_sources_max_actors = 6  # Maximum vehicles alive at the same time per source
        self._junction_spawn_dist = 15  # Distance between spawned vehicles [m]
        self._junction_minimum_source_dist = 15  # Minimum distance between sources and their junction

        self._junction_source_perc = 80  # Probability [%] of the source being created

        # Opposite lane variables
        self._opposite_actors = []
        self._opposite_sources = []
        self._opposite_route_index = 0

        self._opposite_spawn_dist = 40  # Distance between spawned vehicles [m]
        self._opposite_sources_dist = 80  # Distance from the ego to the opposite sources [m]. Twice the spawn distance

        self._active_opposite_sources = True  # Flag to (de)activate all opposite sources

        # Scenario variables:
        self._scenario_stopped_actors = []  # Actors stopped by a hard break scenario
        self._scenario_stopped_back_actors = []  # Actors stopped by a open doors scenario
        self._scenario_max_speed = 0  # Max speed of the Background Activity. Deactivated with a value of 0
        self._scenario_junction_entry = False  # Flag indicating the ego is entering a junction
        self._road_spawn_dist = 15  # Distance between spawned vehicles [m]
        self._scenario_junction_entry_distance = self._road_spawn_dist  # Min distance between vehicles and ego
        self._scenario_removed_lane = False  # Flag indicating a scenario has removed a lane
        self._scenario_remove_lane_offset = 0
        self.frame = 1
        self.other_vecs = []

        self._vehicle_list = []
        self._destroy_list = []

        self._road_spawn_dist = 15  # Distance between spawned vehicles [m]
        if tf_param == None:
            tf_param = {}
            tf_param['radius'] = 1
            tf_param['vehicle_num'] = 0
        self.max_radius = tf_param['radius']
        # 车辆与生成半径约束关系
        self.max_vecs = int(tf_param['radius'] * 0.4) if tf_param['vehicle_num'] > int(tf_param['radius'] * 0.4) else tf_param['vehicle_num']
        # 前边界
        self.front_traffic_bound = 15
        # 反向车道前边界
        self.front_traffic_bound_opp = 15
        # 后边界
        self.back_traffic_bound = 15
        # 反向车道后边界
        self.back_traffic_bound_opp = 15
        self.apll_spawn_points = self._world.get_map().get_spawn_points()

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        # 获取主车初始位置
        ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
        same_dir_wps = get_same_dir_lanes(ego_wp)
        opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
        # # 初始化辆车
        self._initialise_road_behavior(ego_wp,same_dir_wps + opposite_dir_wps)

    def update(self):
        flag = True
        ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
        self.frame += 1
        if flag and self.frame % 5 == 0:
            destroy_indexs = []
            # 临时变量
            front_tmpmax = 0
            opp_front_tmpmax = 0
            bake_tmpmax = 0
            opp_bake_tmpmax = 0
            if self.frame % 100 == 0:
                for i in range(len(self._vehicle_list)):
                    dist = self._vehicle_list[i].get_location().distance(ego_wp.transform.location)
                    vec_wp = self._map.get_waypoint(self._vehicle_list[i].get_location())
                    # 每过一千帧，删除车流车头慢车
                    if flag and self.frame % 1200 == 0 and dist > (80 if self.max_radius < 80 else self.max_radius * 0.6):
                        destroy_indexs.append( self._vehicle_list[i].id )
                        self._destroy_list.append(carla.command.DestroyActor(self._vehicle_list[i]))
                    if dist > (self.max_radius if self.max_radius > 100 else self.max_radius * 1.5) + self.get_speed()*2 :  # 如果车辆与给定坐标的距离大于半径
                        destroy_indexs.append( self._vehicle_list[i].id )
                        self._destroy_list.append(carla.command.DestroyActor(self._vehicle_list[i]))
                    # 更新前后距离
                    elif self.get_local_location(self._ego_actor, self._vehicle_list[i].get_location()).x > 0 and  dist > front_tmpmax and ego_wp.lane_id * vec_wp.lane_id > 0  :
                        front_tmpmax= dist
                    elif self.get_local_location(self._ego_actor, self._vehicle_list[i].get_location()).x > 0 and  dist > opp_front_tmpmax and ego_wp.lane_id * vec_wp.lane_id <  0  :
                        opp_front_tmpmax= dist
                    elif self.get_local_location(self._ego_actor, self._vehicle_list[i].get_location()).x <  0 and  dist > bake_tmpmax and ego_wp.lane_id * vec_wp.lane_id > 0 :
                        bake_tmpmax = dist
                    elif self.get_local_location(self._ego_actor, self._vehicle_list[i].get_location()).x <  0 and  dist > opp_bake_tmpmax and ego_wp.lane_id * vec_wp.lane_id <  0  :
                        opp_bake_tmpmax = dist
                self.front_traffic_bound = front_tmpmax + self.max_radius * 0.1
                self.front_traffic_bound_opp = opp_front_tmpmax + self.max_radius * 0.1
                self.back_traffic_bound = bake_tmpmax + self.max_radius*0.1
                self.back_traffic_bound_opp = opp_bake_tmpmax + self.max_radius*0.1
            if  len(destroy_indexs) > 0:
                self.client.apply_batch(self._destroy_list)
                self._vehicle_list = list(filter(  lambda  x: x.id not in destroy_indexs, self._vehicle_list  ))

            # 补充车辆
            print("len vehicle list:", len(self._vehicle_list))
            if self.frame % 100 == 0 and  len(self._vehicle_list) < self.max_vecs:
                same_dir_wps = get_same_dir_lanes(ego_wp)
                opposite_dir_wps = get_opposite_dir_lanes(ego_wp)
                self._add_road_vecs(ego_wp,same_dir_wps, opposite_dir_wps,True)
        return py_trees.common.Status.RUNNING

    def _add_road_vecs(self,ego_wp,same_dir_wps, opposite_dir_wps,rdm=False):
        spawn_wps = []
        offset_var = self.max_radius * 0.1 if self.max_radius * 0.1 > 15 else 15
        speed_dist = self.get_speed()
        for wp in same_dir_wps:
            next_wp = wp
            # 控制生成车辆车距
            offset = 0
            for _ in range(4):
                # self._road_spawn_dist = 15
                next_wps = next_wp.next(self.front_traffic_bound + self._road_spawn_dist + random.randint(0,4)*3 + speed_dist * 2 + offset  )
                if len(next_wps) <= 0:
                    break
                offset += offset_var
                dist = next_wps[0].transform.location.distance(self._ego_actor.get_location())
                # print("front spawn vec dist: ", dist)
                if dist > self.max_radius + speed_dist * 2 :
                    break
                if len(next_wps) != 1 or self._is_junction(next_wps[0]):
                    break  # Stop when there's no next or found a junction
                next_wp = next_wps[0]
                spawn_wps.insert(0, next_wp)
            prev_wp = wp
            if ego_wp.lane_id == prev_wp.lane_id:
                continue
            offset = 0
            for _ in range(6):
                prev_wps = prev_wp.previous(self.back_traffic_bound + self._road_spawn_dist+ random.randint(0,3)*3 + speed_dist*1 + offset   )
                if len(prev_wps) <= 0:
                    break
                offset +=offset_var
                dist = prev_wps[0].transform.location.distance(self._ego_actor.get_location())
                if dist > self.max_radius + speed_dist * 2 :
                    break
                if len(prev_wps) != 1 or self._is_junction(prev_wps[0]):
                    break  # Stop when there's no next or found a junction
                prev_wp = prev_wps[0]
                spawn_wps.append(prev_wp)
        opp_spawn_wps = []
        for wp in opposite_dir_wps:
            prev_wp = wp
            # for _ in range(self._road_back_vehicles):
            offset = 0
            for _ in range(3):
                prev_wps = prev_wp.previous(  self.front_traffic_bound_opp + self._road_spawn_dist+ random.randint(0,4)*3 + offset   )
                if len(prev_wps) <= 0:
                    break
                offset += offset_var
                dist = prev_wps[0].transform.location.distance(self._ego_actor.get_location())
                if dist > self.max_radius + speed_dist * 2 :
                    break
                if len(prev_wps) != 1 or self._is_junction(prev_wps[0]):
                    break  # Stop when there's no next or found a junction
                prev_wp = prev_wps[0]
                opp_spawn_wps.append(prev_wp)
        spawn_points_filtered = []
        num = 0
        if len(spawn_wps) + len(opp_spawn_wps) < self.max_vecs:
            for i, around_spawn_point in enumerate(self.apll_spawn_points):  # 遍历所有出生点
                tmp_wpt = self._map.get_waypoint(around_spawn_point.location)
                diff_road = around_spawn_point.location.distance(self._ego_actor.get_location())
                if  diff_road < self.max_radius  and diff_road > self.max_radius * 0.5 and  self._map.get_waypoint(around_spawn_point.location).road_id  != same_dir_wps[0].road_id:  # 如果出生点与给定坐标的距离小于半径
                    if num < abs(self.max_vecs -len( self._vehicle_list)):
                        print("spawn points to ego dist:",
                              around_spawn_point.location.distance(self._ego_actor.get_location()))
                        num += 1
                        spawn_points_filtered.append(tmp_wpt)  # 将出生点添加到过滤后的列表中
                    else :
                        break
        print("len(spawn_points_filtered):",len(spawn_points_filtered))
        if len(spawn_wps) > 0:
            random.shuffle(spawn_wps)
            random.shuffle(opp_spawn_wps)

            # spawn_wps = spawn_wps[:int(len(spawn_wps)*0.8)]
            loop = 0
            gl_spawn_wps = []
            spawn_wps_index_first = 0
            spawn_wps_index_second = 100
            opp_spawn_wps_index_first = 0
            opp_spawn_wps_index_second = 100
            while True:
                spawn_wps_index_second = int(np.ceil(0.6 * (self.max_vecs - len(self._vehicle_list)))) if len(spawn_wps) > np.ceil(0.6 * (self.max_vecs - len(self._vehicle_list))) else int((len(spawn_wps) - 1))
                opp_spawn_wps_index_second = int(np.floor((self.max_vecs - len(self._vehicle_list)) * 0.5)) if len(opp_spawn_wps) > np.floor((self.max_vecs - len(self._vehicle_list)) * 0.5) else int((len(opp_spawn_wps) - 1))
                if spawn_wps_index_second < len(spawn_wps) :
                    gl_spawn_wps += spawn_wps[spawn_wps_index_first:spawn_wps_index_second]
                    spawn_wps_index_first = spawn_wps_index_second
                if opp_spawn_wps_index_second < len(opp_spawn_wps) :
                    gl_spawn_wps += opp_spawn_wps[opp_spawn_wps_index_first:opp_spawn_wps_index_second]
                    opp_spawn_wps_index_first = opp_spawn_wps_index_second
                if len(gl_spawn_wps) > self.max_vecs - len(self._vehicle_list) + 1 or loop > 5 :
                    break
                loop +=1
            gl_spawn_wps += spawn_points_filtered[:int(np.floor(self.max_vecs * 0.4)) if len(spawn_wps) > np.floor(self.max_vecs * 0.4) else int( (len(spawn_wps) - 1) ) ]
            tmp_vecs = self._spawn_actors(gl_spawn_wps)
            for i in tmp_vecs:
                if self.get_speed() > 20:
                    self._tm.set_desired_speed(i, float(random.randint(int(self.get_speed() * 0.8 *3.6 ), int(self.get_speed()* 1.6*3.6 )    ) ))
                else:
                    self._tm.set_desired_speed(i, float(
                        random.randint( 60, 100)))
            self._vehicle_list = list(set(self._vehicle_list).union(set( tmp_vecs ) )  )

    def _initialise_road_behavior(self, ego_wp,road_wps,rdm= False):
        """
        Initialises the road behavior, consisting on several vehicle in front of the ego,
        and several on the back and are only spawned outside junctions.
        If there aren't enough actors behind, road sources will be created that will do so later on
        """
        # Vehicles in front
        spawn_wps = []
        for wp in road_wps:

            # Front spawn points
            next_wp = wp
            for _ in range(self._road_front_vehicles):
                next_wps = next_wp.next(self._road_spawn_dist + random.randint(0,15) )
                if len(next_wps) <= 0:
                    break
                dist = next_wps[0].transform.location.distance(wp.transform.location)
                if dist > self.max_radius:
                    break
                if len(next_wps) != 1 or self._is_junction(next_wps[0]):
                    break  # Stop when there's no next or found a junction
                next_wp = next_wps[0]
                spawn_wps.insert(0, next_wp)

            prev_wp = wp
            if ego_wp.lane_id == prev_wp.lane_id:
                continue
            print("spawn_wps:", spawn_wps)
            for _ in range(self._road_back_vehicles):
                prev_wps = prev_wp.previous(self._road_spawn_dist + random.randint(0,15))
                if len(prev_wps) != 1 or self._is_junction(prev_wps[0]):
                    break  # Stop when there's no next or found a junction
                prev_wp = prev_wps[0]
                spawn_wps.append(prev_wp)

        random.shuffle(spawn_wps)
        spawn_wps = spawn_wps[0:self.max_vecs ]
        # spawn_wps = spawn_wps[:int(len(spawn_wps)/2)]
        start = time.time()
        self._vehicle_list = list(set(self._vehicle_list).union(set( self._spawn_actors(spawn_wps) ) )  )
        for i in self._vehicle_list:
            self._tm.set_desired_speed(i , float(random.randint(60, 120)))
        dur_time = time.time() - start
        print("spawn time:",dur_time)


    def _spawn_actors(self, spawn_wps, ego_dist=0):
        """Spawns several actors in batch"""
        spawn_transforms = []
        ego_location = self._ego_actor.get_location()
        for wp in spawn_wps:
            if ego_location.distance(wp.transform.location) < ego_dist:

                continue
            spawn_transforms.append(
                carla.Transform(wp.transform.location + carla.Location(z=self._spawn_vertical_shift),
                                wp.transform.rotation)
            )
        ego_speed = self.get_speed()
        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_transforms), spawn_transforms, True, False, 'background',
            attribute_filter=self._attribute_filter, tick=False,veloc=ego_speed if ego_speed > 2 else 2  )

        if not actors:
            return actors

        for actor in actors:
            self._initialise_actor(actor)

        return actors

    def _is_junction(self, waypoint):
        if not waypoint.is_junction or waypoint.junction_id in self._fake_junction_ids:
            return False
        return True

    def get_speed(self,actor= None):
        if actor == None:
            return np.sqrt( np.square(self._ego_actor.get_velocity().x)  +  np.square(self._ego_actor.get_velocity().y) )
        return np.sqrt(np.square(actor.get_velocity().x) + np.square(actor.get_velocity().y))

    def _initialise_actor(self, actor):
        """
        Save the actor into the needed structures, disable its lane changes and set the leading distance.
        """
        self._tm.auto_lane_change(actor, self._vehicle_lane_change)
        self._tm.update_vehicle_lights(actor, self._vehicle_lights)
        self._tm.distance_to_leading_vehicle(actor, self._vehicle_leading_distance +  random.randint(0,4)*3)
        self._tm.vehicle_lane_offset(actor, self._vehicle_offset)

    def get_local_location(self,vehicle, location) -> carla.Location:
        """将全局坐标系下的坐标转到局部坐标系下

        Args:
            location (Location): 待变换的全局坐标系坐标
        """
        res = np.array(vehicle.get_transform().get_inverse_matrix()).dot(
            np.array(
                [location.x, location.y, location.z, 1]
            )
        )
        return carla.Location(x=res[0], y=res[1], z=res[2])

    def terminate(self, new_status):
        """Destroy all actors"""
        all_actors = list(self._actors_speed_perc)
        for actor in list(all_actors):
            self._destroy_actor(actor)
        super(BAtrafficflow, self).terminate(new_status)