import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional, Union
from gym_sumo.road.lane import LineType, StraightLane, AbstractLane, lane_from_config
from gym_sumo.vehicle.objects import Landmark
from gym_sumo.algo.global_route_planner_sumo import GlobalRoutePlanner
if TYPE_CHECKING:
    from gym_sumo.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[Union[int, float, str], Union[int, float, str], int]
Route = List[LaneIndex]
import networkx as nx

class RoadNetwork(object):
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self, global_route_planner: GlobalRoutePlanner=None):
        self.graph = {}
        self.global_route_planner = global_route_planner

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        try:
            _from, _to, _id = index
            if _id is None and len(self.graph[_from][_to]) == 1:
                _id = 0
            lane = self.graph[_from][_to][_id]
        except Exception as e:
            print(index)
            print(e)
            raise KeyError

        return lane

    def get_closest_lane_index(self, position: np.ndarray, heading: Optional[float] = None,
                               lane_index: LaneIndex = None, target_lane_index: LaneIndex = None) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        if lane_index is None or self.global_route_planner is None:
            for _from, to_dict in self.graph.items():
                for _to, lanes in to_dict.items():
                    for _id, l in enumerate(lanes):
                        distances.append(l.distance_with_heading(position, heading))
                        indexes.append((_from, _to, _id))
        else:
            lanes = [lane_index]
            if target_lane_index is not None:
                lanes.append(target_lane_index)
            sides_lanes = self.side_lanes(lane_index)
            next_lanes = self.get_all_next_lanes(lane_index)
            lanes.extend(sides_lanes)
            lanes.extend(next_lanes)
            for l in lanes:
                lane = self.get_lane(l)
                distances.append(lane.distance_with_heading(position, heading))
                indexes.append(l)
        return indexes[int(np.nanargmin(distances))]

    def get_lane_index_by_road(self, road_id, section_id, lane_id):
        return self.global_route_planner.road_id_to_edge[road_id][section_id][lane_id]

    def check_edges(self, edges: list) -> list:
        result_edges = []
        for e in edges:
            edge = self.global_route_planner.graph.get_edge_data(*e)
            if edge is None:
                print(f'warning: edge {e} is none ')
                continue
            if edge['length'] > 0:
                result_edges.append(e)
        return result_edges

    def next_lane(self, current_index: LaneIndex, route: Route = None, position: np.ndarray = None,
                  np_random: np.random.RandomState = np.random, vehciel_length: float = None, pop=True) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current target lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :pop: 如果为true是才可以pop　route，防止其他地方调用该方法时，重复删除全局路径
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = current_index
        next_id = 0

        # Pick next road according to planned route
        current_lane = self.get_lane(current_index)
        lon, _ = current_lane.local_coordinates(position)
        if route and len(route) > 2:
            if lon > current_lane.length - vehciel_length and pop:
                route.pop(0)
                while len(route) > 2:
                    e = self.global_route_planner.graph.get_edge_data(route[0], route[1])
                    if e.get('length') > 0:
                        break
                    route.pop(0)

            if not pop:
                while len(route) > 2:
                    e = self.global_route_planner.graph.get_edge_data(route[0], route[1])
                    if e.get('length') > 0:
                        break
                    route.pop(0)
            try:
                next_to = (route[0], route[1])
            except Exception as e:
                print(e)
        else:
            if lon > (current_lane.length - vehciel_length):
                e = self.global_route_planner.graph.get_edge_data(current_index[0], current_index[1])
                follow_edge = self.check_edges(e.get('follow_edge'))
                try:
                    next_to = np_random.choice(follow_edge, 1).squeeze()
                except:
                    _next_lane = []
                    left_lane = self.get_left_lane(current_index)
                    if left_lane is not None:
                        left_e = self.global_route_planner.graph.get_edge_data(left_lane[0], left_lane[1])
                        _next_lane.extend(self.check_edges(left_e.get('follow_edge')))

                    right_lane = self.get_right_lane(current_index)
                    if right_lane is not None:
                        right_e = self.global_route_planner.graph.get_edge_data(right_lane[0], right_lane[1])
                        _next_lane.extend(self.check_edges(right_e.get('follow_edge')))

                    if len(_next_lane) <= 0:
                        next_to = (None, None)
                    else:
                        next_to = np_random.choice(_next_lane, 1).squeeze()

        return next_to[0], next_to[1], next_id

    def get_all_next_lanes(self, lane_index):
        edges = self.global_route_planner.graph.out_edges(lane_index[1])
        _, real_lane = self.get_son_node(edges)
        return real_lane

    def can_change_lane(self, lane_index: LaneIndex) -> Tuple[bool, bool]:
        left_change_lane, right_change_lane = False, False
        e = self.global_route_planner.graph.get_edge_data(lane_index[0], lane_index[1])
        if e.get('left_edge') is not None:
            left_change_lane = True

        if e.get('right_edge') is not None:
            right_change_lane = True

        return left_change_lane, right_change_lane

    def get_son_node(self, edges):
        link_lane = []
        real_lane = []
        for edge in edges:
            e = self.global_route_planner.graph.get_edge_data(edge[0], edge[1])
            if e.get('length') > 0:
                real_lane.append((edge[0], edge[1], 0))
            else:
                link_lane.append((edge[0], edge[1], 0))

        return link_lane, real_lane

    # def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
    #     """
    #     Breadth-first search of all routes from start to goal.
    #
    #     :param start: starting node
    #     :param goal: goal node
    #     :return: list of paths from start to goal.
    #     """
    #     queue = [(start, [start])]
    #     while queue:
    #         (node, path) = queue.pop(0)
    #         if node not in self.graph:
    #             yield []
    #         for _next in set(self.graph[node].keys()) - set(path):
    #             if _next == goal:
    #                 yield path + [_next]
    #             elif _next in self.graph:
    #                 queue.append((_next, path + [_next]))

    def shortest_path(self, start: Tuple, goal: Tuple) -> List[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return self.global_route_planner.path_search(start, goal)

    # def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
    #     """
    #     :param lane_index: the index of a lane.
    #     :return: all lanes belonging to the same road.
    #     """
    #     side_lanes = []
    #     e = self.global_route_planner.graph.get_edge_data(lane_index[0], lane_index[1])
    #     left_lane = e.get('left_edge')
    #     right_lane = e.get('right_edge')
    #     if left_lane is not None:
    #         side_lanes.append((left_lane[0], left_lane[1], 0))
    #
    #     if right_lane is not None:
    #         side_lanes.append((right_lane[0], right_lane[1], 0))
    #
    #     return side_lanes

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
                :param lane_index: the index of a lane.
                :return: indexes of lanes next to a an input lane, to its right or left.
                """
        side_lanes = []
        e = self.global_route_planner.graph.get_edge_data(lane_index[0], lane_index[1])
        left_lane = e.get('left_edge')
        right_lane = e.get('right_edge')
        if left_lane is not None:
            side_lanes.append((left_lane[0], left_lane[1], 0))

        if right_lane is not None:
            side_lanes.append((right_lane[0], right_lane[1], 0))
        return side_lanes
    #
    # @staticmethod
    # def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
    #     """Is lane 1 in the same road as lane 2?"""
    #     return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])
    #
    # @staticmethod
    # def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
    #     """Is lane 1 leading to of lane 2?"""
    #     return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])
    #
    # def is_connected_road(self, lane_index_1: LaneIndex, lane_index_2: LaneIndex, route: Route = None,
    #                       same_lane: bool = False, depth: int = 0) -> bool:
    #     """
    #     Is the lane 2 leading to a road within lane 1's route?
    #
    #     Vehicles on these lanes must be considered for collisions.
    #     :param lane_index_1: origin lane
    #     :param lane_index_2: target lane
    #     :param route: route from origin lane, if any
    #     :param same_lane: compare lane id
    #     :param depth: search depth from lane 1 along its route
    #     :return: whether the roads are connected
    #     """
    #     if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
    #             or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
    #         return True
    #     if depth > 0:
    #         if route and route[0][:2] == lane_index_1[:2]:
    #             # Route is starting at current road, skip it
    #             return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
    #         elif route and route[0][0] == lane_index_1[1]:
    #             # Route is continuing from current road, follow it
    #             return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
    #         else:
    #             # Recursively search all roads at intersection
    #             _from, _to, _id = lane_index_1
    #             return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
    #                         for l1_to in self.graph.get(_to, {}).keys()])
    #     return False
    #
    # def lanes_list(self) -> List[AbstractLane]:
    #     return [lane for to in self.graph.values() for ids in to.values() for lane in ids]
    #
    # def lanes_dict(self) -> Dict[str, AbstractLane]:
    #     return {(from_, to_, i): lane
    #             for from_, tos in self.graph.items() for to_, ids in tos.items() for i, lane in enumerate(ids)}
    #
    # @staticmethod
    # def straight_road_network(lanes: int = 4,
    #                           start: float = 0,
    #                           length: float = 10000,
    #                           angle: float = 0,
    #                           speed_limit: float = 30,
    #                           nodes_str: Optional[Tuple[str, str]] = None,
    #                           net: Optional['RoadNetwork'] = None) \
    #         -> 'RoadNetwork':
    #     net = net or RoadNetwork()
    #     nodes_str = nodes_str or ("0", "1")
    #     for lane in range(lanes):
    #         origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
    #         end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
    #         rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    #         origin = rotation @ origin
    #         end = rotation @ end
    #         line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
    #                       LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
    #         net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))
    #     return net
    #
    # def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
    #         -> Tuple[np.ndarray, float]:
    #     """
    #     Get the absolute position and heading along a route composed of several lanes at some local coordinates.
    #
    #     :param route: a planned route, list of lane indexes
    #     :param longitudinal: longitudinal position
    #     :param lateral: : lateral position
    #     :return: position, heading
    #     """
    #     while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
    #         longitudinal -= self.get_lane(route[0]).length
    #         route = route[1:]
    #     return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)
    #
    # def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
    #     _from = np_random.choice(list(self.graph.keys()))
    #     _to = np_random.choice(list(self.graph[_from].keys()))
    #     _id = np_random.randint(len(self.graph[_from][_to]))
    #     return _from, _to, _id

    @classmethod
    # def from_config(cls, config: dict) -> None:
    #     net = cls()
    #     for _from, to_dict in config.items():
    #         net.graph[_from] = {}
    #         for _to, lanes_dict in to_dict.items():
    #             net.graph[_from][_to] = []
    #             for lane_dict in lanes_dict:
    #                 net.graph[_from][_to].append(
    #                     lane_from_config(lane_dict)
    #                 )
    #     return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(
                        lane.to_config()
                    )
        return graph_dict

    def get_left_lane(self, lane_index: Tuple):
        edge = self.global_route_planner.graph.get_edge_data(lane_index[0], lane_index[1])
        return edge.get('left_edge')

    def get_right_lane(self, lane_index: Tuple):
        edge = self.global_route_planner.graph.get_edge_data(lane_index[0], lane_index[1])
        return edge.get('right_edge')

    def get_road_info(self, lane_index: Tuple):
        edge = self.global_route_planner.graph.get_edge_data(lane_index[0], lane_index[1])
        entry_waypoint = edge.get('entry_waypoint')
        return vars(entry_waypoint)

class Road(object):

    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(self,
                 network: RoadNetwork = None,
                 vehicles: List['kinematics.Vehicle'] = None,
                 road_objects: List['objects.RoadObject'] = None,
                 np_random: np.random.RandomState = None,
                 record_history: bool = False) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
        self.ego_vehicles = None

    def set_ego_vehicles(self, vehicle_list):
        self.ego_vehicles = vehicle_list

    def close_objects_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: Optional[int] = None,
                         see_behind: bool = True, sort: bool = True, vehicles_only: bool = False) -> object:
        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]
        obstacles = [o for o in self.objects
                     if np.linalg.norm(o.position - vehicle.position) < distance
                     and -2 * vehicle.LENGTH < vehicle.lane_distance_to(o)]

        objects_ = vehicles if vehicles_only else vehicles + obstacles

        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count:
            objects_ = objects_[:count]
        return objects_

    def close_vehicles_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: Optional[int] = None,
                          see_behind: bool = True, sort: bool = True) -> object:
        return self.close_objects_to(vehicle, distance, count, see_behind, sort, vehicles_only=True)

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        # for i, vehicle in enumerate(self.vehicles):
        if self.ego_vehicles is not None:
            for other in self.vehicles:
                for ego in self.ego_vehicles:
                    ego.handle_collisions(other)
            # for other in self.objects:
            #     vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional['kinematics.Vehicle'], Optional['kinematics.Vehicle']]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        margin = 0 if (len(self.ego_vehicles) > 1 and len(self.ego_vehicles) == 2 and
                       vehicle is self.ego_vehicles[1]) else 1

        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=margin):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()
