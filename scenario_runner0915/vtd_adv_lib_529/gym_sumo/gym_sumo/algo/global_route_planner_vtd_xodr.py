import networkx as nx
import numpy as np
import scipy
import math
from gym_sumo.opendrive_parse.network import Network
from gym_sumo.opendrive_parse.parser import parse_opendrive
from lxml import etree
from gym_sumo.hdmap_engine.element.waypoints import WayPoint
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import sys
from gym_sumo.opendrive_parse.utils import decode_road_section_lane_width_id

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


class GlobalRoutePlanner(object):
    def __init__(self, path='/home/sx/wjc/wjc/datasets/ca_vtd_train_data/zadao/1027_01.xodr', show=False):
        fh = open(path, "r")
        self.openDriveXml = parse_opendrive(etree.parse(fh).getroot())
        fh.close()
        self.loadedRoadNetwork = Network()

        self.graph = None
        self.id_map = None
        self.road_id_to_edge = None
        self.topology = None

        self.kdtree = None
        self.id_wpt = dict()

        self.show = show
        self.decimal = 0
        self._sampling_resolution = 3

        self.exclusion_lane_types = ['biking']

        self._build_topology()
        self._build_kdtree()
        self._build_graph()

    def trace_route(self, origin, destination):
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        route_trace = []
        route = self._path_search(origin, destination)

        last_road_option = RoadOption.LANEFOLLOW
        for i in range(len(route) - 1):
            edge = self.graph.edges[route[i], route[i + 1]]
            path = [] if i != 0 else [edge['entry_waypoint']]

            if edge['type'] == RoadOption.LANEFOLLOW and last_road_option == RoadOption.LANEFOLLOW:
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
            elif last_road_option != RoadOption.LANEFOLLOW:
                path = path + [edge['path']] + [edge['entry_waypoint']]
            else:
                path = path + [edge['entry_waypoint']]

            last_road_option = edge.get('type')

            route_trace.extend(path)
        return route_trace

    def _build_topology(self):
        self.loadedRoadNetwork.load_opendrive(self.openDriveXml)
        self.topology = []
        if self.show:
            fig, ax = plt.subplots(dpi=300)
            plt.minorticks_on()
        for road in self.openDriveXml.roads:
            pre_precalculation = road.planView.get_precalculation
            for lanesection in road.lanes.lane_sections:
                for i, lane in enumerate(lanesection.allLanes):
                    if lane.type in self.exclusion_lane_types:
                        continue

                    if lane.id != 0:
                        road_id = road.id
                        section_id = lanesection.idx
                        lane_id = lane.id
                        junction_id = None
                        if road.junction is not None:
                            junction_id = road.junction.id

                        start_end_label = lanesection.lane_start_end_label[lane.id]

                        start_point, end_point = WayPoint(), WayPoint()
                        long1, heading1, x1, y1, w1 = (pre_precalculation[start_end_label['start_idx'], 0], pre_precalculation[start_end_label['start_idx'], 3],
                                                      lanesection.lane_center_dict[lane.id][0][0], lanesection.lane_center_dict[lane.id][0][1], lanesection.lane_width_dict[lane.id][0])
                        start_point.x = x1
                        start_point.y = y1
                        start_point.heading = heading1
                        start_point.lon = long1
                        start_point.width = w1
                        start_point.lane_id, start_point.section_id, start_point.road_id, start_point.junction_id = lane_id, section_id, road_id, junction_id
                        # self.id_wpt[(round(x1), round(y1))] = start_point

                        long2, heading2, x2, y2, w2 = (pre_precalculation[start_end_label['end_idx'], 0],
                                                       pre_precalculation[start_end_label['end_idx'], 3],
                                                       lanesection.lane_center_dict[lane.id][-1][0],
                                                       lanesection.lane_center_dict[lane.id][-1][1],
                                                       lanesection.lane_width_dict[lane.id][-1])
                        end_point.x = x2
                        end_point.y = y2
                        end_point.heading = heading2
                        end_point.lon = long2
                        end_point.width = w2
                        end_point.lane_id, end_point.section_id, end_point.road_id, end_point.junction_id = lane_id, section_id, road_id, junction_id
                        # self.id_wpt[(round(x2), round(y2))] = end_point
                        segment = dict()
                        segment['entry'] = start_point
                        segment['exit'] = end_point
                        segment['entryxyz'] = (round(x1, self.decimal), round(y1, self.decimal))
                        segment['exitxyz'] = (round(x2, self.decimal), round(y2, self.decimal))
                        segment['path'] = list()
                        segment['length'] = float(np.sum([w.length for w in lane.widths]))

                        for i in range(1, len(lanesection.lane_center_dict[lane.id])-1):
                            long, heading, x, y, w = (pre_precalculation[start_end_label['start_idx']+i, 0],
                                                           pre_precalculation[start_end_label['start_idx']+i, 3],
                                                           lanesection.lane_center_dict[lane.id][i][0],
                                                           lanesection.lane_center_dict[lane.id][i][1],
                                                           lanesection.lane_width_dict[lane.id][i])
                            wpt = WayPoint()
                            wpt.x = x
                            wpt.y = y
                            wpt.heading = heading
                            wpt.lon = long
                            wpt.width = w
                            wpt.lane_id, wpt.section_id, wpt.road_id, wpt.junction_id = lane_id, section_id, road_id, junction_id
                            segment['path'].append(wpt)
                            # self.id_wpt[(round(x), round(y))] = wpt

                        if self.show:
                            # 绘制有向图
                            # plt.plot(x, y, marker='o', linestyle='-', color='black', lw=1)
                            ax.plot(np.array(lanesection.lane_center_dict[lane.id])[:, 0], np.array(lanesection.lane_center_dict[lane.id])[:, 1], lw=1)

                        plt.show()

                        self.topology.append(segment)
        # 显示图形
        #     plt.show()
        if self.show:
            plt.savefig('./map.png')

    def _build_kdtree(self):
        wpt_num = len(self.id_wpt)
        # points = []
        # for k, v in self.id_wpt.items():
        #     points.append([v.x, v.y])
        # points = np.array(points)
        # self.kdtree = KDTree(data=points)

    def _build_graph(self):
        self.graph = nx.DiGraph()
        self.id_map = dict()
        self.road_id_to_edge = dict()
        if self.show:
            fig, ax = plt.subplots(dpi=300)

        # 根据车道起始点与终止点添加边
        for segment in self.topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']

            lane_id, section_id, road_id, junction_id = entry_wp.lane_id, entry_wp.section_id, entry_wp.road_id, entry_wp.junction_id

            for vertex in entry_xyz, exit_xyz:
                if vertex not in self.id_map:
                    new_id = len(self.id_map)
                    self.id_map[vertex] = new_id
                    self.graph.add_node(new_id, vertex=vertex)

            n1 = self.id_map[entry_xyz]
            n2 = self.id_map[exit_xyz]

            if road_id not in self.road_id_to_edge:
                self.road_id_to_edge[road_id] = dict()

            if section_id not in self.road_id_to_edge[road_id]:
                self.road_id_to_edge[road_id][section_id] = dict()

            self.road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            self.graph.add_edge(
                n1, n2,
                length=round(segment['length']), path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                type=RoadOption.LANEFOLLOW,
                left_edge=None, right_edge=None, follow_edge=[]
            )
        # 根据不同道路下的车道连接关系点添加边
        for connecting, successors in self.loadedRoadNetwork._link_index._successors.items():
            from_lane_wpt_id, to_lane_wpt_id = None, None

            try:
                from_road_id, from_lane_section_idx, from_lane_link_successorId, _ = decode_road_section_lane_width_id(connecting)
                from_lane_wpt_id = self.road_id_to_edge[from_road_id][from_lane_section_idx][from_lane_link_successorId]
            except Exception as e:
                # print('func:{}(),line:{},'.format(sys._getframe().f_code.co_name, sys._getframe().f_lineno), end="")
                # print(e)
                # print(connecting)
                pass

            if from_lane_wpt_id is None:
                continue
            for successor in successors:

                try:
                    road_id, lane_section_idx, lane_link_successorId, _ = decode_road_section_lane_width_id(successor)
                    to_lane_wpt_id = self.road_id_to_edge[road_id][lane_section_idx][lane_link_successorId]
                except Exception as e:
                    # print('func:{}(),line:{},'.format(sys._getframe().f_code.co_name, sys._getframe().f_lineno), end="")
                    # print(e)
                    # print(successor)
                    pass

                if to_lane_wpt_id is None:
                    continue

                e1 = self.graph.get_edge_data(from_lane_wpt_id[0], from_lane_wpt_id[1])
                e2 = self.graph.get_edge_data(to_lane_wpt_id[0], to_lane_wpt_id[1])

                if e1 is None or e2 is None:
                    continue

                self.graph[from_lane_wpt_id[0]][from_lane_wpt_id[1]]['follow_edge'].append(to_lane_wpt_id)
                if from_lane_wpt_id[1] == to_lane_wpt_id[0]:
                    continue

                self.graph.add_edge(
                    from_lane_wpt_id[1], to_lane_wpt_id[0],
                    length=0, path=[],
                    entry_waypoint=e1.get('exit_waypoint'), exit_waypoint=e2.get('entry_waypoint'),
                    type=RoadOption.LANEFOLLOW,
                    left_edge=None, right_edge=None
                )

        # 根据相同道路下的车道邻接关系点添加边
        for road in self.openDriveXml.roads:
            for lanesection in road.lanes.lane_sections:
                for i, lane in enumerate(lanesection.allLanes):
                    if lane.type in self.exclusion_lane_types:
                        continue

                    road_id = road.id
                    section_id = lanesection.idx
                    lane_id = lane.id
                    if road.junction is not None or lane_id == 0:
                        continue
                    try:
                        current_edge = self.road_id_to_edge[road_id][section_id][lane_id]
                        left_lane = lane.get_left_lane()
                        if left_lane is None:
                            continue

                        left_edge = self.road_id_to_edge[road_id][section_id][left_lane.id]

                        e1 = self.graph.get_edge_data(current_edge[0], current_edge[1])
                        e2 = self.graph.get_edge_data(left_edge[0], left_edge[1])

                        e1_start_wpt = e1.get('entry_waypoint')
                        e1_end_wpt = e1.get('exit_waypoint')
                        e2_start_wpt = e2.get('entry_waypoint')
                        e2_end_wpt = e2.get('exit_waypoint')
                        self.graph[current_edge[0]][current_edge[1]]['left_edge'] = left_edge
                        self.graph.add_edge(
                            current_edge[0], left_edge[0],
                            length=0, path=[],
                            entry_waypoint=e1_start_wpt, exit_waypoint=e2_start_wpt,
                            type=RoadOption.CHANGELANELEFT,
                            left_edge=None, right_edge=None
                        )
                        self.graph.add_edge(
                            current_edge[1], left_edge[1],
                            length=0, path=[],
                            entry_waypoint=e1_end_wpt, exit_waypoint=e2_end_wpt,
                            type=RoadOption.CHANGELANELEFT,
                            left_edge=None, right_edge=None
                        )

                        self.graph[left_edge[0]][left_edge[1]]['right_edge'] = current_edge
                        self.graph.add_edge(
                            left_edge[0], current_edge[0],
                            length=0, path=[],
                            entry_waypoint=e2_start_wpt, exit_waypoint=e1_start_wpt,
                            type=RoadOption.CHANGELANERIGHT,
                            left_edge=None, right_edge=None
                        )
                        self.graph.add_edge(
                            left_edge[1], current_edge[1],
                            length=0, path=[],
                            entry_waypoint=e2_end_wpt, exit_waypoint=e1_end_wpt,
                            type=RoadOption.CHANGELANERIGHT,
                            left_edge=None, right_edge=None
                        )
                    except Exception as e:
                        # print('func:{}(),line:{},'.format(sys._getframe().f_code.co_name, sys._getframe().f_lineno), end="")
                        print(e)

        remove_edges = []
        for edge in self.graph.edges():
            if edge[0] == edge[1]:
                remove_edges.append(edge)

        for edge in remove_edges:
            self.graph.remove_edge(edge[0], edge[1])

        if self.show:
            nx.draw(self.graph, with_labels=True, arrowsize=2, node_size=15, font_size=3)
            plt.savefig('graph.png', dpi=300)
            # plt.show()

    def _distance_heuristic(self, n1, n2):
        l1 = np.array(self.graph.nodes[n1]['vertex'])
        l2 = np.array(self.graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1-l2)

    def _find_loose_ends(self):
        """
        This method finds road segments that have an unconnected end, and
        adds them to the internal graph representation
        """
        count_loose_ends = 0
        hop_resolution = 2

        # 遍历每个道路段
        for segment in self.topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id

            # 检查道路、路段和车道是否在图形表示中存在
            if road_id in self.road_id_to_edge \
                    and section_id in self.road_id_to_edge[road_id] \
                    and lane_id in self.road_id_to_edge[road_id][section_id]:
                pass
            else:
                count_loose_ends += 1

                # 如果道路 ID 不在 _road_id_to_edge 中，则添加
                if road_id not in self.road_id_to_edge:
                    self.road_id_to_edge[road_id] = dict()

                # 如果路段 ID 不在 _road_id_to_edge 中，则添加
                if section_id not in self.road_id_to_edge[road_id]:
                    self.road_id_to_edge[road_id][section_id] = dict()

                # 获取当前端点的图形节点 ID
                n1 = self.id_map[exit_xyz]

                # 创建新的图形节点 ID
                n2 = -1 * count_loose_ends

                # 将新的道路段添加到 _road_id_to_edge 中
                self.road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

                # 获取与当前端点相邻的下一个路标点
                next_wp = end_wp.next(hop_resolution)
                path = []

                # 将相邻路标点添加到路径中，直到遇到不符合条件的路标点
                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)

                # 如果路径不为空，则将路径的最后一个点作为新的图形节点添加到图中
                if path:
                    n2_xyz = (
                        path[-1].transform.location.x,
                        path[-1].transform.location.y,
                        path[-1].transform.location.z
                    )
                    self.graph.add_node(n2, vertex=n2_xyz)
                    self.graph.add_edge(
                        n1, n2,
                        length=len(path) + 1,
                        path=path,
                        entry_waypoint=end_wp,
                        exit_waypoint=path[-1],
                        entry_vector=None,
                        exit_vector=None,
                        net_vector=None,
                        intersection=end_wp.is_junction,
                        type=RoadOption.LANEFOLLOW
                    )

    def _localize(self, location):
        edge = None
        distance, index = self.kdtree.query(np.array(location))
        dd_edge = self.kdtree.data[index]
        try:
            wpt = self.id_wpt[(round(dd_edge[0]), round(dd_edge[1]))]
            road_id, section_id, lane_id = wpt.road_id, wpt.section_id, wpt.lane_id
            edge = self.road_id_to_edge[road_id][section_id][lane_id]
        except Exception as e:
            print('func:{}(),line:{},'.format(sys._getframe().f_code.co_name, sys._getframe().f_lineno), end="")
            print(e)
        return edge

    def _path_search(self, origin, destination):
        start, end = self._localize(origin), self._localize(destination)

        route = nx.astar_path(self.graph, source=start[0], target=end[0], heuristic=self._distance_heuristic, weight='length')
        route.append(end[1])
        return route

    def path_search(self, origin, destination):
        if destination[0] == origin[0] and destination[1] == origin[1]:
            return [destination[0], destination[1]]

        routes = [origin[0]]
        try:
            route = nx.astar_path(self.graph, source=origin[1], target=destination[0],
                                  heuristic=self._distance_heuristic,
                                  weight='length')
        except Exception as e:
            print(e)
            return None
        routes.extend(route)
        routes.append(destination[1])
        return routes

if __name__ == '__main__':
    # (62, 72) (8, 9)
    # path = '/home/sx/wjc/wjc/datasets/ca_vtd_train_data/zadao/1027_01.xodr'
    path = '/home/sx/wjc/wjc/datasets/ca_vtd_train_data/dingzilukou/T_Junction_Two_Way_Four_Lane-0001.xodr'
    # path = '/home/sx/wjc/wjc/datasets/ca_vtd_train_data/有交通灯的十字路口/Intersection_Two_Way_Four_Lane-0001.xodr'
    gp = GlobalRoutePlanner(show=True, path=path)
    gp.path_search((62, 72), (8, 9))
    # rout = gp.trace_route([399, 602], [396, 622])
    # gp._find_loose_ends()

