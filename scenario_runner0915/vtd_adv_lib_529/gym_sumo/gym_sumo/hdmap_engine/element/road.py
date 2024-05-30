from typing import List
from gym_sumo.hdmap_engine.element.traffic_light import TrafficLight
from gym_sumo.hdmap_engine.element.lane_section import LaneSection
from gym_sumo.hdmap_engine.element.stop_line import StopLine
from gym_sumo.hdmap_engine.element.crosswalk import CrossWalk


class Road(object):
    def __init__(self):

        # 道路id
        self.road_id: str = ''
        # 标记是否路口道路
        self.isJunctionRoad: bool = False

        self.predecessor_elementType: str = ''

        self.predecessor_id: str = ''

        self.successor_id: str = ''

        self.successor_elementType: str = ''

        self.length: float = -1.0
        # lanesection
        self.laneSections: List[LaneSection] = list()
        # 停止线
        self.stopLines: List[StopLine] = list()
        # 人行横道
        self.crosswalks: List[CrossWalk] = list()
        # 交通灯
        self.trafficLights: List[TrafficLight] = list()

    @classmethod
    def create(cls):
        pass
