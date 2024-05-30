from typing import List
from gym_sumo.hdmap_engine.element.lane import Lane, BorderType
from gym_sumo.hdmap_engine.element.point import Point

class LaneSection(object):
    def __init__(self):
        # 长度
        self.length: float = -1.0
        # 车道列表
        self.lanes: List[Lane] = list()
        # laneSection id
        self.laneSection_id: int = -1
        # 中心车道线
        self.centerborder: List[Point] = list()
        # 中心车道线border
        self.centerborderTypes: List[BorderType] = list()

    @classmethod
    def create(cls):
        pass

