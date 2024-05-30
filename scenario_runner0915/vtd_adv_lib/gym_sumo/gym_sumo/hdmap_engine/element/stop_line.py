from typing import List
from gym_sumo.hdmap_engine.element.point import Point


class StopLine(object):
    def __init__(self):
        # 停止线　id
        self.id: int = -1
        # 停止线点集
        self.points: List[Point] = list()

    @classmethod
    def create(cls):
        pass
