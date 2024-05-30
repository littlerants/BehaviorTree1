from typing import List
from gym_sumo.hdmap_engine.element.point import Point


class CrossWalk(object):
    def __init__(self):
        # crosswalk id
        self.id: str = ''
        # 人行横道轮廓
        self.outline: List[Point] = list()

    @classmethod
    def create(cls):
        pass