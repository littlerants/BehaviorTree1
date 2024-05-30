from typing import List
from gym_sumo.hdmap_engine.element.point import Point
from gym_sumo.hdmap_engine.element.connection import Connection


class Junction(object):
    def __init__(self):
        self.id = None
        # 路口连接关系
        self.connections: List[Connection] = list()
        # 路口边界轮廓
        self.outline: List[Point] = list()
        # 路口中心坐标、东北天坐标系x, y, z
        self.center: Point = Point()

    @classmethod
    def create(cls):
        pass

