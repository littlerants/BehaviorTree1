from typing import List
from gym_sumo.hdmap_engine.element.point import Point
from gym_sumo.hdmap_engine.element.stop_line import StopLine


class SubSigal(object):
    def __init__(self):
        self.id: int = -1
        # 子信号类型
        self.type: str = ''
        # center point
        self.lon: float = -1.0
        self.lat: float = -1.0
        # 东北天 x y z
        # x坐标
        self.x = None
        # y坐标
        self.y = None
        # 高度 m
        self.z = None

    @classmethod
    def create(cls):
        pass


class TrafficLight(object):
    def __init__(self):
        self.id: int = -1
        # 信号布局，垂直、水平
        self.layout_type: str = ''
        # 信号灯轮廓
        self.outline: List[Point] = list()
        # 信号灯对应的停止线
        self.stop_line: List[StopLine] = list()
        # 信号灯子信号
        self.sub_signals: List[SubSigal] = list()

    @classmethod
    def create(cls):
        pass