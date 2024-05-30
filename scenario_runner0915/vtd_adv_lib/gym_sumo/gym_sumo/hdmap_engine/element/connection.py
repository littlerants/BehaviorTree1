from typing import List
from gym_sumo.hdmap_engine.element.point import Point


class Connection(object):

    def __init__(self):
        # 路口id
        self.junction_id: int = -1
        # 进入入口的道路id
        self.in_road_id: int = -1
        # 出路口的道路id
        self.out_road_id: int = -1
        # 进入路口的通行方式
        self.in_road_turntype: str = ''
        # 离开路口的通行方式
        self.out_road_turntype: str = ''
        # 进入路口的车道id
        self.in_lane_id:str = ''
        # 离开路口的车道id
        self.out_lane_id: str = ''
        # 进入道路的通行方式
        self.in_lane_turntype: str = ''
        # 离开道路的通行方式
        self.out_lane_turntype: str = ''
        # 虚拟车道id
        self.virtual_lane_id: str = ''
        # 虚拟车道参考线（离散点）
        self.center_line_points: List[Point] = list()

    @classmethod
    def create(cls):
        pass
