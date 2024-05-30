from typing import List
from gym_sumo.hdmap_engine.element.point import Point

class WidthsampleAssociate(object):
    def __init__(self):

        self.sOffset: float = -1.0

        self.leftWidth: float = -1.0

        self.rightWidth: float = -1.0

        self.width: float = -1.0


# 车道类型
class BorderType(object):

    def __init__(self):
        self.type: str = ''

        self.color: str = ''

        self.sOffset: float = -1.0

        self.eOffset: float = -1.0


class Lane(object):
    def __init__(self):
        # 车道id
        self.id: int = -1
        # 车道uid
        self.uid: str = ''

        #车道前驱id 可能多个前驱
        self.predecessors: List[str] = list()

        # 车道后继id,可能多个后继
        self.successors: List[str] = list()

        # 左邻近同向车道
        self.leftNeighbor_sameDirect_id: str = ''
        # 左邻近不同向车道
        self.leftNeighbor_reverseDirect_id: str = ''
        # 右邻近同向车道
        self.rightNeighbor_sameDirect_id: str = ''
        # 右邻近不同向车道
        self.rightNeighbor_reverseDirect_id: str = ''
        # 所在道路id
        self.road_id: int = -1
        #车道长度
        self.lane_length: float = -1
        # 车道转向类型
        self.turn_type: str = ''
        # 车道限速 km/h
        self.speed_limit: int = -1
        # 车道类型
        self.type: str = ''
        # 车道方向: forward、backward、bidirection unknow....
        self.direction: str = ""

        # 车道中心点参考线（离散点）
        self.centerLinePoints: List[Point] = list()

        # 车道宽度采样
        self.widthsampleAssociates: List[WidthsampleAssociate] = list()

        # 车道边界类型(xml里的)
        self.borderTypes: List[BorderType] = list()
        # 车道边界点（xml里的）
        self.borderPoints: List[Point] = list()

        # 车道左右边界
        # 左边界点
        self.leftPoints: List[Point] = list()
        # 车道左边界类型
        self.leftborderTypes: List[BorderType] = list()
        # 右边界点
        self.rightPoints: List[Point] = list()
        # 车道右边界类型
        self.rightborderTypes: List[BorderType] = list()

