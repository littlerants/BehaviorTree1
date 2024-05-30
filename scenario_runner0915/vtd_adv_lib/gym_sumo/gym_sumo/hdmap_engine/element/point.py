
class Point(object):
    def __init__(self):
        # 经度
        self.lon: float = -1.0
        # 维度
        self.lat: float = -1.0
        # 东北天 x y z
        # x坐标
        self.x: float = -1.0
        # y坐标
        self.y: float = -1.0
        # 高度 m
        self.z: float = -1.0
        # 车道上的点
        self.lane_id: str = ''
        # road id
        self.road_id: str = ''
        # section id
        self.section_id = ''
        # id
        self.idx: int = -1

    @classmethod
    def create(cls):
        pass
