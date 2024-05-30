

class CustomWaypoint:
    def __init__(self, location, road_id, section_id, lane_id, lane_type, lane_change, right_lane_marking, left_lane_marking):
        self.location = location
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.lane_type = lane_type
        self.lane_change = lane_change
        self.right_lane_marking = right_lane_marking
        self.left_lane_marking = left_lane_marking

    def next(self):
        # 简化实现，返回下一个相邻车道的CustomWaypoint
        return CustomWaypoint(self.location, self.road_id, self.section_id, self.lane_id + 1, self.lane_type, self.lane_change, self.right_lane_marking, self.left_lane_marking)

    def previous(self):
        # 简化实现，返回上一个相邻车道的CustomWaypoint
        return CustomWaypoint(self.location, self.road_id, self.section_id, self.lane_id - 1, self.lane_type, self.lane_change, self.right_lane_marking, self.left_lane_marking)

    def get_left_lane(self):
        # 简化实现，返回左侧车道的CustomWaypoint
        return CustomWaypoint(self.location, self.road_id, self.section_id, self.lane_id - 1, self.lane_type, self.lane_change, self.right_lane_marking, self.left_lane_marking)

    def get_right_lane(self):
        # 简化实现，返回右侧车道的CustomWaypoint
        return CustomWaypoint(self.location, self.road_id, self.section_id, self.lane_id + 1, self.lane_type, self.lane_change, self.right_lane_marking, self.left_lane_marking)

    def get_junction(self):
        # 简化实现，返回与当前CustomWaypoint相关的路口信息
        return f"Junction at {self.location}"

    def __str__(self):
        return f"CustomWaypoint at {self.location} on Road {self.road_id}, Section {self.section_id}, Lane {self.lane_id}"


class WayPoint(object):
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
        # heading
        self.heading: float = -1.0
        # width
        self.width = -1.0
        # width
        self.length = -1.0
        # 车道上的点
        self.lane_id = None
        # road id
        self.road_id = None
        # section id
        self.section_id = None
        # junction id
        self.junction_id = None
        # id
        self.id = None

    @classmethod
    def create(cls):
        pass
