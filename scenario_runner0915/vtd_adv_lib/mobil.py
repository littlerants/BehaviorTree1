#! python3
# -*- encoding: utf-8 -*-
"""
@File    :   idm_mobil.py
@Time    :   2023/08/16 11:09:24
@Author  :   SOTIF team
@Version :   1.0
@Desc    :   单独进行IDM模块的提取,用于计算主车加速度
@Usage   :   python idm_mobil.py
"""

import numpy as np
from collections import deque

class FakeCarlaActor(object):
    """
    用于进行验证的辅助类, 模拟Carla Actor
    实际使用的使用不需要这个,直接使用Carla Actor对象即可
    """

    class Location(object):
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def distance(self, other):
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    class Velocity(Location):
        pass

    class Rotation(object):
        def __init__(self, pitch, yaw, roll):
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

    class Transform(object):
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation

    def __init__(self, x, y, vx, vy, yaw):
        self.location = self.Location(x, y, 0)
        self.rotation = self.Rotation(0, yaw, 0)
        self.velocity = self.Velocity(vx, vy, 0)
        self.transform = self.Transform(self.location, self.rotation)

        # 周围车辆
        self.near_vehicles = {
            "P": None,
            "F": None,
            "P_L": None,
            "F_L": None,
            "P_R": None,
            "F_R": None,
        }

    def get_location(self):
        return self.location

    def get_velocity(self):
        return self.velocity

    def get_transform(self):
        return self.transform
class OBJECT():
    def __init__(self,simTime = -1,simFrame = -1, name='',id=100,lane_id=0,pos_x=0,off_x=1.39,pos_y=0,pos_h=0,hdg=0,vx=0,vy=0,
                v_h=0,w=2,l=3,lane_offset=0,inertial_heading=0,lane_w=0,obj_type = '1',acc_x  = 0, acc_y = 0,
                leftLaneId = -1,rightLaneId = -3,  distToJunc=10000000000.0,light_state='GO',roadId=None,adv_vec=False,lane = None):
        self.simTime = simTime
        self.simFrame = simFrame
        self.name = name
        self.id = id
        self.lane_id =lane_id
        self.pos_x = pos_x
        self.off_x = off_x
        self.pos_y = pos_y
        self.pos_h = pos_h
        self.vx = vx
        self.vy = vy
        self.v_h = v_h
        self.hdg =hdg
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.w = w
        self.l = l
        self.lane_offset = lane_offset
        self.inertial_heading = inertial_heading
        self.lane_w = lane_w
        self.distToJunc = distToJunc
        self.obj_type = obj_type
        self.light_state = light_state
        self.leftLaneId = leftLaneId
        self.rightLaneId = rightLaneId
        self.roadId = roadId
        self.predisToconfrontation_position = 99999999
        self.new_object = True
        self.adv_vec = adv_vec
        self.lane = lane
        self.pos_trajx = deque(maxlen=20)
        self.pos_trajy = deque(maxlen=20)
        # 0 front
        # 1 back
        # 2 left
        # 3 right
        self.direction_to_ego = -1
class LATCHECK(object):
    # ===== IDM参数 =====
    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 2.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -2.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 4.0  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # ===== MOBIL参数 =====
    # Lateral policy parameters
    POLITENESS = 0 # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]

    # LANE_CHANGE_DELAY = 1.0  # [s]
    def __init__(self,gl = None):
        self.gl = gl

    def idm(self,ego_vehicle, front_vehicle):
        acc = self.acceleration(ego_vehicle, front_vehicle)
        acc = np.clip(acc, -self.ACC_MAX, self.ACC_MAX)

        return acc

    def change_lane_policy(self,ego_vehicle,direction):
        # 如果决定变道，则返回方向，否则返回None
        """
        Decide when to change lane.
        """
        # decide to make a lane change

        for direction in ["LEFT", "RIGHT"]:
            # 是否存在这个方向的车道
            if not self.has_lane(ego_vehicle, direction):
                continue
            # Only change lane when the vehicle is moving
            if np.abs(self.get_vehicle_speed(ego_vehicle)) < 1:
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(ego_vehicle, direction):
                return direction

        return None


    def mobil(self, adv_vehicle, new_preceding ,new_following ,old_preceding ,old_following, direction='LEFT' ) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :return: whether the lane change should be performed
        """

        self_a = self.get_vehicle_acc(adv_vehicle)
        # 模拟换道后，对抗车获得的加速度
        self_pred_a = self.acceleration(new_preceding) + self.get_vehicle_acc(new_preceding)*0.5  if new_preceding else self_a + 0.2
        print("换道后车辆加速度：",self_pred_a)
        # 后车加速度
        new_following_a = self.get_vehicle_acc(new_following) if new_following else 0
        # 对抗车换道后，后车可能的加速度
        new_following_pred_a = self.acceleration(new_following) if new_following else 0
        old_following_pred_a = 0
        old_following_a  = 0

        jerk = (
            self_pred_a
            - self_a
            + self.POLITENESS
            * (
                new_following_pred_a
                - new_following_a
                + old_following_pred_a
                - old_following_a
            )
        )
        print("jerk：", jerk, self.LANE_CHANGE_MIN_ACC_GAIN)
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            return False

        # All clear, let's go!
        return True





    def has_lane(self, vehicle, direction):
        # TODO 需要结合Carla进行工程实现
        # 判断是否存在这个方向的车道
        return True


    def acceleration(self, vehicle ):

        acceleration = self.COMFORT_ACC_MAX

        if vehicle:  # 如果存在
            d = self.lane_distance_between(vehicle)
        acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap( vehicle) / self.not_zero(d), 2
            )

        return acceleration


    def desired_gap(self,front_vehicle, projected: bool = False):
        d0 = self.DISTANCE_WANTED + self.get_vehicle_length(front_vehicle)
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        # # dv = (
        # #     np.dot(
        # #         self.get_vehicle_velocity(ego_vehicle) - self.get_vehicle_velocity(front_vehicle),
        # #         self.get_vehicle_direction(ego_vehicle),
        # #     )
        # #     if projected
        # #     else self.get_vehicle_speed(ego_vehicle) - self.get_vehicle_speed(front_vehicle)
        # # )
        dv =  self.get_vehicle_speed(front_vehicle)
        if front_vehicle.pos_x > 0:
            dv *= -1

        d_star = (
            d0
            + tau
            +  dv / (2 * np.sqrt(ab))
        )
        return d_star


    def not_zero(self,x: float, eps: float = 1e-2) -> float:
        if abs(x) > eps:
            return x
        elif x >= 0:
            return eps
        else:
            return -eps


    # 传入Carla Actor对象，获取其对应信息。根据需要修改
    def get_vehicle_velocity(self,vehicle) -> np.ndarray:
        vel = np.array([vehicle.vx, vehicle.vy])
        return vel


    # 传入Carla Actor对象，获取其对应信息。根据需要修改
    def get_vehicle_speed(self,vehicle) -> float:
        speed = np.sqrt(vehicle.vx ** 2 + vehicle.vy ** 2)
        if vehicle.vx < 0:
            speed *= -1
        return speed

    def get_vehicle_acc(self,vehicle) -> float:
        acc = np.sqrt(vehicle.acc_x ** 2 + vehicle.acc_y ** 2)
        if vehicle.acc_x < 0:
            acc *= -1
        return acc
    # 传入Carla Actor对象，获取其对应信息。根据需要修改
    def get_vehicle_direction(self,vehicle) -> np.ndarray:
        heading = vehicle.pos_h
        return np.array([np.cos(heading), np.sin(heading)])


    # 传入Carla Actor对象，获取其对应信息。根据需要修改
    def lane_distance_between(self, front_vehicle) -> float:
        # 需要计算ego_vehicle到front_vehicle沿车道的距离，此处使用欧式距离近似代替
        # 如果要准确，还需要考虑车道形状
        dist = np.sqrt(front_vehicle.pos_x ** 2 + front_vehicle.pos_y ** 2)
        if front_vehicle.pos_x < 0:
            dist *= -1
        return dist

    # 传入Carla Actor对象，获取其对应信息。根据需要修改
    def get_vehicle_length(self,vehicle) -> float:
        return vehicle.l

    # 传入Carla Actor对象，获取其对应信息。根据需要修改
    def get_vehicle_target_speed(self,vehicle) -> float:
        # self.get_vehicle_speed(vehicle) + self.gl.target_acc
        return self.get_vehicle_speed(vehicle) + 2


if __name__ == "__main__":

    # 左前车慢车，对抗车在有车道便道
    # 构造左前车
    PL = OBJECT(pos_x= 10, pos_y= -4, vx=-5,vy=0, acc_x=-1, acc_y=0, l=3)
    FL = OBJECT(pos_x= -12, pos_y=-4, vx=1.57, vy=0, acc_x=3.14, acc_y=0, l=3)
    ADV = OBJECT(vx=0,vy=0,acc_x=0,acc_y=0, l=3)
    lat_check = LATCHECK()

    if lat_check.mobil(ADV,PL,None,None,None):
        print("left lane change can be allow!!!")
    else:
        print("warning collision!!!")

    if lat_check.mobil(ADV,PL,FL,None,None):
        print("left lane change can be allow!!!")
    else:
        print("warning collision!!!")
    print("对抗车后方存在车辆")
    if lat_check.mobil(ADV,None,FL,None,None):
        print("left lane change can be allow!!!")
    else :
        print("warning collision!!!")

