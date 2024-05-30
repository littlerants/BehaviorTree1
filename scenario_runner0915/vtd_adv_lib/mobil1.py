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

# ===== IDM参数 =====
# Longitudinal policy parameters
ACC_MAX = 6.0  # [m/s2]
"""Maximum acceleration."""

COMFORT_ACC_MAX = 3.0  # [m/s2]
"""Desired maximum acceleration."""

COMFORT_ACC_MIN = -5.0  # [m/s2]
"""Desired maximum deceleration."""

DISTANCE_WANTED = 5.0  # [m]
"""Desired jam distance to the front vehicle."""

TIME_WANTED = 1.5  # [s]
"""Desired time gap to the front vehicle."""

DELTA = 4.0  # []
"""Exponent of the velocity term."""

DELTA_RANGE = [3.5, 4.5]
"""Range of delta when chosen randomly."""

# ===== MOBIL参数 =====
# Lateral policy parameters
POLITENESS = 0.0  # in [0, 1]
LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
# LANE_CHANGE_DELAY = 1.0  # [s]


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


def idm(ego_vehicle, front_vehicle):
    acc = acceleration(ego_vehicle, front_vehicle)
    acc = np.clip(acc, -ACC_MAX, ACC_MAX)

    return acc


def change_lane_policy(ego_vehicle):
    # 如果决定变道，则返回方向，否则返回None
    """
    Decide when to change lane.
    """
    # decide to make a lane change

    for direction in ["LEFT", "RIGHT"]:
        # 是否存在这个方向的车道
        if not has_lane(ego_vehicle, direction):
            continue
        # Only change lane when the vehicle is moving
        if np.abs(get_vehicle_speed(ego_vehicle)) < 1:
            continue
        # Does the MOBIL model recommend a lane change?
        if mobil(ego_vehicle, direction):
            return direction

    return None


def mobil(ego_vehicle, direction) -> bool:
    """
    MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

        The vehicle should change lane only if:
        - after changing it (and/or following vehicles) can accelerate more;
        - it doesn't impose an unsafe braking on its new following vehicle.

    :return: whether the lane change should be performed
    """
    # Is the maneuver unsafe for the new following vehicle?
    if direction == "LEFT":
        new_preceding = get_near_vehicles(ego_vehicle)["P_L"]
        new_following = get_near_vehicles(ego_vehicle)["F_L"]
    else:
        new_preceding = get_near_vehicles(ego_vehicle)["P_R"]
        new_following = get_near_vehicles(ego_vehicle)["F_R"]

    new_following_a = acceleration(
        ego_vehicle=new_following, front_vehicle=new_preceding
    )
    new_following_pred_a = acceleration(
        ego_vehicle=new_following, front_vehicle=ego_vehicle
    )
    if new_following_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED:
        return False

    # Do I have a planned route for a specific lane which is safe for me to access?

    old_preceding = get_near_vehicles(ego_vehicle)["P"]
    old_following = get_near_vehicles(ego_vehicle)["F"]
    # 换道后换道车辆加速度（跟随加速度）
    self_pred_a = acceleration(ego_vehicle=ego_vehicle, front_vehicle=new_preceding)

    # Is there an acceleration advantage for me and/or my followers to change lane?
    # 换道车当前跟随加速度(未换道)
    self_a = acceleration(ego_vehicle=ego_vehicle, front_vehicle=old_preceding)
    # 换道车当前车道跟驰车加速度
    old_following_a = acceleration(ego_vehicle=old_following, front_vehicle=ego_vehicle)

    old_following_pred_a = acceleration(
        ego_vehicle=old_following, front_vehicle=old_preceding
    )
    jerk = (
        self_pred_a
        - self_a
        + POLITENESS
        * (
            new_following_pred_a
            - new_following_a
            + old_following_pred_a
            - old_following_a
        )
    )
    if jerk < LANE_CHANGE_MIN_ACC_GAIN:
        return False

    # All clear, let's go!
    return True


def get_near_vehicles(vehicle) -> dict:
    # TODO 需要结合Carla进行工程实现
    # 获取车辆周围的6辆车
    # 分别为左前，左后，左侧，右前，右后，右侧
    # 如果没有车，则为None

    # 格式如下
    near_vehicles = {
        "P": None,  # 前车
        "F": None,  # 后车
        "P_L": None,  # 左前车
        "F_L": None,  # 左后车
        "P_R": None,  # 右前车
        "F_R": None,  # 右后车
    }

    return vehicle.near_vehicles


def has_lane(vehicle, direction):
    # TODO 需要结合Carla进行工程实现
    # 判断是否存在这个方向的车道
    return True


def acceleration(ego_vehicle, front_vehicle):
    if not ego_vehicle:  # 处理ego不存在的情况
        return 0

    acceleration = COMFORT_ACC_MAX * (
        1
        - np.power(
            max(get_vehicle_speed(ego_vehicle), 0)
            / get_vehicle_target_speed(ego_vehicle),
            DELTA,
        )
    )

    if front_vehicle:  # 如果存在前车
        d = lane_distance_between(ego_vehicle, front_vehicle)  # 计算与前车的距离（沿车道距离）
        acceleration -= COMFORT_ACC_MAX * np.power(
            desired_gap(ego_vehicle, front_vehicle) / not_zero(d), 2
        )

    return acceleration


def desired_gap(ego_vehicle, front_vehicle, projected: bool = True):
    d0 = DISTANCE_WANTED + get_vehicle_length(front_vehicle)
    tau = TIME_WANTED
    ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
    dv = (
        np.dot(
            get_vehicle_velocity(ego_vehicle) - get_vehicle_velocity(front_vehicle),
            get_vehicle_direction(ego_vehicle),
        )
        if projected
        else get_vehicle_speed(ego_vehicle) - get_vehicle_speed(front_vehicle)
    )
    d_star = (
        d0
        + get_vehicle_speed(ego_vehicle) * tau
        + get_vehicle_speed(ego_vehicle) * dv / (2 * np.sqrt(ab))
    )
    return d_star


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


# 传入Carla Actor对象，获取其对应信息。根据需要修改
def get_vehicle_velocity(vehicle) -> np.ndarray:
    vel = vehicle.get_velocity()
    vel = np.array([vel.x, vel.y])
    return vel


# 传入Carla Actor对象，获取其对应信息。根据需要修改
def get_vehicle_speed(vehicle) -> float:
    speed = np.sqrt(vehicle.get_velocity().x ** 2 + vehicle.get_velocity().y ** 2)
    return speed


# 传入Carla Actor对象，获取其对应信息。根据需要修改
def get_vehicle_direction(vehicle) -> np.ndarray:
    heading = vehicle.get_transform().rotation.yaw
    return np.array([np.cos(heading), np.sin(heading)])


# 传入Carla Actor对象，获取其对应信息。根据需要修改
def lane_distance_between(ego_vehicle, front_vehicle) -> float:
    # 需要计算ego_vehicle到front_vehicle沿车道的距离，此处使用欧式距离近似代替
    # 如果要准确，还需要考虑车道形状
    dist = ego_vehicle.get_location().distance(front_vehicle.get_location())
    return dist


# 传入Carla Actor对象，获取其对应信息。根据需要修改
def get_vehicle_length(vehicle) -> float:
    # TODO: 需要Carla工程实现，返回车辆长度
    return 3


# 传入Carla Actor对象，获取其对应信息。根据需要修改
def get_vehicle_target_speed(vehicle) -> float:
    # TODO: 需要Carla工程实现，返回车辆目标车速
    return 20


if __name__ == "__main__":
    # test the idm
    print("============测试IDM(需要上游算法提供前车对象)==============")
    print("=====================输出主车加速度=====================")
    print("前车很近很慢")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    front_vehicle = FakeCarlaActor(x=0, y=15, vx=0, vy=3, yaw=90)
    acc = round(idm(ego_vehicle, front_vehicle), 3)
    print(f"{acc} m/s^2\n")

    print("前车很近很快")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    front_vehicle = FakeCarlaActor(x=0, y=15, vx=0, vy=15, yaw=90)
    acc = round(idm(ego_vehicle, front_vehicle), 3)
    print(f"{acc} m/s^2\n")

    print("前车较远很慢")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    front_vehicle = FakeCarlaActor(x=0, y=25, vx=0, vy=3, yaw=90)
    acc = round(idm(ego_vehicle, front_vehicle), 3)
    print(f"{acc} m/s^2\n")

    print("前车较远很快")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    front_vehicle = FakeCarlaActor(x=0, y=35, vx=0, vy=15, yaw=90)
    acc = round(idm(ego_vehicle, front_vehicle), 3)
    print(f"{acc} m/s^2\n")

    print("没有前车，主车速度较慢")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=3, yaw=90)
    front_vehicle = None
    acc = round(idm(ego_vehicle, front_vehicle), 3)
    print(f"{acc} m/s^2\n")

    print("没有前车，主车速度较快")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=19, yaw=90)
    front_vehicle = None
    acc = round(idm(ego_vehicle, front_vehicle), 3)
    print(f"{acc} m/s^2\n")


    print("============测试MOBIL(默认先搜索左侧车道)==============")
    print("=====LEFT:向左变道;RIGHT:向右变道;None:不变道=======\n")

    print("只有很慢的前车")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    ego_vehicle.near_vehicles["P"] = FakeCarlaActor(x=0, y=15, vx=0, vy=2, yaw=90)
    print(f"{change_lane_policy(ego_vehicle)}\n")

    print("只有很快的前车")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    ego_vehicle.near_vehicles["P"] = FakeCarlaActor(x=0, y=55, vx=0, vy=20, yaw=90)
    print(f"{change_lane_policy(ego_vehicle)}\n")

    print("前车和左前车都很慢")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    ego_vehicle.near_vehicles["P"] = FakeCarlaActor(x=0, y=15, vx=0, vy=3, yaw=90)
    ego_vehicle.near_vehicles["P_L"] = FakeCarlaActor(x=0, y=15, vx=0, vy=2, yaw=90)
    print(f"{change_lane_policy(ego_vehicle)}\n")

    print("前车和左前车和右前车都很慢，但左后车右后车很快(安全刹车阈值)")
    ego_vehicle = FakeCarlaActor(x=0, y=5, vx=0, vy=5, yaw=90)
    ego_vehicle.near_vehicles["P"] = FakeCarlaActor(x=0, y=15, vx=0, vy=3, yaw=90)
    ego_vehicle.near_vehicles["P_L"] = FakeCarlaActor(x=0, y=15, vx=0, vy=2, yaw=90)
    ego_vehicle.near_vehicles["P_R"] = FakeCarlaActor(x=0, y=15, vx=0, vy=2, yaw=90)
    ego_vehicle.near_vehicles["F_L"] = FakeCarlaActor(x=0, y=-15, vx=0, vy=7, yaw=90)
    ego_vehicle.near_vehicles["F_R"] = FakeCarlaActor(x=0, y=-15, vx=0, vy=7, yaw=90)
    print(f"{change_lane_policy(ego_vehicle)}\n")
