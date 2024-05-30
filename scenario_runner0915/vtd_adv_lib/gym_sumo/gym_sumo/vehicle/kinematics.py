from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque

from gym_sumo import utils
from gym_sumo.road.road import Road, LaneIndex
from gym_sumo.vehicle.objects import RoadObject, Obstacle, Landmark
from gym_sumo.utils import Vector


class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -40.
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 10
    """ Length of the vehicle state history, for trajectory display"""

    def __init__(self,
                 road: Road,
                 name: str,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering',
                 lane_index: int = None,):
        super().__init__(road, name, position, heading, speed, lane_index=lane_index)
        self.prediction_type = predition_type
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)

    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12+1.0*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        if hasattr(vehicle, 'color'):
            v.color = vehicle.color
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt

        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        self.speed = 0 if self.speed < 0 and not self.get_name.startswith('ped') else self.speed
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road and not self.name.startswith('ped'):
            if abs(self.lane_offset[1]) > 5:
                self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
                self.lane = self.road.network.get_lane(self.lane_index)
            else:
                if self.lane_index != self.target_lane_index:
                    self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading, lane_index=self.lane_index, target_lane_index=self.target_lane_index)
                else:
                    self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading, lane_index=self.lane_index)
                self.lane = self.road.network.get_lane(self.lane_index)

            # if self.road.record_history:
            #     self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if self.prediction_type == 'zero_steering':
            action = {'acceleration': 0.0, 'steering': 0.0}
        elif self.prediction_type == 'constant_steering':
            action = {'acceleration': 0.0, 'steering': self.action['steering']}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang, self.lane.width_at(long), self.lane.heading_at(long), long / self.lane.length, self.lane.get_speed_limit - self.speed])
        else:
            return np.zeros((7, ))

    @property
    def obj_lane_offset(self)->np.ndarray:
        """"""
        if self.target_lane_index == self.lane_index or self.target_lane_index is None:
            return self.lane_offset

        lane = self.road.network.get_lane(self.target_lane_index)
        long, lat = lane.local_coordinates(self.position)
        ang = lane.local_angle(self.heading, long)
        return np.array([long, lat, ang, lane.width_at(long), lane.heading_at(long), long / lane.length, lane.get_speed_limit - self.speed])

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        left_lane_change, right_lane_change = False, False
        if not self.name.startswith('ped'):
            left_lane_change, right_lane_change = self.road.network.can_change_lane(self.lane_index)

        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            # 'cos_h': self.direction[0],
            # 'sin_h': self.direction[1],
            # 'cos_d': self.destination_direction[0],
            # 'sin_d': self.destination_direction[1],
            'long_off': self.lane_offset[0],
            'lat_off': self.lane_offset[1],
            'ang_off': self.lane_offset[2],
            'speed_off': self.lane_offset[6],
            'obj_lat_off': self.obj_lane_offset[1],
            'obj_ang_off': self.obj_lane_offset[2],
            'obj_speed_off': self.obj_lane_offset[6],
            'lane_width': self.lane_offset[3],
            'lane_heading': self.lane_offset[4],
            'proportion_distance': self.lane_offset[5],
            'width': self.WIDTH,
            'length': self.LENGTH,
            'left_change_lane': int(left_lane_change),
            'right_lane_change': int(right_lane_change),
        }

        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
