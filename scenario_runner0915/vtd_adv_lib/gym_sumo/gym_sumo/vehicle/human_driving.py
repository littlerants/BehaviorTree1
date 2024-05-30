from __future__ import division, print_function
from typing import Union
import math
import numpy as np
import matplotlib.pyplot as plt
from gym_sumo.vehicle.controller import ControlledVehicle
from gym_sumo import utils
from gym_sumo.vehicle.behavior import IDMVehicle
from gym_sumo.road.lane import PolyLaneFixedWidth
import numpy as np
from Utils.math import filter_similar_points, init_linear
from scipy.interpolate import UnivariateSpline


class IntersectionHumanLikeVehicle(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 30  # [m/s]

    def __init__(self, road, name, position,
                 heading=0,
                 velocity=0,
                 acc=0,
                 target_lane_index=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 start_step=0,
                 vehicle_ID=None,
                 v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):

        super(IntersectionHumanLikeVehicle, self).__init__(road, name, position, heading, velocity,
                                                           target_lane_index, target_velocity, route,)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.planned_speed = None
        self.planned_heading = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.acc = acc
        self.steering_noise = None
        self.acc_noise = None
        self.MARGIN = 5
        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]
        self.start_step = start_step - 1
        self.step_num = 1
        self.next_position = None

    @classmethod
    def create(cls, road, vehicle_ID, name, position, v_length, v_width, ngsim_traj, heading=0.0, velocity=0.0, acc=0.0,
               target_velocity=15, human=False, IDM=False, start_step=0):
        """
        Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
        """
        v = cls(road, position, name, heading, velocity, acc, target_velocity=target_velocity,
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM,
                start_step=start_step)

        return v

    @classmethod
    def create_from(cls, vehicle: IDMVehicle) -> "IntersectionHumanLikeVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.get_name, vehicle.position,  vehicle.heading, vehicle.speed, vehicle.acc, target_velocity=vehicle.target_speed,
                vehicle_ID=vehicle.vehicle_ID, v_length=vehicle.LENGTH, v_width=vehicle.WIDTH, ngsim_traj=vehicle.ngsim_traj, human=vehicle.human, IDM=vehicle.IDM,
                start_step=vehicle.start_step)

        return v

    def make_linear(self):
        self.linear, self.unique_arr = init_linear(self.planned_trajectory, PolyLaneFixedWidth)

    def act(self, action: Union[dict, str] = None):
        if self.planned_trajectory is not None and not self.IDM:
            try:
                control_heading, acceleration = self.control_vehicle(self.planned_trajectory[self.step_num],
                                                                     self.planned_speed[self.step_num],
                                                                     self.planned_heading[self.step_num])
                self.action = {'steering': control_heading,
                               'acceleration': acceleration}
            except Exception as e:
                print(e)

            if len(self.planned_trajectory) - 1 <= (self.step_num):
                self.next_position = None
            else:
                self.next_position = self.planned_trajectory[self.step_num]

        elif self.IDM:
            super(IntersectionHumanLikeVehicle, self).act()
            # print('self.vehicle.action', self.action)
        else:
            return

        self.step_num += 1

    def control_vehicle(self, next_position, next_speed, next_heading):
        # # Compute displacement
        # displacement = (next_position - self.position)
        # # Average velocity
        # average_velocity = displacement / 0.1
        # # Next speed
        # next_speed_ = np.linalg.norm(average_velocity)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.speed) * heading_rate_command)
        # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        acceleration = 10 * (np.linalg.norm(next_speed) - self.speed)
        return steering_angle, acceleration

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)

        super(IntersectionHumanLikeVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj[:self.sim_steps + 1, :2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return (ADE, FDE)

