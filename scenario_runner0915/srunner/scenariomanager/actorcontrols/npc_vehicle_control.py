#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles
"""

import math
import carla

from carla_api.agents.navigation.behavior_agent import BehaviorAgent
from carla_api.agents.navigation.local_planner import (
    RoadOption,
)
from srunner.scenariomanager.carla_data_provider import (
    CarlaDataProvider,
)


class NpcVehicleControl(BehaviorAgent):

    """
    Controller class for vehicles derived from BehaviorAgent.

    The controller makes use of the LocalPlanner implemented in CARLA.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
    """


    def __init__(self, actor, args=None):
        super(NpcVehicleControl, self).__init__(actor, opt_dict=args)
        self._init_speed = False
        self._target_speed_updated = False
        self._waypoints_updated = False
        self._waypoints = []

        self._init_speed_holding = False
        self._init_flag = False
        self._offset_updated = False
        self._waypoint_need_generate = True

        if "desired_velocity" in args:
            self._behavior.max_speed = args["desired_velocity"] * 3.6
        if "desired_acceleration" in args:
            self._behavior.desired_acceleration = args["desired_acceleration"]
        if "desired_deceleration" in args:
            self._behavior.desired_deceleration = args["desired_deceleration"]
        if "emergency_param" in args:
            self._behavior.emergency_param = args["emergency_param"]
        if "safety_time" in args:
            self._behavior.safety_time = args["safety_time"]
        if "lane_changing_dynamic" in args:
            self._behavior.lane_changing_dynamic = args["lane_changing_dynamic"]
        if "urge_to_overtake" in args:
            self._behavior.urge_to_overtake = args["urge_to_overtake"]
        if "init_speed_holding" in args:
            self._init_speed_holding = args["init_speed_holding"]
        # only to new add actor have these args
        if "init_speed" in args:
            self._init_speed = args["init_speed"]
        if "speed" in args:
            self._target_speed = args["speed"]

        if self._waypoints:
            self._update_plan()

    def update_target_speed(self, speed):
        """
        Update the actor's target speed and set _init_speed to False.

        Args:
            speed (float): New target speed [m/s].
        """
        self._target_speed = (
            speed * 3.6
        )  # _target_speed is defined in basic_agent with Km/h
        self._target_speed_updated = True

    def update_waypoints(self, waypoints, start_time=None):
        """
        Update the actor's waypoints

        Args:
            waypoints (List of carla.Transform): List of new waypoints.
        """
        self._waypoints = waypoints
        self._waypoints_updated = True
        self._waypoint_need_generate = False

    def _update_plan(self):
        """
        Update the plan (waypoint list) of the LocalPlanner
        """
        plan = []
        for transform in self._waypoints:
            if isinstance(transform, carla.Waypoint):      
                plan.append((transform, RoadOption.LANEFOLLOW))
            else:
                waypoint = CarlaDataProvider.get_map().get_waypoint(
                    transform.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Any,
                )
                plan.append((waypoint, RoadOption.LANEFOLLOW))
        self._update_end(plan[-1][0])
        self._local_planner.set_global_plan(plan)

    def update_offset(self, offset, start_time=None):
        """
        Update the actor's waypoints

        Args:
            offset: lane offset.
            start_time: simulation start time.
        """
        self._local_planner._vehicle_controller._lat_controller._offset = offset

    def check_reached_waypoint_goal(self):
        """
        Check if the actor reached the end of the waypoint list

        returns:
            True if the end was reached, False otherwise.
        """
        if self._local_planner and self._local_planner.done():
            return True
        else:
            return False

    def reset(self):
        """
        Reset the controller
        """
        if self._vehicle and self._vehicle.is_alive:
            if self._local_planner:
                self._local_planner.reset_vehicle()
                self._local_planner = None
            self._vehicle = None

    def set_init_speed(self):
        """
        Set _init_speed to True
        """
        self._init_speed = True

    def run_step(self, debug=False):
        """
        Execute on tick of the controller's control loop

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint.
        Upon reaching the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.

        If _init_speed is True, the control command is post-processed to ensure
        that the initial actor velocity is maintained independent of physics.
        """
        control = None
        if self._vehicle is None:
            return
        if not self._vehicle.is_alive:
            return

        self._update_information()

        if self._waypoints_updated:
            self._waypoints_updated = False
            self._update_plan()
        elif self._waypoint_need_generate:
            self._local_planner._compute_next_waypoints()
            #self._waypoint_need_generate = False

        if self._offset_updated:
            self._offset_updated = False
            self.update_offset(self._offset)
        # print("self._target_speed1:", self._target_speed)
        # If target speed is negavite, raise an exception
        if self._target_speed < 0:
            raise NotImplementedError("Negative target speeds are not yet supported")
        if self._init_speed:
            if abs(self._target_speed) > 0:
                yaw = self._vehicle.get_transform().rotation.yaw * (math.pi / 180)
                vx = math.cos(yaw) * (self._target_speed / 3.6)
                vy = math.sin(yaw) * (self._target_speed / 3.6)
                self._vehicle.set_target_velocity(carla.Vector3D(vx, vy, 0))

            if self._init_speed_holding:
                self.update_target_speed((self._target_speed / 3.6))
            else:
                self.update_target_speed((self._behavior.max_speed / 3.6))

            self._init_flag = True
            self._init_speed = False
        elif self._target_speed_updated:
            self._target_speed_updated = False

            self.set_target_speed(self._target_speed)
            control = super().run_step()
        else:
            control = super().run_step()

        if control:
            # if self._init_speed_holding and self._init_flag:
            #     ackermann_control = carla.VehicleAckermannControl(
            #         steer=control.steer,
            #         speed=(self._target_speed / 3.6),
            #         acceleration=5.0,  # use a normal value for acceleration
            #     )
            #     self._vehicle.apply_ackermann_control(ackermann_control)
            # else:
            # print("driver vec control: ",self._vehicle.id, control )
            # control.throttle = 1
            # self._vehicle.apply_control(control)
            return  control