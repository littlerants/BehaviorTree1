# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import carla
import numpy as np
from carla_api.agents.navigation.basic_agent import BasicAgent
from carla_api.agents.navigation.local_planner import RoadOption
from carla_api.agents.tools.misc import get_speed, positive
from carla_api.agents.navigation.behavior_types import Normal

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles.
    Lane changing decisions can be taken by analyzing the surrounding
     environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep
    safety distance from a car in front of it by tracking the instantaneous
    time to collision and keeping it in a certain range.
    """

    def __init__(
        self,
        vehicle,
        opt_dict={},
        map_inst=None,
        grp_inst=None,
    ):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(
            vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst
        )
        self._vehicle_speed = 0
        self._vehicle_loc = None

        # Vehicle information
        self._look_ahead_steps = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = Normal()
        self._sampling_resolution = 4.5
        self.end_waypoint = None
        self._offset_updated = True

    def _update_end(self, end_waypoint):
        print("end waypoint  {}".format(end_waypoint.transform))
        self.end_waypoint = end_waypoint

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()

        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = 5

        (
            self._incoming_waypoint,
            self._incoming_direction,
        ) = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps
        )
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)
 
        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(
            vehicle_list,
            max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
            up_angle_th=180,
            low_angle_th=160,
        )
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (
                (
                    right_turn == carla.LaneChange.Right
                    or right_turn == carla.LaneChange.Both
                )
                and waypoint.lane_id * right_wpt.lane_id > 0
                and right_wpt.lane_type == carla.LaneType.Driving
            ):
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(
                    vehicle_list,
                    max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
                    up_angle_th=180,
                    lane_offset=1,
                )
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(
                        end_waypoint.transform.location, right_wpt.transform.location
                    )
            elif (
                left_turn == carla.LaneChange.Left
                and waypoint.lane_id * left_wpt.lane_id > 0
                and left_wpt.lane_type == carla.LaneType.Driving
            ):
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(
                    vehicle_list,
                    max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
                    up_angle_th=180,
                    lane_offset=-1,
                )
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(
                        end_waypoint.transform.location, left_wpt.transform.location
                    )

    def _overtake(self, waypoint, vehicle_list):
        """
        This method is in charge of overtaking behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        if (
            (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both)
            and waypoint.lane_id * left_wpt.lane_id > 0
            and left_wpt.lane_type == carla.LaneType.Driving
        ):
            new_vehicle_state, _, _ = self._vehicle_obstacle_detected(
                vehicle_list,
                max(self._behavior.min_proximity_threshold, self._target_speed / 3),
                lane_offset=-1,
            )

            if not new_vehicle_state:
                print("Overtaking to the left!")
                self._behavior.overtake_counter = 200
                self.lane_change(
                    direction="left",
                    same_lane_time=0,
                    other_lane_time=20,
                    lane_change_time=2,
                )
        elif (
            right_turn == carla.LaneChange.Right
            and waypoint.lane_id * right_wpt.lane_id > 0
            and right_wpt.lane_type == carla.LaneType.Driving
        ):
            new_vehicle_state, _, _ = self._vehicle_obstacle_detected(
                vehicle_list,
                max(self._behavior.min_proximity_threshold, self._target_speed / 3),
                lane_offset=-1,
            )
            if not new_vehicle_state:
                print("Overtaking to the right!")
                self._behavior.overtake_counter = 200
                self.lane_change(
                    direction="right",
                    same_lane_time=0,
                    other_lane_time=20,
                    lane_change_time=2,
                )

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        if self._behavior.lane_changing_dynamic or not self._ignore_vehicles:
            vehicle_list = []
            tmp_vehicle_list = self._world.get_actors().filter("*vehicle*")
            tmp_static_list = self._world.get_actors().filter("*static*")
            vehicle_list.extend(tmp_vehicle_list)
            vehicle_list.extend(tmp_static_list)
        else:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        vehicle_list = [
            v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id
        ]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list,
                max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
                up_angle_th=180,
                lane_offset=-1,
            )
            # print(
            #     "CHANGELANELEFT.............................  {} ".format(vehicle_state)
            # )
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list,
                max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
                up_angle_th=180,
                lane_offset=1,
            )
            # print(
            #     "CHANGELANERIGHT............................  {} ".format(vehicle_state)
            # )
        else:
            # print("collision_and_car_avoid_manager .............................   ")
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, 18, up_angle_th=30
            )

            if vehicle is not None:
                d_speed = get_speed(vehicle)
            else:
                d_speed = 5

            if (
                vehicle_state
                and self._direction == RoadOption.LANEFOLLOW
                and not waypoint.is_junction
                and self._speed > d_speed
                and self._behavior.overtake_counter == 0
                and (self._behavior.urge_to_overtake or self._behavior.lane_changing_dynamic)
            ):
                # print("overtaking.............")
                self._overtake(waypoint, vehicle_list)
            elif (
                not vehicle_state
                and self._direction == RoadOption.LANEFOLLOW
                and not waypoint.is_junction
                and self._speed > 10
                and self._behavior.tailgate_counter == 0
                and self._behavior.urge_to_overtake
            ):
                # print("tailgating.............")
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        if self._ignore_vehicles:
            return (False, None, -1)

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        def dist(w):
            return w.get_location().distance(waypoint.transform.location)

        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(
                walker_list,
                max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
                up_angle_th=90,
                lane_offset=-1,
            )
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(
                walker_list,
                max(self._behavior.min_proximity_threshold, self._speed_limit / 2),
                up_angle_th=90,
                lane_offset=1,
            )
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(
                walker_list,
                max(self._behavior.min_proximity_threshold, self._speed_limit / 3),
                up_angle_th=60,
            )

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0.0, 1.0)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min(
                [
                    positive(vehicle_speed - self._behavior.speed_decrease),
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist,
                ]
            )
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min(
                [
                    max(self._min_speed, vehicle_speed),
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist,
                ]
            )
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min(
                [
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist,
                ]
            )
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1
        if self._behavior.overtake_counter > 0:
            self._behavior.overtake_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = (
                w_distance
                - max(walker.bounding_box.extent.y, walker.bounding_box.extent.x)
                - max(
                    self._vehicle.bounding_box.extent.y,
                    self._vehicle.bounding_box.extent.x,
                )
            )

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(
            ego_vehicle_wp
        )

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = (
                distance
                - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x)
                - max(
                    self._vehicle.bounding_box.extent.y,
                    self._vehicle.bounding_box.extent.x,
                )
            )

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint and self._incoming_waypoint.is_junction and (
            self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]
        ):
            if self._speed_limit_flag:
                target_speed = min([self._target_speed, self._speed_limit - 2])
            else:
                target_speed = self._target_speed
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            if self._speed_limit_flag and self._speed_limit:
                target_speed = min(self._speed_limit, self._target_speed)
                self._local_planner.set_speed(target_speed)
            else:
                self._local_planner.set_speed(self._target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
