# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


class Cautious(object):
    """Class for Cautious agent."""

    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0


class Normal(object):
    """Class for Normal agent."""

    max_speed = 35
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 4
    min_proximity_threshold = 15
    braking_distance = 20
    tailgate_counter = 0
    overtake_counter = 0
    emergency_param = 0.4
    desired_acceleration = 10
    desired_deceleration = 10 
    lane_changing_dynamic = False
    urge_to_overtake = False



class Aggressive(object):
    """Class for Aggressive agent."""

    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 4
    tailgate_counter = -1
