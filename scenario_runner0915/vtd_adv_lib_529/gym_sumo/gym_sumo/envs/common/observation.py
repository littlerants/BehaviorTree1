from collections import OrderedDict
from itertools import product
from typing import List, Dict, TYPE_CHECKING, Optional, Union, Tuple, Any
from gymnasium import spaces
import numpy as np
import pandas as pd

from gym_sumo import utils
from gym_sumo.envs.common.finite_mdp import compute_ttc_grid
from gym_sumo.envs.common.graphics import EnvViewer
from gym_sumo.road.lane import AbstractLane
from gym_sumo.utils import distance_to_circle, Vector
from gym_sumo.vehicle.controller import MDPVehicle
from gym_sumo.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from gym_sumo.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class ChangAnAdvObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""
    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy', 'heading']
    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 11,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = False,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 include_obstacles: bool = True,
                 **kwargs: dict) -> None:

        """

        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    # def space(self) -> spaces.Space:
    #     return spaces.Box(shape=(1, 56), low=-np.inf, high=np.inf, dtype=np.float32)
    def space(self) -> spaces.Space:
        return spaces.Box(shape=(1, (self.vehicles_count - 1) * len(self.features) + 5), low=-np.inf, high=np.inf,
                          dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.
        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED]

            }

        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> Any:
        if not self.env.road:
            return np.zeros(self.space().shape)
        df_adv = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        df_adv = df_adv[self.features]
        orin_adv_df = df_adv.copy()
        df_adv = self.normalize_obs(df_adv)
        vehicles = []
        # Add nearby traffic
        if self.env.config['obs_v1']:
            close_vehicles = self.env.road.close_objects_to(self.observer_vehicle,
                                                            60,
                                                            count=self.vehicles_count - 1,
                                                            see_behind=self.see_behind,
                                                            sort=self.order == "sorted",
                                                            vehicles_only=not self.include_obstacles)
        vehicles.append(self.env.ego_vehicle)

        if self.env.config['obs_v1']:
            vehicles.extend(close_vehicles)

        traj_obs = []
        traj_obs.extend(df_adv.to_numpy()[0, 2:].tolist())
        if self.env.config['obs_v2']:
            if hasattr(self.env, 'meeting_points'):
                ego_lon, _ = self.env.vehicle.lane.local_coordinates(self.env.meeting_points)
                ego_current_lon = self.env.vehicle.lane_offset[0]
                bv_lon, _ = self.env.ego_vehicle.lane.local_coordinates(self.env.meeting_points)
                bv_current_lon = self.env.ego_vehicle.lane_offset[0]
                ego_t = (ego_lon - ego_current_lon) / utils.not_zero(self.env.vehicle.speed)
                bv_t = (bv_lon - bv_current_lon) / utils.not_zero(self.env.ego_vehicle.speed)
                traj_obs.append(ego_t - bv_t)
                traj_obs.append(ego_lon - ego_current_lon)

        veh_num = 0
        for veh in vehicles:
            df = pd.DataFrame.from_records([veh.to_dict()])
            df = df[self.features]
            dx = orin_adv_df.loc[0, 'x'] - df.loc[0, 'x']
            dy = orin_adv_df.loc[0, 'y'] - df.loc[0, 'y']
            dvx = orin_adv_df.loc[0, 'vx'] - df.loc[0, 'vx']
            dvy = orin_adv_df.loc[0, 'vy'] - df.loc[0, 'vy']

            dx = utils.lmap(dx, [self.features_range['x'][0], self.features_range['x'][1]], [-1, 1])
            dy = utils.lmap(dy, [self.features_range['y'][0], self.features_range['y'][1]], [-1, 1])
            dvx = utils.lmap(dvx, [self.features_range['vx'][0], self.features_range['vx'][1]], [-1, 1])
            dvy = utils.lmap(dvy, [self.features_range['vy'][0], self.features_range['vy'][1]], [-1, 1])
            traj_obs.extend([dx, dy, dvx, dvy])
            veh_num += 1
            if veh_num <= 1:
                traj_obs.extend(df.to_numpy()[0, 4:].tolist())
            if veh_num >= self.env.config["observation"]['vehicles_count']:
                break

        traj_obs = np.array(traj_obs)
        if self.env.config['obs_v1']:
            if veh_num < self.env.config["observation"]['vehicles_count']:
                zero_pad = np.zeros((self.env.config["observation"]['vehicles_count'] - veh_num) * 4)
                traj_obs = np.append(traj_obs, zero_pad)

        return traj_obs.astype(self.space().dtype)

    def close_objects_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: Optional[int] = None,

                         see_behind: bool = True, sort: bool = True, vehicles_only: bool = False) -> object:
        vehicles = [v for v in self.env.road.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]

        objects_ = vehicles
        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count:
            objects_ = objects_[:count]

        return objects_


def observation_factory(env: 'AbstractEnv', config: dict) -> ObservationType:
    if config['type'] == "ChangAnAdv":
        return ChangAnAdvObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
