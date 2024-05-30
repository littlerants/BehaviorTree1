from gym.envs.registration import register

register(
    id='gym_sumo-v0',
    entry_point='gym_sumo.envs:SumoEnv',
)

register(
    id='citysim-sumo-v0',
    entry_point='gym_sumo.envs:CitySimEnv',
)

register(
    id='gail-sumo-v0',
    entry_point='gym_sumo.envs:GailSumoEnv',
)
