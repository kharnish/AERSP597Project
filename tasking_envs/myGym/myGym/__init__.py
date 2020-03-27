from gym.envs.registration import register

register(
    id='Watchbox-v0',
    entry_point='myGym.envs:TaskingEnv',
)
