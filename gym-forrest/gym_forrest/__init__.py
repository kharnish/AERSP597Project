from gym.envs.registration import register

register(
        id='Forrest-v0',
        entry_point='gym_forrest.envs:ForrestEnv'
)
