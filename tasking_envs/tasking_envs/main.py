import numpy as np
import gym
import myGym

env = gym.make('Watchbox-v0')

action = np.random.randint(low=0, high=4, size=1)

for _ in range(500):
    env.render()
    env.step(env.action_space.sample())
env.close()


for e in range(100):
    env.reset()
    env.render()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()
    print(reward)
    if done:
        break
