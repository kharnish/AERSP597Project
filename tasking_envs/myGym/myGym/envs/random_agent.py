"""
random_agent.py

Kelly Harnish
9 Jan 2020

This inputs a random action into the watchbox environment and renders its path until it reaches the destination.
"""
import numpy as np
import gym
import myGym
from gym import spaces
env = gym.make('Watchbox-v0')

print(np.reshape(np.array(0, 0)))

pathList = ([[0, 0]])
while True:
    action = np.random.randint(low=0, high=4, size=1)
    steps, reward, done = env.step(action)
    pathList.append(steps)
    if done:
        break
path = np.asarray(pathList)
# env.render(path)
env.close()
