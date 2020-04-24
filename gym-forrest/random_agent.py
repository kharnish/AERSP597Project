"""
random_agent.py

Kelly Harnish
9 Jan 2020

This inputs a random action into the forrest environment and renders its path until it reaches the destination.
"""
import numpy as np
import gym
import gym_forrest

env = gym.make('Forrest-v1')
state = env.reset()  # starting state
pathList = [np.asarray(state)]  # initialize list for rendering path

while True:
    if env.actionSize == (6,):
        action = np.random.randint(low=0, high=6, size=1)  # choose random action for 3D environment
    else:
        action = np.random.randint(low=0, high=4, size=1)  # choose random action for 2D environment
    steps, reward, done = env.step(action)
    pathList.append(steps)
    if done:
        break
path = np.asarray(pathList)
env.render(path)
env.close()
