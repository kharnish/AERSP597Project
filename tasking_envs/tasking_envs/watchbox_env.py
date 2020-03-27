"""
Grid world environment for satellite tasking


Grid is given as 5x3: [ ] = watchbox, [X] = land (not a watchbox), [S] = start, [T] = target

   0  1  2
0 [S][ ][X]
1 [ ][ ][X]
2 [ ][ ][ ]
3 [ ][X][ ]
4 [ ][ ][T]

States: 15 locations
Actions: 5 movements

state space: (row, column)
"""

import numpy as np
import gym
import myGym
from gym import spaces
from gym.utils import seeding

Map = ["[ ][ ][X]",
       "[ ][ ][X]",
       "[ ][ ][ ]",
       "[ ][X][ ]",
       "[ ][ ][ ]"
       ]

grid_rows = 5
grid_cols = 3
win_state = (5, 2)

up = 0
right = 1
down = 2
left = 3
stay = 4


class WatchGrid(gym.env):

    def __init__(self):
        self.desc = np.asarray(Map)  # ), dtype='c') # don't know if this needs to be a floating point number
        self.size = (grid_cols, grid_rows)
        self.state = (0, 0)  # start in upper left corner

    def step(self, action):

        if action == 0:  # up
            newState = (self.state[0], max(self.state[1] - 1, grid_rows))
        elif action == 1:  # right
            newState = (min(self.state[0] + 1, grid_cols), self.state[1])
        elif action == 2:  # down
            newState = (self.state[0], min(self.state[1] + 1, grid_rows))
        elif action == 3:  # left
            newState = (max(self.state[0] - 1, grid_cols), self.state[1])
        return self.state

    def giveReward(self):
        if self.state == win_state:
            return 1
        else:
            return 0

    def render(self):

        screen_width = 600
        screen_height = 400

        scale = 1
