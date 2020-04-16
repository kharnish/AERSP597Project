"""
forrest_env.py

Kelly Harnish
9 Jan 2020

This is a gym environment for training an RL agent on a simple grid world based of the OpenAI gyms,
with two obstacles and a target. The grid is a a 5x3, where [ ] = open area, [X] = obstacle, [S] = start,
[T] = target

   0  1  2
0 [S][ ][X]
1 [ ][ ][X]
2 [ ][ ][ ]
3 [ ][X][ ]
4 [ ][ ][T]

States: 15 locations
Actions: 4 movements
    0 = up
    1 = right
    2 = down
    3 = left
"""
import numpy as np
import gym
from matplotlib import pyplot as plt

# Define properties
grid_rows = 5
grid_cols = 3
initialState = (0, 0)
winState = (4, 2)
obstacles = np.array([[0, 2], [1, 2], [3, 1]])


class ForrestEnv(gym.Env):
    # metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self):
        self.stateSize = (grid_rows, grid_cols)
        self.state = initialState  # (rows, columns) start in upper left corner
        self.actionSize = 4  # up, right, down, left

    def allowedActions(self):
        # Generate list of actions allowed depending on location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if y > 0:
            actions_allowed.append(0)
        if y < grid_rows - 1:
            actions_allowed.append(2)
        if x > 0:
            actions_allowed.append(3)
        if x < grid_cols - 1:
            actions_allowed.append(1)
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def step(self, action):
        # Given an action, verify that it is legal then calculate new position. Also calculate reward for that action
        def checkLegal(row, col):
            legal = True
            if row > (grid_rows - 1) or row < 0:
                legal = False
            if col > (grid_cols - 1) or col < 0:
                legal = False
            return legal

        new_row, new_col = self.state

        if action == 0:  # up
            if checkLegal(new_row - 1, new_col):
                new_row = new_row - 1
        elif action == 1:  # right
            if checkLegal(new_row, new_col + 1):
                new_col = new_col + 1
        elif action == 2:  # down
            if checkLegal(new_row + 1, new_col):
                new_row = new_row + 1
        elif action == 3:  # left
            if checkLegal(new_row, new_col - 1):
                new_col = new_col - 1

        self.state = (new_row, new_col)
        reward = self._giveReward()

        if self.state == winState:
            isDone = True
        else:
            isDone = False

        return self.state, reward, isDone

    def _giveReward(self):
        # Calculate award based on steps taken and
        award = -0.1  # small punishment for each step taken
        for i in range(len(obstacles)):
            if self.state[0] == obstacles[int(i), 0] and self.state[1] == obstacles[int(i), 1]:
                award -= 100  # large punishment if in obstacle space
        if self.state == winState:
            award += 100  # large reward if target reached
        return award

    def reset(self):
        # Reset to initial conditions
        self.state = initialState
        return self.state

    def render(self, path, mode='human'):

        x = path[:, 1] + 0.5
        y = 4 - path[:, 0] + 0.5
        fig, ax = plt.subplots()
        ax.grid()
        plt.xticks(np.arange(0, 4, 1))
        plt.xlim(0, 3)
        plt.ylim(0, 5)

        # Mark start, target, and obstacles
        plt.plot(winState[1] + 0.5, 4 - winState[0] + 0.5, color='r', marker='H')
        plt.plot(initialState[1] + 0.5, 4 - initialState[0] + 0.5, color='g', marker='*')
        for i in range(len(obstacles)):
            plt.plot(obstacles[i, 1] + 0.5, 4 - obstacles[i, 0] + 0.5, color='k', marker='x')

        # Note current position to remove after next step
        location = plt.scatter(initialState[1] + 0.5, 4 - initialState[0] + 0.5, color='k', marker='o')

        # Animate agent path
        plt.pause(1)
        for i in range(len(x)):
            location.remove()
            location = plt.scatter(x[i], y[i], color='k', marker='o')
            plt.draw()
            plt.pause(0.5)
        plt.pause(3)