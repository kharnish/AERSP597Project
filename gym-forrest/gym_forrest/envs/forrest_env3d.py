"""
forrest_env.py

Kelly Harnish
9 Jan 2020

This is a gym environment for training an RL agent in a 3D grid world based on the OpenAI gyms,
with obstacles and a target. The grid is a a 5x5x5, shown in map.csv

^+y

.+z   +x>

States: 125 locations
Actions: 6 movements
    0 = forward (+x)
    1 = back    (-x)
    2 = right   (+y)
    3 = left    (-y)
    4 = up      (+z)
    5 = down    (-z)
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

grid_rows = 5
grid_cols = 5
grid_height = 5
initialState = (0, 0, 0)
winState = (4, 4, 3)

raw_map = np.loadtxt("map.csv", delimiter=",")
obstacles = np.zeros((1, 3))
for i in range(grid_rows*grid_height):
    for j in range(grid_cols):
        if raw_map[i, j] == 1:
            obstacles = np.append(obstacles, [[i % 5, j, np.math.floor(i/5)]], axis=0)
obstacles = np.delete(obstacles, 0, axis=0)  # delete first index, which was from creating the array


class ForrestEnv3d(gym.Env):
    # metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self):
        self.stateSize = (grid_rows, grid_cols, grid_height)
        self.state = initialState  # (columns, rows, height) start in upper left corner
        self.actionSize = (6,)

    def allowed_actions(self):
        # Generate list of actions allowed depending on location
        actions_allowed = []
        x, y, z = self.state[0], self.state[1], self.state[2]
        if x < grid_rows - 1:
            actions_allowed.append(0)
        if x > 0:
            actions_allowed.append(1)
        if y < grid_cols - 1:
            actions_allowed.append(2)
        if y > 0:
            actions_allowed.append(3)
        if z < grid_height - 1:
            actions_allowed.append(4)
        if z > 0:
            actions_allowed.append(5)
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def step(self, action):
        # Given an action, verify that it is legal then calculate new position. Also calculate reward for that action
        def check_legal(row, col, hei):
            legal = True
            if col > (grid_cols - 1) or col < 0:
                legal = False
            if row > (grid_rows - 1) or row < 0:
                legal = False
            if hei > (grid_height - 1) or hei < 0:
                legal = False
            return legal

        new_col, new_row, new_hei = self.state

        if action == 0:
            if check_legal(new_row, new_col + 1, new_hei):
                new_col = new_col + 1
        elif action == 1:
            if check_legal(new_row, new_col - 1, new_hei):
                new_col = new_col - 1
        elif action == 2:
            if check_legal(new_row + 1, new_col, new_hei):
                new_row = new_row + 1
        elif action == 3:
            if check_legal(new_row - 1, new_col, new_hei):
                new_row = new_row - 1
        elif action == 4:
            if check_legal(new_row, new_col, new_hei + 1):
                new_hei = new_hei + 1
        elif action == 5:
            if check_legal(new_row, new_col, new_hei - 1):
                new_hei = new_hei - 1

        self.state = (new_col, new_row, new_hei)
        reward = self.give_reward()

        if self.state == winState:
            is_done = True
        else:
            is_done = False

        return self.state, reward, is_done

    def give_reward(self):
        # Calculate award based on steps taken and
        award = -0.1  # small punishment for each step taken
        for ii in range(len(obstacles)):
            if [self.state[0], self.state[1], self.state[2]] == [obstacles[ii, 0], obstacles[ii, 1], obstacles[ii, 2]]:
                award -= 100  # large punishment if in obstacle space
        if self.state == winState:
            award += 100  # large reward if target reached
        return award

    def reset(self):
        # Reset to initial conditions
        self.state = initialState
        return self.state

    def render(self, path, mode='human'):

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

        # define figure
        fig, ax = plt.subplots()
        ax.grid()
        # plt.xticks(np.arange(0, 5, 1))
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.title("Floor 0")

        # Mark initial start and obstacles
        """plt.scatter(winState[1] + 0.5, winState[0] + 0.5, color='r', marker='H', s=20)
        plt.scatter(initialState[1] + 0.5,  initialState[0] + 0.5, color='g', marker='*', s=20)
        for ii in range(len(obstacles)):
            if obstacles[ii, 2] == initialState[2]:
                plt.scatter(obstacles[ii, 1] + 0.5,  obstacles[ii, 0] + 0.5, color='k', marker='x', s=30)
        plt.scatter(initialState[1] + 0.5,  initialState[0] + 0.5, color='k', marker='o', s=150)
"""
        def animate(frame):
            x = path[:, 0] + 0.5
            y = path[:, 1] + 0.5
            z = path[:, 2]

            ax.clear()
            ax.grid()
            plt.xlim(0, 5)
            plt.ylim(0, 5)
            plt.title("Floor " + str(z[frame]))

            if winState[2] == z[frame]:
                plt.plot(winState[1] + 0.5,  winState[0] + 0.5, color='r', marker='H', markersize=20)
            if initialState[2] == z[frame]:
                plt.plot(initialState[1] + 0.5,  initialState[0] + 0.5, color='g', marker='*', markersize=20)
            for iii in range(len(obstacles)):
                if obstacles[iii, 2] == z[frame]:
                    plt.plot(obstacles[iii, 1] + 0.5,  obstacles[iii, 0] + 0.5, color='k', marker='x', markersize=30)

            plt.scatter(x[frame], y[frame], color='k', marker='o', s=150)

        ani = FuncAnimation(fig, animate, frames=len(path), interval=2000)
        ani.save('path3D.mp4', writer=writer)
        #plt.show()

