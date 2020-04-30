"""
QAgent.py

Kelly Harnish
9 Jan 2020

This uses a Q-learning algorithm to train an epsilon-greedy agent to find the shortest path from the starting
position to the target position while avoiding obstacles.

Based on code authored by Anson Wong,
from https://towardsdatascience.com/training-an-agent-to-beat-grid-world-fac8a48109a8
and https://github.com/ankonzoid/LearningX/tree/master/classical_RL/gridworld,
Accessed 7 Jan 2020
"""
import numpy as np
import gym
import gym_forrest
import random
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, env):
        # Store state and action dimension
        self.state_dim = env.stateSize
        self.action_dim = env.actionSize
        # Agent learning parameters
        self.epsilon = 1.0          # initial exploration probability
        self.epsilon_decay = 0.99   # epsilon decay after each episode
        self.beta = 0.99            # learning rate
        self.gamma = 0.99           # reward discount factor
        # Initialize Q[[s],a] table
        if env.actionSize == 6:
            self.Q = np.zeros((self.state_dim[0], self.state_dim[1], self.state_dim[2], self.action_dim), dtype=float)
        else:
            self.Q = np.zeros((self.state_dim[0], self.state_dim[1], self.action_dim), dtype=float)

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            xp = np.random.choice(env.allowed_actions())
            return xp
        else:
            # exploit on allowed actions
            state = env.state
            actions_allowed = env.allowed_actions()
            if self.action_dim == 6:
                Q_s = self.Q[state[0], state[1], state[2], actions_allowed]  # for 3D environment
            else:
                Q_s = self.Q[state[0], state[1], actions_allowed]  # for 2D environment
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[sa])

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']

        if self.action_dim == 6:
            greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1], self.state_dim[2]), dtype=int)
            for i in range(self.state_dim[0]):
                for j in range(self.state_dim[1]):
                    for k in range(self.state_dim[2]):
                        greedy_policy[i, j, k] = np.argmax(self.Q[i, j, k, :])
            print("\nGreedy policy(y, x):")
            print(greedy_policy)
        else:
            greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
            for i in range(self.state_dim[0]):
                for j in range(self.state_dim[1]):
                    greedy_policy[i, j] = np.argmax(self.Q[i, j, :])
            print("\nGreedy policy(y, x):")
            print(greedy_policy)


def main():
    env = gym.make('Forrest-v1')
    agent = Agent(env)

    # Train agent
    print("\nTraining agent...\n")
    N_episodes = 750
    reward_total = []
    reward_average = []

    for episode in range(N_episodes):
        # Generate an episode
        iter_episode, reward_episode = 0, 0
        state = env.reset()  # starting state
        pathList = [np.asarray(state)]  # initialize list for rendering path

        while True:
            action = agent.get_action(env)  # get action
            state_next, reward, done = env.step(action)  # evolve state by action
            pathList.append(state_next)
            agent.train((state, action, state_next, reward, done))  # train agent
            iter_episode += 1
            reward_episode += reward
            if done:
                break
            if iter_episode == 5000:
                break
            state = state_next  # transition to next state
        reward_total.append(reward_episode)
        reward_average.append(np.mean(reward_total[max(0, episode - 100):(episode + 1)]))

        # Decay agent exploration parameter
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

        # Print
        if episode % 10 == 0:
            print("Episode:", episode, "\tEpsilon:", '%.6f' % agent.epsilon, "\tEpisode Iters:" , iter_episode, "\t\tEpisode reward:",
                  '%.3f' % reward_total[episode], "\t\tAvg reward (last 100):", '%.6f' % reward_average[episode])
        if reward_average[episode] > 98:  # try 99.35 for 2D and 98 for 3D
            print("Total Episodes: ", episode)
            break

    plt.rcParams["font.family"] = "serif"
    plt.plot(reward_total, '--g', label='Total Reward')
    plt.plot(reward_average, 'b', label='Average Reward')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # Print greedy policy
    # agent.display_greedy_policy()

    # Render optimal path
    path = np.asarray(pathList)
    if env.actionSize == 6:
        plt.savefig('Q_agent_3D.png')
        env.render(path, 'Q_agent_3D.mp4')
    else:
        plt.savefig('Q_agent_2D.png')
        env.render(path, 'Q_agent_2D.mp4')


# Driver
if __name__ == '__main__':
    main()
