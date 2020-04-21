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
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[[s],a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

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
            Q_s = self.Q[state[0], state[1], actions_allowed]
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
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for i in range(self.state_dim[0]):
            for j in range(self.state_dim[1]):
                greedy_policy[i, j] = np.argmax(self.Q[i, j, :])
        print("\nGreedy policy(y, x):")
        print(greedy_policy)


def main():
    env = gym.make('Forrest-v0')
    agent = Agent(env)

    # Train agent
    print("\nTraining agent...\n")
    N_episodes = 500
    reward_total = np.zeros((N_episodes))

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
            state = state_next  # transition to next state
        reward_total[episode] = reward_episode

        # Decay agent exploration parameter
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

        # Print
        # if (episode == 0) or (episode + 1) % 10 == 0:
            # print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}".format(
        # episode + 1, N_episodes, agent.epsilon, iter_episode, reward_episode))

    plt.plot(reward_total)
    plt.xlabel('Agent')
    plt.ylabel('Reward')
    # plt.show()

    # Print greedy policy
    agent.display_greedy_policy()

    # Render optimal path
    path = np.asarray(pathList)
    env.render(path)


# Driver
if __name__ == '__main__':
    main()
