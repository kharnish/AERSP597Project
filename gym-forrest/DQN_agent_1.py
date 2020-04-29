"""
DQN_agent_1.py

Kelly Harnish
27 April 2020

This uses a DQN learning algorithm to train an agent to find the shortest path from the starting position to the
target position while avoiding obstacles.

Based on code authored by Siwei Xu,
https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
Accessed 7 Jan 2020
"""

import numpy as np
import tensorflow as tf
import gym
import gym_forrest
import os
# import datetime
# from gym import wrappers
import matplotlib.pyplot as plt


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def main():
    env = gym.make('Forrest-v0')
    gamma = 0.99
    copy_step = 25
    num_states, num_actions = env.get_size()
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 1000  # max number of episodes
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    reward_total = []
    reward_average = []
    for n in range(N):
        rewards = 0
        iter = 0
        done = False
        observations = env.reset()  # initial state
        pathList = [np.asarray(observations)]  # initialize list for rendering path
        epsilon = max(min_epsilon, epsilon * decay)
        losses = list()

        while not done:
            action = TrainNet.get_action(observations, epsilon)
            prev_observations = observations
            observations, reward, done = env.step(action)
            pathList.append(observations)
            rewards += reward
            if done:
                reward = -200
                env.reset()
            exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
            TrainNet.add_experience(exp)
            loss = TrainNet.train(TargetNet)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())
            iter += 1
            if iter % copy_step == 0:
                TargetNet.copy_weights(TrainNet)

        losses = np.mean(losses)
        reward_total.append(rewards)
        reward_average.append(np.mean(reward_total[max(0, n - 100):(n + 1)]))

        if n % 100 == 0:
            print("Episode:", n, "\t\tEpisode reward:", '%.3f' % reward_total[n], "\tEpsilon:", '%.6f' % epsilon,
                  "\tAvg reward (last 100):", '%.6f' % reward_average[n], "Episode loss: ", losses)
        if reward_average[n] > -500:
            break
    print("avg reward for last 100 episodes:", reward_average[n])
    # env.close()

    # Plot reward over episodes
    plt.plot(reward_total, '--g', label='Total Reward')
    plt.plot(reward_average, 'b', label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # Render optimal path
    path = np.asarray(pathList)
    env.render(path, 'DQN_agent_2D.mp4')


# Driver
if __name__ == '__main__':
    main()
