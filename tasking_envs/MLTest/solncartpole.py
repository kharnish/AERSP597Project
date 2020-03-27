'''
Based the following two code bases:
 - Berkeley's CS188 pacman project code
   http://ai.berkeley.edu/
 - Victor Mayoral Vilches's RL tutorial 
   https://github.com/vmayoral/basic_reinforcement_learning

@author: Heechul Yun (heechul.yun@gmail.com)

Retrieved from : https://gist.github.com/heechul/9f8f43c229fc790af4a8f073108ed49f
3 January 2020
'''

import gym
import random
import numpy
import pandas
import functools

env = gym.make('CartPole-v0')

class QLearningAgent:
    def __init__(self, actions, epsilon=0.1, gamma=0.90, alpha=0.5, **args):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # exploration probability
        self.actions = actions
        self.qs = {}  # state table

    def getQValue(self, state, action):
        if not (state in self.qs) or not (action in self.qs[state]):
            return 0.0
        else:
            return self.qs[state][action]

    def getLegalActions(self, state):
        return self.actions

    # def getAction(self, state):
    #     action = None
    #     if util.flipCoin(self.epsilon):
    #         legalActions = self.getLegalActions(state)
    #         action = random.choice(legalActions)
    #     else:
    #         action = self.computeActionFromQValues(state)
    #     return action

    def getAction(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        q = [self.getQValue(state, a) for a in legalActions]
        maxQ = max(q)

        # this is the trick.
        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - 0.5 * mag for i in range(len(legalActions))]
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(legalActions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        return legalActions[i]

    def update(self, state, action, nextState, reward):
        """
        Update q-value of the given state
        """
        if not (state in self.qs):
            self.qs[state] = {}
        if not (action in self.qs[state]):
            self.qs[state][action] = reward
        else:
            maxqnew = max([self.getQValue(nextState, a) for a in self.getLegalActions(nextState)])
            diff = reward + self.gamma * maxqnew - self.qs[state][action]
            newQ = self.qs[state][action] + self.alpha * diff
            self.qs[state][action] = newQ

        # print "(s, a, s', r) = [%3d (%3.1f, %3.1f), %d, %3d (%3.1f, %3.1f), %.1f]" % \
        #     (state, self.getQValue(state,0), self.getQValue(state, 1), action, \
        #      nextState, self.getQValue(nextState,0), self.getQValue(nextState, 1), \
        #      reward)


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]


last100Scores = [0] * 100
last100ScoresIndex = 0
last100Filled = False

# Number of states is huge so in order to simplify the situation
# we discretize the space to: 10 ** number_of_features
n_bins = 8
n_bins_angle = 10
cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

last_time_steps = numpy.ndarray(0)

agent = QLearningAgent(actions=range(env.action_space.n),
                       alpha=0.5, gamma=0.90, epsilon=0.1)

# I commented this out because it was giving me an error and didn't seem to be needed
# env.monitor.start('cartpole-exp-1', force=True)

for i_episode in range(1001):
    state = env.reset()

    # if i_episode > 100:
    #     agent.epsilon = 0.01

    for t in range(200):
        if i_episode == 1000:
            env.render()

        # choose an action
        stateId = build_state([to_bin(state[0], cart_position_bins),
                               to_bin(state[1], cart_velocity_bins),
                               to_bin(state[2], pole_angle_bins),
                               to_bin(state[3], angle_rate_bins)])
        action = agent.getAction(stateId)

        # perform the action
        state, reward, done, info = env.step(action)
        nextStateId = build_state([to_bin(state[0], cart_position_bins),
                                   to_bin(state[1], cart_velocity_bins),
                                   to_bin(state[2], pole_angle_bins),
                                   to_bin(state[3], angle_rate_bins)])

        if done == False:
            # update q-learning agent
            agent.update(stateId, action, nextStateId, reward)
        else:
            reward = -200
            agent.update(stateId, action, nextStateId, reward)
            last100Scores[last100ScoresIndex] = t
            last100ScoresIndex += 1
            if last100ScoresIndex >= 100:
                last100Filled = True
                last100ScoresIndex = 0
            if not last100Filled:
                print("Episode ", i_episode, " finished after {} timesteps".format(t + 1))
            else:
                print("Episode ", i_episode, " finished after {} timesteps".format(t + 1), " last 100 average: ",
                      (sum(last100Scores) / len(last100Scores)))
            last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
            break

l = last_time_steps.tolist()
l.sort()
print("Overall score: {:0.2f}".format(last_time_steps.mean()))
print("Best 100 score: {:0.2f}".format(functools.reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

env.close()
