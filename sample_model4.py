#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from gym import Env, logger
from gym.spaces import Discrete, Tuple
from random import random, randint
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter
from sklearn.neural_network import MLPClassifier

np.set_printoptions(precision=2)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_size=(4, 4, 4), random_state=1)


class MyEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.prev_action = 0
        self.state = np.zeros(4).astype(np.int32)

    def render(self):
        pass
        # print('action: ', self.prev_action, 'state:', self.state)

    def step(self, action):
        self.prev_action = action
        assert self.action_space.contains(action)
        self.state[action] += 1

        reward = 0
        if self.state[action] == 10:
            reward = 1
        elif self.state[action] > 10:
            reward = -1
        is_done = np.all(self.state > 9) or np.any(self.state > 15)
        if is_done:
            reward = 1 if np.all(self.state > 9) else -1
        return np.copy(self.state), reward, is_done, {}
        
    def reset(self):
        self.state = np.zeros(4).astype(np.int32)


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, importance):
        self.epsilon = 0.2
        self.action_space = action_space
        self.q = {}
        self.importance = importance

    def act(self, observation, reward, done, train=True):
        state = tuple(observation)

        if self.q.get(state) is None:
            self.q[state] = np.zeros(4) + 0.5 # [0.5, 0.5, 0.5, 0.5]

        if train and random() < self.epsilon:
            # print('random action selected')
            return randint(0, 3)
        else:
            max_value = np.max(self.q[state])
            next_action = np.random.choice(np.where(self.q[state] == max_value)[0])
            return next_action

    def learn(self, obs1, action, reward, obs2, done):
        state1 = tuple(obs1)
        state2 = tuple(obs2)
        if self.q.get(state1) is None:
            self.q[state1] = np.zeros(4) + 0.5
        if self.q.get(state2) is None:
            self.q[state2] = np.zeros(4) + 0.5

        if done:
            self.q[state1][action] = reward * self.importance[action] if reward >= 0 else -1
            # self.q[state1] = normalize(self.q[state1].reshape(1, -1))[0]
        else:
            self.q[state1][action] = (reward  * self.importance[action] if reward >= 0 else -1) + 0.7 * np.max(self.q[state2])
            # self.q[state1] = normalize(self.q[state1].reshape(1, -1))[0]


    def get_action_value(self, obs):
        return self.q.get(tuple(obs))


def train1():
    env = MyEnv()

    env.reset()
    importance = [1,2,3,4]
    agent = RandomAgent(env.action_space, importance)

    _obs, _rew, _done = (np.zeros(4).astype(np.int32), None, None)

    for _ in range(10000 * 5):
        if _done:
            env.reset()
            _obs, _rew, _done = (np.zeros(4).astype(np.int32), None, None)

        action = agent.act(_obs, _rew, _done)
        _next_obs, _rew, _done, _info = env.step(action)
        print('s1:', tuple(_obs), 'av:', agent.get_action_value(_obs), 'a:', action, 'r:', _rew, 's2:', tuple(_next_obs), 'done:', _done, end=' ')
        agent.learn(_obs, action, _rew, _next_obs, _done)
        print('av:', agent.get_action_value(_obs))
        _obs = _next_obs


    print()
    print('----test----')
    env.reset()
    _obs, _rew, _done = (np.zeros(4).astype(np.int32), None, None)
    for i in range(40):
        action = agent.act(_obs, _rew, _done, train=False)
        _next_obs, _rew, _done, _info = env.step(action)
        print('s1:', tuple(_obs), 'av:', agent.get_action_value(_obs), 'a:', action, 'r:', _rew, 's2:', tuple(_next_obs), 'done:', _done)

        _obs = _next_obs

if __name__ == '__main__':
    train1()
