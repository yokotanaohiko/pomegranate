'''
Symptom・Causeのチェックをするかどうかの判断の強化学習のプログラム

state:
    [流れてきた数, Symptom有の数, Symptom無の数, Cause有の数, Cause無の数, Symptomをチェックするかどうか(0,1の二値)]
action:
    [Symptomをチェックする, Causeをチェックする, 何もしない]
'''

import os
import sys
from enum import Enum
from random import random, randint
import pickle

import numpy as np
from gym import Env
from gym.spaces import Discrete

from logistic_regression import get_logistic_regression_coef

np.set_printoptions(precision=2)


class Action(Enum):
    '''行動の値'''
    symptom_check=0
    cause_check=1
    no_check=2

class Unit:
    '''ある機種についての情報'''
    def __init__(self, symptom_prob=0.3, coef=0.7):
        '''
        symptom_prob: symptom発生確率
        coef: symptomを観測した時のcauseの発生確率(条件付き確率)
        '''
        self.symptom = random() < symptom_prob
        self.cause = random() < coef if self.symptom else False
    def has_symptom(self):
        '''Symptomを持っているかどうかを判定'''
        return self.symptom
    def has_cause(self):
        '''Causeを持っているかどうかを判定'''
        return self.cause

class MyEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.prev_action = 0
        self.state = np.array([1, 0, 0, 0, 0, 1])
        self.current_model = Unit()
        self.step_count = 0

        self.coef_threshold = 0.7
        self.significance_level = 0.05

    def render(self):
        '''何もしないように、gymライブラリのrenderメソッドをオーバーライド'''
        pass

    def step(self, action):
        self.prev_action = action
        assert self.action_space.contains(action)
        is_symptom_turn = self.state[5] > 0
        is_done = False
        reward = 0
        if is_symptom_turn:
            if Action(action) == Action.symptom_check:
                reward = -1 # Symptomのチェックにはコストがかかるため
                if self.current_model.has_symptom():
                    self.state[1] += 1
                    self.state[5] = 0
                else:
                    self.state[2] += 1

                    reward, is_done = self.calc_reward()
                    # SymptomCheckをしない場合、次のインスタンスをrefillする
                    self.refill_model()
            if Action(action) == Action.cause_check:
                raise Exception('SymptomチェックのタイミングでCauseチェックはできません')

        else:
            if Action(action) == Action.cause_check:
                reward = -1 # Causeのチェックにはコストがかかるため
                if self.current_model.has_cause():
                    self.state[3] += 1
                else:
                    self.state[4] += 1

                self.state[5] = 1
                reward, is_done = self.calc_reward()
                # CauseCheckが終わったら、次のインスタンスをrefillする
                self.refill_model()
            if Action(action) == Action.no_check:
                self.refill_model()

            if Action(action) == Action.symptom_check:
                raise Exception('CauseチェックのタイミングでSymptomチェックはできません')

        if Action(action) == Action.no_check:
            is_coef_calculatable = self.state[0] > self.state[1] and self.state[3] > 0
            if not is_coef_calculatable:
                reward = -1 # coefを計算できるようになるまではペナルティを与える

        self.step_count += 1

        if self.step_count > 30: # 学習を早くするため、30stepで打ち切る
            is_done = True
            reward = reward or -1
        return np.copy(self.state), reward, is_done, {}

    def reset(self):
        self.prev_action = 0
        self.state = np.array([1, 0, 0, 0, 0, 1])
        self.current_model = Unit()
        self.step_count = 0
        return self.state

    def refill_model(self):
        '''次の端末を検証フローに流す'''
        self.current_model = Unit()
        self.state[0] += 1

    def calc_reward(self):
        '''相関係数(厳密には違うが)を計算する'''
        is_coef_calculatable = self.state[0] > self.state[1] and self.state[3] >= 1 
        if is_coef_calculatable:
            coef, p_value = get_logistic_regression_coef(self.state[0], self.state[1], self.state[3])
            if coef > self.coef_threshold:
                if p_value < self.significance_level:
                    return 100, True
        return -1, False

    def get_current_state(self):
        return self.state


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.epsilon = 0.2
        self.action_space = action_space
        self.q = {}

    def act(self, observation, reward, done, train=True):
        state = tuple(observation)
        self.ensure_action_value(observation)

        if state[5] > 0:
            if train and random() < self.epsilon:
                return [Action.symptom_check, Action.no_check][randint(0, 1)].value
            else:
                return Action.symptom_check.value if self.q[state][0] > self.q[state][2] else Action.no_check.value
        else:
            if train and random() < self.epsilon:
                return [Action.cause_check, Action.no_check][randint(0, 1)].value
            else:
                return Action.cause_check.value if self.q[state][1] > self.q[state][2] else Action.no_check.value

        raise Exception('act:想定しないケース', state)

    def learn(self, obs1, action, reward, obs2, done):
        state1 = tuple(obs1)
        state2 = tuple(obs2)

        self.ensure_action_value(obs1)
        self.ensure_action_value(obs2)

        if done:
            self.q[state1][action] = reward if reward >= 0 else -1
        else:
            self.q[state1][action] = (reward if reward >= 0 else -1) + 0.7 * np.max(self.q[state2])


    def get_action_value(self, obs):
        return self.q.get(tuple(obs))

    def ensure_action_value(self, obs):
        state = tuple(obs)
        if self.q.get(state) is None:
            if state[5] > 0: # SymptomCheckをすべき時
                self.q[state] = np.array([0.5, -100, 0.5])
            else: # CauseCheckをすべき時
                self.q[state] = np.array([-100, 0.5, 0.5])
            return

    def save_action_value(self):
        with open('action_value.pickle', 'wb') as f:
            pickle.dump(self.q, f)

    def load_action_value(self):
        '''事前に学習したq関数を読み込む'''
        if os.path.exists('action_value.pickle'):
            with open('action_value.pickle', 'rb') as f:
                self.q = pickle.load(f)
        else:
            self.q = {}

def train1():
    env = MyEnv()

    env.reset()
    agent = RandomAgent(env.action_space)
    agent.load_action_value()
    print(agent.q)

    _obs, _rew, _done = (env.reset(), None, None)

    for _ in range(100000):
        if _done:

            _obs, _rew, _done = (env.reset(), None, None)

        action = agent.act(_obs, _rew, _done)
        _next_obs, _rew, _done, _info = env.step(action)
        print('s1:', tuple(_obs), 'av:', agent.get_action_value(_obs), 'a:', action, 'r:', _rew, 's2:', tuple(_next_obs), 'done:', _done, end=' ')
        agent.learn(_obs, action, _rew, _next_obs, _done)
        print('av:', agent.get_action_value(_obs))
        _obs = _next_obs
    agent.save_action_value()

def test():
    env = MyEnv()

    env.reset()
    agent = RandomAgent(env.action_space)
    agent.load_action_value()

    def print_action_value(obs):
        print('obs:', obs, 'av:', agent.get_action_value(obs), 'next:', Action(agent.act(obs, None, None, False)))

    print_action_value(np.array([1,0,0,0,0,1]))
    print_action_value(np.array([1,1,0,0,0,0]))
    print_action_value(np.array([2,1,0,1,0,1]))
    print_action_value(np.array([2,1,0,0,1,1]))
    print_action_value(np.array([3,0,2,0,0,1]))
    print_action_value(np.array([3,1,1,1,0,1]))
    print_action_value(np.array([3,1,1,0,1,1]))
    print_action_value(np.array([3,2,0,2,0,1]))
    print_action_value(np.array([3,2,0,1,1,1]))
    print_action_value(np.array([3,2,0,0,2,1]))
    print_action_value(np.array([4,0,3,0,0,1]))
    print_action_value(np.array([4,1,2,1,0,1]))
    print_action_value(np.array([4,1,2,0,1,1]))
    print_action_value(np.array([4,2,1,2,0,1]))
    print_action_value(np.array([4,2,1,1,1,1]))
    print_action_value(np.array([4,2,1,0,2,1]))
    print_action_value(np.array([4,3,0,3,0,1]))
    print_action_value(np.array([4,3,0,1,2,1]))
    print_action_value(np.array([4,3,0,2,1,1]))
    print_action_value(np.array([4,3,0,0,3,1]))

    _done = False
    _obs, _rew, _done = (env.reset(), None, None)
    while not _done:
        action = agent.act(_obs, _rew, _done)
        _next_obs, _rew, _done, _info = env.step(action)
        print('s1:', tuple(_obs), 'av:', agent.get_action_value(_obs), 'a:', action, 'r:', _rew, 's2:', tuple(_next_obs), 'done:', _done)
        _obs = _next_obs


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test()
    else:
        train1()
