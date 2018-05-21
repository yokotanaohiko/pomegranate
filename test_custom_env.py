
from gym import Env, logger
from gym.spaces import Discrete, Tuple


class MyEnv(Env):
    def __init__(self):
        self.action_space = Discrete(5)
        self.observable_space = Discrete(3)
        self.prev_action = 0

    def render(self):
        print(self.prev_action)

    def step(self, action):
        self.prev_action = action
        assert self.action_space.contains(action)
        return 0, 1, True, {}
        
    def reset(self):
        return 0


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


env = MyEnv()

env.reset()
agent = RandomAgent(env.action_space)

_obs, _rew, _done = (None, None, None)

for i in range(10):
    action = agent.act(_obs, _rew, _done)
    _obs, _rew, _done, _info = env.step(action)
    env.render()

