# Environments
# Put all custom environments here
import numpy as np
import gym
import logging
logger = logging.getLogger(__name__)
# from gym_tetris import TetrisEnvironment
from environment import TetrisEnvironment
# can just download premade tetris environment online

# to register, look at torchkit (good example of base environment and multiple subtypes)

class FeatureWrapper(object):
    def __init__(self,env):
        self._env = env
        self._terminal = env._terminal

    def set_state(self, state):
        self._env.set_state(state)

    def step(self,action):
        s_tp1,r_tp1,terminal,inf = self._env.step(action)
        self._terminal = self._env._terminal
        board_features = self._env.get_features()
        return board_features,r_tp1,terminal,inf

    def get_action_set(self):
        return self._env.get_action_set()

    def reset(self):
        self._env.reset()
        self._terminal = self._env._terminal

    # @property
    def state(self):
        return self._env.get_features()
