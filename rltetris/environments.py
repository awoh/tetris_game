# Environments
# Put all custom environments here
import numpy as np
import gym
import logging
logger = logging.getLogger(__name__)
from tetris import TetrisEnvironment

# can just download premade tetris environment online

# to register, look at torchkit (good example of base environment and multiple subtypes)

def features(state):
    """returns list of features for given state """
    pass


Class FeatureWrapper():

	__init__(self,env):
		self._env = env


    def set_state(self, state):
        self._env.setState(state)


	def step(action):
		S_tp1,t_tp1,terminal,inf = self._env.step(action)

		return features(s_tp1),r_tp1,terminal,inf

    def get_action_set(self):
        return self._env.get_action_set()
