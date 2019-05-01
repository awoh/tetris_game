# Environments
# Put all custom environments here
import numpy as np
import gym
import logging
logger = logging.getLogger(__name__)
from tetris import environment as env

# can just download premade tetris environment online

# to register, look at torchkit (good example of base environment and multiple subtypes)

Class FeatureWrapper():

	__init__(self,env):
		self._env = env


    def set_state(self, state):
        self._env.setState(state)


	def step(action):
		S_tp1,t_tp1,terminal,inf = self._env.step(action)
		Return features(s_tp1),r_tp1,terminal,inf
