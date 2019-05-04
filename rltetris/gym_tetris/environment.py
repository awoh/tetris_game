# Environments
# Put all custom environments here
import numpy as np
import gym
from gym.envs.registration import register
from gym import spaces
import logging
import random
from collections import namedtuple
logger = logging.getLogger(__name__)

# Local imports

from .tetris import TetrisEngine

def register_envs():
    register(
        id='Tetris-v0',
        entry_point='gym_tetris.environment:TetrisEnvironment',
        kwargs={"width": 10, "height": 22, "num_shapes": 8}
    )


class TetrisEnvironment(gym.Env):
    """
    Tetris game, takes as arguments the board size and allowed shapes

    Actions are a tuple (r,x) where r is 0..3 and x is in 0..width
    """
    def __init__(self,width,height,num_shapes):

        # Action Space, State Space
        obs1 = spaces.MultiDiscrete([num_shapes,num_shapes])
        obs2 = spaces.Box(low=0, high=num_shapes, shape=(height, width), dtype=np.intc)
        self.observation_space = spaces.Tuple((obs1,obs2))
        self.action_space = spaces.MultiDiscrete([4,width])
        self._engine = TetrisEngine()
        self._terminal = False

        # initial state config
        self.reset()

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self._terminal:
            raise RuntimeError('Game is already over')
        # update state
        a = self._engine.action_map[action]
        r_tp1 = self._engine.moveRotateDrop(a[0],a[1])

        self._terminal = self._engine.done
        return self.get_features, r_tp1,self._terminal,{}
        return self.state,r_tp1,self._terminal,{}

    def get_features(self):
        return self._engine.getFeatures()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self._time = 0
        return self._engine.reset()

    def set_state(self, state):
        self._engine.setState(state)
        # self._terminal = self._engine.done

    @property
    def state(self):
        return self._engine.state

    @property
    def time(self):
        return self._time

    def _render(self, mode='human', close=False):
        return

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_action_set(self):
        """
        Determines A based on the board state, A = {(rotation, column)}
        we want size A = 40 (try every possible rotation/column pair)

        Returns: list of size 40, 0 = action not possible, 1 = action possible at every position
        """
        A = np.empty(shape=40)  # r = rotation, c = column, minY= offset from top of board
        for r in range(4):
            for c in range(self._engine.width):
                # minX, maxX, minY, maxY = board.nextShape.getBoundingOffsets(0)
                minX, maxX, minY, maxY = self._engine.state.currentShape.getBoundingOffsets(r)
                if(self._engine.tryMoveCurrent(r, c, -minY)):
                    A[r*10 +c] = 1 # can make move
                else:
                    A[r*10+c] = 0 # action not possible
        return A

register_envs()

if __name__ =='__main__':
    pass
