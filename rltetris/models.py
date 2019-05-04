import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from gym_tetris.tetris import TetrisEngine, TetrisState, Shape, ShapeKind

import logging
logger = logging.getLogger(__name__)

# Define Policy Interface
class BasePolicy(object):
    """
    Base policy class
    """
    def __init__(self,env, num_features, num_actions):
        self.weights = [[0]*num_features]*num_actions  # weights are a matrix of featuers * num_actions
        # SHOULD PASS IN ENVIRONMENT!!
        self._env = env        # policy, want possible actions, so pass in environment


    def save_model(self,path):
        raise NotImplementedError()

    def load_model(self,path):
        # loading and saving from disk
        raise NotImplementedError()

    # def action(self,*args,**kwargs):
    #     # choose argmax of policy
    #     # want a vector of actions that are/aren't allowed and then only
    #     # do argmax on ones that are allowed.
    #     # multiply state * every array in weights and take max of those
    #     np.argmax()
    #     return action

    def eval(weights, state):
        """ calculates score for given weights and state"""
        return np.dot(weights, state)


    def get_params(self):
        # get weights
        return self.weights

    def set_params(self,*args,**kwargs):
        # setting weights
        self.weights = weights



# Define Value function interface
class BaseValue(object):
    """
    Base value class
    """
    def __init__(self,num_features,**kwargs):
        self.model = LinearRegression()
        self.weights = [0]*num_features
        pass

    def save_model(self,path):
        raise NotImplementedError()

    def load_model(self,path):
        raise NotImplementedError()

    def eval(self,state):
        return self.model.predict(state)   #STATE IS REALLY JUST FEATURES OF STATE

    def get_params(self):
        return self.weights

    def set_params(self,*args,**kwargs):
        self.weights = weights

# Implement DUPolicy for tetris initial state rollouts
class DUPolicy(BasePolicy):

    def __init__(self,env, num_features, num_actions):
         # [land_height, eroded_cells, row_transitions, Col_transitions, holes, wells, hole depth, rows w/holes]
        self.weights = [-12.63, 6.60, -9.22,-19.77,-13.08,-10.49,-1.61, -24.04,0]
        # need weights to correspond to the number of features (so 9 for easy board)
        self._env = env

    def action(self,state):
        # choose argmax of policy
        # want a vector of actions that are/aren't allowed and then only
        # do argmax on ones that are allowed.
        # multiply state * every array in weights and take max of those
        A = self._env.get_action_set()

        scores = np.empty(shape = len(A))
        for i in range(len(A)):
            if A[i] == 0:
                scores[i] = -sys.maxsize -1
            else:
                new_s = TetrisState(np.ndarray.copy(state.board),state.x,state.y,state.direction,state.currentShape,state.nextShape, state.width)
                self._env.set_state(new_s)  #set the copied board as the state

                self._env.step(i)
                scores[i] = np.dot(self.weights, self._env.get_features())

        best_actions = np.argwhere(scores == np.amax(scores)).flatten()
        action = np.random.choice(best_actions)    # need to get random one in best_actions
        return action


# Extend classes above to implement policies, and value function approx
class LinearPolicy(BasePolicy):
    # policy.act
    # weights for this policy are a 1D array ( flattened from 2D array of features * actions)

    def action(self,state):
        """State is list of features for the state. Returns index of action to use """
        vector_weights = np.reshape(self.weights, newshape = (40, len(state))) #number of actions is 40
        action_scores = [np.dot(w, state) for w in vector_weights]
        best_actions = np.argwhere(action_scores == np.amax(action_scores)).flatten()
        action = np.random.choice(best_actions)    # need to get random one in best_actions
        return action


class LinearVFA(BaseValue):


    def fit(input, out):
        """ fit the model using parameters
        input: the inputs to the regresssion (states/features)
        out: the outputs to the regression (values)
        """
        self.model.fit(input, out)





# Implement Random Policy for discrete control spaces
# could use this instead of DUPolicy for non-tetris games
class RandomPolicy(BasePolicy):
    pass

# You could also just find some Blackjack policy (trained model) online
# to generate the initial states
class ProBlackjackPolicy(BasePolicy):
    pass
