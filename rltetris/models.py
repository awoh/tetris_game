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
        self.weights = [0]*(num_actions*num_features)  # weights are a matrix of featuers * num_actions, but flattened to 1D
        self._env = env        # policy, want possible actions, so pass in environment


    def save_model(self,path):
        np.save(path, self.weights)

    def load_model(self,path):
        # loading and saving from disk
        data = np.load(path)

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

    def set_params(self,weights):
        # setting weights
        self.weights = weights



# Define Value function interface
class BaseValue(object):
    """
    Base value class
    DON'T NEED ACTUAL MODEL, CAN JUST LOAD AND SET WEIGHTS AND THEN DO DOT PRODUCT
    """
    def __init__(self,num_features,**kwargs):
        self.weights = [0]*num_features
        self.model = LinearRegression()

    def save_model(self,path):
        np.save(path, self.weights)

    def load_model(self,path):
        data = np.load(path)

    def eval(self,state):
        """Use weights to just get value (weights dot features)"""
        return np.dot(self.weights, state)
        # return self.model.predict(state)   #STATE IS REALLY JUST FEATURES OF STATE

    def get_params(self):
        return self.weights

    def set_params(self,weights):
        """Don't know how to manually load coefficients to model!!! """
        self.weights = weights


# Implement DUPolicy for tetris initial state rollouts
class DUPolicy(BasePolicy):

    def __init__(self,env, num_features, num_actions):
         # [land_height, eroded_cells, row_transitions, Col_transitions, holes, wells, hole depth, rows w/holes]
        # need weights to correspond to the number of features (so 9 for easy board)
        du_weights = [-12.63, 6.60, -9.22,-19.77,-13.08,-10.49,-1.61, -24.04]
        weight_blanks = [0]*(num_features - len(du_weights))  #includes for both missing blanks and blocks
        self.weights = du_weights + weight_blanks
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
                new_s = TetrisState(np.ndarray.copy(state.board),state.x,state.y,
                                    state.direction,state.currentShape,state.nextShape, state.width,
                                    state.height_of_last_piece,state.num_last_lines_cleared,
                                    state.num_last_piece_cleared,state.last_piece_drop_coords)
                self._env.set_state(new_s)  #set the copied board as the state
                self._env.step(i)
                scores[i] = np.dot(self.weights, self._env.get_features())
        self._env.set_state(state)  #reset back to original state
        best_actions = np.argwhere(scores == np.amax(scores)).flatten()
        action = np.random.choice(best_actions)    # need to get random one in best_actions
        return action


# Extend classes above to implement policies, and value function approx
class LinearPolicy(BasePolicy):
    # policy.act
    # weights for this policy are a 1D array ( flattened from 2D array of features * actions)
    def __init__(self,env, num_features, num_actions):
        self.weights = [0]*(num_actions*num_features)  # weights are a matrix of featuers * num_actions, but flattened to 1D
        self._env = env        # policy, want possible actions, so pass in environment

        self.weights[7 + (self._env._engine.width*2)] = -10    # making max height bad to see if helps


    def action(self,state):
        """State is list of features for the state. Returns index of action to use """
        vector_weights = np.reshape(self.weights, newshape = (self._env._engine.width*4, len(state))) #number of actions is 40
        action_scores = [np.dot(w, state) for w in vector_weights]

        # want only actions that are valid, so make all others awful
        valid_actions = self._env.get_action_set()
        for i in range(len(valid_actions)):
            if valid_actions[i] == 0:
                action_scores[i] = -sys.maxsize -1

        # print(action_scores)

        best_actions = np.argwhere(action_scores == np.amax(action_scores)).flatten()
        action = np.random.choice(best_actions)    # need to get random one in best_actions
        return action


class LinearVFA(BaseValue):


    # def fit(self,input, out):
    #     """ fit the model using parameters
    #     input: the inputs to the regresssion (states/features)
    #     out: the outputs to the regression (values)
    #     """
    #
    #     self.model.fit(input, out)
    #     return self.model.coef_

    def set_params(self,states, vals):
        """Use linear model to get params """
        # new_val = linear_model.LinearRegression()
        self.model.fit(states, vals)
        self.weights = self.model.coef_






# Implement Random Policy for discrete control spaces
# could use this instead of DUPolicy for non-tetris games
class RandomPolicy(BasePolicy):
    def __init__(self,env, num_features, num_actions):
        self.weights = [0]*(num_features)  # weights are a matrix of featuers * num_actions, but flattened to 1D
        self._env = env        # policy, want possible actions, so pass in environment

    def action(self,state):
        """Choose random action from list of potential actions"""


        # want only actions that are valid, so make all others awful
        valid_actions = self._env.get_action_set()
        possible_actions = np.argwhere(valid_actions ==1).flatten()  #get indices of all valid actions
        action = np.random.choice(possible_actions)    # need to get random one in best_actions
        return action

# You could also just find some Blackjack policy (trained model) online
# to generate the initial states
class ProBlackjackPolicy(BasePolicy):
    pass
