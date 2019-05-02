import numpy as np

import logging
logger = logging.getLogger(__name__)

# Define Policy Interface
class BasePolicy(object):
    """
    Base policy class
    """
    def __init__(self,**kwargs):
        self.weights = [0]  # weights are a matrix of featuers * num_actions
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
    def __init__(self,**kwargs):
        self.model = LinearRegression()
        self.weights = [0]
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

    def __init__(self, **kwargs):
         # [land_height, eroded_cells, row_transitions, Col_transitions, holes, wells, hole depth, rows w/holes]
        self.weights = [-12.63, 6.60, -9.22,-19.77,-13.08,-10.49,-1.61, -24.04]


    def action(self,state):
        # choose argmax of policy
        # want a vector of actions that are/aren't allowed and then only
        # do argmax on ones that are allowed.
        # multiply state * every array in weights and take max of those
        next_states = [0]*len(state)
        A = self._env.get_action_set()
        for i in range(len(A)):
            if A[i] == 0:
                next_states[i] = [-sys.maxsize -1]*len(state)    # make horrible score so won't pick
            else:
                new_s = self._env.step(i)
                next_states[i] = new_s
        action = np.argmax([eval(self.weights, s) for s in next_states])   #get index for best state, i.e. best action
        return action


# Extend classes above to implement policies, and value function approx
class LinearPolicy(BasePolicy):
    # policy.act

    def action(self,state):
        """Returns index of action to use """
        action = np.argmax([eval(w, state) for w in self.weights])
        return action


class LinearVFA(BaseValue):


    def fit(in, out):
    """ fit the model using parameters
    in: the inputs to the regresssion (states/features)
    out: the outputs to the regression (values)
     """
        self.model.fit(in, out)





# Implement Random Policy for discrete control spaces
# could use this instead of DUPolicy for non-tetris games
class RandomPolicy(BasePolicy):
    pass

# You could also just find some Blackjack policy (trained model) online
# to generate the initial states
class ProBlackjackPolicy(BasePolicy):
    pass
