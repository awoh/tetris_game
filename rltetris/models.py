import numpy as np

import logging
logger = logging.getLogger(__name__)

# Define Policy Interface
class BasePolicy(object):
    """
    Base policy class
    """
    def __init__(self,**kwargs):
        # self.model = LinearRegression()
        self.weights = [0]  # weights are a matrix of featuers * num_actions
        # SHOULD PASS IN ENVIRONMENT!!
        pass

    def save_model(self,path):
        raise NotImplementedError()

    def load_model(self,path):
        # loading and saving from disk
        raise NotImplementedError()

    def action(self,*args,**kwargs):
        # choose argmax of policy
        # want a vector of actions that are/aren't allowed and then only
        # do argmax on ones that are alowed.
        # policy
        raise NotImplementedError()

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
    pass

# Implement Random Policy for discrete control spaces
# could use this instead of DUPolicy for non-tetris games
class RandomPolicy(BasePolicy):
    pass

# You could also just find some Blackjack policy (trained model) online
# to generate the initial states
class ProBlackjackPolicy(BasePolicy):
    pass

# Extend classes above to implement policies, and value function approx
class LinearPolicy(BasePolicy):
    # policy.act
    pass


class LinearVFA(BaseValue):
    pass
