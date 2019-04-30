import numpy as np
import cma
from sklearn import linear_model, svm

import logging
logger = logging.getLogger(__name__)

# Base Algorithm Class
class Algorithm(object):

    def __init__(self,policy,critic,args,**kwargs):
        self._args = args
        self._policy = policy
        self._critic = critic

    def update_policy(self,batch):
        raise NotImplementedError()

    def update_critic(self,batch):
        raise NotImplementedError()


# Objective function for CMA-ES
# Need to bind the first arg using lambda function before optimizing
# NOTE - You may want to add additional arguments
# omega0 - initial param
# batch - train data
def _policy_loss_cbmpi(omega0,batch):
    pass


# CBMPI/DPI
# Fill This In
# algorithm is really just the approximate value fucntion update and greeedy policy update in paper (fig. 3)
# all that's passsed in is a bunch of data
class CBMPI(Algorithm):
    """Probably won't have it inherit Algorithm class since we don't want to try
    other algorithms """

    def __init__(self,policy,critic,args,**kwargs):
        self._args = args
        self._policy = policy
        self._critic = critic

    def update_policy(self,batch):
        """updating policy (uses CMA-ES) """
        # This is where you need to call CMA-ES
        # https://pypi.org/project/cma/
        # you need to give it an objective function to evaluate
        # you can pass extra args to be forwarded to _policy_loss_cbmpi (see docs)
        # suggested to use fmin2 -- it is a simplified interface
        # need to fill in rest of call to fmin2
        inital_params = self._policy.get_params()
        policy_loss = lambda x : _policy_loss_cbmpi(x,batch)
        new_params, es = cma.fmin2(policy_loss,inital_params,...)
        self._policy.set_params(new_params)

        raise NotImplementedError()

    def update_critic(self,batch):
        """updating value function """
        # add estimatesd reward here

            v=0    # if haven't developed a model for value function yet...idk
            if(critic != None):
                s_features = S_i.getFeatures()    # get state, so can predict using model
                v = critic.sample(s_features)    #HOW TO GET VALUE FROM CRITIC??????
                tot_reward += (gamma**m) * v # compute the unbiased estimate (v), prev_v based on m moves away from s

        new_val = linear_model.LinearRegression()
        new_val.fit(val_features, val_outputs)
        # print("coefs: " + str(curr_val.coef_))   # prints the weights in the model
        self._critic.set_params(new_val.get_params()) = new_val      # update current value function
