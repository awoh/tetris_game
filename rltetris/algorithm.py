import numpy as np
import cma
from sklearn import linear_model, svm

import logging
logger = logging.getLogger(__name__)

# # Base Algorithm Class
# class Algorithm(object):
#
#     def __init__(self,policy,critic,args,**kwargs):
#         self._args = args
#         self._policy = policy
#         self._critic = critic
#
#     def update_policy(self,batch):
#         raise NotImplementedError()
#
#     def update_critic(self,batch):
#         raise NotImplementedError()





# CBMPI/DPI
# Fill This In
# algorithm is really just the approximate value fucntion update and greeedy policy update in paper (fig. 3)
# all that's passsed in is a bunch of data
# class CBMPI(Algorithm):
class CBMPI(object):
    """Probably won't have it inherit Algorithm class since we don't want to try
    other algorithms """

    def __init__(self,policy,critic,args,**kwargs):
        self._args = args
        self._policy = policy
        self._critic = critic







        # add final reward to it here (so just get rewards from get_vh)
        for i in range(len(v_hats)):
            # add reward to q_hats, too
            for j in range(len(q_hats[j])):
                estimated_q_val = critic.sample(q_states[(i,j)]) + q_hats[(i,j)]
                q_hats[(i,j)] = estimated_q_val

    # Objective function for CMA-ES
    # Need to bind the first arg using lambda function before optimizing
    # NOTE - You may want to add additional arguments
    # omega0 - initial param
    # batch - train data
    def _policy_loss_cbmpi(self, omega0,batch):
        # compute ugly thing from
        pass

    def update_policy(self,batch, q_hats):
        """updating policy (uses CMA-ES) """
        # This is where you need to call CMA-ES
        # https://pypi.org/project/cma/
        # you need to give it an objective function to evaluate
        # you can pass extra args to be forwarded to _policy_loss_cbmpi (see docs)
        # suggested to use fmin2 -- it is a simplified interface
        # need to fill in rest of call to fmin2
        inital_params = self._policy.get_params()
        policy_loss = lambda x : _policy_loss_cbmpi(x,batch)        # x is the weights of the policy, may need to bind policy too
        new_params, es = cma.fmin2(policy_loss,inital_params,...)   # need weights of features for every action (so new_params = feat*action)
        self._policy.set_params(new_params)

        raise NotImplementedError()

    def update_critic(self,batch, v_hats, v_states):
        """updating value function """
        # add estimatesd reward here
        # v-hats is really the estimated reward
        for i in range(len(v_hats)):
            estimated_v_val = critic.sample(v_states[i]) + v_hats[i]
            v_hats[i] = estimated_v_val

        new_val = linear_model.LinearRegression()
        new_val.fit(val_features, val_outputs)
        # print("coefs: " + str(curr_val.coef_))   # prints the weights in the model
        self._critic.set_params(new_val.get_params())      # update current value function
