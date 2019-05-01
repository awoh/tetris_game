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


    # Objective function for CMA-ES
    # Need to bind the first arg using lambda function before optimizing
    # NOTE - You may want to add additional arguments
    # omega0 - initial param
    # batch - train data
    def _policy_loss_cbmpi(self, omega0,batch):
        # compute ugly thing from paper
        pass

    def update_policy(self,batch):
        """updating policy (uses CMA-ES)
        q_hats: [[Q_0,Q_1,...Q_a],...], list of every actions's Q value for every init state.(size: N*|A|)
        q_states:[[S_0,S_1,...S_a],...], where S_a is set of features for a state (size: N* (|A|*features))
        """

        # add final reward to it here (so just get rewards from get_vh)
        for i in range(len(q_hats)):
            # add reward to q_hats, too
            for j in range(len(q_hats[j])):
                estimated_q_val = self._critic.sample(q_states[(i,j)]) + q_hats[(i,j)]
                q_hats[(i,j)] = estimated_q_val

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

    def update_critic(self,batch):
        """updating value function """
        # add estimatesd reward here
        # v-hats is really the estimated reward
        for i in range(len(batch)):
            state = batch[i][0]
            roll_val = batch[i][1]
            estimated_v_val = self._critic.sample(state) + roll_val
            v_hats[i][1] = estimated_v_val
        states = batch[:,0]
        vals = batch[:,1]
        # new_val = linear_model.LinearRegression()
        self._critic.fit(states, val)
        new_params = self._critic.model.get_params()
        self._critic.set_params(new_params)
