import numpy as np
import cma
from sklearn import linear_model, svm

import logging
logger = logging.getLogger(__name__)

# CBMPI/DPI
# Fill This In
# algorithm is really just the approximate value fucntion update and greeedy policy update in paper (fig. 3)
# all that's passsed in is a bunch of data
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
    def _policy_loss_cbmpi(self, omega0,batch,policy):
        """Loss function for CBMI algorithms. Returns the empirical error for a dataset
            Used for evaluating policies in CMA-ES
        """
        N = len(batch)
        policy.set_params(omega0) #replace policy with input policy for loss
        loss =0
        print("BATCH")
        print(batch)
        for i in range(N):
            state = batch[i][0]
            print("STATE: ")
            print(state)
            q_hats = np.array(batch[i][1])  #list of all q_hats for given state

            max_q = np.amax(q_hats) # get max Q(s_i, a)

            # will give integer corresponding to action --> tells you index of q_hat from array q_hats
            # print(batch[i][0])
            policy_action = policy.action(state)
            # print(policy_action)
            policy_q = q_hats[policy_action]
            q_diff = max_q - policy_q\

            loss += q_diff

        loss = loss/N
        print("LOSS: " + str(loss))
        return loss

    def update_policy(self,init_states, q_batch):
        """updating policy (uses CMA-ES)
        q_hats: [[Q_0,Q_1,...Q_a],...], list of every actions's Q value for every init state.(size: N*|A|)
        q_states:[[S_0,S_1,...S_a],...], where S_a is set of features for a state (size: N* (|A|*features))

        fmin2(objective_function, x0, sigma0) minimizes objective_function starting at x0 and with standard deviation sigma0 (step-size)
         --> returns: x_best:numpy.ndarray, es:cma.CMAEvolutionStrategy)
          https://pypi.org/project/cma/
        """

        # batch for policy function is comprised of states and the q values
        # add final reward to it here (so just get rewards from get_vh)
        for i in range(len(q_batch)):
            state_q_len = len(q_batch[i])
            for j in range(state_q_len):
                estimated_q_val = self._critic.eval(q_batch[i][j][0]) + q_batch[i][j][1]
                q_batch[i][j][1] = estimated_q_val
        len_state = len(init_states[0])

        batch = [ [[0]*(len_state), [0]*(state_q_len)] ] *len(q_batch)   #each inner array is: [S_i, [Q...]]
        for i in range(len(q_batch)):
            batch[i][0] =  init_states[i]
            batch[i][1] =  q_batch[i][:,1]   # all q values for that state

        # This is where you need to call CMA-ES,  you need to give it an objective function to evaluate
        # you can pass extra args to be forwarded to _policy_loss_cbmpi (see docs)
        policy_loss = lambda x : self._policy_loss_cbmpi(x,batch,self._policy)        # x is the weights of the policy, may need to bind policy too
        initial_params = self._policy.get_params()

        sigma0, pop_size = 0.5, len_state *15     # parameters set by paper number of features * 15
        opts={'popsize': pop_size}

        # new_params, es = cma.fmin2(policy_loss,self._policy.get_params(),sigma0, options={'popsize': pop_size})   # need weights of features for every action (so new_params = feat*action)
        new_params = cma.CMAEvolutionStrategy(self._policy.get_params(), sigma0,opts).optimize(policy_loss).result[0]

        # NEW PARAMS IS OF LENGTH FEATURES * NUM_ACTIONS
        self._policy.set_params(new_params)


    def update_critic(self,init_states,v_batch):
        """updating value function """
        # add estimatesd reward here
        for i in range(len(v_batch)):
            state = v_batch[i][0]
            roll_val = v_batch[i][1]
            estimated_v_val = self._critic.eval(state) + roll_val
            v_batch[i][1] = estimated_v_val

        vals = v_batch[:,1]

        self._critic.set_params(init_states, vals)
