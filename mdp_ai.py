"""
Improved AI for Tetris - uses MDP

Authors: Sarah Eisenach and Niall Williams
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tetris_model import BOARD_DATA, Shape, BoardData
import math
from datetime import datetime
import numpy as np
from sklearn import linear_model, svm

class TetrisAI(object):


    # each game, AI does a random # of random moves in beginning

    # start with a policy (random moves)

    # start collecting experiments (api function):
        # train a classification learner (svm) where max action= label for state
        # state has 2 components (board description and type of piece)

    # evaluation function to evaluate all states:
        # f is defined as linear combo of set of features, phi
        # f(.) = phi * (.) * theta, theta = parameter vector (policy/controller)
    def nextMove():
        pass

    def getRandomState(num_moves):
        # let regular tetris_ai calculateScore get random initial moves for state in D
        # (use the weights provided in the DU controller from other paper)
        board = BoardData()
        for i in range(num_moves):
            board.createNewPiece()
            move = nextMove()
            board.tryMoveCurrent(move)

        return board

    def rollout(s, steps, curr_val):
        # generate rollout for state of size m (go up to m steps away) (go fromm 0 to m-1)
        curr_state = s
        tot_reward = 0
        # need to go from 0 to m-1, m is an upper bound (since game may end earlier)
        for i in range(steps):
            # get state, action, reward tuple

            # WHEN DO YOU UPDATE ACTION?? (AFTER REWARD OR BEFORE?),
            # IS INITIAL STATE INCLUDED IN REWARD??
            curr_reward = calculateReward(curr_state)   # curr_reward is the immediate reward (score)
            tot_reward += exp(gamma,i) * curr_reward    # add gamma*reward to sum of rewards

            # make next move
            a = curr_policy.predict(curr_state)
            curr_state = board.makeMove(a)

        # compute the unbiased estimate (v), prev_v based on m moves away from s
        tot_reward += exp(gamma, steps) * curr_val.predict(curr_state)

        return tot_reward


    """starts with init policy and value and generates a sequence of value-new_policy
    pairs (v_k = evaluation step and policy_k+1 = greedy step) """
    def mpi():
        curr_policy     # policy_1 (arbitrary)
        curr_val        # value_0 function (arbitrary)
        A = {}      #action set (32 possible actions)
        M = 1
        done = False

        # i don't know what k is... maybe just do while not done (until new_policy = curr_policy)
        while !done:
        # for k  in range(idk):
            # at every iteration, build new value function and policy function

            state_set = np.array([])
            val_set = np.array([])
            q_set = np.array([])

            num_moves = random.randint(5,15)    # number of moves to make when creating init state

            # sampling N states from distribution (each iteration, generating new random state)
            for i in range(N):

                # generate random initial state by making num_moves number of moves
                s = getRandomState(num_moves)

                # generate rollout for state of size m (go up to m steps away) (go fromm 0 to m-1)
                v_hat = rollout(s, m, curr_val)

                # add (s, v) to training set for regressor (x = state_set, y = val_set)
                state_set = np.append(state_set, [s])
                val_set = np.append(val_set, [v_hat])

                state_q = np.array([])
                # for every possible action from state s, follow policy
                for a in A:
                    # build M rollouts  (get rewards for all future states (1 -> m+1))
                    tot_Q = 0
                    for i in range(M):
                        # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
                        R = rollout(s, m+1, curr_val)
                        tot_Q += R

                    # calculate Q_hat
                    Q_hat = tot_Q / M
                    state_q = np.append(state_q, [Q_hat])   # add Q_hat value to list of Q's for given state

                q_set = np.append(q_set, [state_q])     # output values for classifier (list of Q values for each state)
                    # output values: max of q_set - q_hat for given action

            # learn v_k w/ regressor
            new_val = svm.LinearSVR(max_iter = 1000)
            new_val.fit(state_set, val_set)
            curr_val = new_val      # update current value function

            # learn new policy w/ classifier
            # input: state, output: set of q_hats (all values for actions of a given set)
            # take min of the output set and get action
            new_policy = svm.LinearSVC(max_iter = 1000)
            new_policy.fit(state_set, q_set)
            curr_policy = new_policy      # update current policy function


            # if curr_policy is approximately new_policy, end
            if(curr_policy == new_policy):
                return curr_policy
                # break
