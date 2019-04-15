"""
Improved AI for Tetris - uses MDP

Authors: Sarah Eisenach and Niall Williams
"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tetris_model import BOARD_DATA, Shape
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


    """starts with init policy and value and generates a sequence of value-new_policy
    pairs (v_k = evaluation step and policy_k+1 = greedy step) """
    def mpi():
        curr_policy     # policy_1 (arbitrary)
        curr_val        # value_0 function (arbitrary)
        done = False

        # i don't know what k is... maybe just do while not done (until new_policy = curr_policy)
        # while !done:
        for k  in range(i):
            # at every iteration, build new value function and policy function

            # build a rollout set D by sampling N states from distribution
            # let regular tetris_ai calculateScore get random initial moves for state in D
            # (use the weights provided in the DU controller from other paper)
            for i in range(N):
                # generate random initial state
                s = state
                D.append(s)

            state_set = {}
            val_set = {}

            # for each state in D
            for s in D:
                # generate rollout for state of size m (go up to m steps away) (go fromm 0 to m-1)
                curr_state = s
                for i in range(m):
                    # get state, action, reward tuple

                    # add gamma*reward to sum of rewards
                    curr_reward = calculateReward(curr_state)
                    tot_reward += exp(gamma,i) * curr_reward
                    curr_state = board.makeMove(a)

                # compute the unbiased estimate (v), prev_v based on m moves away from s
                curr_state = board.makeMove(a)
                v_hat = tot_reward + exp(gamma, m)* curr_val.predict(curr_state)

                # add (s, v) to training set
                state_set += s
                val_set += v_hat

                # for every possible action from state s, follow policy
                for a in A:
                    # build M rollouts  (get rewards for all future states (1 -> m+1))
                    for i in range(M):
                        # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
                        curr_state = s
                        # from 0 to m
                        for j in range(m+1):
                            a = curr_policy.predict(curr_state)  # get next action
                            r = calculateReward()      # calc reward based on this action
                            curr_state = board.makeMove(a)   # updating the state (k steps away from initial state)
                            tot_R += exp(gamma, j) * r      # SAME AS v_hat (i think)


                        # calc R (s, a), add to tot_Q
                        a = curr_policy.predict()
                        curr_state = board.makeMove(a)
                        R = tot_R + exp(gamma, m+1) * curr_val.predict(curr_state)
                        tot_Q += R

                    # calculate Q_hat
                    Q_hat = tot_Q / M

            # learn v_k w/ regressor
            new_val = svm.LinearSVM(max_iter = 1000)
            new_val.fit(state_set, val_set)

            # learn new policy w/ classifier
            new_policy = svm.LinearSVC(max_iter = 1000)
            new_policy.fit(x_train, y_train)

            curr_val = new_val
            curr_policy = new_policy


            # if curr_policy is approximately new_policy, end
            if(curr_policy == new_policy):
                return
