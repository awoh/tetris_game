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
import pickle

class TetrisAI(object):

    # each game, AI does a random # of random moves in beginning

    # start with a policy (random moves)

    # start collecting experiments (api function):
        # train a classification learner (svm) where max action= label for state
        # state has 2 components (board description and type of piece)

    # evaluation function to evaluate all states:
        # f is defined as linear combo of set of features, phi
        # f(.) = phi * (.) * theta, theta = parameter vector (policy/controller)
    def nextMove(policy, board):
        """
        Return the next move for the board.
        Used for generating the initial states
        Initially use the weights provided in the DU controller from other paper as policy
        Maybe change this after it learns some...
        """
        pass

    def makeMove(next_move, board):
        """
        Code taken from timerEvent() in tetris_game.py
        Makes the move on the board
        """
        k = 0
        while board.currentDirection != next_move[0] and k < 4:
            board.rotateRight()
            k += 1
        k = 0
        while board.currentX != next_move[1] and k < 5:
            if board.currentX > next_move[1]:
                board.moveLeft()
            elif board.currentX < next_move[1]:
                board.moveRight()
            k += 1

        lines = board.dropDown()    # creates new piece after moving down
        return lines, board

    def getRandomState():
        """
        Creates initial random state for training
        """
        # let regular tetris_ai calculateScore get random initial moves for state in D
        num_moves = random.randint(5,15)    # number of moves to make when creating init state
        board = BoardData()
        board.createNewPiece()  # create initial piece
        for i in range(num_moves):
            next_move = nextMove(policy, board)
            lines, board = makeMove(next_move, board)

        return board


    def rollout(board, steps, curr_val, curr_policy, init_action):
        """
        Generate rollout for a given state (goes steps number of actions into future)
        Returns the total reward for the rollout
        """
        curr_state = s
        tot_reward = 0
        # need to go from 0 to m-1, m is an upper bound (since game may end earlier)
        for i in range(steps):
            # get state, action, reward tuple

            # WHEN DO YOU UPDATE ACTION?? (AFTER REWARD OR BEFORE?),
            # IS INITIAL STATE INCLUDED IN REWARD??
            curr_reward = calculateReward(curr_state)   # curr_reward is the immediate reward (score) --> what is this??
            tot_reward += exp(gamma,i) * curr_reward    # add gamma*reward to sum of rewards

            # make next move... if first iteration, then use provided action (handles for R(s,a) calculations)
            if i == 0:
                next_move = init_action
            else:
                next_move = getAction(curr_policy.predict((curr_state.getData, curr_state.currentShape)))
            curr_state = makeMove(next_move, curr_state)    # makes move, and then creates next piece

        # compute the unbiased estimate (v), prev_v based on m moves away from s
        tot_reward += exp(gamma, steps) * curr_val.predict((curr_state.getData, curr_state.currentShape))

        return tot_reward

    def getActionSet(board):
        """
        Determines A based on the board state, A = {(rotation, column)}
        max size of A = 34 (for L, J, and T)
        we want size A = 40 (try every possible rotation/column pair)
        """
        A = list()
        for r in range(4):
            for c in range(10):
                if(board.tryMoveCurrent(r, c, 0)):
                    A.append((r,c)) # add to set
                else:
                    A.append((-1,-1))
        return A

    def getAction():
        """
        return action for policy, chooses max from classifier output
        """
        pass

    def play(policy):
        board = BoardData()
        game_lines = 0
        while board.createNewPiece():
            next_move = getAction(policy.predict((board.getData, board.currentShape)))
            lines, board = makeMove(next_move, board)
            game_lines += lines

        return game_lines

    def mpi(A, N, M, m, init_policy, init_val):
        """
        starts with init policy and value and generates a sequence of value-new_policy
        pairs (v_k = evaluation step and policy_k+1 = greedy step)
        """
        done = False
        curr_policy_score = 0

        # i don't know what k is... maybe just do while not done (until new_policy = curr_policy)
        while !done:
        # for k  in range(idk):
            # at every iteration, build new value function and policy function

            state_set = np.array([])
            val_set = np.array([])
            q_set = np.array([])

            # sampling N states from distribution (each iteration, generating new random state)
            for i in range(N):

                # generate random initial state by making num_moves number of moves
                s = getRandomState()

                # generate rollout for state of size m (go up to m steps away)
                action = getAction(curr_policy.predict((s.getData, s.currentShape)))
                v_hat = rollout(s, m, curr_val, curr_policy, action)

                # add (s, v) to training set for regressor (x = state_set, y = val_set)
                state_set = np.append(state_set, [(s.getData, s.currentShape)])
                val_set = np.append(val_set, [v_hat])

                state_q = np.array([])

                # for every possible action from state s, make action and then follow policy for m steps
                A = getActionSet(s)
                for a in A:
                    # build M rollouts  (get rewards for all future states (1 -> m+1))
                    tot_Q = 0
                    for i in range(M):
                        # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
                        R = rollout(s, m+1, curr_val, curr_policy, a)
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

            # evaluate policy by averaging score over 20 games
            tot_lines = 0
            for i in range(20):
                num_lines = play(new_policy)
                tot_lines += num_lines

            new_policy_score = tot_lines/20

            # if curr_policy is approximately new_policy, save policy into file end
            policy_change = new_policy_score - curr_policy_score
            if(policy_change < error_threshold and policy_change >=0):
                # return curr_policy
                # save the model to disk, CAN USE PICKLE OR JOBLIB
                pickle.dump(new_policy, open('finalized_policy.sav', 'wb'))
                break

            # update current policy function and score
            curr_policy = new_policy
            curr_policy_score = new_policy_score

def main:

    init_policy    # policy_1 (arbitrary)
    init_val      # value_0 function (arbitrary)
    A = {}      #action set (32 possible actions)...what are these???
    M = 1
    N = 100
    m = 5
    mpi(A, N, M, m, init_policy, init_val)

    # load the model from disk
    policy = pickle.load(open('finalized_policy.sav', 'rb'))
