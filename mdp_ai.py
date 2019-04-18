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
import sys
import pickle
import random



# each game, AI does a random # of random moves in beginning

# start with a policy (random moves)

# start collecting experiments (api function):
    # train a classification learner (svm) where max action= label for state
    # state has 2 components (board description and type of piece)

# evaluation function to evaluate all states:
    # f is defined as linear combo of set of features, phi
    # f(.) = phi * (.) * theta, theta = parameter vector (policy/controller)
def nextInitialMove(weights, board):
    """
    Return the next move for the board.
    Used for generating the initial states
    Initially use the weights provided in the DU controller from other paper as policy
    Maybe change this after it learns some...
    """
    action_set = getActionSet(board)
    scores = np.array([])
    for a in action_set:
        # do action
        score =0
        if a == (-1,-1, 0):
            score = -sys.maxsize -1
        else:
            lines, new_board = makeMove(a, board)
            score = calculateReward(weights, board.getFeatures())   # evaluate and add to list
        scores = np.append(scores, [score])

    best = np.argwhere(scores == np.amax(scores))
    best = best.flatten().tolist()
    max = np.random.choice(best)
    return action_set[max]

def calculateReward(weights, features):
    tot_score = 0
    for i in range(len(weights)):
        tot_score += weights[i]*features[i]
    return tot_score


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
        policy = [1]*7
        next_move = nextInitialMove(policy, board)
        lines, board = makeMove(next_move, board)
        print(i)

    return board

def getActionSet(board):
    """
    Determines A based on the board state, A = {(rotation, column)}
    max size of A = 34 (for L, J, and T)
    we want size A = 40 (try every possible rotation/column pair)
    """
    A = list()  # r = rotation, c = column
    for r in range(4):
        for c in range(10):
            if(board.tryMoveCurrent(r, c, 0)):
                A.append((r,c,0)) # add to set
            else:
                A.append((-1,-1,0))
    return A

def getAction(board, policy, action_set):
    """
    return action for policy, chooses max from classifier output
    """
    # if policy doesn't exist yet, choose action randomly, else get from policy model
    if policy == None:
        possible_actions = [i for i in action_set if i[0] > -1]
        rand_i = random.randint(0, len(possible_actions)-1)
        action = possible_actions[rand_i]
        # print(action)
    else:
        piece = [0]*7   # one hot encode piece
        piece[board.currentShape.shape -1] = 1
        tot_features = np.append(board.getFeatures(), [piece])
        action_scores = policy.predict([tot_features])
        best_scores = np.argwhere(action_scores == np.amax(action_scores)).flatten().tolist()
        max_score = np.random.choice(best_scores)
        action =  action_set[max_score]
    return action

def rollout(curr_state, steps, curr_val, curr_policy, init_action, action_set, gamma):
    """
    Generate rollout for a given state (goes steps number of actions into future)
    Returns the total reward for the rollout
    """
    tot_reward = 0
    curr_reward = 0 # PROBABLY SHOULD CHANGE???????????

    # need to go from 0 to m-1, m is an upper bound (since game may end earlier)
    for i in range(steps):
        # get action... if first iteration, then use provided action (handles for R(s,a) calculations)
        if i == 0:
            next_move = init_action
        else:
            next_move = getAction(curr_state, curr_policy, action_set)

        curr_reward, curr_state = makeMove(next_move, curr_state)    # make move, cur_reward = lines cleared by action

        # curr_reward = calculateReward(curr_state), curr_reward is the immediate reward (score, ie # lines cleared)
        tot_reward += (gamma**i) * curr_reward    # add gamma*reward to sum of rewards

    # if haven't developed a model for value function yet...idk
    if(curr_val == None):
        v = 0
    else:
        s_features = curr_state.getFeatures()    # get state, so can predict using model
        v = curr_val.predict(s_features)

    tot_reward += (gamma**steps) * v # compute the unbiased estimate (v), prev_v based on m moves away from s
    return tot_reward

def play(policy):
    board = BoardData()
    game_lines = 0
    while board.createNewPiece():
        A = getActionSet(board)
        next_move = getAction(board, policy, A)
        lines, board = makeMove(next_move, board)
        game_lines += lines

    return game_lines

def mpi(N, M, m, error_threshold, gamma, num_evaluations):
    """
    starts with init policy and value and generates a sequence of value-new_policy
    pairs (v_k = evaluation step and policy_k+1 = greedy step)
    """
    done = False
    curr_policy_score = 0
    # i don't know what k is... maybe just do while not done (until new_policy = curr_policy)
    while not done:
    # for k  in range(idk):
        # at every iteration, build new value function and policy function
        s = getRandomState()
        # print(s.getFeatures())
        num_features = len(s.features)
        val_features = np.empty((0,num_features))
        val_outputs = np.empty((0,1))

        tot_features = num_features + 7
        policy_features = np.empty((0, tot_features))
        policy_outputs = np.empty((0,40)) # output values: max of q_set - q_hat for given action

        curr_val = None
        curr_policy = None  # weights are all the same, so would pick random action

        # sampling N states from distribution (each iteration, generating new random state)
        for i in range(N):

            # generate random initial state by making num_moves number of moves
            s = getRandomState()
            s_features = s.getFeatures()
            A = getActionSet(s)

            # generate rollout for state of size m (go up to m steps away)
            action = getAction(s, curr_policy, A)
            v_hat = rollout(s, m, curr_val, curr_policy, action, A, gamma)

            # add (s, v) to training set for regressor (x = state_set, y = val_set)
            # s is represented by the features determined in board
            val_features = np.append(val_features, np.array([s_features]), axis=0)
            val_outputs = np.append(val_outputs, np.array([[v_hat]]), axis=0)

            state_q = np.array([])

            # for every possible action from state s, make action and then follow policy for m steps
            for a in A:
                tot_Q = 0

                # if action not possible...
                if a == (-1,-1,0):
                    pass

                # build M rollouts  (get rewards for all future states (1 -> m+1))
                for i in range(M):
                    # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
                    R = rollout(s, m+1, curr_val, curr_policy, a, A, gamma)
                    tot_Q += R

                Q_hat = tot_Q / M   # calculate Q_hat
                state_q = np.append(state_q, [Q_hat])   # add Q_hat value to list of Q's for given state

            # piece is one-hot encoded
            piece = [0]*7
            piece[s.currentShape.shape -1] = 1
            tot_features = np.append(s_features, [piece])
            # print("tot_features: ")
            # print(tot_features)

            policy_features = np.append(policy_features, np.array([tot_features]), axis=0)
            policy_outputs = np.append(policy_outputs, np.array([state_q]), axis=0)     # output values for classifier (list of Q values for each state)
            # print("policy outputs: ")
            # print(policy_outputs)
        # learn v_k w/ regressor
        new_val = linear_model.LinearRegression()
        # print("value func data: ")
        # print(val_features)
        # print(val_outputs)
        new_val.fit(val_features, val_outputs)
        curr_val = new_val      # update current value function
        # print(curr_val.coef_)   # prints the weights in the model

        # learn new policy w/ classifier
        # input: state, output: set of q_hats (all values for actions of a given set)
        new_policy = linear_model.LinearRegression()
        new_policy.fit(policy_features, policy_outputs)

        # evaluate policy by averaging score over 20 games
        tot_lines = 0
        for i in range(num_evaluations):
            num_lines = play(new_policy)
            tot_lines += num_lines

        new_policy_score = tot_lines/num_evaluations
        print("policy score: " + str(new_policy_score))

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

def main():

    # A = {}      #action set (32 possible actions)...what are these???
    M = 1
    N = 5
    m = 5
    gamma = 1
    error_threshold = 0.01
    num_evaluations = 20
    mpi(N, M, m, error_threshold, gamma, num_evaluations)

    # load the model from disk
    policy = pickle.load(open('finalized_policy.sav', 'rb'))

if __name__ == '__main__':
    main()
