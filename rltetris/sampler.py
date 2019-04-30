# Sampler Module
import gym
import numpy as np
import logging
import random
import copy
from tetris import TetrisEngine, Shape, ShapeKind
from environment import TetrisEnvironment
logger = logging.getLogger(__name__)


# Do rollout using policy to get random states
def sample_random_states(env,policy,N):
    # Step 0 - Allocate return arrays
    states = None

    # Step 2 - run the environment and collect final states

    return states

# function to call all rollouts for all states
# def all_rollouts():
#     rollout_from_state(env, )
#     return batch

# pass start states to function to get samples to update value function
def get_vh(env, D_k, plc,critic,m,gamma):
    v_hats = np.empty(shape = len(D_k))
    # go thru every state in D_k
    for i in range(len(D_k)):
        S_m, reward = rollout_from_state(env, D_k[i], plc,critic, m, gamma)   # get rollout for state
        v_hats[i] = reward
    return v_hats

def get_qh(env, D_k, plc,critic,m,gamma):
    # same thing as get_vh
    num_actions = 40    # there are 40 potential actions
    q_hats = np.empty(shape = [len(D_k), num_actions]

    # go thru every state in D_k
    for i in range(len(D_k)):
        curr_state = D_k[i]
        A = get_action_set(curr_state)

       # for every possible action from state s, make action and then follow policy for m steps
       for j in range(len(A)):
           tot_Q = 0
           a = A[j]
           # if action not possible...
           if a == (-1,-1,0):
               pass

           # build M rollouts  (get rewards for all future states (1 -> m+1))
           # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
           for i in range(M):
               S_m, reward = rollout_from_state(env, D_k[i], plc, critic m+1, gamma, a)   # get rollout for state
               tot_Q += R

        Q_hat = tot_Q / M   # calculate Q_hat
        q_hats[(i,j)] = Q_hat   # assign for given state, action pair, q_hat value
    return q_hats




# Use start_action to optionally pass the start action. If it is None, policy should be used
def rollout_from_state(env,start,plc,m,gamma,start_action=None):
    """
    Generate rollout for a given state (goes steps number of actions into future)
    Returns the total reward for the rollout
    """
    # S_m, R_disc = None, None

    # Step 1 - set start state
    copy_state(start, S_i)
    env._engine.setState(S_i)
    copy_state(start, S_i)
    env.set_state(S_i)

    tot_reward = 0
    curr_reward = 0 # PROBABLY SHOULD CHANGE???????????

    # Step 2 - Use for loop to run,  need to go from 0 to m-1, m is an upper bound (since game may end earlier)
    for i in range(m-1):
        # Use start_action to optionally pass the start action. If it is None, policy should be used
        if start_action != None:
            next_move = start_action
        else:
            # use policy
            next_move = getAction(S_i, plc, get_action_set(S_i))

        # DO I DO ENV.SET_STATE?????
        env_state, curr_reward, _, _ = env.step(action)

        # if reached end of game before doing m steps
        if env._terminal:
            return S_i, tot_reward

        # curr_reward = S_i.moveRotateDrop(next_move[0], next_move[1])    # make move, cur_reward = lines cleared by action
        tot_reward += (gamma**i) * curr_reward    # add gamma*reward to sum of rewards

    # use policy, and make final move
    next_move = getAction(S_i, plc, get_action_set(S_i))
    env.step(next_move)
    # S_i.moveRotateDrop(next_move[0], next_move[1])




    env.set_state(start)        # reset environment to start state
    return S_i, tot_reward


def copy_state(board, copy):
    copy.width = board.width
    copy.height = board.height

    np.copyto(copy.state.board, board.state.board)
    copy.state.x = board.state.x
    copy.state.y = board.state.y
    copy.state.direction = board.state.direction
    copy.state.currentShape = board.state.currentShape
    copy.state.nextShape = board.state.nextShape
    copy.state.width = board.state.width
    copy.state.height = board.state.height

    np.copyto(copy.shapeStat, board.shapeStat)
    copy.done = board.done

    # return new_s


def get_action_set(board):
    """
    Determines A based on the board state, A = {(rotation, column)}
    max size of A = 34 (for L, J, and T)
    we want size A = 40 (try every possible rotation/column pair)
    """
    A = np.array(shape=40)  # r = rotation, c = column, minY= offset from top of board
    for r in range(4):
        for c in range(10):
            # minX, maxX, minY, maxY = board.nextShape.getBoundingOffsets(0)
            minX, maxX, minY, maxY = board.currentShape.getBoundingOffsets(r)
            if(board.tryMoveCurrent(r, c, -minY)):
                A[r*10 +c] = (r,c, -minY) # add to set
            else:
                A[r*10+c] = (-1,-1,0)
    return A
