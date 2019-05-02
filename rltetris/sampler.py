# Sampler Module
import gym
import numpy as np
import logging
import random
import copy
from tetris import TetrisEngine, TetrisState, Shape, ShapeKind
from environment import TetrisEnvironment
logger = logging.getLogger(__name__)


# Do rollout using policy to get random states
def sample_random_states(env,policy,N):
    # Step 0 - Allocate return arrays
    states = np.empty(shape = N)

    # Step 2 - run the environment and collect final states
    for i in range(N):
        env.reset() # reset environment

        # make x number of moves following DU policy
        for j in range(x):
            action = policy.action()
            new_state = env.step(action)    # get state of env

        states[i] = new_state

    return states


# pass start states to function to get samples to update value function
def get_vh(env, D_k, plc,m,gamma, num_features):
    v_batch = [ [[0]*num_features, 0] ]*len(D_k)  # every state has [S, v], with S being []
    # v_hats = [0]*len(D_k)
    # S_ms = [[0]*num_features]*len(D_k)
    # go thru every state in D_k
    for i in range(len(D_k)):
        S_m, reward = rollout_from_state(env, D_k[i], plc,critic, m, gamma)   # get rollout for state
        v_batch[i] = [S_m, reward]
    v_batch = np.array(v_batch)
    return v_batch

def get_qh(env, D_k, plc,m,gamma, num_features, num_actions):
    q_batch = [ [ [[0]*num_features, 0] ]*num_actions ]*len(D_k)
    # q_hats = np.empty(shape = [len(D_k), num_actions])
    # s_ms = np.empty(shape = [len(D_k), num_actions, num_features])

    # go thru every state in D_k
    for i in range(len(D_k)):
        curr_state = D_k[i]
        env.set_state(curr_state)
        A = get_action_set(curr_state)

        # for every possible action from state s, make action and then follow policy for m steps
        for j in range(len(A)):
            tot_Q = 0
            a = A[j]

            
            # if action not possible...
            if a == 0:
                # q_batch[i][j] = [reward, S_m]
                # q_hats[(i,j)] = reward   # assign for given state, action pair, q_hat value
                # s_ms[(i,j)] = S_m
                pass

            # build M rollouts  (get rewards for all future states (1 -> m+1))
            # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
            S_m, reward = rollout_from_state(env, curr_state, plc, critic m+1, gamma, a)   # get rollout for state
            q_batch[i][j] = [S_m, reward]

    q_batch = np.array(q_batch)
    return q_batch




# Use start_action to optionally pass the start action. If it is None, policy should be used
def rollout_from_state(env,start,plc,m,gamma,start_action=None):
    """
    Generate rollout for a given state (goes steps number of actions into future)
    Returns the total reward for the rollout
    """
    # S_m, R_disc = None, None

    # Step 1 - set start state
    # CURRENTLY CREATING NEW TETRIS STATE
    S_i = copy_state(start)
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
            # use wrapper environemnt, so S_i is really set of reatures
            next_move = plc.action(S_i)

        env_state, curr_reward, _, _ = env.step(action)

        # if reached end of game before doing m steps
        if env._terminal:
            return S_i, tot_reward

        # curr_reward = S_i.moveRotateDrop(next_move[0], next_move[1])    # make move, cur_reward = lines cleared by action
        tot_reward += (gamma**i) * curr_reward    # add gamma*reward to sum of rewards

    # use policy, and make final move
    next_move = plc.action(S_i)
    env.step(next_move)

    env.set_state(start)        # reset environment to start state
    return S_i, tot_reward


def copy_state(s):
    copy_board = ndarray.copy(s.board)
    new_s = TetrisState(copy_board,s.x,s.y,s.direction,s.currentShape,s.nextShape)
    return new_s

    # np.copyto(copy.board, board.board)
    # copy.x = board.x
    # copy.y = board.y
    # copy.direction = board.direction
    # copy.currentShape = board.currentShape
    # copy.nextShape = board.nextShape
    # copy.width = board.width
    # copy.height = board.height

    # np.copyto(copy.shapeStat, board.shapeStat)
    # copy.done = board.done
