# Sampler Module
import gym
import numpy as np
import logging
import random
import copy
from gym_tetris import TetrisEnvironment
from gym_tetris.tetris import TetrisEngine, TetrisState, Shape, ShapeKind
logger = logging.getLogger(__name__)



def sample_random_states(env,policy,N):
    """Does rollout using policy to get N random states"""

    # Step 0 - Allocate return arrays
    states = np.empty(shape = N, dtype=TetrisState)
    x = random.randint(10,20)    # number of moves to make when creating init state

    # Step 2 - run the environment and collect final states
    for i in range(N):
        env.reset() # reset environment

        # make x number of moves following DU policy
        for j in range(x):
            if env._terminal:
                break
            action = policy.action(env.state)
            # print("ACTION: " + str(action))
            env.step(action)    # get state of env

        # print(env.state.board)
        states[i] = env.state

    return states


# pass start states to function to get samples to update value function
def get_vh(env, D_k, plc,m,gamma, num_features):
    v_batch = [ [[0]*num_features, 0] ]*len(D_k)  # every state has [S, v], with S being []
    # v_hats = [0]*len(D_k)
    # S_ms = [[0]*num_features]*len(D_k)
    # go thru every state in D_k
    for i in range(len(D_k)):
        S_m, reward = rollout_from_state(env, D_k[i], plc, m, gamma)   # get rollout for state
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

        A = env.get_action_set()

        # for every possible action from state s, make action and then follow policy for m steps
        for j in range(len(A)):
            tot_Q = 0

            # if action not possible...
            if A[j] == 0:
                # q_batch[i][j] = [reward, S_m]
                # q_hats[(i,j)] = reward   # assign for given state, action pair, q_hat value
                # s_ms[(i,j)] = S_m
                continue
            # build M rollouts  (get rewards for all future states (1 -> m+1))
            # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
            S_m, reward = rollout_from_state(env, curr_state, plc, m+1, gamma, j)   # get rollout for state
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
    env.reset()
    S_i = copy_state(start)
    env.set_state(S_i)
    env_state = env.state()

    tot_reward = 0
    curr_reward = 0 # PROBABLY SHOULD CHANGE???????????

    # Step 2 - Use for loop to run,  need to go from 0 to m-1, m is an upper bound (since game may end earlier)
    for i in range(m-1):

        # print(env._env._engine.state.board)
        # Use start_action to optionally pass the start action. If it is None, policy should be used
        if start_action != None:
            next_move = start_action
        else:
            # use policy
            # use wrapper environemnt, so S_i is really set of reatures
            next_move = plc.action(env_state)
            # print("NEXT MOVE: " + str(next_move))

        env_state, curr_reward, _, _ = env.step(next_move)

        # if reached end of game before doing m steps
        if env._terminal:
            return env_state, tot_reward

        # curr_reward = S_i.moveRotateDrop(next_move[0], next_move[1])    # make move, cur_reward = lines cleared by action
        # print("curr reward: " + str(curr_reward))
        tot_reward += (gamma**i) * curr_reward    # add gamma*reward to sum of rewards

    # use policy, and make final move
    next_move = plc.action(env_state)
    env_state,_,_,_ = env.step(next_move)

    env.set_state(start)        # reset environment to start state
    return env_state, tot_reward


def copy_state(s):
    copy_board = np.ndarray.copy(s.board)
    new_s = TetrisState(copy_board,s.x,s.y,s.direction,s.currentShape,s.nextShape,
                        s.width,s.height_of_last_piece,s.num_last_lines_cleared,
                        s.num_last_piece_cleared,s.last_piece_drop_coords)
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
