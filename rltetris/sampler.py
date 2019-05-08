# Sampler Module
import gym
import numpy as np
import logging
import random
import copy
from gym_tetris import TetrisEnvironment
from gym_tetris.tetris import TetrisEngine, TetrisState, Shape, ShapeKind
logger = logging.getLogger(__name__)

def sample_random_states(env,policy, rnd_plc,N):
    """Does rollout using policy to get N random states
     FOR EASY GAME...JUST DO RANDOM POLICY TO MAKE INITIAL STATE SINCE DU CONTROLLER TO GOOD (i.e. just o piece)
     """

    # Step 0 - Allocate return arrays
    states = np.empty(shape = N, dtype=TetrisState)
    x = 10

    # Step 2 - run the environment and collect final states
    i=0
    while i < N-4:
        x = random.randint(10,20)    # number of moves to make when creating init state
        # print("X: "+str(x))
        env.reset() # reset environment

        # make x number of moves following DU policy
        for j in range(x):
            if env._terminal:
                continue

            # coin toss - 80% du, 20% random
            coin_toss = random.random() # between 0 and 1
            if coin_toss < 0.8:
                plc = policy
            else:
                plc = rnd_plc

            action = plc.action(env.state)
            # print("ACTION: " + str(action) +" PIECE: "+ str(env.state.currentShape.kind))
            env.step(action)    # get state of env
            # print("NEW STATE")
            # print(env.state.board)

        if not env._terminal:
            states[i] = env.state
            i+=1


    env.reset() # reset environment
    for i in range(2):
        env.step(1)
    states[N-4] = env.state

    env.reset() # reset environment
    for i in range(4):
        env.step(1)
    states[N-3] = env.state

    env.reset() # reset environment
    for i in range(3):
        env.step(1)
    env.step(3)
    states[N-2] = env.state

    env.reset() # reset environment
    for i in range(2):
        env.step(1)
        env.step(3)
    states[N-1] = env.state


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
                q_batch[i][j] = [env._env.get_features(), -1000]  #make it really bad

            # build M rollouts  (get rewards for all future states (1 -> m+1))
            # build rollout set (size m+1) from this state (going further in future), i.e. [(s, a, r)...]
            else:
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

        # print(env._env.state.board)
        tot_reward += (gamma**i) * curr_reward    # add gamma*reward to sum of rewards
        # print("tot reward: " + str(tot_reward))

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
