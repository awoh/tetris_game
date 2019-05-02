# Train Model using Modified Policy Iteration
import utils as utils
import os
import gym
import json
import time
import random
import numpy as np
import sampler as smp

import logging
logger = logging.getLogger(__name__)

# local imports
import environments
from algorithm import CBMPI
import models

from gym_tetris import TetrisEnvironment

def eval_policy(env, plc):
    """takes in environment and policy and runs game """
    env.reset()
    lines_cleared = 0
    while env._terminal != True:
        action = plc.action(env._engine.state)
        _, reward, _, _ = env.step()
        lines_cleared += reward
    return lines_cleared

if __name__ == '__main__':
    #############################
    # SETUP
    parser = utils.train_argparser()
    args = parser.parse_args()
    train_config = utils.train_params_from_args(args)


    episode_results_path = os.path.join(train_config['odir'],'episode_results.npy')

    utils.make_directory(train_config['odir'])
    with open(os.path.join(train_config['odir'],'train_config.json'), 'w') as fp:
        json.dump(train_config, fp, sort_keys=True, indent=4)

    log_level = logging.DEBUG if train_config['debug'] else logging.INFO
    if not train_config['console']:
        logging.basicConfig(filename=os.path.join(train_config['odir'],'train_log.log'),
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
    else:
        logging.basicConfig(level=log_level)

    ###############################
    # Create the environment
    env = gym.make(train_config['env'])
    if train_config['seed'] is not None:
        random.seed(train_config['seed'])
        # check to see if random.seed also sets numpy.seed, if not, set numpy.seed

# prints out what configuration you just ran on the command line
    log_str = '\r\n###################################################\r\n' + \
              '\tEnvironment: %s\r\n' % train_config['env'] + \
              '\tRandom Seed: %s\r\n' % str(train_config['seed']) + \
              '\tN, M, m: %d, %d, %d \r\n' % (train_config['N'],train_config['M'],train_config['m']) + \
              '\tConsole Logging, Debug Mode: (%s,%s) \r\n' % (str(train_config['console']),str(train_config['debug'])) + \
              '\tLog Interal, Total Updates, Save Interval: (%d,%d,%d) \r\n' % (train_config['log_interval'],train_config['num_updates'],train_config['save_interval']) + \
              '###################################################'
    logger.info(log_str)

    quit()
    # TODO - Modify the code below
    # THERE MAY BE ERRORS -- CHECK THIS

    num_actions = 40    # there are 40 potential actions
    num_features = 9 # DU + square piece
    init_plc = models.DUPolicy(env,num_features, num_actions)
    critic,plc = models.LinearVFA(num_features),models.LinearPolicy(env,num_features, num_actions)


    algo = CBMPI(plc,critic,train_config)
    # episode_results = np.array([]).reshape((0,6))     # will allow for training curve like in paper
    episode_results = np.array([]).reshape((train_config['num_updates']*num_eval,4)) # allocate nujmpy array for all of iterations and evaluations initially, so can add more to it later
    # will want to
    cur_update = 0
    finished_episodes = 0
    start = time.time()
    # results_data = np.array([]).reshape((train_config['num_updates'],))

    # create wrapper for environment
    w_env = FeatureWrapper(env)

    while cur_update < train_config['num_updates']:

        # get set D_k (get start states), use DU Policy
        init_states = smp.sample_random_states(env, init_plc, train_config['N'])

        # pass start states to function to get samples to update value function
        # SHOULD COMBINE V_HATS AND V_STATES IN THE GET_VH SO JUST ONE BATCH OUTPUT!!

        v_batch = smp.get_vh(w_env,init_states,plc,m,gamma,num_features)
        q_batch = smp.get_qh(w_env,init_states,plc,m,gamma,num_features, num_actions)

        algo.update_critic(init_states,v_batch)    # update critic first
        algo.update_policy(init_states, q_batch)

        # run evaluation code, save results, log resutls
         # save entire list to some file (instead of just average, provides additional info)
        # have function that takes environment name (train_config['n']) and policy
        # run and save results
        for i in range(num_eval):
            # [(iteration_number,discounted_reward,lines_cleared)]
            result = eval_policy(env,plc)

            finished_episodes += 1
            # total_samples = cur_update * samples_per_update   #WHAT IS THIS???
            # stores: total_updates, total_episodes, total_samples, current_episode_length, current_total_reward, current_cumulative_reward

            # after update, generate list (for each of evaluations, put entry in list saying (iteration, lines cleared, discounted reward)
            episode_results[finished_episodes] = np.array([cur_update,finished_episodes,result],ndmin=2)
            # episode_results = np.concatenate((episode_results,np.array([cur_update,finished_episodes,total_samples,res],ndmin=2)),axis=0)
            # episode_results = np.concatenate((episode_results,np.array([cur_update,finished_episodes,total_samples,el,tr,cr],ndmin=2)),axis=0)
            np.save(episode_results_path, episode_results)
            logger.info('Update Number: %06d, Finished Episode: %04d ---  Result: %.3f'% (cur_update,finished_episodes,result))

        # checkpoint
        if cur_update % train_config['save_interval'] == 0:
            plc.save_model(os.path.join(train_config['odir'],'model_update_%06d.npy'))
        cur_update += 1

    end = time.time()
    print(end - start)
