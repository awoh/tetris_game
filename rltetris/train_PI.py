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
from environments import FeatureWrapper
from algorithm import CBMPI
import models

from gym_tetris import TetrisEnvironment


def eval_policy(env, plc):
    """takes in environment and policy and runs game """
    env.reset()
    lines_cleared = 0
    while not env._terminal:
        action = plc.action(env.state())
        # print("action: "+str(action))
        # GETS STUCK HERE SOMETIMES IN AN INFINITE LOOP...NOT SURE WHY!!
        _, reward, _, _ = env.step(action)
        if not env._terminal:
            lines_cleared += reward

        # print("reward: "+str(lines_cleared))
        # print(env._terminal)


    # print("eval state")
        # print(env._env.state.board)
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
    # width = 10
    width = 6
    num_actions = width * 4    # there are 40 potential actions (if width = 10)
    num_eval = 20
    # num_features = 8+(2*width+1) +7 # DU + bertsekas + 7 blocks (even though only using 2 rn)
    # num_features = 8 + 7 # bertsekas + pieces
    num_features = 9 # without pieces, just bertsekas
    init_plc = models.DUPolicy(env,num_features, num_actions)
    critic,plc = models.LinearVFA(num_features),models.LinearPolicy(env,num_features, num_actions)


    algo = CBMPI(plc,critic,train_config)
    # episode_results = np.array([]).reshape((0,6))     # will allow for training curve like in paper
    episode_results = np.empty(shape = [train_config['num_updates']*num_eval,3]) # allocate nujmpy array for all of iterations and evaluations initially, so can add more to it later
    # will want to
    cur_update = 0
    finished_episodes = 0
    start = time.time()
    # results_data = np.array([]).reshape((train_config['num_updates'],))

    # create wrapper for environment
    w_env = FeatureWrapper(env)
    m = train_config['m']
    gamma = train_config['gamma']

    while cur_update < train_config['num_updates']:
        env.reset()
        w_env.reset()

        # get set D_k (get start states), use DU Policy
        rnd_plc = models.RandomPolicy(env,num_features,num_actions)
        init_states = smp.sample_random_states(env, init_plc, rnd_plc, train_config['N'])
        # init_states = smp.sample_random_states(env, models.RandomPolicy(env,num_features,num_actions), train_config['N'])
        # quit()
        init_features = [0]*len(init_states)
        # get features for every state
        for i in range(len(init_states)):
            env.set_state(init_states[i])
            # print(init_states[i].board)
            init_features[i] = env.get_features()
        # print(init_features)
        # v_batch = smp.get_vh(w_env,init_states,plc,m,gamma,num_features)
        # print(v_batch)

        q_batch = smp.get_qh(w_env,init_states,plc,m,gamma,num_features, num_actions)

        print(q_batch)
        algo.update_critic(init_features,v_batch)    # update critic first
        print("CRITIC WEIGHTS: ")
        print(critic.weights)
        algo.update_policy(init_states, q_batch)
        print("POLICY WEIGHTS: ")
        print(plc.weights)


        # run evaluation code, save results, log results
         # save entire list to some file (instead of just average, provides additional info)
        # have function that takes environment name (train_config['n']) and policy
        # run and save results
        for i in range(num_eval):
            # [(iteration_number,discounted_reward,lines_cleared)]
            result = eval_policy(w_env,plc)
            print(w_env._env.state.board)
            print("result: "+ str(result))
            finished_episodes += 1
            # total_samples = cur_update * samples_per_update
            # stores: total_updates, total_episodes, total_samples, current_episode_length, current_total_reward, current_cumulative_reward

            # after update, generate list (for each of evaluations, put entry in list saying (iteration, lines cleared, discounted reward)
            episode_results[finished_episodes-1] = np.array([cur_update,finished_episodes,result],ndmin=2)
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
