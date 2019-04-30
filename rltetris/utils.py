import os
import argparse
import random


def make_directory(dirpath):
    os.makedirs(dirpath,exist_ok=True)

def train_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--odir", type=str, default=None, help="output directory")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-g","--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("-s",'--save_interval', type=float, default=1e3, help="Model Save Interval")
    parser.add_argument("-l",'--log_interval', type=int, default=50, help="Log Interval")
    parser.add_argument("-E",'--env', type=str, default="Blackjack-v0", help="Environment to use")      # to run Tetris: "Tetris-v0"
    parser.add_argument("-co", "--console", action="store_true", help="log to console")
    parser.add_argument('--seed', type=int, default=543, metavar='N',help='random seed (default: 543). -1 indicates no seed')
    parser.add_argument("-N", type=int, default=20, help="Number of start states")
    parser.add_argument("-M",  type=int, default=1, help="Rollouts per state action pair")
    parser.add_argument("-m",  type=int, default=5, help="Rollout Depth")
    parser.add_argument("--niter", type=int, default=20, help="Number of iterations")

    return parser


def train_params_from_args(args):
    train_config = {'gamma' : args.gamma,
                         'debug' : args.debug,
                         'seed' : args.seed if args.seed != -1 else random.randint(0,1e8), # make a random seed
                         'num_updates' : int(args.niter),
                         'N' : args.N,
                         'M' : args.M,
                         'm' : args.m,
                         'env' : args.env,
                         'console' : args.console,
                         'log_interval': args.log_interval,
                         'odir' : args.odir if args.odir is not None else 'out/experiment_%s' % time.strftime("%Y.%m.%d_%H.%M.%S"),
                         'save_interval' : int(args.save_interval)}

    return train_config
