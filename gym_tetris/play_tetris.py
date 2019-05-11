import sys
import random
import argparse
from PyQt5.QtWidgets import QApplication
from tetris import TetrisGame
from tetris_ai import TetrisRBAI
import gym
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rltetris import models
from rltetris import environments
# import environment

model = "../rltetris/out/experiment_2019.05.10_17.12.41/model_update_000004.npy"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai', action="store_true", help="Use the rule based AI to play")
    parser.add_argument('--speed', type=int, default=200, help="Game speed for human players")
    parser.add_argument('--seed', type=int, default=-1, help="Set random seed")

    args = parser.parse_args()
    args.speed = 10 if args.ai else args.speed
    if args.seed != -1:
        random.seed(args.seed)

    app = QApplication([])
    env = gym.make("Tetris-v0")

    # w_env = environments.FeatureWrapper(env)
    ai = models.LinearPolicy(env, 9,6)
    # ai = models.DUPolicy(env, 9, 6)
    ai.load_model(model)
    print(ai.weights)
    # ai = TetrisRBAI if args.ai else None
    tetris = TetrisGame(speed=args.speed,AIType=ai)
    sys.exit(app.exec_())
