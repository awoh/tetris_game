import sys
import random
import argparse
from PyQt5.QtWidgets import QApplication
from tetris import TetrisGame
from tetris_ai import TetrisRBAI
sys.path.append("/foo")
import models
import environments
import environment
import gym

model = "../out/experiment_2019.05.08_18.11.27/model_update_%06d.npy"


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
    ai.load_model(model)
    # ai = TetrisRBAI if args.ai else None
    tetris = TetrisGame(speed=args.speed,AIType=ai)
    sys.exit(app.exec_())
