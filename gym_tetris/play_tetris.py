import sys
import random
import argparse
from PyQt5.QtWidgets import QApplication
from tetris import TetrisGame
from tetris_ai import TetrisRBAI


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
    ai = TetrisRBAI if args.ai else None
    tetris = TetrisGame(speed=args.speed,AIType=ai)
    sys.exit(app.exec_())
