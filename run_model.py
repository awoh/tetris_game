import os, sys, time
import tetris_game, tetris_model, tetris_ai

from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QPainter, QColor

NUM_EPISODES = 3
episode_count = 0
episode_data = []

def main():
    pass

def prettyPrint(board):
    ret = ''

    for i in range(20):
        for j in range(10):
            if board[i*10 + j] != 0:
                ret += '#, ' 
            else:
                ret += str(board[i*10 + j]) + ', '
        ret += '\n'

    return ret

if __name__ == "__main__":
    while episode_count < NUM_EPISODES:
        app = QApplication([])
        # tetris = tetris_game.Tetris(episode_data)
        tetris = tetris_game.Tetris()
        app.exec_()

        print('DATA:\n' + prettyPrint(tetris.getBoardData().backBoard))
        # print('DATA:\n' + str((tetris_model.episode_data)))
        # print('DATA:\n' + str(len(tetris_model.getEpisodeData())))
        app = None
        tetris = None
        episode_count += 1
        print('count: ' + str(episode_count))