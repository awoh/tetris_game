#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

from . import TetrisEngine, Shape, ShapeKind

class TetrisGame(QMainWindow):

    def __init__(self,speed=200,AIType=None):
        super().__init__()
        self.isStarted = False
        self.isPaused = False
        self.nextMove = None
        self.gridSize = 22
        self.speed = speed
        self.lastShape = ShapeKind.NONE.value
        self._engine = TetrisEngine()
        self._agent = None
        if AIType is not None:
            self._agent = AIType(self._engine.width,self._engine.height)

        self.initUI()

    def initUI(self):
        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)

        hLayout = QHBoxLayout()
        self.tboard = Board(self, self.gridSize, self._engine.width, self._engine.height, self._state)
        hLayout.addWidget(self.tboard)

        self.sidePanel = SidePanel(self, self.gridSize, self._engine.width, self._engine.height, self._state)
        hLayout.addWidget(self.sidePanel)

        self.statusbar = self.statusBar()
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.start()
        self.center()
        self.setWindowTitle('Tetris')
        self.show()
        self.setFixedSize(self.tboard.width() + self.sidePanel.width(),
                          self.sidePanel.height() + self.statusbar.height())


    @property
    def _state(self):
        return self._engine.state

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def start(self):
        if self.isPaused:
            return

        self.isStarted = True
        self.tboard.score = 0
        self._engine.reset()
        self.tboard.msg2Statusbar.emit(str(self.tboard.score))
        self.timer.start(self.speed, self)

    def pause(self):
        if not self.isStarted:
            return

        self.isPaused = not self.isPaused

        if self.isPaused:
            self.timer.stop()
            self.tboard.msg2Statusbar.emit("paused")
        else:
            self.timer.start(self.speed, self)

        self.updateWindow()

    def updateWindow(self):
        self.tboard.updateData(self._state)
        self.sidePanel.updateData(self._state)
        self.update()

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            if self._agent and not self.nextMove:
                self.nextMove = self._agent.action(self._state)
            if self.nextMove:
                k = 0
                while self._state.direction != self.nextMove[0] and k < 4:
                    self._engine.rotateRight()
                    k += 1
                k = 0
                while self._state.x != self.nextMove[1] and k < 5:
                    if self._state.x > self.nextMove[1]:
                        self._engine.moveLeft()
                    elif self._state.x < self.nextMove[1]:
                        self._engine.moveRight()
                    k += 1
            # lines = self._engine.dropDown()
            lines = self._engine.moveDown()
            self.tboard.score += lines
            if self.lastShape != self._state.currentShape:
                self.nextMove = None
                self.lastShape = self._state.currentShape
            self.updateWindow()
        else:
            super(TetrisGame, self).timerEvent(event)

    def keyPressEvent(self, event):
        if not self.isStarted or self._state.currentShape.kind == ShapeKind.NONE.value:
            super(TetrisGame, self).keyPressEvent(event)
            return

        key = event.key()

        if key == Qt.Key_P:
            self.pause()
            return

        if self.isPaused:
            return
        elif key == Qt.Key_Left:
            self._engine.moveLeft()
        elif key == Qt.Key_Right:
            self._engine.moveRight()
        elif key == Qt.Key_Up:
            self._engine.rotateLeft()
        elif key == Qt.Key_Space:
            self.tboard.score += self._engine.dropDown()
        else:
            super(TetrisGame, self).keyPressEvent(event)

        self.updateWindow()


def drawSquare(painter, x, y, val, s):
    colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                  0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]

    if val == 0:
        return

    color = QColor(colorTable[val])
    painter.fillRect(x + 1, y + 1, s - 2, s - 2, color)

    painter.setPen(color.lighter())
    painter.drawLine(x, y + s - 1, x, y)
    painter.drawLine(x, y, x + s - 1, y)

    painter.setPen(color.darker())
    painter.drawLine(x + 1, y + s - 1, x + s - 1, y + s - 1)
    painter.drawLine(x + s - 1, y + s - 1, x + s - 1, y + 1)


class SidePanel(QFrame):
    def __init__(self, parent, gridSize, width, height, state):
        super().__init__(parent)
        self._state = state
        self.setFixedSize(gridSize * 5, gridSize * height)
        self.move(gridSize * width, 0)
        self.gridSize = gridSize

    def updateData(self,state):
        self._state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        minX, maxX, minY, maxY = self._state.nextShape.getBoundingOffsets(0)

        dy = 3 * self.gridSize
        dx = (self.width() - (maxX - minX) * self.gridSize) / 2

        val = self._state.nextShape.kind
        for x, y in self._state.nextShape.getCoords(0, 0, -minY):
            drawSquare(painter, x * self.gridSize + dx, y * self.gridSize + dy, val, self.gridSize)


class Board(QFrame):
    msg2Statusbar = pyqtSignal(str)

    def __init__(self, parent, gridSize, width, height, state):
        super().__init__(parent)
        self._width = width
        self._height = height
        self._state = state
        self.setFixedSize(gridSize * self._width, gridSize * self._height)
        self.gridSize = gridSize
        self.score = 0

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw backboard
        for x in range(self._width):
            for y in range(self._height):
                val = self._state.getValue(x, y)
                drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw current shape
        for x, y in self._state.getCurrentShapeCoord():
            val = self._state.currentShape.kind
            drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw a border
        painter.setPen(QColor(0x777777))
        painter.drawLine(self.width()-1, 0, self.width()-1, self.height())
        painter.setPen(QColor(0xCCCCCC))
        painter.drawLine(self.width(), 0, self.width(), self.height())

    def updateData(self,state):
        self._state = state
        self.msg2Statusbar.emit(str(self.score))
        self.update()
