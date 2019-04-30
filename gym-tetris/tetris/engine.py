#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from . import Shape, ShapeKind

class TetrisState(object):
    def __init__(self,board,x,y,direction,currentShape,nextShape):
        self.board = board
        self.x = x
        self.y = y
        self.direction = direction
        self.currentShape = currentShape
        self.nextShape = nextShape
        self.width = 10
        self.height = 22

    def getValue(self, x, y):
        return self.board[y,x]

    def getCurrentShapeCoord(self):
        return self.currentShape.getCoords(self.direction, self.x, self.y)

class TetrisEngine(object):
    # maybe add parameter to say which shapes are allowed
    def __init__(self,width=10,height=22):
        self.width = width
        self.height = height
        self.state = TetrisState(np.zeros((height,width),dtype=np.intc),-1,-1,0,Shape(),Shape.random())
        self.shapeStat = [0] * 8
        self.done = False

    def getData(self):
        return self.state.board[:]
        
    def setState(self, state):
        self.state = state

    def getCurrentShapeCoord(self):
        return self.state.currentShape.getCoords(self.state.direction, self.state.x, self.state.y)

    def createNewPiece(self):
        minX, maxX, minY, maxY = self.state.nextShape.getBoundingOffsets(0)
        result = False
        if self.tryMoveCurrent(0, 5, -minY):
            self.state.x = 5
            self.state.y = -minY
            self.state.direction = 0
            self.state.currentShape = self.state.nextShape
            self.state.nextShape = Shape(random.randint(1, 7))
            result = True
        else:
            self.state.currentShape = Shape()
            self.state.x = -1
            self.state.y = -1
            self.state.direction = 0
            result = False
        self.shapeStat[self.state.currentShape.kind] += 1
        return result

    def tryMoveCurrent(self, direction, x, y):
        return self.tryMove(self.state.currentShape, direction, x, y)

    def tryMove(self, shape, direction, x, y):
        for x, y in shape.getCoords(direction, x, y):
            if x >= self.width or x < 0 or y >= self.height or y < 0:
                return False
            if self.state.board[y,x] > 0:
                return False
        return True

    def moveDown(self):
        lines = 0
        if self.tryMoveCurrent(self.state.direction, self.state.x, self.state.y + 1):
            self.state.y += 1
        else:
            self.mergePiece()
            lines = self.removeFullLines()
            self.createNewPiece()
        return lines

    def dropDown(self):
        while self.tryMoveCurrent(self.state.direction, self.state.x, self.state.y + 1):
            self.state.y += 1
        self.mergePiece()
        lines = self.removeFullLines()
        self.createNewPiece()
        return lines

    def moveRotateDrop(self,direction,x):
        can_move = False

        while self.tryMoveCurrent(direction, x, self.state.y + 1):
            self.state.y += 1
            can_move = True
        if not can_move:
            self.done = True
            return

        # perform merge
        for xx, yy in self.state.currentShape.getCoords(direction, x, self.state.y):
            self.state.board[yy,xx] = self.state.currentShape.kind

        lines = self.removeFullLines()
        self.createNewPiece()
        return lines

    def moveLeft(self):
        if self.tryMoveCurrent(self.state.direction, self.state.x - 1, self.state.y):
            self.state.x -= 1

    def moveRight(self):
        if self.tryMoveCurrent(self.state.direction, self.state.x + 1, self.state.y):
            self.state.x += 1

    def rotateRight(self):
        if self.tryMoveCurrent((self.state.direction + 1) % 4, self.state.x, self.state.y):
            self.state.direction += 1
            self.state.direction %= 4

    def rotateLeft(self):
        if self.tryMoveCurrent((self.state.direction - 1) % 4, self.state.x, self.state.y):
            self.state.direction -= 1
            self.state.direction %= 4

    def removeFullLines(self):
        board_mask = self.state.board > 0
        rsums = np.sum(board_mask,axis=1)
        rmask = rsums < self.width
        num_left = np.sum(rmask)
        num_full = self.height - num_left

        if num_full > 0:
            new_board = np.zeros_like(self.state.board)
            new_board[(-num_left):,:] = self.state.board[rmask,:]
            self.state.board = new_board

        return num_full

    def mergePiece(self):
        for x, y in self.state.currentShape.getCoords(self.state.direction, self.state.x, self.state.y):
            self.state.board[y,x] = self.state.currentShape.kind

        self.state.x = -1
        self.state.y = -1
        self.state.direction = 0
        self.state.currentShape = Shape()

    def reset(self):
        self.state = TetrisState(np.zeros((self.height,self.width),dtype=np.intc),
            -1,-1,0,Shape(),Shape.random())
        self.createNewPiece()
        self.done = False
