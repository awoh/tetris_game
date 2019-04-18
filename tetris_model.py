#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np

# from run_model import episode_data
episode_data = []

class Shape(object):
    shapeNone = 0
    shapeI = 1
    shapeL = 2
    shapeJ = 3
    shapeT = 4
    shapeO = 5
    shapeS = 6
    shapeZ = 7

    # ORIGINAL
    # shapeCoord = (
    #     ((0, 0), (0, 0), (0, 0), (0, 0)),
    #     ((0, -1), (0, 0), (0, 1), (0, 2)),      # I piece
    #     ((0, -1), (0, 0), (0, 1), (1, 1)),      # J piece
    #     ((0, -1), (0, 0), (0, 1), (-1, 1)),     # L piece
    #     ((0, -1), (0, 0), (0, 1), (1, 0)),      # T piece
    #     ((0, 0), (0, -1), (1, 0), (1, -1)),     # O piece
    #     ((0, 0), (0, -1), (-1, 0), (1, -1)),    # Z piece
    #     ((0, 0), (0, -1), (1, 0), (-1, -1))     # S piece
    # )

    # MODIFIED (fixed spawn orientations)
    shapeCoord = (
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((-2, 0), (-1, 0), (0, 0), (1, 0)),       # I piece (1)
        ((-2, -1), (-2, 0), (-1, 0), (0, 0)),     # J piece (2)
        ((0, -1), (-2, 0), (-1, 0), (0, 0)),      # L piece (3)
        ((-2, -1), (-1, -1), (0, -1), (-1, -2)),  # T piece (4)
        ((-1, 0), (-1, -1), (0, 0), (0, -1)),     # O piece (5)
        ((-1, 0), (-1, -1), (-2, 0), (0, -1)),    # S piece (6)
        ((-1, 0), (-1, -1), (0, 0), (-2, -1))     # Z piece (7)
    )

    def __init__(self, shape=0):
        self.shape = shape

    def getRotatedOffsets(self, direction):
        tmpCoords = Shape.shapeCoord[self.shape]
        if direction == 0 or self.shape == Shape.shapeO:
            return ((x, y) for x, y in tmpCoords)

        if direction == 1:
            return ((-y, x) for x, y in tmpCoords)

        if direction == 2:
            if self.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
                return ((x, y) for x, y in tmpCoords)
            else:
                return ((-x, -y) for x, y in tmpCoords)

        if direction == 3:
            if self.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
                return ((-y, x) for x, y in tmpCoords)
            else:
                return ((y, -x) for x, y in tmpCoords)

    def getCoords(self, direction, x, y):
        return ((x + xx, y + yy) for xx, yy in self.getRotatedOffsets(direction))

    def getBoundingOffsets(self, direction):
        tmpCoords = self.getRotatedOffsets(direction)
        minX, maxX, minY, maxY = 0, 0, 0, 0
        for x, y in tmpCoords:
            if minX > x:
                minX = x
            if maxX < x:
                maxX = x
            if minY > y:
                minY = y
            if maxY < y:
                maxY = y
        return (minX, maxX, minY, maxY)


class BoardData(object):
    width = 10
    height = 20

    def __init__(self):
        self.backBoard = [0] * BoardData.width * BoardData.height

        self.height_of_last_piece = 22
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()
        episode_data = []
        self.backBoard2D = [[0]*BoardData.width]*BoardData.height
        self.features = []

        self.pieces_consumed = 0

        # Create batch of all 7 tetrominoes
        self.shape_queue = list(range(1,8))
        # print(self.shape_queue)
        random.shuffle(self.shape_queue)
        if self.shape_queue[0] == 6 or self.shape_queue[0] == 7:
            self.ensureGoodFirstPiece()

        self.nextShape = Shape(self.shape_queue.pop(0)) # Get the first piece in the queue

        self.shapeStat = [0] * 8

    def ensureGoodFirstPiece(self):
        '''
        Ensures that we don't begin with a Z or S piece because those cannot be placed
        without creating holes.
        '''
        i = 1
        while not (self.shape_queue[i] != 6 or self.shape_queue[i] != 7):
            i += 1

        temp = self.shape_queue[0]
        self.shape_queue[0] = self.shape_queue[i]
        self.shape_queue[i] = temp

    def getData(self):
        return self.backBoard[:]

    def getValue(self, x, y):
        return self.backBoard[x + y * BoardData.width]

    def getCurrentShapeCoord(self):
        return self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY)

    def getNextShape(self):
        # Make new batch if we ran out of pieces
        if len(self.shape_queue) == 0:
            self.shape_queue = list(range(1,8))
            random.shuffle(self.shape_queue)

        # print('Removed a piece: ' + str(self.shape_queue))
        self.pieces_consumed += 1

        return Shape(self.shape_queue.pop(0))


    def createNewPiece(self):
        episode_data.append((self.backBoard, self.nextShape.shape))

        minX, maxX, minY, maxY = self.nextShape.getBoundingOffsets(0)
        result = False
        if self.tryMoveCurrent(0, 5, -minY):
            self.currentX = 5
            self.currentY = -minY
            self.currentDirection = 0
            self.currentShape = self.nextShape
            self.nextShape = self.getNextShape()
            result = True
        else:
            # FIXME: this is where we know when we've lost. restart program here
            # os.execv('tetris_game.py', sys.argv)
            self.currentShape = Shape()
            self.currentX = -1
            self.currentY = -1
            self.currentDirection = 0
            result = False
        self.shapeStat[self.currentShape.shape] += 1
        return result

    def tryMoveCurrent(self, direction, x, y):
        return self.tryMove(self.currentShape, direction, x, y)

    def tryMove(self, shape, direction, x, y):
        for x, y in shape.getCoords(direction, x, y):
            if x >= BoardData.width or x < 0 or y >= BoardData.height or y < 0:
                return False
            if self.backBoard[x + y * BoardData.width] > 0:
                return False
        return True

    def moveDown(self):
        lines = 0
        if self.tryMoveCurrent(self.currentDirection, self.currentX, self.currentY + 1):
            self.currentY += 1
        else:
            self.mergePiece()
            lines = self.removeFullLines()
            if not self.createNewPiece():
                return -1
        return lines

    def dropDown(self):
        while self.tryMoveCurrent(self.currentDirection, self.currentX, self.currentY + 1):
            self.currentY += 1
        self.mergePiece()
        lines = self.removeFullLines()
        self.createNewPiece()
        return lines

    def moveLeft(self):
        if self.tryMoveCurrent(self.currentDirection, self.currentX - 1, self.currentY):
            self.currentX -= 1

    def moveRight(self):
        if self.tryMoveCurrent(self.currentDirection, self.currentX + 1, self.currentY):
            self.currentX += 1

    def rotateRight(self):
        if self.tryMoveCurrent((self.currentDirection + 1) % 4, self.currentX, self.currentY):
            self.currentDirection += 1
            self.currentDirection %= 4

    def rotateLeft(self):
        if self.tryMoveCurrent((self.currentDirection - 1) % 4, self.currentX, self.currentY):
            self.currentDirection -= 1
            self.currentDirection %= 4

    def removeFullLines(self):
        newBackBoard = [0] * BoardData.width * BoardData.height
        newY = BoardData.height - 1
        lines = 0
        for y in range(BoardData.height - 1, -1, -1):
            blockCount = sum([1 if self.backBoard[x + y * BoardData.width] > 0 else 0 for x in range(BoardData.width)])
            if blockCount < BoardData.width:
                for x in range(BoardData.width):
                    newBackBoard[x + newY * BoardData.width] = self.backBoard[x + y * BoardData.width]
                newY -= 1
            else:
                lines += 1
        if lines > 0:
            self.backBoard = newBackBoard
        return lines

    def mergePiece(self):
        min_y = 22
        for x, y in self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY):
            if y < min_y:
                min_y = y
            self.backBoard[x + y * BoardData.width] = self.currentShape.shape

        self.height_of_last_piece = min_y
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()

    def clear(self):
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()
        self.backBoard = [0] * BoardData.width * BoardData.height
    def update2DBoard(self):
        self.backBoard2D = np.array(self.backBoard).reshape((self.height, self.width))

    def getFeatures(self):
        # TODO:
        self.backBoard2D = np.array(self.backBoard).reshape((self.height, self.width))
        self.features = []

        self.features.append(self.countRowTransitions())
        self.features.append(self.countColTransitions())
        self.features.append(self.countNumHoles())
        self.features.append(self.getNumWells())
        self.features.append(self.getHoleDepths())

        temp_heights = self.getColHeights()
        for i in range(len(temp_heights)):
            self.features.append(temp_heights[i])
        # self.features.append(self.getColHeights())
        temp_differences = self.getHeightDifferences()
        for i in range(len(temp_differences)):
            self.features.append(temp_differences[i])
        # self.features.append(self.getHeightDifferences())

        # self.features.append(self.countCellsAbove())

        self.features.append(self.getMaxHeight())

        return self.features

    def getMaxHeight(self):
        width = self.width
        height = self.height

        for y in range(height):
            for x in range(width):
                if self.backBoard2D[y, x] != 0:
                    return y

        return 0

    def countRowTransitions(self):
        num_transitions = 0
        width = self.width
        max_height = self.getMaxHeight()

        for y in range(self.height-1, max_height-1, -1): # +1 ?
            for x in range(width):
                cur_cell = self.backBoard2D[y, x]
                if cur_cell == 0:
                    left_cell = None
                    right_cell = None

                    if x != 0:
                        left_cell = self.backBoard2D[y, x-1]
                    if x != width - 1:
                        right_cell = self.backBoard2D[y, x+1]

                    if left_cell != 0:
                        num_transitions += 1
                    if right_cell != 0:
                        num_transitions += 1

        return num_transitions

    def countColTransitions(self):
        num_transitions = 0
        width = self.width
        height = self.height

        for y in range(height-1, -1, -1):
            for x in range(width):
                cur_cell = self.backBoard2D[y, x]
                if cur_cell == 0:
                    bottom_cell = None
                    top_cell = None

                    if y != 0:
                        top_cell = self.backBoard2D[y-1, x]
                    if y != height-1:
                        bottom_cell = self.backBoard2D[y+1, x]

                    if bottom_cell != 0:
                        num_transitions += 1
                    if top_cell != 0 and y != 0:
                        num_transitions += 1

        return num_transitions

    def countNumHoles(self):
        num_holes = 0
        width = self.width
        height = self.height

        for y in range(height-1, -1, -1):
            for x in range(width):
                cur_cell = self.backBoard2D[y, x]
                if cur_cell == 0:
                    top_cell = None

                    if y != 0:
                        top_cell = self.backBoard2D[y-1, x]

                    if top_cell != 0 and y != 0:
                        num_holes += 1

        # print('===================')
        # print(self.backBoard2D)
        # print('num holes: ' + str(num_holes))
        # print('===================')
        return num_holes

    def getColHeights(self):
        heights = []
        width = self.width
        height = self.height

        for x in range(width):
            height_found = False
            for y in range(height):
                if self.backBoard2D[y, x] != 0:
                    height_found = True
                    heights.append(height - y)
                    break
            if not height_found:
                heights.append(0)

        return heights

    def getHeightDifferences(self):
        # Could be combined with getColHeights() to save a tiny bit of time
        differences = []
        width = self.width
        height = self.height

        heights = self.getColHeights()
        for i in range(len(heights) - 1):
            differences.append(abs(heights[i] - heights[i+1]))

        # print('===================')
        # print(self.backBoard2D)
        # print('differences: ' + str(differences))
        # print('===================')
        return differences

    def getNumWells(self):
        num_wells = 0
        width = self.width
        height = self.height

        heights = self.getColHeights()
        for c in range(len(heights)):
            if c == 0:
                if self.backBoard2D[height-heights[c]-1, c+1] != 0:
                    num_wells += 1
            elif c == len(heights)-1:
                if self.backBoard2D[height-heights[c]-1, c-1] != 0:
                    num_wells += 1
            else:
                left_cell = self.backBoard2D[height-heights[c]-1, c-1]
                right_cell = self.backBoard2D[height-heights[c]-1, c+1]
                if left_cell != 0 and right_cell != 0:
                    num_wells += 1

        # print('===================')
        # print(self.backBoard2D)
        # print('num_wells: ' + str(num_wells))
        # print('===================')
        return num_wells

    def countCellsAbove(self, cur_x, cur_y):
        count = 0

        for y in range(cur_y-1, -1, -1):
            if self.backBoard2D[y, cur_x] != 0:
                count += 1

        return count

    def getHoleDepths(self):
        hole_depths = 0
        width = self.width
        height = self.height

        heights = self.getColHeights()
        for c in range(len(heights)):
            if heights[c] > 1: # Holes only possible if blocks are at at least height 2
                for y in range(height-1, -1, -1):
                    if self.backBoard2D[y, c] == 0:
                        hole_depths += self.countCellsAbove(c, y)

        # print('===================')
        # print(board)
        # print('hole_depths: ' + str(hole_depths))
        # print('===================')
        return hole_depths


BOARD_DATA = BoardData()
