#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tetris_model import BOARD_DATA, Shape
import math
from datetime import datetime
import numpy as np


class TetrisAI(object):

    def nextMove(self):
        t1 = datetime.now()
        if BOARD_DATA.currentShape == Shape.shapeNone:
            return None

        currentDirection = BOARD_DATA.currentDirection
        currentY = BOARD_DATA.currentY
        _, _, minY, _ = BOARD_DATA.nextShape.getBoundingOffsets(0)
        nextY = -minY

        # print('=======')
        strategy = None

        # d0Range and d1Range refer to how many different configurations of the tetromino we can get ONLY via rotations.
        # d0Range is for the current piece, and d1Range is for the next piece
        if BOARD_DATA.currentShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            d0Range = (0, 1)  
        elif BOARD_DATA.currentShape.shape == Shape.shapeO:
            d0Range = (0,) # Rotating the square piece does nothing
        else:
            d0Range = (0, 1, 2, 3)

        if BOARD_DATA.nextShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            d1Range = (0, 1)
        elif BOARD_DATA.nextShape.shape == Shape.shapeO:
            d1Range = (0,)
        else:
            d1Range = (0, 1, 2, 3)

        num_options_count = 0
        for d0 in d0Range:
            minX, maxX, _, _ = BOARD_DATA.currentShape.getBoundingOffsets(d0)
            for x0 in range(-minX, BOARD_DATA.width - maxX):
                # print("here: {} , {}".format(-minX, BOARD_DATA.width - maxX))
                num_options_count += 1
                board = self.calcStep1Board(d0, x0)
                for d1 in d1Range:
                    minX, maxX, _, _ = BOARD_DATA.nextShape.getBoundingOffsets(d1)
                    dropDist = self.calcNextDropDist(board, d1, range(-minX, BOARD_DATA.width - maxX))
                    for x1 in range(-minX, BOARD_DATA.width - maxX):
                        score = self.calculateScore(np.copy(board), d1, x1, dropDist)
                        if not strategy or strategy[2] < score:
                            strategy = (d0, x0, score)
        print('===', datetime.now() - t1)
        # print(num_options_count)
        t1,t2,t3 = strategy # t1 is rotation, t2 is which column to place, t3 is the score of the board
        return (0,5,0)
        # return strategy

    def calcNextDropDist(self, data, d0, xRange):
        res = {}
        for x0 in xRange:
            if x0 not in res:
                res[x0] = BOARD_DATA.height - 1
            for x, y in BOARD_DATA.nextShape.getCoords(d0, x0, 0):
                yy = 0
                while yy + y < BOARD_DATA.height and (yy + y < 0 or data[(y + yy), x] == Shape.shapeNone):
                    yy += 1
                yy -= 1
                if yy < res[x0]:
                    res[x0] = yy
        return res

    def calcStep1Board(self, d0, x0):
        board = np.array(BOARD_DATA.getData()).reshape((BOARD_DATA.height, BOARD_DATA.width))
        self.dropDown(board, BOARD_DATA.currentShape, d0, x0)
        return board

    def dropDown(self, data, shape, direction, x0):
        dy = BOARD_DATA.height - 1
        for x, y in shape.getCoords(direction, x0, 0):
            yy = 0
            while yy + y < BOARD_DATA.height and (yy + y < 0 or data[(y + yy), x] == Shape.shapeNone):
                yy += 1
            yy -= 1
            if yy < dy:
                dy = yy
        # print('dropDown: shape {0}, direction {1}, x0 {2}, dy {3}'.format(shape.shape, direction, x0, dy))
        self.dropDownByDist(data, shape, direction, x0, dy)

    def dropDownByDist(self, data, shape, direction, x0, dist):
        for x, y in shape.getCoords(direction, x0, 0):
            data[y + dist, x] = shape.shape

    def calculateScore(self, step1Board, d1, x1, dropDist):
        # print('calculateScore')
        t1 = datetime.now()
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        self.dropDownByDist(step1Board, BOARD_DATA.nextShape, d1, x1, dropDist[x1])
        # print(datetime.now() - t1)

        # Term 1: lines to be removed
        fullLines, nearFullLines = 0, 0
        roofY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width
        vHoles, vBlocks = 0, 0
        for y in range(height - 1, -1, -1):
            hasHole = False
            hasBlock = False
            for x in range(width):
                if step1Board[y, x] == Shape.shapeNone:
                    hasHole = True
                    holeCandidates[x] += 1
                else:
                    hasBlock = True
                    roofY[x] = height - y
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]
                        holeCandidates[x] = 0
                    if holeConfirm[x] > 0:
                        vBlocks += 1
            if not hasBlock:
                break
            if not hasHole and hasBlock:
                fullLines += 1
        vHoles = sum([x ** .7 for x in holeConfirm])
        maxHeight = max(roofY) - fullLines
        # print(datetime.now() - t1)

        roofDy = [roofY[i] - roofY[i+1] for i in range(len(roofY) - 1)]

        if len(roofY) <= 0:
            stdY = 0
        else:
            stdY = math.sqrt(sum([y ** 2 for y in roofY]) / len(roofY) - (sum(roofY) / len(roofY)) ** 2)
        if len(roofDy) <= 0:
            stdDY = 0
        else:
            stdDY = math.sqrt(sum([y ** 2 for y in roofDy]) / len(roofDy) - (sum(roofDy) / len(roofDy)) ** 2)

        absDy = sum([abs(x) for x in roofDy])
        maxDy = max(roofY) - min(roofY)
        # print(datetime.now() - t1)
        # self.getHoleDepths(step1Board) # FIXME: remove this line. it's for testing

        score = fullLines * 1.8 - vHoles * 1.0 - vBlocks * 0.5 - maxHeight ** 1.5 * 0.02 \
            - stdY * 0.0 - stdDY * 0.01 - absDy * 0.2 - maxDy * 0.3
        # print('++++++++++++++++++')
        # print('score: {} | fullLines: {} | vHoles: {} | vBlocks: {} | maxHeight: {} | stdY: {} | stdDY: {} | absDy: {} | roofY: {} | d1: {} | x1: {}'.format(score, fullLines, vHoles, vBlocks, maxHeight, stdY, stdDY, absDy, roofY, d1, x1))
        # print('++++++++++++++++++')
        return score

    def getMaxHeight(self, board):
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        for y in range(height):
            for x in range(width):
                if board[y, x] != 0:
                    return y

        return 0

    def countRowTransitions(self, board):
        num_transitions = 0
        width = BOARD_DATA.width
        max_height = self.getMaxHeight(board)

        for y in range(BOARD_DATA.height-1, max_height-1, -1): # +1 ?
            for x in range(width):
                cur_cell = board[y, x]
                if cur_cell == 0:
                    left_cell = None
                    right_cell = None

                    if x != 0:
                        left_cell = board[y, x-1]
                    if x != width - 1:
                        right_cell = board[y, x+1]

                    if left_cell != 0:
                        num_transitions += 1
                    if right_cell != 0:
                        num_transitions += 1

    def countColTransitions(self, board):
        num_transitions = 0
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        for y in range(height-1, -1, -1):
            for x in range(width):
                cur_cell = board[y, x]
                if cur_cell == 0:
                    bottom_cell = None
                    top_cell = None

                    if y != 0:
                        top_cell = board[y-1, x]
                    if y != height-1:
                        bottom_cell = board[y+1, x]

                    if bottom_cell != 0:
                        num_transitions += 1
                    if top_cell != 0 and y != 0:
                        num_transitions += 1

        return num_transitions

    def countNumHoles(self, board):
        num_holes = 0
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        for y in range(height-1, -1, -1):
            for x in range(width):
                cur_cell = board[y, x]
                if cur_cell == 0:
                    top_cell = None

                    if y != 0:
                        top_cell = board[y-1, x]

                    if top_cell != 0 and y != 0:
                        num_holes += 1

        print('===================')
        print(board)
        print('num holes: ' + str(num_holes))
        print('===================')
        return num_holes

    def getColHeights(self, board):
        heights = []
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        for x in range(width):
            height_found = False
            for y in range(height):
                if board[y, x] != 0:
                    height_found = True
                    heights.append(height - y)
                    break
            if not height_found:
                heights.append(0)

        return heights

    def getHeightDifferences(self, board):
        # Could be combined with getColHeights() to save a tiny bit of time
        differences = []
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        heights = self.getColHeights(board)
        for i in range(len(heights) - 1):
            differences.append(abs(heights[i] - heights[i+1]))

        print('===================')
        print(board)
        print('differences: ' + str(differences))
        print('===================')
        return differences

    def getNumWells(self, board):
        num_wells = 0
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        heights = self.getColHeights(board)
        for c in range(len(heights)):
            if c == 0:
                if board[height-heights[c]-1, c+1] != 0:
                    num_wells += 1
            elif c == len(heights)-1:
                if board[height-heights[c]-1, c-1] != 0:
                    num_wells += 1
            else:
                left_cell = board[height-heights[c]-1, c-1]
                right_cell = board[height-heights[c]-1, c+1]
                if left_cell != 0 and right_cell != 0:
                    num_wells += 1

        print('===================')
        print(board)
        print('num_wells: ' + str(num_wells))
        print('===================')
        return num_wells

    def countCellsAbove(self, cur_x, cur_y, board):
        count = 0

        for y in range(cur_y-1, -1, -1):
            if board[y, cur_x] != 0:
                count += 1

        return count

    def getHoleDepths(self, board):
        hole_depths = 0
        width = BOARD_DATA.width
        height = BOARD_DATA.height
        
        heights = self.getColHeights(board)
        for c in range(len(heights)):
            if heights[c] > 1: # Holes only possible if blocks are at at least height 2
                for y in range(height-1, -1, -1):
                    if board[y, c] == 0:
                        hole_depths += self.countCellsAbove(c, y, board)

        print('===================')
        print(board)
        print('hole_depths: ' + str(hole_depths))
        print('===================')
        return hole_depths
        

# TODO: 
#     landing height of falling piece FIXME: this is updated in mergePiece() in tetris_model.py
#     number of eroded piece cells
#     num holes (again???)
#     number of wells FIXME: do the walls count as wells? I vote yes.
#     number of rows with holes
#     pattern diversity feature (???)
#     RBF height features

TETRIS_AI = TetrisAI()

