#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import math
from . import Shape, ShapeKind

class TetrisState(object):
    def __init__(self,board,x,y,direction,currentShape,nextShape, width,last_piece_h, last_lines_clear, last_piece_clear,last_piece_coord):
        self.board = board
        self.x = x
        self.y = y
        self.direction = direction
        self.currentShape = currentShape
        # self.nextShape = nextShape
        self.nextShape = Shape(5)    #FOR TRIVIAL TETRIS GAME!!

        self.width = 6
        # self.width = width
        self.height = 22

        self.height_of_last_piece = 22     # ATTRIBUTES FOR FEATURES!!
        self.num_last_lines_cleared = 0
        self.num_last_piece_cleared = 0
        self.last_piece_drop_coords = np.empty(shape=4, dtype=(int,2))

    def getValue(self, x, y):
        return self.board[y,x]

    def getCurrentShapeCoord(self):
        return self.currentShape.getCoords(self.direction, self.x, self.y)

class TetrisEngine(object):
    # maybe add parameter to say which shapes are allowed
    def __init__(self,width=10,height=22):
        self.width = width
        self.height = height
        self.state = TetrisState(np.zeros((height,width),dtype=np.intc),-1,-1,0,Shape(),Shape.random(), width,22,0,0,np.empty(shape=4, dtype=(int,2)))
        self.shapeStat = [0] * 8
        self.done = False



        action_list = []
        for r in range(4):
            for c in range(self.width):
                action_list.append((r,c,0))

        self.action_map = {i: action_list[i] for i in range(len(action_list))}

    def getData(self):
        return self.state.board[:]

    def setState(self, state):
        self.state = state

    def reset(self):
        self.state = TetrisState(np.zeros((self.height,self.width),dtype=np.intc),
            -1,-1,0,Shape(),Shape.random(), self.width,22,0,0,np.empty(shape=4, dtype=(int,2)))
        self.createNewPiece()
        self.done = False
        return self.state

    def getCurrentShapeCoord(self):
        return self.state.currentShape.getCoords(self.state.direction, self.state.x, self.state.y)

    def createNewPiece(self):
        minX, maxX, minY, maxY = self.state.nextShape.getBoundingOffsets(0)
        result = False
        if self.tryMoveCurrent(0, int((self.width-1)/2), -minY):
            self.state.x = int((self.width-1)/2)
            self.state.y = -minY
            self.state.direction = 0
            self.state.currentShape = self.state.nextShape
            # self.state.nextShape = Shape(random.randint(1, 7))
            self.state.nextShape = Shape(5)   #FOR TRIVIAL STATE!!
            # self.state.nextShape = Shape(np.random.choice([1,5]))   #for o + i piece
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
        min_y = 22
        i=0
        for xx, yy in self.state.currentShape.getCoords(direction, x, self.state.y):
            if yy < min_y:
                min_y = yy
            self.state.board[yy,xx] = self.state.currentShape.kind
            self.state.last_piece_drop_coords[i] = (xx,yy) # tracks position of last dropped piece
            i+=1

        self.state.height_of_last_piece = min_y
        self.state.x= -1
        self.state.y = -1
        self.state.direction = 0
        self.state.currentShape = Shape()

        lines = self.removeFullLines()
        self.state.num_last_lines_cleared = lines
        created_piece = self.createNewPiece()

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

        self.state.num_last_piece_cleared = 0

        for coord in self.state.last_piece_drop_coords:
            if not rmask[coord[1]]:
                self.state.num_last_piece_cleared += 1
        # print("REMOVED: " + str(self.state.num_last_piece_cleared))

        if num_full > 0:
            new_board = np.zeros_like(self.state.board)
            new_board[(-num_left):,:] = self.state.board[rmask,:]
            self.state.board = new_board

        return num_full

    def mergePiece(self):
        min_y = 22
        # self.last_piece_drop_coords = []
        i=0
        for x, y in self.state.currentShape.getCoords(self.state.direction, self.state.x, self.state.y):
            if y < min_y:
                min_y = y
            self.state.board[y,x] = self.state.currentShape.kind
            self.state.last_piece_drop_coords[i] = (x,y) # tracks position of dropped piece
            i += 1


        self.state.height_of_last_piece = min_y
        self.state.x = -1
        self.state.y = -1
        self.state.direction = 0
        self.state.currentShape = Shape()

    def get_du_features(self):
        """35 features total """
        # TODO:

        self.features = []

        # DU FEATURES
        self.features.append(self.height -self.state.height_of_last_piece)     # landingHeight
        self.features.append(self.state.num_last_lines_cleared*self.state.num_last_piece_cleared)    # eroded cells
        self.features.append(self.countRowTransitions()) #number of times alternates piece and empty
        self.features.append(self.countColTransitions())
        self.features.append(self.countNumHoles()) # each empty, covered cell is a distinct hole
        self.features.append(self.getNumWells())
        self.features.append(self.getHoleDepths())
        self.features.append(self.countRowsWithHoles())  # rows with holes

        return self.features

    def getFeatures(self):
        """35 features total """
        # TODO:

        self.features = []

        # DU FEATURES
        self.features.append(self.height -self.state.height_of_last_piece)     # landingHeight
        self.features.append(self.state.num_last_lines_cleared*self.state.num_last_piece_cleared)    # eroded cells
        self.features.append(self.countRowTransitions()) #number of times alternates piece and empty
        self.features.append(self.countColTransitions())
        self.features.append(self.countNumHoles()) # each empty, covered cell is a distinct hole
        self.features.append(self.getNumWells())
        self.features.append(self.getHoleDepths())
        self.features.append(self.countRowsWithHoles())  # rows with holes

        self.features.append(self.countPatternDiversity())  # pattern diversity feature ???

        # # BERTSEKAS FEATURES (+num holes from above)
        # # height of each column
        # temp_heights = self.getColHeights() # 10
        # for i in range(len(temp_heights)):
        #     self.features.append(temp_heights[i])
        #
        # # height difference between columns
        # temp_differences = self.getHeightDifferences() # 9
        # for i in range(len(temp_differences)):
        #     self.features.append(temp_differences[i])
        #
        # self.features.append(self.getMaxHeight())
        # self.features.append(1)  # constant feature

        #
        #

        #
        # # rbf features (5)
        # for i in range(5):
        #     c = np.mean(np.array([temp_heights]))
        #     h = self.height
        #     numer = -1 * abs(c - (i * h)/4)**2
        #     denom = 2 * ((h / 5)**2)
        #     rbf_height = math.exp(-1*((c - (i*h)/4)**2)/(2*(h/5)**2))
        #     self.features.append(numer / denom)

        # # PIECE FEATURES (7)
        # pieces = [0]*7
        # pieces[self.state.currentShape.kind-1] = 1
        # self.features += pieces

        return self.features

    def getMaxHeight(self):
        width = self.width
        height = self.height
        board = self.getData()

        for y in range(height):
            for x in range(width):
                if board[y, x] != 0:
                    return y

        return 0

    def countRowTransitions(self):
        num_transitions = 0
        width = self.width
        max_height = self.getMaxHeight()
        board = self.getData()

        for y in range(self.height-1, max_height-1, -1): # +1 ?
            for x in range(width):
                cur_cell = board[y, x]
                if cur_cell == 0:
                    left_cell = None
                    right_cell = None

                    if x != 0:
                        left_cell = board[y, x-1]
                    if x != width - 1:
                        right_cell = board[y, x+1]

                    if left_cell != 0 and left_cell != None:
                        num_transitions += 1
                    if right_cell != 0 and right_cell != None:
                        num_transitions += 1

        return num_transitions

    def countColTransitions(self):
        num_transitions = 0
        width = self.width
        height = self.height
        board = self.getData()

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

                    if bottom_cell != 0 and bottom_cell != None:
                        num_transitions += 1
                    if top_cell != 0 and y != 0 and top_cell != None:
                        num_transitions += 1

        return num_transitions

    def countNumHoles(self):
        num_holes = 0
        width = self.width
        height = self.height
        board = self.getData()

        for y in range(height-1, -1, -1):
            for x in range(width):
                cur_cell = board[y, x]
                if cur_cell == 0:
                    covered = False
                    if y != 0:
                        for i in range(y-1, -1, -1):
                            if board[i,x] != 0:
                                covered = True
                                break

                    if covered and y != 0:
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
        board = self.getData()

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

    def getNumWells(self):
        num_wells = 0
        height = self.height
        board = self.getData()

        heights = self.getColHeights()
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

        # print('===================')
        # print(self.backBoard2D)
        # print('num_wells: ' + str(num_wells))
        # print('===================')
        return num_wells

    def countCellsAbove(self, cur_x, cur_y):
        count = 0
        board = self.getData()

        for y in range(cur_y-1, -1, -1):
            if board[y, cur_x] != 0:
                count += 1

        return count

    def getHoleDepths(self):
        hole_depths = 0
        height = self.height
        board = self.getData()

        heights = self.getColHeights()
        for c in range(len(heights)):
            if heights[c] > 1: # Holes only possible if blocks are at at least height 2
                for y in range(height-1, -1, -1):
                    if board[y, c] == 0:
                        hole_depths += self.countCellsAbove(c, y)

        # print('===================')
        # print(board)
        # print('hole_depths: ' + str(hole_depths))
        # print('===================')
        return hole_depths

    def countRowsWithHoles(self):
        num_rows = 0
        width = self.width
        height = self.height
        board = self.getData()

        for y in range(height-1, -1, -1):
            for x in range(width):
                cur_cell = board[y, x]
                if cur_cell == 0:
                    covered = False
                    if y != 0:
                        for i in range(y-1, -1, -1):
                            if board[i,x] != 0:
                                covered = True
                                break

                    if covered and y != 0:
                        num_rows += 1
                        break
        return num_rows

    def getHeightDifferences(self):
        # Could be combined with getColHeights() to save a tiny bit of time
        differences = []

        heights = self.getColHeights()
        for i in range(len(heights) - 1):
            differences.append(abs(heights[i] - heights[i+1]))

        # print('===================')
        # print(self.backBoard2D)
        # print('differences: ' + str(differences))
        # print('===================')
        return differences

    def countPatternDiversity(self):
        diversity_count = 0
        width = self.width
        height = self.height
        board = self.getData()

        string_rows = []
        for row in board:
            string_rows.append(''.join(str(x) for x in row))

        # Check unique rows
        diversity_count += len(set(string_rows))
        # Check unique columns
        string_cols = []
        for x in range(width):
            temp = ''
            for y in range(height):
                temp += str(board[y][x])
            string_cols.append(temp)
        diversity_count += len(set(string_cols))

        return diversity_count
