import numpy as np
import math

from tetris import Shape, ShapeKind

# NOTE: Suffix IP indicates the operation modifies the data in place

def dropDownByDist_IP(board, shape, direction, x0, dist):
    for x, y in shape.getCoords(direction, x0, 0):
        board[y + dist, x] = shape.kind

def dropDown_IP(board, shape, direction, x0):
    ht = board.shape[0]
    dy = ht - 1
    for x, y in shape.getCoords(direction, x0, 0):
        yy = 0
        while yy + y < ht and (yy + y < 0 or board[(y + yy), x] == ShapeKind.NONE.value):
            yy += 1
        yy -= 1
        if yy < dy:
            dy = yy
    # print("dropDown: shape {0}, direction {1}, x0 {2}, dy {3}".format(shape.shape, direction, x0, dy))
    dropDownByDist_IP(board, shape, direction, x0, dy)


def calcNextDropDist(board, nextShape, d0, xRange):
    ht = board.shape[0]
    res = {}
    for x0 in xRange:
        if x0 not in res:
            res[x0] = ht - 1
        for x, y in nextShape.getCoords(d0, x0, 0):
            yy = 0
            while yy + y < ht and (yy + y < 0 or board[(y + yy), x] == ShapeKind.NONE.value):
                yy += 1
            yy -= 1
            if yy < res[x0]:
                res[x0] = yy
    return res

def calculateScore(step1Board, nextShape, d1, x1, dropDist):
    height,width = step1Board.shape

    dropDownByDist_IP(step1Board, nextShape, d1, x1, dropDist[x1])

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
            if step1Board[y, x] == ShapeKind.NONE.value:
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

    score = fullLines * 1.8 - vHoles * 1.0 - vBlocks * 0.5 - maxHeight ** 1.5 * 0.02 \
        - stdY * 0.0 - stdDY * 0.01 - absDy * 0.2 - maxDy * 0.3
    # print(score, fullLines, vHoles, vBlocks, maxHeight, stdY, stdDY, absDy, roofY, d0, x0, d1, x1)
    return score
