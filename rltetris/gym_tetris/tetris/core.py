import numpy as np
from enum import Enum
import random

from gym_tetris.utils1 import memoized_as_tuple


_shapeCoord = (
    ((0, 0), (0, 0), (0, 0), (0, 0)),
    ((0, -1), (0, 0), (0, 1), (0, 2)),
    ((0, -1), (0, 0), (0, 1), (1, 1)),
    ((0, -1), (0, 0), (0, 1), (-1, 1)),
    ((0, -1), (0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, -1), (1, 0), (1, -1)),
    ((0, 0), (0, -1), (-1, 0), (1, -1)),
    ((0, 0), (0, -1), (1, 0), (-1, -1))
)

class ShapeKind(Enum):
    NONE = 0
    I = 1
    L = 2
    J = 3
    T = 4
    O = 5
    S = 6
    Z = 7

    @classmethod
    def random(cls):
        return cls(random.randint(1, 7))


@memoized_as_tuple
def getRotatedOffsets(shapeKind, direction):
    tmpCoords = _shapeCoord[shapeKind]
    if direction == 0 or shapeKind == ShapeKind.O.value:
        return [(x, y) for x, y in tmpCoords]

    if direction == 1:
        return [(-y, x) for x, y in tmpCoords]

    if direction == 2:
        if shapeKind  in (ShapeKind.I.value, ShapeKind.Z.value, ShapeKind.S.value):
            return [(x, y) for x, y in tmpCoords]
        else:
            return [(-x, -y) for x, y in tmpCoords]

    if direction == 3:
        if shapeKind in (ShapeKind.I.value, ShapeKind.Z.value, ShapeKind.S.value):
            return [(-y, x) for x, y in tmpCoords]
        else:
            return [(y, -x) for x, y in tmpCoords]


@memoized_as_tuple
def getBoundingOffsets(shapeKind, direction):
    tmpCoords = getRotatedOffsets(shapeKind,direction)
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


@memoized_as_tuple
def getCoords(kind,direction,x,y):
    return [(x + xx, y + yy) for xx, yy in getRotatedOffsets(kind,direction)]


class Shape(object):
    __slots__ = ['kind']

    @classmethod
    def random(cls):
        return cls(random.randint(1, 7))

    def __init__(self, kind=0):
        self.kind = kind

    def getCoords(self, direction, x, y):
        return getCoords(self.kind,direction,x,y)

    def getBoundingOffsets(self, direction):
        return getBoundingOffsets(self.kind,direction)
