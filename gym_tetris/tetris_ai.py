#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from datetime import datetime
import numpy as np

from tetris import Shape, ShapeKind
import features as uf

class TetrisRBAI(object):

    def __init__(self,width,height):
        self._width = width
        self._height = height
        # Allocate space AOT, reduce calls to allocate memory
        self._data1 = np.zeros((height,width),dtype=np.intc)
        self._data2 = np.zeros((height,width),dtype=np.intc)

    def action(self,state):
        t1 = datetime.now()
        CS = state.currentShape
        NS = state.nextShape
        if CS == ShapeKind.NONE.value:
            return None

        s1b = self._data1
        s1b2 = self._data2

        strategy = None
        if CS.kind in (ShapeKind.I.value, ShapeKind.Z.value, ShapeKind.S.value):
            d0Range = (0, 1)
        elif CS.kind == ShapeKind.O.value:
            d0Range = (0,)
        else:
            d0Range = (0, 1, 2, 3)

        if NS.kind in (ShapeKind.I.value, ShapeKind.Z.value, ShapeKind.S.value):
            d1Range = (0, 1)
        elif NS.kind == ShapeKind.O.value:
            d1Range = (0,)
        else:
            d1Range = (0, 1, 2, 3)


        for d0 in d0Range:
            minX, maxX, _, _ = CS.getBoundingOffsets(d0)
            for x0 in range(-minX, self._width - maxX):
                # Drop down the shape in place
                np.copyto(s1b,state.board)
                uf.dropDown_IP(s1b, CS, d0, x0)

                for d1 in d1Range:
                    minX, maxX, _, _ = NS.getBoundingOffsets(d1)
                    dropDist = uf.calcNextDropDist(s1b, NS, d1, range(-minX, self._width - maxX))
                    for x1 in range(-minX, self._width - maxX):
                        np.copyto(s1b2,s1b)
                        score = uf.calculateScore(s1b2, NS, d1, x1, dropDist)
                        if not strategy or strategy[2] < score:
                            strategy = (d0, x0, score)
        print("===", datetime.now() - t1)
        return strategy
