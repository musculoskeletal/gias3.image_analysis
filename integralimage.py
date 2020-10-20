"""
FILE: integralimage.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: calculation of integral images.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

"""
calculates an integral image
"""

USECYTHON = False

import numpy as np

if USECYTHON:
    import pyximport

    pyximport.install(
        setup_args={"include_dirs": np.get_include()},
        language_level=3
    )
    from gias2.image_analysis import integralimagec as IIC


def makeIntegralArray2(image):
    if len(image.shape) != 2:
        raise ValueError('input image dimension must be 2')

    X, Y = image.shape
    newImage = np.zeros((X + 1, Y + 1), dtype=np.int64)

    for i in range(X):
        for j in range(Y):
            newImage[i + 1, j + 1] = - newImage[i, j] + newImage[i + 1, j] + newImage[i, j + 1] + image[i, j]

    return newImage


def makeIntegralArray3(image):
    if len(image.shape) != 3:
        raise ValueError('input image dimension must be 3')

    X, Y, Z = image.shape
    newImage = np.zeros((X + 1, Y + 1, Z + 1), dtype=np.int64)

    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                newImage[i + 1, j + 1, k + 1] = + newImage[i, j, k] - newImage[i + 1, j, k] - newImage[i, j + 1, k] - \
                                                newImage[i, j, k + 1] \
                                                + newImage[i + 1, j + 1, k] + newImage[i + 1, j, k + 1] + newImage[
                                                    i, j + 1, k + 1] + image[i, j, k]

    return newImage


class IntegralImage2(object):
    _useCython = USECYTHON

    def __init__(self, image, useCython=USECYTHON):
        self._useCython = useCython
        if self._useCython:
            self.II = IIC.makeIntegralArray2(image.astype(IIC.DTYPE))
        else:
            self.II = makeIntegralArray2(image)
        self.shape = image.shape

    def getSum(self, x, y, l, w):
        s = self.II[x, y] - self.II[x + l, y] - self.II[x, y + w] + self.II[x + l, y + w]
        return s


class IntegralImage3(object):
    _useCython = USECYTHON

    def __init__(self, image, useCython=USECYTHON):
        self._useCython = useCython
        if self._useCython:
            self.II = IIC.makeIntegralArray3(image.astype(IIC.DTYPE))
        else:
            self.II = makeIntegralArray3(image)
        self.shape = image.shape

    def getSum(self, x, y, z, l, w, h):
        s = - self.II[x, y, z] + self.II[x + l, y, z] + self.II[x, y + w, z] + self.II[x, y, z + h] \
            - self.II[x + l, y + w, z] - self.II[x + l, y, z + h] - self.II[x, y + w, z + h] + self.II[
                x + l, y + w, z + h]
        return s
