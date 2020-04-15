"""
FILE: harr.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
3D Haar-like feature extraction. Given an integral image, each function returns
the difference in light and dark regions of one of 7 3D  haar-like features.

For each function the arguments are:
integralImage: an IntegralImage object
X: list-like, the indices of the upper left corner (smallest indices) of the
volume of interest
size: list-like, the size of the volume of interest in each direction

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import numpy as np


def haar3D1(integralImage, X, size):
    """
    x split
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l // 2, w, h)
    s2 = integralImage.getSum(x + l // 2, y, z, l // 2, w, h)
    return s1, s2


def haar3D2(integralImage, X, size):
    """
    y split
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l, w // 2, h)
    s2 = integralImage.getSum(x, y + w // 2, z, l, w // 2, h)
    return s1, s2


def haar3D3(integralImage, X, size):
    """
    z split
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l, w, h // 2)
    s2 = integralImage.getSum(x, y, z + h // 2, l, w, h // 2)
    return s1, s2


def haar3D4(integralImage, X, size):
    """
    xy checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l // 2, w // 2, h) + integralImage.getSum(x + l // 2, y + w // 2, z, l // 2,
                                                                                 w // 2, h)
    s2 = integralImage.getSum(x, y + w // 2, z, l // 2, w // 2, h) + integralImage.getSum(x + l // 2, y, z, l // 2,
                                                                                          w // 2, h)
    return s1, s2


def haar3D5(integralImage, X, size):
    """
    yz checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l, w // 2, h // 2) + integralImage.getSum(x, y + w // 2, z + h // 2, l, w // 2,
                                                                                 h // 2)
    s2 = integralImage.getSum(x, y + w // 2, z, l, w // 2, h // 2) + integralImage.getSum(x, y, z + h // 2, l, w // 2,
                                                                                          h // 2)
    return s1, s2


def haar3D6(integralImage, X, size):
    """
    xz checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l // 2, w, h // 2) + integralImage.getSum(x + l // 2, y, z + h // 2, l // 2, w,
                                                                                 h // 2)
    s2 = integralImage.getSum(x, y, z + h // 2, l // 2, w, h // 2) + integralImage.getSum(x + l // 2, y, z, l // 2, w,
                                                                                          h // 2)
    return s1, s2


def haar3D7(integralImage, X, size):
    """
    xyz checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l // 2, w // 2, h // 2) + integralImage.getSum(x + l // 2, y + w // 2, z, l // 2,
                                                                                      w // 2, h // 2) \
         + integralImage.getSum(x, y + w // 2, z + h // 2, l // 2, w // 2, h // 2) + integralImage.getSum(x + l // 2, y,
                                                                                                          z + h // 2,
                                                                                                          l // 2,
                                                                                                          w // 2,
                                                                                                          h // 2)
    s2 = integralImage.getSum(x, y + w // 2, z, l // 2, w // 2, h // 2) + integralImage.getSum(x + l // 2, y, z, l // 2,
                                                                                               w // 2, h // 2) \
         + integralImage.getSum(x, y, z + h // 2, l // 2, w // 2, h // 2) + integralImage.getSum(x + l // 2, y + w // 2,
                                                                                                 z + h // 2, l // 2,
                                                                                                 w // 2, h // 2)
    return s1, s2


def haar3D8(integralImage, X, size):
    """
    x 3-split
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l // 3, w, h) + integralImage.getSum(x + (2 * l // 3), y, z, l // 3, w, h)
    s2 = integralImage.getSum(x + (l // 3), y, z, l // 3, w, h)
    return s1, s2


def haar3D9(integralImage, X, size):
    """
    y 3-split
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l, w // 3, h) + integralImage.getSum(x, y + (2 * w // 3), z, l, w // 3, h)
    s2 = integralImage.getSum(x, y + (w // 3), z, l, w // 3, h)
    return s1, s2


def haar3D10(integralImage, X, size):
    """
    z 3-split
    """
    l, w, h = size
    x, y, z = X

    s1 = integralImage.getSum(x, y, z, l, w, h // 3) + integralImage.getSum(x, y, z + (2 * h // 3), l, w, h // 3)
    s2 = integralImage.getSum(x, y, z + (h // 3), l, w, h // 3)
    return s1, s2


features = (
    haar3D1, haar3D2, haar3D3, haar3D4, haar3D5,
    haar3D6, haar3D7, haar3D8, haar3D9, haar3D10,
)


# most accurate in HRV.py testing
def extractAllHaar3DDiff(integralImage, X, size):
    F = np.array([h(integralImage, X, size) for h in features])
    return F[:, 0] - F[:, 1]


# least accurate in HRV.py testing
def extractAllHaar3DRelDiff(integralImage, X, size):
    F = np.array([h(integralImage, X, size) for h in features])
    return (F[:, 0] - F[:, 1]) / F[:, 0]


# decently accurate in HRV.py testing
def extractAllHaar3DSign(integralImage, X, size):
    F = np.array([h(integralImage, X, size) for h in features])
    return np.sign(F[:, 0] - F[:, 1]).astype(np.int8)
