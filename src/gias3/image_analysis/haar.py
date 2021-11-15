"""
FILE: harr.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
3D Haar-like feature extraction. Given an integral image, each function returns
the difference in light and dark regions of one of 7 3D  haar-like features.

For each function the arguments are:
integral_image: an IntegralImage object
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


def haar3D1(integral_image, X, size):
    """
    x split
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l // 2, w, h)
    s2 = integral_image.getSum(x + l // 2, y, z, l // 2, w, h)
    return s1, s2


def haar3D2(integral_image, X, size):
    """
    y split
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l, w // 2, h)
    s2 = integral_image.getSum(x, y + w // 2, z, l, w // 2, h)
    return s1, s2


def haar3D3(integral_image, X, size):
    """
    z split
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l, w, h // 2)
    s2 = integral_image.getSum(x, y, z + h // 2, l, w, h // 2)
    return s1, s2


def haar3D4(integral_image, X, size):
    """
    xy checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l // 2, w // 2, h) + integral_image.getSum(x + l // 2, y + w // 2, z, l // 2,
                                                                                 w // 2, h)
    s2 = integral_image.getSum(x, y + w // 2, z, l // 2, w // 2, h) + integral_image.getSum(x + l // 2, y, z, l // 2,
                                                                                          w // 2, h)
    return s1, s2


def haar3D5(integral_image, X, size):
    """
    yz checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l, w // 2, h // 2) + integral_image.getSum(x, y + w // 2, z + h // 2, l, w // 2,
                                                                                 h // 2)
    s2 = integral_image.getSum(x, y + w // 2, z, l, w // 2, h // 2) + integral_image.getSum(x, y, z + h // 2, l, w // 2,
                                                                                          h // 2)
    return s1, s2


def haar3D6(integral_image, X, size):
    """
    xz checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l // 2, w, h // 2) + integral_image.getSum(x + l // 2, y, z + h // 2, l // 2, w,
                                                                                 h // 2)
    s2 = integral_image.getSum(x, y, z + h // 2, l // 2, w, h // 2) + integral_image.getSum(x + l // 2, y, z, l // 2, w,
                                                                                          h // 2)
    return s1, s2


def haar3D7(integral_image, X, size):
    """
    xyz checkered
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l // 2, w // 2, h // 2) + integral_image.getSum(x + l // 2, y + w // 2, z, l // 2,
                                                                                      w // 2, h // 2) \
         + integral_image.getSum(x, y + w // 2, z + h // 2, l // 2, w // 2, h // 2) + integral_image.getSum(x + l // 2, y,
                                                                                                          z + h // 2,
                                                                                                          l // 2,
                                                                                                          w // 2,
                                                                                                          h // 2)
    s2 = integral_image.getSum(x, y + w // 2, z, l // 2, w // 2, h // 2) + integral_image.getSum(x + l // 2, y, z, l // 2,
                                                                                               w // 2, h // 2) \
         + integral_image.getSum(x, y, z + h // 2, l // 2, w // 2, h // 2) + integral_image.getSum(x + l // 2, y + w // 2,
                                                                                                 z + h // 2, l // 2,
                                                                                                 w // 2, h // 2)
    return s1, s2


def haar3D8(integral_image, X, size):
    """
    x 3-split
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l // 3, w, h) + integral_image.getSum(x + (2 * l // 3), y, z, l // 3, w, h)
    s2 = integral_image.getSum(x + (l // 3), y, z, l // 3, w, h)
    return s1, s2


def haar3D9(integral_image, X, size):
    """
    y 3-split
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l, w // 3, h) + integral_image.getSum(x, y + (2 * w // 3), z, l, w // 3, h)
    s2 = integral_image.getSum(x, y + (w // 3), z, l, w // 3, h)
    return s1, s2


def haar3D10(integral_image, X, size):
    """
    z 3-split
    """
    l, w, h = size
    x, y, z = X

    s1 = integral_image.getSum(x, y, z, l, w, h // 3) + integral_image.getSum(x, y, z + (2 * h // 3), l, w, h // 3)
    s2 = integral_image.getSum(x, y, z + (h // 3), l, w, h // 3)
    return s1, s2


features = (
    haar3D1, haar3D2, haar3D3, haar3D4, haar3D5,
    haar3D6, haar3D7, haar3D8, haar3D9, haar3D10,
)


# most accurate in HRV.py testing
def extractAllHaar3DDiff(integral_image, X, size):
    F = np.array([h(integral_image, X, size) for h in features])
    return F[:, 0] - F[:, 1]


# least accurate in HRV.py testing
def extractAllHaar3DRelDiff(integral_image, X, size):
    F = np.array([h(integral_image, X, size) for h in features])
    return (F[:, 0] - F[:, 1]) / F[:, 0]


# decently accurate in HRV.py testing
def extractAllHaar3DSign(integral_image, X, size):
    F = np.array([h(integral_image, X, size) for h in features])
    return np.sign(F[:, 0] - F[:, 1]).astype(np.int8)
