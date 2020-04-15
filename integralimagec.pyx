"""
FILE: integralimagec.pyx
LAST MODIFIED: 24-12-2015 
DESCRIPTION: calculation of integral images implemented in Cython

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

@cython.boundscheck(False)
def makeIntegralArray2(np.ndarray[DTYPE_t, ndim=2] image):
    assert image.dtype == DTYPE

    cdef unsigned int X = image.shape[0]
    cdef unsigned int Y = image.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] newImage = np.zeros((X + 1, Y + 1), dtype=DTYPE)
    cdef unsigned int i, j

    for i in range(X):
        for j in range(Y):
            newImage[i + 1, j + 1] = - newImage[i, j] + newImage[i + 1, j] + newImage[i, j + 1] + image[i, j]

    return newImage

@cython.boundscheck(False)
def makeIntegralArray3(np.ndarray[DTYPE_t, ndim=3] image):
    assert image.dtype == DTYPE

    cdef unsigned int X = image.shape[0]
    cdef unsigned int Y = image.shape[1]
    cdef unsigned int Z = image.shape[2]
    cdef np.ndarray[DTYPE_t, ndim=3] newImage = np.zeros((X + 1, Y + 1, Z + 1), dtype=DTYPE)
    cdef unsigned int i, j, k

    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                newImage[i + 1, j + 1, k + 1] = + newImage[i, j, k] - newImage[i + 1, j, k] - newImage[i, j + 1, k] - \
                                                newImage[i, j, k + 1] \
                                                + newImage[i + 1, j + 1, k] + newImage[i + 1, j, k + 1] + newImage[
                                                    i, j + 1, k + 1] + image[i, j, k]

    return newImage
