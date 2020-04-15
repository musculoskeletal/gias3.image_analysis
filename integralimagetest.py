"""
FILE: integralimagetest.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: unit test for integral images

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import unittest

import numpy as np

from gias2.image_analysis import integralimage


class TestIntegralImage(unittest.TestCase):

    def testIntegralImage3(self):
        I = np.ones([5, 5, 5], dtype=int)
        II = integralimage.IntegralImage3(I)
        s = II.getSum(1, 1, 1, 2, 2, 2)
        self.assertEqual(s, 8)

    def testIntegralImage2(self):
        I = np.ones([5, 5], dtype=int)
        II = integralimage.IntegralImage2(I)
        s = II.getSum(1, 1, 2, 2)
        self.assertEqual(s, 4)


if __name__ == '__main__':
    unittest.main()
