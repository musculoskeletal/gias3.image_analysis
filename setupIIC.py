"""
FILE: setupIIC.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: setup file for Cython implementation of integralimage

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
from distutils.extension import Extension

from distutils.core import setup

from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("integralimagec", ["integralimagec.pyx"])]
)
