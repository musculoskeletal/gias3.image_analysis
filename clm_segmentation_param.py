"""
FILE: clm_segmentation.param.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: default parameters for CLM segmentation.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

iteration1 = {
    "RFFilename": '',  # filename of random forests
    "sampleMode": 'grid',  # how samples are distributed around each landmark
    "sampleN": 10,  # number of samples to take around each landmark
    "sampleWindowSize": 20,  # size length of cubic sample volumes
    "sampleDMax": 30,  # max distance to take samples from around each landmark
    "votingMode": 'comstd',  # how votes from decision trees are processed into data points for fitting
    "passMaxDist": 10,  # distance within which a landmark's datapoint is considered converged
    "minPassFrac": 0.8,  # minimum fraction of datapoints needed to pass to consider segmentation converged
}

iteration2 = {
    "RFFilename": '',  # filename of random forests
    "sampleMode": 'grid',  # how samples are distributed around each landmark
    "sampleN": 10,  # number of samples to take around each landmark
    "sampleWindowSize": 20,  # size length of cubic sample volumes
    "sampleDMax": 30,  # max distance to take samples from around each landmark
    "votingMode": 'comstd',  # how votes from decision trees are processed into data points for fitting
    "passMaxDist": 10,  # distance within which a landmark's datapoint is considered converged
    "minPassFrac": 0.8,  # minimum fraction of datapoints needed to pass to consider segmentation converged
}

iteration3 = {
    "RFFilename": '',  # filename of random forests
    "sampleMode": 'grid',  # how samples are distributed around each landmark
    "sampleN": 10,  # number of samples to take around each landmark
    "sampleWindowSize": 20,  # size length of cubic sample volumes
    "sampleDMax": 30,  # max distance to take samples from around each landmark
    "votingMode": 'comstd',  # how votes from decision trees are processed into data points for fitting
    "passMaxDist": 10,  # distance within which a landmark's datapoint is considered converged
    "minPassFrac": 0.8,  # minimum fraction of datapoints needed to pass to consider segmentation converged
}

params = [iteration1, iteration2, iteration3]
