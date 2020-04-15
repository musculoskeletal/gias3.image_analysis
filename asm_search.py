"""
FILE: asm_search.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: various profile search functions for ASM segmentation

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import numpy as np


def scanProfile(dP, PC, modes):
    """ finds the position of signal within P what best matches the signal
    represented by principalComponent object PC, by minimising the
    mahalanobis distance. modes is a list of the modes on which to
    calculate the mahalanobis distance.
    """

    ND = len(PC.mean)  # length of model signal
    if np.mod(ND, 2):  # index of centre of signal
        NMid = ND / 2 + 1
    else:
        NMid = ND / 2

    nShift = len(dP) - ND  # number of positions to trial
    M = np.zeros(nShift)

    ##
    # ~ dP = dP/abs(dP).sum()	# normalise whole profile

    # calculate m distance at each position
    for s in range(nShift):
        # ~ dp = calcDeriv( P[s:s+ND] )

        # sample dP, and normalise
        dp = dP[s:s + ND]
        ##
        dp = dp / abs(dp).sum()  # normalise per sample
        dp = np.where(np.isfinite(dp), dp, 0.0)

        M[s] = PC.mahalanobis(dp, modes)
    # ~ print M[s]

    # get centre index of best match
    xMin = M.argmin()
    return xMin + NMid, M[xMin], M


def profileSearchElementPoints(epI, PPC, pModes, dP):
    """
    profile search on the element points specificed by epI.

    Arguments:
    ePi: a list of the indices of elements to do profiel search on. Indices correspond to the dim0 of the dP array
    PPC: a list of PCs for all possible elements point profiles
    dP: array of sampled element point profiles
    profWeight: weight to penalise distant profile matches. Not useful
    pModes: a list of mode numbers to use for each profile PC

    Returns:
    x: 1d array of profile match positions
    m: 1d array of match mahalanobis distances
    M: 2d array of the mahalanobis distances calculated for each sampled profile
    """

    # search along each normal profile to find best match
    x = np.zeros(epI.shape[0], dtype=int)
    m = np.zeros(epI.shape[0], dtype=float)
    M = []
    for i, epi in enumerate(epI):
        # find best match location along profile
        x[i], m[i], mi = scanProfile(dP[epi], PPC[epi], pModes[epi])
        M.append(mi)

    M = np.array(M)

    return x, m, M


def profileSearchElementOneSide(epI, PPC, pModes, dP):
    """
    profile search in the specified elements, contraining matches to be all on one side of each element

    Arguments:
    ePi: 2d list of the global indices of the element points in the elements of interest. len(epI)=number of elements of interest
    PPC: a list of PCs for all possible elements point profiles
    dP: array of sampled element point profiles
    pModes: a list of mode numbers to use for each profile PC

    Returns:
    x: 1d array of profile match positions
    m: 1d array of match mahalanobis distances
    M: 2d array of the mahalanobis distances calculated for each sampled profile
    """

    # search along each normal profile to find best match
    x = []
    m = []
    M = []
    dPLen = dP.shape[1]
    dPLen2 = dPLen / 2
    pLen = PPC[0].getMean().shape[0]
    pLen2 = pLen / 2
    padLen = dPLen2 - pLen2

    # loop through each element's element points
    for epEi in epI:

        # do normal matching
        xE, mE, ME = profileSearchElementPoints(epEi, PPC, pModes, dP)

        # pick +ve or -ve side (which ever has more)
        pSides = np.sign(xE - dPLen2)
        side = pSides.sum()

        if (side == 0) or (abs(side) == len(epEi)):
            # if 50/50 split, or all are on one side, then done
            pass
        else:
            sideXMean = int(xE[np.where(np.sign(pSides) == np.sign(side))].mean())
            sidemMean = mE[np.where(np.sign(pSides) == np.sign(side))].mean()

            # find element points with wrong side matches
            # find their best match on the right side by analysing their M vectors
            if side <= 0:
                redoEPi = np.where(pSides > 0)[0]
                for ri in redoEPi:
                    t = ME[ri][:padLen].argmin()
                    xE[ri] = t + pLen2
                    mE[ri] = ME[ri][t]

            # ~ # replace wrong side match positions with average right side position
            # ~ xE[ri] = sideXMean
            # ~ mE[ri] = sidemMean
            else:
                redoEPi = np.where(pSides < 0)[0]
                for ri in redoEPi:
                    t = ME[ri][padLen:].argmin() + padLen
                    xE[ri] = t + pLen2
                    mE[ri] = ME[ri][t]

        # record results of this element
        x.append(xE)
        m.append(mE)
        M.append(ME)

    return np.hstack(x), np.hstack(m), np.vstack(M)


def profileSearchElementMedian(epI, PPC, pModes, dP, landmarkMask, outSD=1.0):
    """
    profile search in the specified elements, detects outliers and finds alternative matches for them

    Arguments:
    ePi: 2d list of the global indices of the element points in the elements of interest. len(epI)=number of elements of interest
    PPC: a list of PCs for all possible elements point profiles
    pModes: a list of mode numbers to use for each profile PC
    dP: array of sampled element point profiles
    landmarkMask: a boolean array of length number_of_landmarks. True denotes a value landmark (inbound and unmasked)
    outSD: standard deviation outside which a point is considered an outlier

    Returns:
    x: 1d array of profile match positions
    m: 1d array of match mahalanobis distances
    M: 2d array of the mahalanobis distances calculated for each sampled profile
    """

    minValids = 5

    # search along each normal profile to find best match
    x = []
    m = []
    M = []
    dPLen = dP.shape[1]
    dPLen2 = dPLen / 2
    pLen = PPC[0].getMean().shape[0]
    pLen2 = pLen / 2
    padLen = dPLen2 - pLen2

    # loop through each element's element points
    for epEi in epI:

        epMask = landmarkMask[epEi]  # [True,False,True,True]
        # pass if less than minValids landmarks in this element are not masked
        if sum(epMask) < minValids:
            continue
        else:
            epEi2 = epEi[epMask]  # [100,102,103]

        # do normal matching
        xE, mE, ME = profileSearchElementPoints(epEi2, PPC, pModes, dP)

        # calc match position statistics
        xEMean = xE.mean()
        xEStd = xE.std()
        xEMedian = np.median(xE)

        # identify outliers
        outI = np.where(abs(xE - xEMedian) > outSD * xEStd)[0]

        # for each outlier
        for i in outI:
            # find new match closest to median match position
            t = findTrough(ME[i]) + pLen2

            if t.shape[0]:
                xE[i] = t[abs(t - xEMedian).argmin()]
                mE[i] = ME[i][xE[i] - pLen2]
            else:
                # no trough found
                pass

        # record results of this element
        x.append(xE)
        m.append(mE)
        M.append(ME)

    return np.hstack(x), np.hstack(m), np.vstack(M)


def findTrough(x):
    """ find troughs in x by looking at consecutive triplets of points
    where x[i-1]>x[i]<x[i+1]
    """
    t = np.where((x[1:-1] < x[2:]) & (x[1:-1] < x[:-2]))[0] + 1
    if len(t) == 0:
        t = np.array([np.argmin(x)])
    return t
