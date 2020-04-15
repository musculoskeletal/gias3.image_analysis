"""
FILE: asm_search_c.pyx
LAST MODIFIED: 24-12-2015 
DESCRIPTION: various profile search functions for ASM segmentation implemented
in Cython.

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
DTYPEF = np.double
ctypedef np.double_t DTYPEF_t
# ctypedef double DTYPEF_t
DTYPEI = np.int
ctypedef np.int_t DTYPEI_t
# DTYPEB = np.bool8
# ctypedef np.bool8_t DTYPEB_t

cdef DTYPEF_t PRECISION = 1e-16

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef DTYPEI_t argmin(np.ndarray[DTYPEF_t, ndim=1] x):
    cdef DTYPEI_t n = x.shape[0]
    cdef DTYPEI_t i
    cdef DTYPEI_t imin = 0
    cdef DTYPEF_t xmin = x[0]
    for i in range(n):
        if x[i] < xmin:
            imin = i
            xmin = x[i]

    return imin

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void project(np.ndarray[DTYPEF_t, ndim=1] d,
                  np.ndarray[DTYPEF_t, ndim=1] u,
                  np.ndarray[DTYPEF_t, ndim=2] B,
                  np.ndarray[DTYPEI_t, ndim=1] modes,
                  int nModes, int nVar,
                  np.ndarray[DTYPEF_t, ndim=1] W):
    cdef DTYPEF_t w
    cdef unsigned int mi, vi, m

    for mi in range(nModes):
        w = 0.0
        m = modes[mi]
        for vi in range(nVar):
            w += B[vi, m] * (d[vi] - u[vi])
        W[mi] = w

    return

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef DTYPEF_t mahalanobis(np.ndarray[DTYPEF_t, ndim=1] d,
                          np.ndarray[DTYPEF_t, ndim=1] u,
                          np.ndarray[DTYPEF_t, ndim=2] B,
                          np.ndarray[DTYPEF_t, ndim=1] E,
                          np.ndarray[DTYPEI_t, ndim=1] modes,
                          int nModes,
                          int nVar):
    cdef DTYPEF_t d2, w, e
    cdef unsigned int mi, vi, m

    d2 = 0.0
    for mi in range(nModes):
        w = 0.0
        m = modes[mi]
        e = E[m]
        for vi in range(nVar):
            w += B[vi, m] * (d[vi] - u[vi])

        if e > PRECISION:
            d2 += (w * w) / e

    return d2

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef DTYPEF_t mahalanobis2(np.ndarray[DTYPEF_t, ndim=1] W,
                           np.ndarray[DTYPEF_t, ndim=1] E,
                           np.ndarray[DTYPEI_t, ndim=1] modes,
                           int nModes):
    cdef DTYPEF_t d2, w, e
    cdef unsigned int mi, m

    d2 = 0.0
    for mi in range(nModes):
        m = modes[mi]
        w = W[mi]
        e = E[m]
        if e > PRECISION:
            d2 += (w * w) / e

    return d2

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void reconstruct(np.ndarray[DTYPEF_t, ndim=1] W,
                      np.ndarray[DTYPEF_t, ndim=2] B,
                      np.ndarray[DTYPEF_t, ndim=1] u,
                      np.ndarray[DTYPEI_t, ndim=1] modes,
                      int nModes,
                      int nVar,
                      np.ndarray[DTYPEF_t, ndim=1] r):
    cdef int vi, mi, m

    for vi in range(nVar):
        r[vi] = u[vi]
        for mi in range(nModes):
            m = modes[mi]
            r[vi] += B[vi, mi] * W[mi]

    return

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef DTYPEF_t diffrms(np.ndarray[DTYPEF_t, ndim=1] x1,
                      np.ndarray[DTYPEF_t, ndim=1] x2,
                      int n):
    cdef DTYPEF_t d, s, m

    for i in range(n):
        d = x1[i] - x2[i]
        s += d * d

    return (s / n) ** 0.5

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def scanProfile(np.ndarray[DTYPEF_t, ndim=1] dP, PC,
                np.ndarray[DTYPEI_t, ndim=1] modes):
    """ finds the position of signal within P what best matches the signal
    represented by principalComponent object PC, by minimising the
    mahalanobis distance. modes is a list of the modes on which to
    calculate the mahalanobis distance.
    """

    # assert dP.dtype == DTYPEF and modes.dtype == DTYPEI

    cdef unsigned int ND = PC.mean.shape[0]  # length of model signal
    cdef unsigned int NModes = modes.shape[0]  # number of modes for PCA
    cdef unsigned int nShift = dP.shape[0] - ND  # number of positions to trial
    cdef np.ndarray[DTYPEF_t, ndim=1] M = np.zeros(nShift, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] dp = np.zeros(ND, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] u = PC.mean.astype(DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=2] B = PC.modes.astype(DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] E = PC.weights.astype(DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] W = np.zeros(NModes, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=1] r = np.zeros(ND, dtype=DTYPEF)
    cdef unsigned int s, i
    cdef DTYPEI_t xmin, NMid
    cdef DTYPEF_t p, psum, md, rms

    if ND % 2 is 1:  # index of centre of signal
        NMid = ND / 2 + 1
    else:
        NMid = ND / 2

    # calculate m distance at each position
    for s in range(nShift):
        psum = 0.0

        # get subsample
        for i in range(ND):
            dp[i] = dP[s + i]
            psum += abs(dP[s + i])

        # normalise subsample
        if psum < PRECISION:
            for i in range(ND):
                dp[i] = 0.0
        else:
            for i in range(ND):
                dp[i] = dp[i] / psum

        M[s] = mahalanobis(dp, u, B, E, modes, NModes, ND)
    # project(dp, u, B, modes, NModes, ND, W)
    # reconstruct(W, B, u, modes, NModes, ND, r)
    # md = mahalanobis2(W, E, modes, NModes)
    # rms = diffrms(r, dp, ND)
    # M[s] = md * rms

    # get centre index of best match
    xMin = argmin(M)
    return xMin + NMid, M[xMin], M

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def profileSearchElementPoints(np.ndarray[DTYPEI_t, ndim=1] epI, PPC, pModes,
                               np.ndarray[DTYPEF_t, ndim=2] dP):
    """
    profile search on the element points specificed by epI.

    Arguments:
    ePi: a list of the indices of elements to do profile search on. Indices
         correspond to the dim0 of the dP array
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
    cdef unsigned int nPoints = epI.shape[0]
    cdef unsigned int ND = dP.shape[1] - PPC[0].mean.shape[0]
    cdef np.ndarray[DTYPEI_t, ndim=1] x = np.zeros(nPoints, dtype=DTYPEI)
    cdef np.ndarray[DTYPEF_t, ndim=1] m = np.zeros(nPoints, dtype=DTYPEF)
    cdef np.ndarray[DTYPEF_t, ndim=2] M = np.zeros((nPoints, ND), dtype=DTYPEF)
    for i in range(nPoints):
        # find best match location along profile
        x[i], m[i], mi = scanProfile(dP[epI[i]], PPC[epI[i]], pModes[epI[i]])
        M[i, :] = mi

    return x, m, M

def profileSearchElementOneSide(epI, PPC, pModes, dP):
    """
    profile search in the specified elements, contraining matches to be all on one side of each element

    Arguments:
    ePi: 2d list of the global indices of the element points in the elements of interest.
         len(epI)=number of elements of interest
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

                #~ # replace wrong side match positions with average right side position
                #~ xE[ri] = sideXMean
                #~ mE[ri] = sidemMean
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

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def profileSearchElementMedian(epIList, PPC, pModes,
                               np.ndarray[DTYPEF_t, ndim=2] dP,
                               landmarkMask,
                               DTYPEF_t outSD):
    """
    profile search in the specified elements, detects outliers and finds alternative
    matches for them

    Arguments:
    ePi: 2d list of the global indices of the element points in the elements of interest.
         len(epI)=number of elements of interest
    PPC: a list of PCs for all possible elements point profiles
    pModes: a list of mode numbers to use for each profile PC
    dP: array of sampled element point profiles
    landmarkMask: a boolean array of length number_of_landmarks. True denotes a value landmark
                  (inbound and unmasked)
    outSD: standard deviation outside which a point is considered an outlier

    Returns:
    x: 1d array of profile match positions
    m: 1d array of match mahalanobis distances
    M: 2d array of the mahalanobis distances calculated for each sampled profile
    """
    cdef unsigned int nElems = len(epIList)
    cdef np.ndarray[DTYPEI_t, ndim=1] epEI
    cdef np.ndarray[DTYPEI_t, ndim=1] epEI2
    cdef np.ndarray[DTYPEI_t, ndim=1] xE
    cdef np.ndarray[DTYPEF_t, ndim=1] mE
    cdef np.ndarray[DTYPEF_t, ndim=2] ME
    cdef np.ndarray[DTYPEI_t, ndim=1] outI
    cdef np.ndarray[DTYPEI_t, ndim=1] t
    cdef unsigned int minValids = 1
    cdef unsigned int ei, nOutliers, i, outi

    # search along each normal profile to find best match
    x = []
    m = []
    M = []
    cdef unsigned int dPLen = dP.shape[1]
    cdef unsigned int dPLen2 = dPLen / 2
    cdef unsigned int pLen = PPC[0].mean.shape[0]
    cdef unsigned int pLen2 = pLen / 2
    cdef unsigned int padLen = dPLen2 - pLen2
    cdef DTYPEF_t xEStd, xEMedian

    # loop through each element's element points
    for ei in range(nElems):
        epEI = np.array(epIList[ei], dtype=DTYPEI)
        epMask = landmarkMask[epEI]  #[True,False,True,True]
        # pass if less than minValids landmarks in this element are not masked
        if sum(epMask) < minValids:
            # print 'DING'
            # print sum(epMask)
            continue
        else:
            epEI2 = epEI[epMask]  #[100,102,103]

        # do normal matching
        xE, mE, ME = profileSearchElementPoints(epEI2, PPC, pModes, dP)

        # calc match position statistics
        xEStd = xE.std()
        xEMedian = np.median(xE)

        # identify outliers
        outI = np.where(abs(xE - xEMedian) > outSD * xEStd)[0]
        nOutliers = outI.shape[0]

        # for each outlier
        for i in range(nOutliers):
            # find new match closest to median match position
            outi = outI[i]
            t = findTrough(ME[i]) + pLen2
            if t.shape[0] > 0:
                xE[outi] = t[argmin(abs(t - xEMedian))]
                mE[outi] = ME[outi][xE[outi] - pLen2]
            else:
                # no trough found
                pass

        # record results of this element
        x.append(xE)
        m.append(mE)
        M.append(ME)

    return np.hstack(x), np.hstack(m), np.vstack(M)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef np.ndarray[DTYPEI_t, ndim=1] findTrough(np.ndarray[DTYPEF_t, ndim=1] x):
    """ find troughs in x by looking at consecutive triplets of points
    where x[i-1]>x[i]<x[i+1]
    """
    cdef unsigned int n = x.shape[0]
    cdef unsigned int i
    t = []
    for i in range(1, n - 1):
        if (x[i - 1] > x[i]) & (x[i + 1] > x[i]):
            t.append(i)

    return np.array(t, dtype=DTYPEI)
