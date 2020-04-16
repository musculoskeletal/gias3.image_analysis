"""
FILE: haarregressionvoting.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: sampling of 3D haar-like features from 3D images.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging
import pickle
import warnings

import numpy as np
import sys
from sklearn.ensemble import ExtraTreesRegressor

from gias2.image_analysis import haar
from gias2.image_analysis import image_tools
from gias2.image_analysis import integralimage

log = logging.getLogger(__name__)


class SamplingWarning(Exception):
    pass


class SamplingError(Exception):
    pass


def extractHaarFeatures(II, p, windowSize, haarMode='diff'):
    X = np.round(p - np.array(windowSize) / 2.0).astype(int)
    if haarMode == 'diff':
        return haar.extractAllHaar3DDiff(II, X, windowSize)
    elif haarMode == 'reldiff':
        return haar.extractAllHaar3DRelDiff(II, X, windowSize)
    elif haarMode == 'sign':
        return haar.extractAllHaar3DSign(II, X, windowSize)


def makeHaarFeatureExtractor(II, haarMode):
    if haarMode == 'diff':
        def haarFeatureExtractor(p, w):
            X = np.round(p - w / 2.0).astype(int)
            return haar.extractAllHaar3DDiff(II, X.T, w.T.astype(int))
    elif haarMode == 'reldiff':
        def haarFeatureExtractor(p, w):
            X = np.round(p - w / 2.0).astype(int)
            return haar.extractAllHaar3DRelDiff(II, X.T, w.T.astype(int))
    elif haarMode == 'sign':
        def haarFeatureExtractor(p, w):
            X = np.round(p - w / 2.0).astype(int)
            return haar.extractAllHaar3DSign(II, X.T, w.T.astype(int))
    else:
        raise ValueError('invalid haarMode')

    return haarFeatureExtractor


class HaarImage(image_tools.Scan):
    """
    Class for 3D image for sampling 3D Haar features. Inherits from 
    image_tools.Scan. An integral image is generated on instantiation
    """

    windowSize = (10, 10, 10)  # size of volume to sample around each landmark
    samplesPerPoint = 10  # number volumes to sample around each landmark
    reader = None

    def __init__(self, I=None, voxelSpacing=None, voxelOrigin=None, isMasked=False):
        """
        instantiation input args:
        I: 3D numpy array
        voxelSpacing: three tuple
        voxelOrigin: three tuple
        isMasked: boolean, where I is a masked array
        haarMode: ['diff', 'reldiff', 'sign'] how haar features are calculated
        """

        self._displacementGrids = {}
        self.isMasked = isMasked
        if I is not None:
            self._setImageArray(I, voxelSpacing, voxelOrigin)

        self.haarFeatureExtractor = None
        self.haarFeatureExtractorHaarMode = None

    def __del__(self):
        del self.I
        del self.II

    def _setImageArray(self, I, voxelSpacing=None, voxelOrigin=None):
        if len(I.shape) != 3:
            raise ValueError('image must be 3D')

        self.I = I
        self.II = integralimage.IntegralImage3(self.I)

        if voxelSpacing is None:
            self.voxelSpacing = np.array([1.0, 1.0, 1.0])
        else:
            self.voxelSpacing = voxelSpacing

        if voxelOrigin is None:
            self.voxelOrigin = np.array([0.0, 0.0, 0.0])
        else:
            self.voxelOrigin = voxelOrigin

    def setHaarExtractor(self, haarMode):
        self.haarFeatureExtractorHaarMode = haarMode
        self.haarFeatureExtractor = makeHaarFeatureExtractor(self.II, haarMode)

    def extractHaarAboutPoint(self, p, windowSize, sampleMode='diff'):
        """
        extracts Haar features from a volume centered about image
        indices p, with size defined by windowSize.

        inputs:
        p: 3-tuple, image voxel indices
        windowSize: 3-tuple, sample volume size in number of voxels

        returns:
        a list of haar features
        """
        p = np.array(p)
        windowSize = np.array(windowSize)
        X = np.round(p - windowSize / 2.0).astype(int)
        if sampleMode == 'diff':
            return haar.extractAllHaar3DDiff(self.II, X, windowSize)
        elif sampleMode == 'reldiff':
            return haar.extractAllHaar3DRelDiff(self.II, X, windowSize)
        elif sampleMode == 'sign':
            return haar.extractAllHaar3DSign(self.II, X, windowSize)
        else:
            raise ValueError('invalid sampleMode')

    def extractHaarAboutPointRandom(self, p, n, windowSize, dMax, zShift=False, negSpacing=False, haarMode='diff'):
        """
        randomly extract features in volumes randomly displaced about p.
        Returns list of features and the displacement vectors.

        displacements are uniformly (random) distributed about p with maximum displacement
        in physical (not image) units of dMax.

        Returns a list of displacements in physical units, and a list of
        corresponding image features.

        If sample requires an index out of range, random displacement is re-drawn until 
        one within range is obtained. Therefore, no displacements requiring sampling out 
        of range will be generated.

        inputs:
        p: 3-tuple, image voxel indices
        n: integer, number of random samples
        windowSize: 3-tuple, sample volume size in number of voxels
        dMax: float, maximum displacement of random sample from p
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        
        returns:
        displacements: a n x 3 array of displacement vectors
        features: a list of lists of haar features
        """

        windowSize = np.array(windowSize)
        p = np.array(p)
        displacements = np.random.uniform(low=-dMax, high=dMax, size=(n, 3))
        sampleIndices = self.coord2Index(p + displacements, zShift, negSpacing)
        # randomly modify windowSize too?
        # randomly alter orientation?

        features = []
        for i, sInd in enumerate(sampleIndices):
            # print sInd
            try:
                features.append(extractHaarFeatures(self.II, sInd, windowSize, haarMode))
            except IndexError:
                retry = 1
                while retry:
                    log.debug('retry', retry)
                    dRetry = np.random.uniform(low=-dMax, high=dMax, size=(3))
                    sIndRetry = self.coord2Index(p + dRetry, zShift, negSpacing)

                    try:
                        features.append(extractHaarFeatures(self.II, sIndRetry, windowSize, haarMode))
                    except IndexError:
                        retry += 1
                    else:
                        displacements[i] = dRetry
                        retry = 0
                        log.debug(sIndRetry)

        if len(features) == 0:
            warnings.warn("No suitable sampling locations, p = " + str(p))

        return displacements, features

    def extractHaarAboutPointRandomMulti(self, P, n, windowSize, dMax, zShift=False, negSpacing=False, haarMode='diff',
                                         windowSizeVar=None):
        """
        randomly extract features in volumes randomly displaced about points P.
        Returns list of lists of features and the displacement vectors.

        displacements are uniformly (random) distributed about p with maximum displacement
        in physical (not image) units of dMax.

        Returns a list of displacements in physical units, and a list of
        corresponding image features.

        If sample requires an index out of range, random displacement is re-drawn until 
        one within range is obtained. Therefore, no displacements requiring sampling out 
        of range will be generated.

        inputs:
        p: 3-tuple, image voxel indices
        n: integer, number of random samples
        windowSize: 3-tuple, sample volume size in number of voxels
        dMax: float, maximum displacement of random sample from p
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        
        returns:
        displacements: a n x 3 array of displacement vectors
        features: a list of lists of haar features
        """

        nPoints = P.shape[0]
        nSamples = n * nPoints
        windowSize = np.array(windowSize)
        maxRetry = 10000

        # generate window sizes
        if windowSizeVar != None:
            windowSizes = windowSize * np.random.uniform(low=1.0 - windowSizeVar,
                                                         high=1.0 + windowSizeVar,
                                                         size=nSamples)[:, np.newaxis]
        else:
            windowSizes = windowSize * np.ones(nSamples)[:, np.newaxis]

        windowSizes = windowSizes.reshape((nPoints, n, 3))
        windowSizes2 = windowSizes / 2.0

        # generate displacements
        displacements = np.random.uniform(low=-dMax, high=dMax, size=(nSamples, 3)).reshape(
            (nPoints, n, 3))  # shape = (nPoints, samples per point, 3)
        samplePoints = displacements + P[:, np.newaxis, :]
        sampleIndices = self.coord2Index(samplePoints.reshape((nSamples, 3)), zShift, negSpacing).reshape(
            (nPoints, n, 3))

        # randomly alter orientation?

        # redo out of bounds samples
        for pi, I in enumerate(sampleIndices):
            for ii, ind in enumerate(I):
                if not (self.checkIndexInBounds(ind + windowSizes2[pi, ii]) and self.checkIndexInBounds(
                        ind - windowSizes2[pi, ii])):
                    retry = True
                    retryCount = 1
                    while retry:
                        sys.stdout.write('\rretry ' + str(retryCount))
                        sys.stdout.flush()

                        # regen displacement
                        dRetry = np.random.uniform(low=-dMax, high=dMax, size=(3))
                        indRetry = self.coord2Index(P[pi] + dRetry, zShift, negSpacing)
                        # regen window size
                        if windowSizeVar != None:
                            wsRetry = windowSize * np.random.uniform(low=1.0 - windowSizeVar,
                                                                     high=1.0 + windowSizeVar)
                        else:
                            wsRetry = windowSize

                        wsRetry2 = wsRetry / 2.0

                        if (self.checkIndexInBounds(indRetry + wsRetry2) and self.checkIndexInBounds(
                                indRetry - wsRetry2)):
                            displacements[pi, ii, :] = dRetry
                            sampleIndices[pi, ii, :] = indRetry
                            windowSizes[pi, ii, :] = wsRetry
                            retry = False
                        else:
                            retryCount += 1
                            if retryCount > maxRetry:
                                raise SamplingError('Unable to sample in bounds')

        # output shape = (nPoints*n, number of features)
        self.setHaarExtractor(haarMode)
        features = self.haarFeatureExtractor(sampleIndices.reshape((nSamples, 3)),
                                             windowSizes.reshape((nSamples, 3))).T
        # shape = (nPoints, n, number of features)
        features = features.reshape((nPoints, n, -1))

        if len(features) == 0:
            warnings.warn("No suitable sampling locations")

        return displacements, features

    def extractHaarAboutPointGridSphere(self, p, n, windowSize, dMax, zShift=False, negSpacing=False):
        """
        Extract features in volumes distributed in a regular grid within a sphere of radius
        dMax about point P. n is the number of samples along the diameter. Sample volumes
        outside of image are skipped.

        Returns a list of displacements in physical units, and a list of
        corresponding image features.

        inputs:
        p: 3-tuple, image voxel indices
        n: integer, number of random samples
        windowSize: 3-tuple, sample volume size in number of voxels
        dMax: float, maximum displacement of random sample from p
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        
        returns:
        displacements: a n x 3 array of displacement vectors
        features: a list of lists of haar features
        """

        try:
            displacements = self._displacementGrids[(dMax, n)]
        except KeyError:
            displacements = _generateSampleGrid(dMax, n)
            self._displacementGrids[(dMax, n)] = displacements

        # sample
        sampleIndices = self.coord2Index(p + displacements, zShift, negSpacing)

        features = []
        featureDisplacements = []
        for i, d in enumerate(sampleIndices):
            try:
                features.append(extractHaarFeatures(self.II, d, windowSize))
            except IndexError:
                # out of bounds sample, ignore
                pass
            else:
                featureDisplacements.append(displacements[i])

        # if all samples were out of bounds
        if len(features) == 0:
            # print sampleIndices.max(0)
            # print sampleIndices.min(0)
            warnings.warn("No suitable sampling locations, p = " + str(p))

        return featureDisplacements, features

    def extractHaarAboutPointGridSphereMulti(self, P, n, windowSize, dMax, zShift=False, negSpacing=False,
                                             haarMode='diff'):
        """
        Extract features in volumes distributed in a regular grid within a sphere of radius
        dMax about points P. n is the number of samples along the diameter. Sample volumes
        outside of image are skipped.

        Returns a list of displacements in physical units, and a list of
        corresponding image features.

        inputs:
        p: 3-tuple, image voxel indices
        n: integer, number of random samples
        windowSize: 3-tuple, sample volume size in number of voxels
        dMax: float, maximum displacement of random sample from p
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        
        returns:
        displacements: a n x 3 array of displacement vectors
        features: a list of lists of haar features
        """
        nPoints = P.shape[0]
        windowSize = np.array(windowSize)
        windowSize2 = windowSize / 2.0
        self.setHaarExtractor(haarMode)

        try:
            disp = self._displacementGrids[(dMax, n)]
        except KeyError:
            disp = _generateSampleGrid(dMax, n)
            self._displacementGrids[(dMax, n)] = disp

        # generate sampling points
        sampleIndices = []
        displacements = []
        nSamples = []
        for p in P:
            samplePoints = p + disp
            sampleIndicesTemp = self.coord2Index(samplePoints, zShift, negSpacing)
            inBounds = np.array([(self.checkIndexInBounds(ind + windowSize2) and \
                                  self.checkIndexInBounds(ind - windowSize2)) for ind in sampleIndicesTemp])
            inBoundsI = np.where(inBounds == True)[0]
            sampleIndices.append(sampleIndicesTemp[inBoundsI, :])
            displacements.append(disp[inBoundsI, :])
            nSamples.append(sampleIndices[-1].shape[0])

        sampleIndicesFlat = np.vstack(sampleIndices)
        windowSizes = windowSize + np.zeros_like(sampleIndicesFlat)
        # output shape = (nPoints*n, number of features)
        featuresTemp = self.haarFeatureExtractor(sampleIndicesFlat, windowSizes).T
        # shape = (nPoints, n, number of features)
        features = []
        i = 0
        for nS in nSamples:
            features.append(featuresTemp[i:i + nS, :])
            i += nS

        # if all samples were out of bounds
        if len(features) == 0:
            # print sampleIndices.max(0)
            # print sampleIndices.min(0)
            warnings.warn("No suitable sampling locations, p = " + str(p))

        return displacements, features


def _generateSampleGrid(dMax, n):
    # make cube grid about origin
    x, y, z = np.mgrid[-dMax:dMax:complex(0, n),
              -dMax:dMax:complex(0, n),
              -dMax:dMax:complex(0, n),
              ]
    X = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # filter out grid points outside sphere
    d = np.sqrt((X ** 2.0).sum(1))
    displacements = X[np.where(d <= dMax)[0]]
    return displacements


class TrainCLMRFs(object):
    """
    Class for training 3D Haar feature random forests. Given a iterable that 
    returns a training image and training landmarks, this class extracts Haar 
    features from the image at random displacements around the landmarks, then 
    trains a random forest for each landmark using the Haar features and random 
    displacements.
    """

    def __init__(self, nSamples, windowSize, dMax, haarMode='diff', windowSizeVar=None, zShift=True, negSpacing=False):
        """
        input:
        nSamples: integer, number of random samples to take around each landmark point. 
        windowSize: 3-tuple of integers, sample volume size in number of voxels.
        dMax: float, maximum displacement of random sample from p 
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        """

        self.nSamples = nSamples
        self.windowSize = windowSize
        self.dMax = dMax
        self.haarMode = haarMode
        self.windowSizeVar = windowSizeVar

        self.pointDisplacements = None
        self.pointFeatures = None
        self.RFs = []

        self.zShift = zShift
        self.negSpacing = negSpacing

    def setTrainingSamples(self, samples, nPoints):
        """
        Define training samples

        inputs
        sample: iteratble, should return a scan object and a list of coordinates 
                when its next method is called. Coordinates should be in coordinates 
                in physical space which can be mapped to image voxel indices by calling 
                scan.coord2Index.
        nPoints: the number of landmark points per image
        """
        self.trainingSamples = samples
        self.nPoints = nPoints
        self.pointDisplacements = []
        self.pointFeatures = []

    def sampleTrainingImages(self):
        """
        Run sampling process. Each image is loaded and sampled in sequence
        """

        for i, (scan, P) in enumerate(self.trainingSamples):
            log.debug('calculating integral image')
            trainingImage = HaarImage(scan.I, scan.voxelSpacing, scan.voxelOrigin)

            log.debug('sampling features')
            try:
                pointDisplacements, \
                pointFeatures = trainingImage.extractHaarAboutPointRandomMulti(
                    P, self.nSamples, self.windowSize, self.dMax,
                    zShift=self.zShift, negSpacing=self.negSpacing,
                    haarMode=self.haarMode,
                    windowSizeVar=self.windowSizeVar)
            except SamplingError:
                log.debug('WARNING: skipped due to out of bounds sampling')
            else:
                self.pointDisplacements.append(pointDisplacements)
                self.pointFeatures.append(pointFeatures)

        self.pointDisplacements = np.hstack(self.pointDisplacements)
        self.pointFeatures = np.hstack(self.pointFeatures)

        log.debug('displacements shape:', self.pointDisplacements.shape)
        log.debug('features shape:', self.pointFeatures.shape)

    def trainRFs(self, **kwargs):
        """
        trains a RF regressor given features and displacements obtained
        from running sampleTrainingImages.

        inputs:
        **kwargs: keyword arguments for sklearn.ensemble.ExtraTreesRegressor 
        """

        log.debug('training RFs')
        self.RFs = []
        for i in range(self.nPoints):
            sys.stdout.flush()
            sys.stdout.write('\rpoint %5i/%5i' % (i + 1, self.nPoints))
            # print len(self.pointFeatures[i]), self.pointFeatures[i][0]
            # print len(self.pointDisplacements[i]), self.pointDisplacements[i][0]

            pointDisplacements = self.pointDisplacements[i, :, :]
            pointFeatures = self.pointFeatures[i, :, :]

            # print pointDisplacements.shape
            # print pointFeatures.shape

            RF = ExtraTreesRegressor(**kwargs)
            RF = RF.fit(pointFeatures, pointDisplacements)
            self.RFs.append(RF)

    def saveRFs(self, filename):
        """
        Save trained random forests.
        inputs:
        filename: string
        """
        saveRFs(self.RFs, filename)


def saveRFs(RFs, filename):
    """
    Save a list of random forests.
    inputs:
    RFs: a list of sklearn.ensemble.ExtraTreesRegressor instances.
    filename: string
    """
    with open(filename, 'w') as f:
        pickle.dump(RFs, f, protocol=2)


def loadRFs(filename):
    """
    Load a list of random forests.
    inputs:
    filename: string

    return:
    RFs: a list of sklearn.ensemble.ExtraTreesRegressor instances. 
    """
    with open(filename, 'r') as f:
        RFs = pickle.load(f)

    return RFs


# ========================================================#
# voting collecting functions                            #
# ========================================================#

def collectVoteCoMStd(displacementVotes, samplePoints):
    """
    for each landmark, its datapoint is the centre of mass of all points from regressed samples.
    The weight associated with each datapoint is the std of distances forom the centre of mass.
    """
    ######################
    # left-right flip hack
    # displacementVotes[:,0] = displacementVotes[:,0]*-1.0
    ######################
    voteCoords = samplePoints - displacementVotes  # minus because RFs are trained on displacements from landmarks to a sample point, now it is the reverse
    CoM = voteCoords.mean(0)
    std = np.sqrt(((voteCoords - CoM) ** 2.0).sum(1)).std()
    return CoM, std
