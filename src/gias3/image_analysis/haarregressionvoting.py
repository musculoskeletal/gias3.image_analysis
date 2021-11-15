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

from gias3.image_analysis import haar
from gias3.image_analysis import image_tools
from gias3.image_analysis import integralimage

log = logging.getLogger(__name__)


class SamplingWarning(Exception):
    pass


class SamplingError(Exception):
    pass


def extractHaarFeatures(II, p, window_size, haar_mode='diff'):
    X = np.round(p - np.array(window_size) / 2.0).astype(int)
    if haar_mode == 'diff':
        return haar.extractAllHaar3DDiff(II, X, window_size)
    elif haar_mode == 'reldiff':
        return haar.extractAllHaar3DRelDiff(II, X, window_size)
    elif haar_mode == 'sign':
        return haar.extractAllHaar3DSign(II, X, window_size)


def makeHaarFeatureExtractor(II, haar_mode):
    if haar_mode == 'diff':
        def haarFeatureExtractor(p, w):
            X = np.round(p - w / 2.0).astype(int)
            return haar.extractAllHaar3DDiff(II, X.T, w.T.astype(int))
    elif haar_mode == 'reldiff':
        def haarFeatureExtractor(p, w):
            X = np.round(p - w / 2.0).astype(int)
            return haar.extractAllHaar3DRelDiff(II, X.T, w.T.astype(int))
    elif haar_mode == 'sign':
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

    def __init__(self, I=None, voxel_spacing=None, voxel_origin=None, is_masked=False):
        """
        instantiation input args:
        I: 3D numpy array
        voxelSpacing: three tuple
        voxelOrigin: three tuple
        isMasked: boolean, where I is a masked array
        haarMode: ['diff', 'reldiff', 'sign'] how haar features are calculated
        """

        self._displacementGrids = {}
        self.isMasked = is_masked
        if I is not None:
            self._setImageArray(I, voxel_spacing, voxel_origin)

        self.haarFeatureExtractor = None
        self.haarFeatureExtractorHaarMode = None

    def __del__(self):
        del self.I
        del self.II

    def _setImageArray(self, I, voxel_spacing=None, voxel_origin=None):
        if len(I.shape) != 3:
            raise ValueError('image must be 3D')

        self.I = I
        self.II = integralimage.IntegralImage3(self.I)

        if voxel_spacing is None:
            self.voxelSpacing = np.array([1.0, 1.0, 1.0])
        else:
            self.voxelSpacing = voxel_spacing

        if voxel_origin is None:
            self.voxelOrigin = np.array([0.0, 0.0, 0.0])
        else:
            self.voxelOrigin = voxel_origin

    def setHaarExtractor(self, haar_mode):
        self.haarFeatureExtractorHaarMode = haar_mode
        self.haarFeatureExtractor = makeHaarFeatureExtractor(self.II, haar_mode)

    def extractHaarAboutPoint(self, p, window_size, sample_mode='diff'):
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
        window_size = np.array(window_size)
        X = np.round(p - window_size / 2.0).astype(int)
        if sample_mode == 'diff':
            return haar.extractAllHaar3DDiff(self.II, X, window_size)
        elif sample_mode == 'reldiff':
            return haar.extractAllHaar3DRelDiff(self.II, X, window_size)
        elif sample_mode == 'sign':
            return haar.extractAllHaar3DSign(self.II, X, window_size)
        else:
            raise ValueError('invalid sampleMode')

    def extractHaarAboutPointRandom(self, p, n, window_size, d_max, z_shift=False, neg_spacing=False, haar_mode='diff'):
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

        window_size = np.array(window_size)
        p = np.array(p)
        displacements = np.random.uniform(low=-d_max, high=d_max, size=(n, 3))
        sampleIndices = self.coord2Index(p + displacements, z_shift, neg_spacing)
        # randomly modify windowSize too?
        # randomly alter orientation?

        features = []
        for i, sInd in enumerate(sampleIndices):
            # print sInd
            try:
                features.append(extractHaarFeatures(self.II, sInd, window_size, haar_mode))
            except IndexError:
                retry = 1
                while retry:
                    log.debug('retry', retry)
                    dRetry = np.random.uniform(low=-d_max, high=d_max, size=(3))
                    sIndRetry = self.coord2Index(p + dRetry, z_shift, neg_spacing)

                    try:
                        features.append(extractHaarFeatures(self.II, sIndRetry, window_size, haar_mode))
                    except IndexError:
                        retry += 1
                    else:
                        displacements[i] = dRetry
                        retry = 0
                        log.debug(sIndRetry)

        if len(features) == 0:
            warnings.warn("No suitable sampling locations, p = " + str(p))

        return displacements, features

    def extractHaarAboutPointRandomMulti(self, P, n, window_size, d_max, z_shift=False, neg_spacing=False, haar_mode='diff',
                                         window_size_var=None):
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
        window_size = np.array(window_size)
        maxRetry = 10000

        # generate window sizes
        if window_size_var is not None:
            windowSizes = window_size * np.random.uniform(low=1.0 - window_size_var,
                                                          high=1.0 + window_size_var,
                                                          size=nSamples)[:, np.newaxis]
        else:
            windowSizes = window_size * np.ones(nSamples)[:, np.newaxis]

        windowSizes = windowSizes.reshape((nPoints, n, 3))
        windowSizes2 = windowSizes / 2.0

        # generate displacements
        displacements = np.random.uniform(low=-d_max, high=d_max, size=(nSamples, 3)).reshape(
            (nPoints, n, 3))  # shape = (nPoints, samples per point, 3)
        samplePoints = displacements + P[:, np.newaxis, :]
        sampleIndices = self.coord2Index(samplePoints.reshape((nSamples, 3)), z_shift, neg_spacing).reshape(
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
                        dRetry = np.random.uniform(low=-d_max, high=d_max, size=(3))
                        indRetry = self.coord2Index(P[pi] + dRetry, z_shift, neg_spacing)
                        # regen window size
                        if window_size_var is not None:
                            wsRetry = window_size * np.random.uniform(low=1.0 - window_size_var,
                                                                      high=1.0 + window_size_var)
                        else:
                            wsRetry = window_size

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
        self.setHaarExtractor(haar_mode)
        features = self.haarFeatureExtractor(sampleIndices.reshape((nSamples, 3)),
                                             windowSizes.reshape((nSamples, 3))).T
        # shape = (nPoints, n, number of features)
        features = features.reshape((nPoints, n, -1))

        if len(features) == 0:
            warnings.warn("No suitable sampling locations")

        return displacements, features

    def extractHaarAboutPointGridSphere(self, p, n, window_size, d_max, z_shift=False, neg_spacing=False):
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
            displacements = self._displacementGrids[(d_max, n)]
        except KeyError:
            displacements = _generateSampleGrid(d_max, n)
            self._displacementGrids[(d_max, n)] = displacements

        # sample
        sampleIndices = self.coord2Index(p + displacements, z_shift, neg_spacing)

        features = []
        featureDisplacements = []
        for i, d in enumerate(sampleIndices):
            try:
                features.append(extractHaarFeatures(self.II, d, window_size))
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

    def extractHaarAboutPointGridSphereMulti(self, P, n, window_size, d_max, z_shift=False, neg_spacing=False,
                                             haar_mode='diff'):
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
        window_size = np.array(window_size)
        windowSize2 = window_size / 2.0
        self.setHaarExtractor(haar_mode)

        try:
            disp = self._displacementGrids[(d_max, n)]
        except KeyError:
            disp = _generateSampleGrid(d_max, n)
            self._displacementGrids[(d_max, n)] = disp

        # generate sampling points
        sampleIndices = []
        displacements = []
        nSamples = []
        for p in P:
            samplePoints = p + disp
            sampleIndicesTemp = self.coord2Index(samplePoints, z_shift, neg_spacing)
            inBounds = np.array([(self.checkIndexInBounds(ind + windowSize2) and
                                  self.checkIndexInBounds(ind - windowSize2)) for ind in sampleIndicesTemp])
            inBoundsI = np.where(inBounds is True)[0]
            sampleIndices.append(sampleIndicesTemp[inBoundsI, :])
            displacements.append(disp[inBoundsI, :])
            nSamples.append(sampleIndices[-1].shape[0])

        sampleIndicesFlat = np.vstack(sampleIndices)
        windowSizes = window_size + np.zeros_like(sampleIndicesFlat)
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


def _generateSampleGrid(d_max, n):
    # make cube grid about origin
    x, y, z = np.mgrid[-d_max:d_max:complex(0, n),
              -d_max:d_max:complex(0, n),
              -d_max:d_max:complex(0, n),
              ]
    X = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # filter out grid points outside sphere
    d = np.sqrt((X ** 2.0).sum(1))
    displacements = X[np.where(d <= d_max)[0]]
    return displacements


class TrainCLMRFs(object):
    """
    Class for training 3D Haar feature random forests. Given a iterable that 
    returns a training image and training landmarks, this class extracts Haar 
    features from the image at random displacements around the landmarks, then 
    trains a random forest for each landmark using the Haar features and random 
    displacements.
    """

    def __init__(self, n_samples, window_size, d_max, haar_mode='diff', window_size_var=None, z_shift=True, neg_spacing=False):
        """
        input:
        nSamples: integer, number of random samples to take around each landmark point. 
        windowSize: 3-tuple of integers, sample volume size in number of voxels.
        dMax: float, maximum displacement of random sample from p 
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        """

        self.nSamples = n_samples
        self.windowSize = window_size
        self.dMax = d_max
        self.haarMode = haar_mode
        self.windowSizeVar = window_size_var

        self.pointDisplacements = None
        self.pointFeatures = None
        self.RFs = []

        self.zShift = z_shift
        self.negSpacing = neg_spacing

    def setTrainingSamples(self, samples, n_points):
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
        self.nPoints = n_points
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
                    z_shift=self.zShift, neg_spacing=self.negSpacing,
                    haar_mode=self.haarMode,
                    window_size_var=self.windowSizeVar)
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

def collectVoteCoMStd(displacement_votes, sample_points):
    """
    for each landmark, its datapoint is the centre of mass of all points from regressed samples.
    The weight associated with each datapoint is the std of distances forom the centre of mass.
    """
    ######################
    # left-right flip hack
    # displacementVotes[:,0] = displacementVotes[:,0]*-1.0
    ######################
    voteCoords = sample_points - displacement_votes  # minus because RFs are trained on displacements from landmarks to a sample point, now it is the reverse
    CoM = voteCoords.mean(0)
    std = np.sqrt(((voteCoords - CoM) ** 2.0).sum(1)).std()
    return CoM, std
