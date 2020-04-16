"""
FILE: clm_segmentation.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
classes and functions for segmenting surfaces using a constrained local model.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging
import pickle

import numpy as np

from gias2.image_analysis import haarregressionvoting as HRV


log = logging.getLogger(__name__)


class CLMSegmentationParams(object):
    """
    Parameters container class for CLMSegmentation.
    """
    RFFilename = ''  # filename of random forests
    sampleMode = 'grid'  # how samples are distributed around each landmark
    sampleN = 5  # number of samples to take around each landmark, or number along one edge for grid mode
    sampleWindowSize = [20, 20, 20]  # size length of cubic sample volumes
    sampleDMax = 30  # max distance to take samples from around each landmark
    votingMode = 'comstd'  # how votes from decision trees are processed into data points for fitting
    passMaxDist = 5.0  # distance within which a landmark's datapoint is considered converged
    minPassFrac = 0.8  # minimum fraction of datapoints needed to pass to consider segmentation converged
    imageZShift = False  # do z shift during coord to image index conversion
    imageNegSpacing = False  # do negative spacing during coord to image index conversion
    maxIt = 5  # maximum segmentation iterations
    filterLandmarks = True  # ignore landmarks that are out of bounds or in masked regions.
    haarMode = 'diff'

    def __init__(self, **params):
        if params != None:
            self.setParams(params)

    def setParams(self, params):
        for param, value in list(params.items()):
            self.__setattr__(param, value)


class CLMSegmentation(object):
    """
    Class for constrained local model image segmentation on 3D images using 
    random forests and a mesh-based object representation. 

    self.HRVimage, self.fitMesh, self.getMeshCoords, and self.RFs must be defined 
    on initialisation or via self.setHRVImage, self.setMeshFitter, 
    self.setMeshCoordinatesEvaluator, and self.loadRFs, respectively. 

    Once the above is satisfied, calling self.segment with initial mesh parameters 
    will begin iterative segmentation. In each iteration, self.getMeshCoords is 
    called with mesh parameters to get landmark coords at which to carry out 
    RF regression. Newly segmented points from RF regression will be used in calling 
    self.fitMesh, which should return updated mesh parameters as well as fit errors.

    Iterations terminate when
    - a defined percentage of landmarks have their newly segmented point within a 
      central percentage of their search volume,
    - a maximum number of iterations is reached,
    - or when fitting error has not changed between the last two iterations.
    """
    I = None
    HRVImage = None
    imageSpacing = None
    params = None
    getMeshCoords = None
    fitMesh = None
    meshParamsFinal = None
    RFs = None

    def __init__(self, HRVImage=None, params=None, getMeshCoords=None, fitMesh=None):
        """
        inputs:
        HRVImage: HaarImage instance containing the image to be segmented.
        params: CLMSegmentationParams instance containing segmentation parameters
        getMeshCoords: function for evaluating mesh coordinates
        fitMesh: function for fitting mesh to segmented points
        """

        self.hasMaskedImage = False
        self.filterLandmarks = True

        if HRVImage != None:
            self.setHRVImage(HRVImage)

        self.params = params
        try:
            self.filterLandmarks = self.params.filterLandmarks
        except AttributeError:
            pass

        self.setMeshFitter(fitMesh)
        self.setMeshCoordinatesEvaluator(getMeshCoords)

    def setMeshCoordinatesEvaluator(self, E):
        """
        Define function for evaluating mesh. Function should take as input (meshParams)
        and return a n x 3 array where n is the number of points to segment.
        """
        self.getMeshCoords = E

    def setMeshFitter(self, F):
        """ 
        Define function for fitting mesh. Function should take arguments (data, initialParams, 
        dataWeights) and return (fittedParams, RMS error, error stdev)
        """
        self.fitMesh = F

    def setHRVImage(self, HRVImage):
        """
        Define image for segmentation. Image must be in the form of a HaarImage instance 
        which contains an integral image and a correct coord2Index method.
        """
        self.HRVImage = HRVImage
        self.hasMaskedImage = HRVImage.isMasked
        if self.hasMaskedImage:
            self.filterLandmarks = True

    def loadRFs(self, filename):
        """
        Load the random forests to be used for RF regression voting.
        """
        with open(filename, 'r') as f:
            self.RFs = pickle.load(f)

    def _sampleAndRegressFilter(self, Li, L):
        """ 
        samples image around each landmark coordinate in L, regress against point's RF,
        and collect votes. Indices of landmarks that fail regression are removed from Li.
        """

        if self.params.sampleMode == 'grid':
            disp, features = self.HRVImage.extractHaarAboutPointGridSphereMulti(L,
                                                                                self.params.sampleN,
                                                                                self.params.sampleWindowSize,
                                                                                self.params.sampleDMax,
                                                                                zShift=self.params.imageZShift,
                                                                                negSpacing=self.params.imageNegSpacing,
                                                                                haarMode=self.params.haarMode,
                                                                                )
        elif self.params.sampleMode == 'random':
            disp, features = self.HRVImage.extractHaarAboutPointRandomMulti(L,
                                                                            self.params.sampleN,
                                                                            self.params.sampleWindowSize,
                                                                            self.params.sampleDMax,
                                                                            zShift=self.params.imageZShift,
                                                                            negSpacing=self.params.imageNegSpacing,
                                                                            haarMode=self.params.haarMode,
                                                                            )

        samplePointsTemp = [disp[i] + L[i] for i in range(len(L))]
        regressResults = []
        validLi = []
        samplePoints = []
        for i, li in enumerate(Li):
            if len(features[i]) != 0:
                regressResults.append(self._regressAtLandmark(li, features[i]))
                validLi.append(li)
                samplePoints.append(samplePointsTemp[i])

        return regressResults, samplePoints, validLi

    def _sampleAndRegress(self, L):
        """ 
        samples image around each landmark coordinate in L, regress against point's RF,
        and collect votes
        """

        if self.params.sampleMode == 'grid':
            disp, features = self.HRVImage.extractHaarAboutPointGridSphereMulti(L,
                                                                                self.params.sampleN,
                                                                                self.params.sampleWindowSize,
                                                                                self.params.sampleDMax,
                                                                                zShift=self.params.imageZShift,
                                                                                negSpacing=self.params.imageNegSpacing,
                                                                                sampleMode=self.params.sampleMode,
                                                                                )
        elif self.params.sampleMode == 'random':
            disp, features = self.HRVImage.extractHaarAboutPointRandomMulti(L,
                                                                            self.params.sampleN,
                                                                            self.params.sampleWindowSize,
                                                                            self.params.sampleDMax,
                                                                            zShift=self.params.imageZShift,
                                                                            negSpacing=self.params.imageNegSpacing,
                                                                            sampleMode=self.params.sampleMode,
                                                                            )

        samplePointsTemp = [disp[i] + L[i] for i in range(len(L))]
        regressResults = []
        samplePoints = []
        for i in range(len(L)):
            if len(features[i]) != 0:
                regressResults.append(self._regressAtLandmark(i, features[i]))
                samplePoints.append(samplePointsTemp[i])
            else:
                regressResults.append(None)
                samplePoints.append(None)

        return regressResults, samplePoints

    # def _sampleImageAboutLandmark(self, l):
    #   """
    #   sample image around landmark (grid or random).
    #   Returns a list of feature vectors, and a list of coordinates of the 
    #   centres of the sample volumes.
    #   """
    #   if self.params.sampleMode=='grid':
    #       disp, features = self.HRVImage.extractHaarAboutPointGridSphere(l,\
    #                                                                self.params.sampleN,\
    #                                                                self.params.sampleWindowSize,\
    #                                                                self.params.sampleDMax,\
    #                                                                zShift=self.params.imageZShift,\
    #                                                                negSpacing=self.params.imageNegSpacing,\
    #                                                                )
    #   elif self.params.sampleMode=='random':
    #       disp, features = self.HRVImage.extractHaarAboutPointRandom(l,\
    #                                                                  self.params.sampleN,\
    #                                                                  self.params.sampleWindowSize,\
    #                                                                  self.params.sampleDMax,\
    #                                                                  zShift=self.params.imageZShift,\
    #                                                                  negSpacing=self.params.imageNegSpacing,\
    #                                                                  )

    #   if len(features)!=0:
    #       features = np.array(features)
    #       P = np.array(disp) + l          # physical coords
    #   else:
    #       P = []

    #   return features, P

    def _regressAtLandmark(self, li, F):
        """
        run RF regression for landmark li using a list of features F. li is an integer.
        """
        return np.array([self.RFs[li].predict(f) for f in F]).squeeze()

    def _processVotesFilter(self, regressResults, samplePoints):
        """
        for each landmark, process its regressed votes into data points
        and weights. Indices of landmarks that have failed regression are removed.
        """
        if self.params.votingMode == 'comstd':
            voteProcessor = HRV.collectVoteCoMStd
        else:
            raise NotImplementedError

        dataPoints = np.zeros((len(regressResults), 3), dtype=float)
        weights = np.zeros((len(regressResults),), dtype=float)
        maskPoints = []
        for i in range(len(regressResults)):
            dataPoints[i], weights[i] = voteProcessor(regressResults[i], samplePoints[i])

        log.debug('dataPoints shape', dataPoints.shape)
        log.debug('weights shape', weights.shape)
        if self.params.votingMode == 'comstd':
            # convert stds into useful fitting weights (higher SD, lower weight, normalise to 0-1)
            weights = 1.0 / (weights / weights.max())

        # weights for out of bound points
        # weights[maskPoints] = 0.0

        return dataPoints, weights

    def _processVotes(self, regressResults, samplePoints, landmarks):
        """
        for each landmark, process its regressed votes into data points
        and weights
        """
        if self.params.votingMode == 'comstd':
            voteProcessor = HRV.collectVoteCoMStd

        dataPoints = np.zeros((len(regressResults), 3), dtype=float)
        weights = np.zeros((len(regressResults),), dtype=float)
        maskPoints = []
        for i in range(len(regressResults)):
            if regressResults[i] == None:
                # out of bound points
                dataPoints[i], weights[i] = landmarks[i], 0.0
                maskPoints.append(i)
            else:
                dataPoints[i], weights[i] = voteProcessor(regressResults[i], samplePoints[i])

        if self.params.votingMode == 'comstd':
            # convert stds into useful fitting weights (higher SD, lower weight, normalise to 0-1)
            weights = 1.0 / (weights / weights.max())

        # weights for out of bound points
        weights[maskPoints] = 0.0

        return dataPoints, weights

    def seg1Landmark(self, L, li):
        """
        segment just one landmark.

        inputs:
        L: 3-tuple of landmark coordinates
        li: integer, index of the landmark with respect to all landmarks

        returns:
        features: list of image features sampled around L
        samplePoints: coordinates of points around L where features were extracted
        regressResults: results from the RF regression
        dataPoint: the segmented data point
        weight: weighting for the segmented datapoint
        """

        features, samplePoints = self._sampleImageAboutLandmark(L)
        regressResults = self._regressAtLandmark(li, features)

        if self.params.votingMode == 'comstd':
            voteProcessor = HRV.collectVoteCoMStd
        else:
            raise NotImplementedError

        dataPoint, weight = voteProcessor(regressResults, samplePoints)

        return features, samplePoints, regressResults, dataPoint, weight

    def _filterValidLandmarks(self, landmarks):
        landmarkIndices = np.arange(len(landmarks))

        # first reject any point outside of the image volume
        inds = self.HRVImage.coord2Index(np.array(landmarks), \
                                         zShift=self.params.imageZShift, \
                                         negSpacing=self.params.imageNegSpacing)

        if self.hasMaskedImage:
            validLandmarkIndices = [li for li in landmarkIndices if (
                        self.HRVImage.checkIndexInBounds(inds[li]) and not self.HRVImage.checkIndexIsMasked(inds[li]))]
            validLandmarks = landmarks[validLandmarkIndices, :]
        else:
            validLandmarkIndices = [li for li in landmarkIndices if self.HRVImage.checkIndexInBounds(inds[li])]
            validLandmarks = landmarks[validLandmarkIndices, :]

        return validLandmarkIndices, validLandmarks

    def segment(self, meshParams0, verbose=1, debug=0):
        """
        Run the main segmentation loop. See class docstring for general overview.
        Due to masked images or out of FOV objects, the number of segmented dataPoints (n')
        may be less than the number of landmarks (n) evaluated from the mesh.

        inputs:
        meshParams0: initial mesh parameters

        returns:
        self.meshParamsFinal: best mesh parameters
        data: n' x 3 array, final segmented datapoints
        W: length n' array, weights for each segmented datapoint
        goodLandmarkIndices: indices of segmented datapoints wrt to all 
                             landmarks evaluated from the mesh
        rmsFinal: Best rms mesh fit error as reported by self.fitMesh
        sdFinal: Best mesh fit error s.d. as reported by self.fitMesh
        passFrac: Best percentage of landmarks with segmented point within tolerable distance
        outputHistory: A dictionary of variables recorded at each iteration.
        """

        log.debug('\nstarting segmentation')
        it = 0
        meshRMSOld = -1.0
        meshSDOld = -1.0
        meshParams = np.array(meshParams0)
        converged = False
        outputHistory = {'meshParams': [],
                         'meshRMS': [],
                         'meshSD': [],
                         'passFrac': [],
                         'mahaDist': [],
                         }

        dataHistory = {'data': [],
                       'W': [],
                       'goodLandmarkIndices': [],
                       }

        # in each iteration:
        while it < self.params.maxIt:

            # evaluation landmark coordinates
            landmarks = self.getMeshCoords(meshParams)

            if debug:
                log.debug('landmark bounds:')
                log.debug(landmarks.min(0))
                log.debug(landmarks.max(0))

            # filter out out-of-bounds landmarks and landmarks in masked image regions in using a masked image
            if self.filterLandmarks:
                validLandmarkIndices, validLandmarks = self._filterValidLandmarks(landmarks)
            else:
                validLandmarks = landmarks
                goodLandmarks = landmarks
                validLandmarkIndices = np.arange(len(landmarks))
                goodLandmarkIndices = validLandmarkIndices

            # sample and regress at each landmark
            if self.filterLandmarks:
                regressResults, samplePoints, goodLandmarkIndices = self._sampleAndRegressFilter(validLandmarkIndices,
                                                                                                 validLandmarks)
                goodLandmarks = landmarks[goodLandmarkIndices]
            else:
                regressResults, samplePoints = self._sampleAndRegress(validLandmarks)

            if debug:
                log.debug('n regressResults', len(regressResults))
                log.debug('n samplePoints', len(samplePoints))
                log.debug('n goodLandmarks', len(goodLandmarks))
                log.debug('n goodLandmarkIndices', len(goodLandmarkIndices))

            # post-process regression results (votes or datapoints) for fitting
            if self.filterLandmarks:
                data, W = self._processVotesFilter(regressResults, samplePoints)
            else:
                data, W = self._processVotes(regressResults, samplePoints, landmarks)

            if debug:
                log.debug('data bounds:')
                log.debug(data.min(0))
                log.debug(data.max(0))

            # fit shape model
            newMeshParams, meshRMS, meshSD = self.fitMesh(data, meshParams.copy(), W, goodLandmarkIndices)
            # mahaDist = np.sqrt((newMeshParams[6:]**2.0).sum())    # assuming mesh params are tx,ty,tz,rx,ry,rz,pc1,pc2,...

            if debug:
                log.debug('newMeshParams shape', newMeshParams.shape)
                log.debug('fitted bounds:')
                log.debug(newMeshParams.min(0))
                log.debug(newMeshParams.max(0))

            # check against stopping criteria
            stopSeg, passFrac = self._stopCritCoMStd(goodLandmarks, data)

            # record results
            outputHistory['meshParams'].append(newMeshParams)
            outputHistory['meshRMS'].append(meshRMS)
            outputHistory['meshSD'].append(meshSD)
            outputHistory['passFrac'].append(passFrac)
            # outputHistory['mahaDist'].append(mahaDist)
            dataHistory['data'].append(data)
            dataHistory['W'].append(W)
            dataHistory['goodLandmarkIndices'].append(goodLandmarkIndices)

            if verbose:
                log.debug('\nit: %(it)03i  passFrac: %(pFrac)5.3f  MeshRMS: %(meshRMS)5.3f  MeshSD: %(meshSD)5.3f\n' \
                      % {'it': it, 'pFrac': passFrac, 'meshRMS': meshRMS, 'meshSD': meshSD})
                # print 'mesh params: '+' '.join( ['%(0)2.3f'%{'0':i} for i in newMeshParams] )

            # prepare for next iteration
            it += 1
            if stopSeg or (meshRMS == meshRMSOld):
                converged = True
                break
            else:
                meshRMSOld = meshRMS
                meshSDOld = meshSD
                meshParams = newMeshParams

        # if converged, use latest outputs
        if converged:
            self.meshParamsFinal = meshParams.copy()
            rmsFinal = meshRMS
            sdFinal = meshSD
        else:
            # use highest cFrac params
            bestIt = np.argmax(outputHistory['passFrac'])
            if verbose:
                log.debug('using results from iteration', bestIt + 1)
            self.meshParamsFinal = outputHistory['meshParams'][bestIt]
            rmsFinal = outputHistory['meshRMS'][bestIt]
            sdFinal = outputHistory['meshSD'][bestIt]
            passFrac = outputHistory['passFrac'][bestIt]
            data = dataHistory['data'][bestIt]
            W = dataHistory['W'][bestIt]
            goodLandmarkIndices = dataHistory['goodLandmarkIndices'][bestIt]
        if verbose:
            log.debug('DONE')

        return self.meshParamsFinal, data, W, goodLandmarkIndices, rmsFinal, sdFinal, passFrac, outputHistory

    def _stopCritCoMStd(self, landmarks, data):
        """
        stopping criteria, based on how central the votes are in the sampling volume.
        Stop if x percent of all points are within y of landmarks
        """
        dist = np.sqrt(((landmarks - data) ** 2.0).sum(1))
        nPass = (dist < self.params.passMaxDist).sum()
        passFrac = float(nPass) / len(data)

        return (passFrac >= self.params.minPassFrac), passFrac
