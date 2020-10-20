"""
FILE: asm_segmentation.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
classes and functions for segmenting surfaces using active-shape modelling

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging
import sys
import time
import numpy
from scipy import ndimage

from gias2.learning import PCA

log = logging.getLogger(__name__)

try:
    from matplotlib import pyplot as plot
    import matplotlib as mpl
except ImportError:
    log.debug('No Matplotlib, plotting functions will not work.')

usePyxScan = True


def genSamplingPoints(X, N, d, xLim):
    """ at each point in X[i], in direction N[i], calc the coordinates
    of d points between X[i]+xLim[0]*N[i] and x[i]+xLim[1]*N[i]
    """
    t = numpy.linspace(xLim[0], xLim[1], d)
    p = (t[:, numpy.newaxis, numpy.newaxis] * N + X).transpose((1, 0, 2))
    return p


def sampleImage(image, x):
    """ samples image at set of point lists x. x.shape(nLists,pointsperlist,3)
    """
    # linear interpolation, therefore prefilter=false.
    return ndimage.map_coordinates(image, numpy.transpose(x, (2, 0, 1)),
                                   output=float, order=1, mode='nearest',
                                   prefilter=False)


def calcDerivNormalise(P):
    """ calculates the derivative of signals P, normalise against 
    sum of absolution values of signal.
    """
    dP = P[:, 1:] - P[:, :-1]
    dP = numpy.hstack((dP, dP[:, -1, numpy.newaxis]))
    dP[:, 1:-1] = (dP[:, 1:-1] + dP[:, :-2]) / 2.0  # average back and forward diff for interior points
    dP = dP / abs(dP).sum(1)[:, numpy.newaxis]
    dP = numpy.where(numpy.isfinite(dP), dP, 0.0)
    return dP


def calcDerivArray(P):
    """ calculates the derivative of signal P
    """
    dP = P[:, 1:] - P[:, :-1]
    dP = numpy.hstack((dP, dP[:, -1][:, numpy.newaxis]))
    dP[:, 1:-1] = (dP[:, 1:-1] + dP[:, :-2]) / 2.0  # average back and forward diff for interior points
    return dP


def _calcPPCPModes(epI, PPC, cutOff):
    pModes = []
    for i in epI:
        # calculate number of modes to use for profile matching
        cumSpec = numpy.cumsum(PPC[i].getNormSpectrum())
        pModes.append(numpy.arange(numpy.where(cumSpec > cutOff)[0][0]))

    return pModes


def _asmStopCrit(matchIndices, nSamples, window=0.1, threshold=0.95):
    """ returns true if thresh proportion of xpads are in the middle
    window proportion of pLength
    """
    n0 = nSamples / 2.0 - nSamples * window
    n1 = nSamples / 2.0 + nSamples * window

    ##
    # print n0, n1
    ##

    passes = ((n0 <= matchIndices) & (matchIndices <= n1)).sum()
    passFrac = float(passes) / len(matchIndices)
    if passFrac > threshold:
        return True, passFrac
    else:
        return False, passFrac


def weightMDist(m, upper):
    """ assign weights to datapoints for fitting based on their
    MDistance
    """
    mCap = numpy.where(m > upper, upper, m)
    W = 1.0 - mCap / upper  # 2x to shaft data weighted halfway
    return W


if usePyxScan:
    import numpy
    import pyximport

    pyximport.install(
        setup_args={"include_dirs": numpy.get_include()},
        language_level=3
    )
    from gias2.image_analysis import asm_search_c

    # reload(asm_search_c)
    scanProfile = asm_search_c.scanProfile
    profileSearchElementPoints = asm_search_c.profileSearchElementPoints
    profileSearchElementOneSide = asm_search_c.profileSearchElementOneSide
    profileSearchElementMedian = asm_search_c.profileSearchElementMedian
    log.debug('using cython for search')
else:
    from .asm_search import *

    log.debug('using python for search')


# =====================================================================#
class ASMSegmentationParams(object):
    """
    Parameters container class for ASMSegmentation.
    """
    PPCFilename = ''  # filename of appearance pc file
    GD = [10, 10]
    ND = 40  # number of samples per profile
    NLim = [-20.0,
            20.0]  # (min, max) of extent of MODEL profile samples, e.g. (-10,10) for -10mm to 10mm either side of mesh
    NPad = 20  # extra sample points either side of profile to sample
    NRes = None  # auto-generated
    NL = None  # auto-generated
    matchMode = 'default'  # 'elementmedian', 'oneside', or 'default'
    MDistWeight = True  # use mahalanobis distance of profile matches to weight its datapoint
    MDistWeightUpper = 5.0  # datapoints with m distance above this are weighted zero
    passWindow = 0.1  # central proportion of a profile in which search is considered converged
    minPassFrac = 0.9  # minimum fraction of datapoints needed to pass to consider segmentation converged
    maxIt = 10  # maximum segmentation iterations
    filterLandmarks = True  # ignore landmarks that are out of bounds or in masked regions.
    imageZShift = False  # do z shift during coord to image index conversion
    imageNegSpacing = False  # do negative spacing during coord to image index conversion
    verbose = 1
    PPCVarCutoff = 0.9

    def __init__(self, **params):
        if params != None:
            self.setParams(params)

        self._postProcess()

    def setParams(self, params):
        for param, value in list(params.items()):
            self.__setattr__(param, value)

        self._postProcess()

    def _postProcess(self):
        self.NLim = numpy.array(self.NLim)
        self.NRes = (self.NLim[1] - self.NLim[0]) / self.ND
        self.NL = self.NLim[1] - self.NLim[0]


class ASMSegmentation(object):
    PPC = None
    I = None
    spacing = None
    getMeshCoords = None
    getMeshNormals = None
    elementXIndices = None
    elementXIndicesFlat = None
    fitMesh = None

    def __init__(self, image=None, params=None, getMeshCoords=None, getMeshNormals=None, fitMesh=None):

        self.hasMaskedImage = False
        self.filterLandmarks = True

        if image != None:
            self.setImage(image)

        self.params = params
        try:
            self.filterLandmarks = self.params.filterLandmarks
        except AttributeError:
            pass

        self.setMeshCoordinatesEvaluator(getMeshCoords)
        self.setMeshNormalEvaluator(getMeshNormals)
        self.setMeshFitter(fitMesh)

    def setMeshCoordinatesEvaluator(self, E):
        self.getMeshCoords = E

    def setMeshNormalEvaluator(self, E):
        self.getMeshNormals = E

    def setElementXIndices(self, I):
        self.elementXIndices = I
        self.elementXIndicesFlat = numpy.hstack(self.elementXIndices)

    def setMeshFitter(self, F):
        """ function F should take arguments (data, initialParams, dataWeights)
        """
        self.fitMesh = F

    def setProfilePC(self, P):
        self.PPC = P

    def loadProfilePC(self, filename=None):
        if filename == None:
            filename = self.params.PPCFilename

        self.PPC = PCA.PCList()
        self.PPC.load(self.params.PPCFilename)

    def setImage(self, image):
        self.image = image
        self.hasMaskedImage = self.image.isMasked
        if self.hasMaskedImage:
            self.filterLandmarks = True

    def _evaluateLandmarks(self, meshParams):
        self.XN = self.getMeshNormals(meshParams)
        self.XMesh = self.getMeshCoords(meshParams)

    def _filterValidLandmarks(self, landmarks):
        # landmarkIndices = numpy.arange(len(landmarks))

        # first reject any point outside of the image volume
        inds = self.image.coord2Index(numpy.array(landmarks), \
                                      zShift=self.params.imageZShift, \
                                      negSpacing=self.params.imageNegSpacing)

        if self.hasMaskedImage:
            landmarkMask = numpy.array(
                [(self.image.checkIndexInBounds(l) and not self.image.checkIndexIsMasked(l)) for l in inds], dtype=bool)
        else:
            landmarkMask = numpy.array([self.image.checkIndexInBounds(l) for l in inds], dtype=bool)

        return landmarkMask

    def _sampleImage(self):
        """ samples image along mesh normals at GF material points with
        the current set of field parameters. Smoothes samples and
        calculates the derivatives
        """

        nLandmarks = self.XMesh.shape[0]
        nSamples = self.params.ND + self.params.NPad * 2

        # generated in real coords
        # shape = (nlandmarks, nsamples, 3)
        self.XSample = genSamplingPoints(self.XMesh,
                                         self.XN,
                                         nSamples,
                                         [self.params.NLim[0] - self.params.NPad * self.params.NRes,
                                          self.params.NLim[1] + self.params.NPad * self.params.NRes],
                                         )

        # convert to image coordinates
        # self.XSampleImg = self.image.coord2Index(self.XSample,
        #                   zShift=self.params.imageZShift,
        #                   negSpacing=self.params.imageNegSpacing,
        #                   )
        self.XSampleImg = self.image.coord2Index(
            numpy.vstack(self.XSample),
            zShift=self.params.imageZShift,
            negSpacing=self.params.imageNegSpacing,
            roundInt=False,
        ).reshape((nLandmarks, nSamples, -1))

        # self.P = sampleImage( self.image.I, self.XSampleImg )

        self.P = ndimage.map_coordinates(
            self.image.I,
            numpy.vstack(self.XSampleImg).T,
            output=float, order=1, mode='nearest',
            prefilter=False).reshape((nLandmarks, -1))

        self.dP = calcDerivArray(self.P)

    def _match2data(self, matchInd, landmarkMask):

        validLandmarks = numpy.where(landmarkMask)[0]
        XSampleValid = self.XSample[landmarkMask, :]
        data = XSampleValid[numpy.arange(len(validLandmarks), dtype=int), matchInd]
        return data

    def segment(self, meshParams0, verbose=1, debug=0, callback=None):
        """
        Run the main segmentation loop.
        meshParams0: array - initial mesh parameters
        """

        if (len(self.PPC.L) - 1) < max(self.elementXIndicesFlat):
            raise ValueError('Maximum landmark index ({}) greater than number of profile models ({}). Check PPC.' \
                             .format(max(self.elementXIndicesFlat), (len(self.PPC.L) - 1)))

        log.debug('\nstarting segmentation')
        it = 0
        mRMSOld = 0.0
        meshRMSOld = 0.0
        meshSDOld = 0.0
        meshParams = numpy.array(meshParams0)
        converged = False
        outputHistory = {'meshParams': [],
                         'meshRMS': [],
                         'meshSD': [],
                         'passFrac': [],
                         'mahaDist': [],
                         'mRMS': [],
                         }

        dataHistory = {'data': [],
                       'W': [],
                       'm': [],
                       'M': [],
                       'landmarkMask': [],
                       }

        # calculate the modes needed for each profile PC based on pSpecCutOff
        ppcModes = self.PPC.getModesFracVariance(self.params.PPCVarCutoff)

        # in each iteration:
        while it < self.params.maxIt:

            if debug:
                t0 = time.time()
                tprev = time.time()

            # evaluate landmark positions and normals
            self._evaluateLandmarks(meshParams)
            if debug:
                log.debug('landmark eval done (%6.3fs)' % (time.time() - tprev))
                tprev = time.time()

            # sample image along normals at a particular GD, ND, NLim for valid landmarks
            # self.P and self.dP are of shape (n valid landmarks, profile length)
            self._sampleImage()
            if debug:
                log.debug('image sampling done (%6.3fs)' % (time.time() - tprev))
                tprev = time.time()

            # filter out out-of-bounds landmarks and landmarks in masked image regions in using a masked image
            if self.filterLandmarks:
                landmarkMask = self._filterValidLandmarks(self.XMesh)
                if not numpy.any(landmarkMask):
                    raise RuntimeError('All landmarks masked')
            else:
                landmarkMask = numpy.ones(self.XMesh.shape[0], dtype=bool)

            if debug:
                log.debug('landmark filtering done (%6.3fs)' % (time.time() - tprev))
                tprev = time.time()

            # match profiles
            if self.params.matchMode == 'default':
                landmarkIndices = numpy.where(landmarkMask)[0]
                matchInd, m, M = profileSearchElementPoints(landmarkIndices, self.PPC.L, ppcModes, self.dP)
            elif self.params.matchMode == 'oneside':
                matchInd, m, M = profileSearchElementOneSide(self.elementXIndices, self.PPC.L, ppcModes, self.dP)
            elif self.params.matchMode == 'elementmedian':
                matchInd, m, M = profileSearchElementMedian(self.elementXIndices, self.PPC.L, ppcModes, self.dP,
                                                            landmarkMask, 1.0)
            else:
                raise ValueError('unrecognised matchMode')

            if debug:
                log.debug('profile search done (%6.3fs)' % (time.time() - tprev))
                tprev = time.time()

            # convert best match positions to data points
            data = self._match2data(matchInd, landmarkMask)
            if debug:
                log.debug('match to data done (%6.3fs)' % (time.time() - tprev))
                tprev = time.time()

            # rigid + mode fit GF to data points
            if self.params.MDistWeight:
                W = weightMDist(m, self.params.MDistWeightUpper)
            else:
                W = numpy.ones(len(m))

            # debug
            if debug:
                log.debug('profile search x (1st 10):', matchInd[:10])
                # ~ print 'profile search m:', m
                # ~ print 'profile search M:', M
                # ~ print 'profile data:', data
                # ~ print 'profile W:', W

            newMeshParams, meshRMS, meshSD = self.fitMesh(data, x0=meshParams.copy(),
                                                          weights=W,
                                                          landmarkIndices=numpy.where(landmarkMask)[0])

            if debug:
                log.debug('mesh fit done (%6.3fs)' % (time.time() - tprev))
                tprev = time.time()

            dx = matchInd
            stopSeg, passFrac = _asmStopCrit(matchInd,
                                             self.params.ND + 2 * self.params.NPad,
                                             window=self.params.passWindow,
                                             threshold=self.params.minPassFrac)

            mRMS = numpy.sqrt(m.mean())
            outputHistory['meshParams'].append(newMeshParams)
            outputHistory['meshRMS'].append(meshRMS)
            outputHistory['meshSD'].append(meshSD)
            outputHistory['passFrac'].append(passFrac)
            outputHistory['mRMS'].append(mRMS)
            dataHistory['data'].append(data)
            dataHistory['W'].append(W)
            dataHistory['m'].append(m)
            dataHistory['M'].append(M)
            dataHistory['landmarkMask'].append(landmarkMask)

            it += 1

            if verbose:
                log.debug(
                    '\nit: %(it)03i  M-distance RMS: %(mRMS)5.3f  passFrac: %(passFrac)5.3f  MeshRMS: %(meshRMS)5.3f  MeshSD: %(meshSD)5.3f' \
                    % {'it': it, 'mRMS': mRMS, 'passFrac': passFrac, 'meshRMS': meshRMS, 'meshSD': meshSD})
                log.debug('mesh params: ' + ' '.join(['%(0)2.3f' % {'0': i} for i in newMeshParams]))

            if callback:
                callback(newMeshParams, data, meshRMS, passFrac)

            if stopSeg or ((mRMS == mRMSOld) and (meshRMS == meshRMSOld)):
                converged = True
                break
            else:
                mRMSOld = mRMS
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
            bestIt = numpy.argmax(outputHistory['passFrac'])
            if verbose:
                log.debug('using results from iteration', bestIt + 1)
            self.meshParamsFinal = outputHistory['meshParams'][bestIt]
            rmsFinal = outputHistory['meshRMS'][bestIt]
            sdFinal = outputHistory['meshSD'][bestIt]
            passFrac = outputHistory['passFrac'][bestIt]
            data = dataHistory['data'][bestIt]
            W = dataHistory['W'][bestIt]
            m = dataHistory['m'][bestIt]
            M = dataHistory['M'][bestIt]
            landmarkMask = dataHistory['landmarkMask'][bestIt]
        if verbose:
            log.debug('DONE')

        return self.meshParamsFinal, data, W, landmarkMask, \
               rmsFinal, sdFinal, passFrac, m, M, outputHistory

    def showProfileMatch(self, profileI, nPModes):
        PData = self.P[profileI]
        dPData = self.dP[profileI]
        ppc = self.PPC.L[profileI]
        dPMean = ppc.getMean()
        mx, m, M = scanProfile(dPData, ppc, list(range(nPModes)))

        f = plot.figure()
        ax1 = f.add_subplot(411)
        ax2 = f.add_subplot(412)
        ax3 = f.add_subplot(413)
        ax4 = f.add_subplot(414)

        ax1.plot(PData)
        ax2.plot(dPData)
        ax3.plot(dPMean)
        ax4.plot(M)
        plot.show()

        log.debug('match index:', mx)
        log.debug('match M-Distance:', m)


# ==========================================================================#
class TrainASMPPCs(object):
    """
    Class for training ASM profile PCs. Given a iterable that returns a 
    training image, training landmarks, and normals, this class extracts profiles
    from the image at landmarks normal to the surface, then trains a 
    pca model for each landmark using the profiles.
    """

    def __init__(self, nSamples, xLim, zShift=True, negSpacing=False):
        """
        input:
        nSamples: integer, number of samples per profile. 
        xLim: 2 tuple, distance to sample either side of the landmark.
        zShift: boolean, apply zShift in coord2Index mapping
        negSpacing: boolean, apply negSpacing in coord2Index mapping
        msSize: None or int, kernel size for median smoothing of image
        """

        self.nSamples = nSamples
        self.xLim = xLim

        self._dP = None
        self._landmarkMasks = None
        self._PPCs = None
        self.nLandmarks = None
        self.trainingSamples = None
        self._asm = None

        self.zShift = zShift
        self.negSpacing = negSpacing

        self.params = ASMSegmentationParams(
            ND=self.nSamples,
            NLim=self.xLim,
            NPad=0,
            filterLandmarks=True,
            imageZShift=self.zShift,
            imageNegSpacing=self.negSpacing,
        )

    def setTrainingSamples(self, samples, nLandmarks):
        """
        Define training samples

        inputs
        sample: iterable, should return a scan object, a list of coordinates, 
                and a list of normalised normal vectors when its next method 
                is called. Coordinates and normals should be in in physical 
                space which can be mapped to image voxel indices by calling 
                scan.coord2Index.
        nLandmarks: the number of landmark points per image
        """
        self.trainingSamples = samples
        self.nLandmarks = nLandmarks

    def sampleTrainingImages(self, debug=False):
        """
        Run sampling process. Each image is loaded and sampled in sequence
        """

        self._dP = []
        self._landmarkMasks = []

        for i, (scan, L, N) in enumerate(self.trainingSamples):
            log.debug('sampling profiles')

            def getLandmarks(x):
                return L

            def getNormals(x):
                return N

            self._asm = ASMSegmentation(scan, params=self.params,
                                        getMeshCoords=getLandmarks,
                                        getMeshNormals=getNormals)

            self._asm._evaluateLandmarks(None)
            self._asm._sampleImage()
            landmarkMask = self._asm._filterValidLandmarks(L)
            self._dP.append(self._asm.dP)
            self._landmarkMasks.append(landmarkMask)

        self._dP = numpy.array(self._dP, dtype=float)
        self._landmarkMasks = numpy.array(self._landmarkMasks, dtype=bool)
        self._dP = numpy.ma.masked_array(self._dP,
                                         numpy.repeat((~self._landmarkMasks)[:, :, numpy.newaxis], self.nSamples, 2),
                                         dtype=float)

        log.debug('profiles shape:', self._dP.shape)

    def trainPPCs(self):
        """
        trains a RF regressor given features and displacements obtained
        from running sampleTrainingImages.
        """

        log.debug('training profile pcs')
        self.PPCs = PCA.PCList()
        for i in range(self.nLandmarks):
            sys.stdout.flush()
            sys.stdout.write('\rpoint %5i/%5i' % (i + 1, self.nLandmarks))

            dP = self._dP[:, i, :]
            pca = PCA.PCA()
            pca.setData(dP.T)
            pca.svd_decompose()
            self.PPCs.append(pca.PC)

    def savePPCs(self, filename):
        """
        Save trained profile pca list.
        inputs:
        filename: string
        """
        self.PPCs.save(filename)

    def showProfile(self, profileI, subjectI):
        dPSubject = self._dP.data[subjectI, profileI, :]
        dPMean = self._dP[:, profileI, :].mean(0)

        f = plot.figure()
        ax1 = f.add_subplot(311)
        ax2 = f.add_subplot(312)
        ax3 = f.add_subplot(313)

        ax1.plot(dPSubject)
        ax2.plot(dPMean)

        if self.PPCs != None:
            pcMean = self.PPCs.L[profileI].getMean()
            ax3.plot(pcMean)

        plot.show()

        log.debug('profile masked:', ~self._landmarkMasks[subjectI, profileI])
