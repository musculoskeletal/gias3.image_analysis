"""
FILE: fw_segmentation_tools
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
Common tools for using fieldwork meshes in segmentation.

Combines meshes and clm (and eventually asm) segmentaters
in easier to use functions.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import copy
import logging

import numpy as np
import time

from gias2.common import transform3D, math
from gias2.fieldwork.field import geometric_field
from gias2.fieldwork.field import geometric_field_fitter as GFF
from gias2.fieldwork.field.tools import fitting_tools
from gias2.image_analysis import asm_segmentation as ASM
from gias2.image_analysis import clm_segmentation as CLM
from gias2.image_analysis import image_tools
from gias2.learning import PCA_fitting
from gias2.registration import alignment_fitting

log = logging.getLogger(__name__)


def makeImageSpaceGF(scan, GF, negSpacing=False, zShift=True):
    """
    Transform a mesh from physical coords to image voxel indices
    """
    newGF = copy.deepcopy(GF)
    p = GF.get_all_point_positions()
    pImg = scan.coord2Index(p, negSpacing=negSpacing, zShift=zShift, roundInt=False)
    newGF.set_field_parameters(pImg.T[:, :, np.newaxis])

    return newGF


def makeImageSpaceSimpleMesh(scan, sm, negSpacing=False, zShift=True):
    """
    Transform a mesh from physical coords to image voxel indices
    """
    newSM = copy.deepcopy(sm)
    newSM.v = scan.coord2Index(sm.v, negSpacing=negSpacing, zShift=zShift, roundInt=False)
    return newSM


def makeImageSpacePoints(scan, pts, negSpacing=False, zShift=True):
    """
    Transform a mesh from physical coords to image voxel indices
    """
    return scan.coord2Index(pts, negSpacing=negSpacing, zShift=zShift, roundInt=False)


# ============================================================================#
# mesh evaluation functions                                                  #
# ============================================================================#

def makeGFEvaluator(mode, GF, **kwargs):
    """
    Generates mesh evaluation functions used in asm and clm segmentations.
    Is called by initialiseGFCLM, does not need to be used directly. Returns a 
    function for evaluating the GF and a function for getting GF parameters. 
    The evaluation function return a set of points at which image feature 
    detection is carried out. Both functions accept one input which modifies 
    the mesh in some way, and is dependent on the evaluation mode.

    inputs:
    modes: string of the evaluator type. Valid modes:
    PCXiGrid: evaluate the mesh at a grid of xi coordinates per element given rigid transformation and PC weights
    XiGrid: evaluate the mesh at a grid of xi coordinates per element given nodal parameters
    nodes: evaluates nodal coordinates given nodal parameters
    PCNodes: evaluate nodal coordinates reconstructed from given rigid transformation and PC weights

    GF: geometric_field instance that will be evaluated.

    Keyword arguments and the modes that need them:
    PC          - PCXiGrid, PCNodes; principleComponent instance
    PCModes     - PCXiGrid, PCNodes; list of integers corresponding to principle component number
    GD          - PCXiGrid, XiGrid; 2-tuple of integers, Xi discretisation.

    returns:
    GFEval: GF evaluation function
    getGFParams: GF parameters getter 
    """
    if mode == 'PCXiGrid':

        GFSparseEval = geometric_field.makeGeometricFieldEvaluatorSparse(GF, kwargs['GD'])
        PC = kwargs['PC']
        PCModes = kwargs['PCModes']

        def GFEval(X):
            # print 'GFEval X:', X

            if len(X) > 6:
                p = PC.reconstruct(PC.getWeightsBySD(PCModes, X[6:]), PCModes)
            else:
                p = GF.get_field_parameters()
            # reconstruct rigid transform
            p = alignment_fitting.transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, X[:6])
            return GFSparseEval(p.T[:, :, np.newaxis]).T

        def getGFParams(X):
            if len(X) > 6:
                p = PC.reconstruct(PC.getWeightsBySD(PCModes, X[6:]), PCModes)
            else:
                p = GF.get_field_parameters()
            # reconstruct rigid transform
            p = alignment_fitting.transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, X[:6])
            return p.T[:, :, np.newaxis]

    elif mode == 'XiGrid':
        GFSparseEval = geometric_field.makeGeometricFieldEvaluatorSparse(GF, kwargs['GD'])

        def GFEval(X):
            return GFSparseEval(X).T

        def getGFParams(X):
            return X

    elif mode == 'nodes':
        def GFEval(X):
            return X

        def getGFParams(X):
            return X.T[:, :, np.newaxis]

    elif mode == 'PCNodes':
        PC = kwargs['PC']
        PCModes = kwargs['PCModes']

        def GFEval(X):
            if len(X) > 6:
                p = PC.reconstruct(PC.getWeightsBySD(PCModes, X[6:]), PCModes)
            else:
                p = GF.get_field_parameters()
            # reconstruct rigid transform
            p = transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, X[:6])
            return p

        def getGFParams(X):
            if len(X) > 6:
                p = PC.reconstruct(PC.getWeightsBySD(PCModes, X[6:]), PCModes)
            else:
                p = GF.get_field_parameters()
            # reconstruct rigid transform
            p = transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, X[:6])
            return p.T[:, :, np.newaxis]

    return GFEval, getGFParams


def makeGFNormalEvaluator(mode, GF, **kwargs):
    """
    Generates mesh normal evaluation functions used in asm segmentations.
    Is called by initialiseGFASM, does not need to be used directly. Returns a 
    function for evaluating the normal of GF. 
    The evaluation function return a set of normalised vectors. The function 
    accepts one input which modifies  the mesh in some way, and is dependent 
    on the evaluation mode.

    inputs:
    modes: string of the evaluator type. Valid modes:
    PCXiGrid: evaluate the normals at a grid of xi coordinates per element 
              given rigid transformation and PC weights
    XiGrid: evaluate the normals at a grid of xi coordinates per element given nodal parameters
    nodes: evaluates normals at nodes given nodal parameters
    PCNodes: evaluate normals at nodes reconstructed from given rigid transformation and PC weights

    GF: geometric_field instance that will be evaluated.

    Keyword arguments and the modes that need them:
    PC          - PCXiGrid, PCNodes; principleComponent instance
    PCModes     - PCXiGrid, PCNodes; list of integers corresponding to principle component number
    GD          - PCXiGrid, XiGrid; 2-tuple of integers, Xi discretisation.

    returns:
    GFEval: GF evaluation function
    getGFParams: GF parameters getter 
    """

    if mode == 'PCXiGrid':

        # shape (3,nderivs,-1)
        dXEval = geometric_field.makeGeometricFieldDerivativesEvaluatorSparse(GF, kwargs['GD'], dim=3)
        PC = kwargs['PC']
        PCModes = kwargs['PCModes']

        def _normalEval(p):
            D = dXEval(p)
            d10 = D[:, 0]
            d01 = D[:, 1]
            d10Norm = math.norms(d10.T)
            d01Norm = math.norms(d01.T)
            return np.cross(d10Norm, d01Norm)

        def GFNormalEval(X):

            if len(X) > 6:
                p = PC.reconstruct(PC.getWeightsBySD(PCModes, X[6:]), PCModes)
            else:
                p = GF.get_field_parameters()
            # reconstruct rigid transform
            p = alignment_fitting.transform3D.transformRigid3DAboutCoM(p.reshape((3, -1)).T, X[:6])
            return _normalEval(p.T[:, :, np.newaxis])

    elif mode == 'XiGrid':
        dXEval = geometric_field.makeGeometricFieldDerivativesEvaluatorSparse(GF, kwargs['GD'], dim=3)

        def GFNormalEval(X):
            D = dXEval(X)
            d10 = D[:, 0]
            d01 = D[:, 1]
            d10Norm = math.norms(d10.T)
            d01Norm = math.norms(d01.T)
            return np.cross(d10Norm, d01Norm)

    return GFNormalEval


# ============================================================================#
# mesh fit functions                                                         #
# ============================================================================#

def makeMeshFit(mode, **kwargs):
    """
    Generate mesh fitting functions used in asm and clm segmentation.
    Is called by initialiseGFCLM, does not need to be used directly.

    input:
    models: string matching a fitting mode. Valid modes:
    PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit

    Keyword arguments and the modes that need them:
    SSM               - PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit
    SSMModes          - PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit
    GF                - PCEPEP, PCDPEP, PCEPDP, nodal
    GD                - PCEPEP, PCDPEP, PCEPDP, nodal
    mahalanobisWeight - PCEPEP, PCDPEP, PCEPDP, PCPointFit
    epIndex           - PCEPEP, PCDPEP, PCEPDP
    GFCoordEval       - PCEPEP, PCDPEP, PCEPDP

    returns:
    fitter: a fitting function that accepts as input data to fit to, initial 
            parameters, data weights, and indices of data points to use in the fit.


    """

    log.debug('creating fitting mode: ' + mode)
    if mode == 'PCEPEP':
        return _makeMeshFitPCFit(GFF.makeObjEPEP, kwargs['GF'], kwargs['GD'],
                                 kwargs['SSM'], kwargs['SSMModes'],
                                 kwargs['mahalanobisWeight'], kwargs['epIndex'],
                                 kwargs['GFCoordEval'])
    elif mode == 'PCDPEP':
        return _makeMeshFitPCFit(GFF.makeObjDPEP, kwargs['GF'], kwargs['GD'],
                                 kwargs['SSM'], kwargs['SSMModes'],
                                 kwargs['mahalanobisWeight'], kwargs['epIndex'],
                                 kwargs['GFCoordEval'])
    elif mode == 'PCEPDP':
        return _makeMeshFitPCFit(GFF.makeObjEPDP, kwargs['GF'], kwargs['GD'],
                                 kwargs['SSM'], kwargs['SSMModes'],
                                 kwargs['mahalanobisWeight'], kwargs['epIndex'],
                                 kwargs['GFCoordEval'])
    elif mode == 'PCPointProject':
        return _makeMeshFitPointProject(kwargs['SSM'], kwargs['SSMModes'])
    elif mode == 'PCPointFit':
        return _makeMeshFitPointPCFit(kwargs['SSM'], kwargs['SSMModes'],
                                      kwargs['mahalanobisWeight'], kwargs['initRotation'],
                                      kwargs['doScale'], kwargs['landmarkTargets'],
                                      kwargs['landmarkEvaluator'], kwargs['landmarkWeights'])
    elif mode == 'pointNoFit':
        return _makeMeshFitPointNoFit()


def _makeMeshFitPCFit(objMaker, GF, GD, SSM, fitModes, mWeight, epIndex, GFCoordEval, xtol=1e-6, retFullError=False):
    PCFitter = PCA_fitting.PCFit()
    PCFitter.setPC(SSM)
    PCFitter.xtol = xtol
    if epIndex == None:
        segElements = list(GF.ensemble_field_function.mesh.elements.keys())
        epI = GF.getElementPointIPerTrueElement(GD, segElements)

    if retFullError:
        def meshFitPCFit(data, x0, weights, landmarkIndices=None):
            # print 'meshFitPCFit x0:', x0
            # obj = objMaker(GF, data, GD, dataWeights=weights, epIndex=epIndex, evaluator=GFCoordEval )
            obj = objMaker(GF, data, GD, dataWeights=weights, epIndex=landmarkIndices, evaluator=None)
            GXOpt, GPOpt = PCFitter.rigidModeNRotateAboutCoMFit(obj, modes=fitModes[1:], x0=x0, mWeight=mWeight,
                                                                funcArgs=())
            GF.set_field_parameters(GPOpt.copy().reshape((3, -1, 1)))
            # error calculation
            fullError = obj(GPOpt.copy())
            meshRMS = np.sqrt(fullError.mean())
            meshSD = np.sqrt(fullError.std())
            return GXOpt, meshRMS, meshSD, fullError
    else:
        def meshFitPCFit(data, x0, weights, landmarkIndices=None):
            # print 'meshFitPCFit x0:', x0
            # obj = objMaker(GF, data, GD, dataWeights=weights, epIndex=epIndex, evaluator=GFCoordEval )
            obj = objMaker(GF, data, GD, dataWeights=weights, epIndex=landmarkIndices, evaluator=None)
            GXOpt, GPOpt = PCFitter.rigidModeNRotateAboutCoMFit(obj, modes=fitModes[1:], x0=x0, mWeight=mWeight,
                                                                funcArgs=())
            GF.set_field_parameters(GPOpt.copy().reshape((3, -1, 1)))
            # error calculation
            meshRMS = np.sqrt(obj(GPOpt.copy()).mean())
            meshSD = np.sqrt(obj(GPOpt.copy())).std()
            return GXOpt, meshRMS, meshSD

    return meshFitPCFit


def _makeMeshFitPointProject(SSM, projectModes):
    def meshFitPointProject(data, x0, weights, landmarkIndices=None):

        if landmarkIndices != None:
            landmarkIndices = np.array(landmarkIndices)
            variables = np.hstack(
                [landmarkIndices, landmarkIndices * 2, landmarkIndices * 3])  # because variables are x y z coords
        else:
            variables = None

        # print 'dongdong', landmarkIndices
        # pdb.set_trace()

        # project against SSM
        pcWeights, reconDataT, dataT, reconData = PCA_fitting.project3DPointsToSSM(data, SSM, projectModes,
                                                                                   projectVariables=variables,
                                                                                   landmarkIs=landmarkIndices,
                                                                                   verbose=1)
        # errors
        if landmarkIndices != None:
            errors = np.sqrt(((reconDataT[landmarkIndices, :] - data) ** 2.0).sum(1))
        else:
            errors = np.sqrt(((reconDataT - data) ** 2.0).sum(1))
        rms = np.sqrt((errors ** 2.0).mean())
        stdev = errors.std()
        return reconDataT, rms, stdev

    return meshFitPointProject


def _makeMeshFitPointPCFit(SSM, fitModes, mahalanobisWeight=0.0, initRotation=None, doScale=False,
                           landmarkTargets=None, landmarkEvaluator=None, landmarkWeights=None):
    def meshFitPointPCFit(data, x0, weights, landmarkIndices=None):

        # project against SSM
        pcWeights, reconDataT, dataT, reconData = PCA_fitting.fitSSMTo3DPoints( \
            data, SSM, fitModes, fitPointIndices=landmarkIndices, mWeight=mahalanobisWeight, \
            initRotation=initRotation, doScale=doScale, landmarkTargets=landmarkTargets, \
            landmarkEvaluator=landmarkEvaluator, landmarkWeights=landmarkWeights, \
            verbose=True)

        log.debug(pcWeights)

        # errors
        if landmarkIndices != None:
            errors = np.sqrt(((reconDataT[landmarkIndices, :] - data) ** 2.0).sum(1))
        else:
            errors = np.sqrt(((reconDataT - data) ** 2.0).sum(1))
        rms = np.sqrt((errors ** 2.0).mean())
        stdev = errors.std()
        return reconDataT, rms, stdev

    return meshFitPointPCFit


def _makeMeshFitPointPCFitBad(SSM, fitModes, mWeight=0.0):
    pcFit = PCA_fitting.PCFit(SSM)
    pcFit.xtol = 1e-6

    def meshFitPointPCFit(data, x0, weights, landmarkIndices=None):

        def _makeObj(data, landmarkIndices=None):

            if landmarkIndices == None:
                def _objAllPoints(p):
                    fittedPoints = p.reshape((3, -1)).T
                    E = ((data - fittedPoints) ** 2.0).sum(1) * weights
                    return E

                return _objAllPoints
            else:
                def _objSubsetPoints(p):
                    fittedPoints = p.reshape((3, -1)).T[landmarkIndices, :]
                    E = ((data - fittedPoints) ** 2.0).sum(1) * weights
                    return E

                return _objSubsetPoints

        obj = _makeObj(data, landmarkIndices)
        xOpt, pOpt = pcFit.rigidModeNRotateAboutCoMFit(obj, modes=fitModes, x0=x0, mWeight=mWeight, maxfev=0,
                                                       funcArgs=())
        pOpt = pOpt.reshape((3, -1)).T

        # errors
        if landmarkIndices != None:
            errors = np.sqrt(((pOpt[landmarkIndices, :] - data) ** 2.0).sum(1))
        else:
            errors = np.sqrt(((pOpt - data) ** 2.0).sum(1))
        rms = np.sqrt((errors ** 2.0).mean())
        stdev = errors.std()
        return xOpt, rms, stdev

    return meshFitPointPCFit


def _makeMeshFitPointNoFit():
    def meshFitPointNoFit(data, x0, weights, landmarkIndices):
        return data, np.random.rand(), np.random.rand()

    return meshFitPointNoFit


def _makeMeshFitNodal(objMode, GF, EPD, sobD, sobW, ND, NW, fixedNodes=None, xtol=None, maxIt=None, maxItPerIt=None,
                      nClosestPoints=None, treeArgs=None):
    if treeArgs is None:
        treeArgs = {}

    def meshFitNodal(data, x0):
        GF, gfFitPOpt, meshFitRMS, meshFitError = fitting_tools.fitSurfacePerItSearch(objMode,
                                                                                      GF,
                                                                                      data,
                                                                                      EPD,
                                                                                      sobD,
                                                                                      sobW,
                                                                                      ND,
                                                                                      NW,
                                                                                      fixedNodes=fixedNodes,
                                                                                      xtol=xtol,
                                                                                      itMax=maxIt,
                                                                                      itMaxPerIt=maxItPerIt,
                                                                                      nClosestPoints=nClosestPoints,
                                                                                      treeArgs=treeArgs,
                                                                                      fullErrors=True
                                                                                      )

        return gfFitPOpt, meshFitRMS, meshFitError.std()


# ====================================================#
# Main CLM Segmentation Functions                    #
# ====================================================#
def initialiseGFCLM(CLMParams, GF, GFEvalMode, GFFitMode, GD, shapeModel, shapeModelModes,
                    segElements, mahalanobisWeight, GFInitialRotation=None, doScale=False,
                    landmarkTargets=None, landmarkEvaluator=None, landmarkWeights=None):
    """
    function for initialising a CLM with a GF and shape model.

    inputs:
    CLMParams: CLMSegmentationParams instance
    GF: geometric_field instance
    GFEvalMode: string matching a mode for makeGFEvaluator (PCXiGrid, XiGrid, node, PCNodes)
    GFFitMode: string mathcing a mode for makeMeshFit (PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit)
    GD: 2-tuple, xi discretisation
    shapeModel: principleComponent instance or similar, None if not applicable.
    shapeModelModes: list of integers, the mode numbers to use for fitting and or evaluation, None if not applicable.
    segElements: list of element numbers to perform segmentation on, None if not applicable.
    mahalanobisWeight: float, None if not applicable.

    returns:
    clm: initialised CLMSegmentation instance
    GFCoordEval: GF evaluator function
    GFGetParams: GF parameter getter function
    GFFitter: GF fitting function
    """

    # define evaluator functions
    GFCoordEval, GFGetParams = makeGFEvaluator( \
        GFEvalMode, \
        GF, \
        PC=shapeModel, \
        PCModes=shapeModelModes, \
        GD=GD, \
        )

    # define xi coordinates for fitting (optional)
    if segElements == None:
        epI = None
    elif segElements == 'all':
        epI = None
    else:
        epI = GF.getElementPointIPerTrueElement(GD, segElements)

    # create PC fitter
    GFFitter = makeMeshFit(
        GFFitMode,
        SSM=shapeModel,
        SSMModes=shapeModelModes,
        GF=GF,
        GD=GD,
        mahalanobisWeight=mahalanobisWeight,
        epIndex=epI,
        GFCoordEval=GFCoordEval,
        initRotation=GFInitialRotation,
        doScale=doScale,
        landmarkTargets=landmarkTargets,
        landmarkEvaluator=landmarkEvaluator,
        landmarkWeights=landmarkWeights
    )

    # instantiate CLM segmenter
    clm = CLM.CLMSegmentation(
        params=CLMParams,
        getMeshCoords=GFCoordEval,
        fitMesh=GFFitter
    )

    clm.loadRFs(CLMParams.RFFilename)

    return clm, GFCoordEval, GFGetParams, GFFitter


def runGFCLM(clm, scan, GF, GFFitMode, GFGetParams, shapeModel, shapeModelModes,
             GFInitialRotation, imageCropPad, filterLandmarks, verbose=0):
    """
    function for running a CLM with a GF and shape model

    inputs:
    clm: CLMSegmentation instance
    scan: Scan instance containing image to segment
    GF: geometric_field instance
    GFFitMode: string mathcing a mode for makeMeshFit (PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit)
    GFGetParams: function for getting GF params. Generated by makeGFEvaluator.
    shapeModel: principleComponent instance or similar, None if not applicable.
    shapeModelModes: list of integers, the mode numbers to use for fitting and or evaluation, None if not applicable.
    GFInitialRotation: 3-tuple, initial rotations in each axis to apply to the GF
    imageCropPad: number of voxels to pad around mesh when cropping the image around it
    verbose: extra messages.

    returns:
    CLMOutput: dictionary of clm segmentation output
    GF: final segmented GF instance
    croppedScan: cropped scan used for segmentation
    """

    # get x0
    if GFFitMode in ['PCPointProject', 'pointNoFit', 'PCPointFit']:
        x0 = GF.get_all_point_positions()
    elif GFFitMode in ['PCEPEP', 'PCDPEP', 'PCEPDP']:
        # must get PC mode weights for the current shape, therefore, have to align current mesh
        # must also get rigid transform from mean shape
        x0 = PCA_fitting.fitSSMTo3DPoints(
            GF.get_all_point_positions(),
            shapeModel,
            shapeModelModes,
            mWeight=0.5
        )[0]

        # # align GF to mean GF
        # targetPoints = shapeModel.getMean().reshape((3,-1)).T
        # dataPoints = GF.get_all_point_positions()
        # alignX0 = np.hstack([ targetPoints.mean(0)-dataPoints.mean(0), np.array(GFInitialRotation) ])
        # GF2MeanRigidT, dataAligned = alignment_fitting.fitRigid( dataPoints, targetPoints, alignX0, verbose=verbose )

        # # project aligned params on shape model
        # alignedData = dataAligned.T.ravel()
        # alignedDataC = alignedData - shapeModel.getMean()
        # GFPCWeights = shapeModel.project( alignedDataC, shapeModelModes )
        # GFPCSD = shapeModel.calcSDFromWeights( shapeModelModes, GFPCWeights )

        # # find transformation back to image location of GF
        # targetPoints = GF.get_all_point_positions()
        # dataPoints = GFGetParams( np.hstack([np.zeros(6), GFPCSD]) ).squeeze().T
        # reverseAlignX0 = np.hstack([ targetPoints.mean(0)-dataPoints.mean(0), -np.array(GFInitialRotation) ])
        # mean2GFRigidT, dataTempAligned = alignment_fitting.fitRigid( dataPoints, targetPoints, reverseAlignX0, verbose=1 )

        # x0 = np.hstack([ mean2GFRigidT, GFPCSD ])
        if verbose:
            log.debug('x0:', x0)

    # crop/subsample image around initial model for segmentation
    initPoints = GF.get_all_point_positions()  ###
    croppedScan, cropOffset = image_tools.cropImageAroundPoints(initPoints, scan, imageCropPad, \
                                                                croppedName=scan.name + '_cropped',
                                                                transformToIndexSpace=True)
    # cropOffset -= self.scan.voxelOffset
    HRVImage = CLM.HRV.HaarImage(croppedScan.I, croppedScan.voxelSpacing, croppedScan.voxelOrigin,
                                 isMasked=croppedScan.isMasked)
    clm.setHRVImage(HRVImage)
    if filterLandmarks != None:
        clm.filterLandmarks = filterLandmarks

    CLMOutput = clm.segment(x0, verbose=verbose, debug=0)
    outputVars = ['segXOpt', 'segData', 'segDataWeight', 'segDataLandmarkIndices' 'segRMS', 'segSD', 'segPFrac',
                  'segHistory']
    CLMOutput = dict(list(zip(outputVars, CLMOutput)))
    CLMOutput['segPOpt'] = GFGetParams(CLMOutput['segXOpt'].copy())
    GF.set_field_parameters(CLMOutput['segPOpt'])

    return CLMOutput, GF, croppedScan


def doCLM(CLMParams, scan, GF, GFEvalMode, GFFitMode, GD, shapeModel, shapeModelModes,
          segElements, mahalanobisWeight, imageCropPad, verbose=0,
          filterLandmarks=None, GFInitialRotation=None, doScale=None,
          landmarkTargets=None, landmarkEvaluator=None, landmarkWeights=None):
    """
    function for initialising then running a CLM with a GF and shape model. Combines 
    initialiseGFCLM and runGFCLM.

    inputs:
    CLMParams: CLMSegmentationParams instance
    scan: Scan instance containing image to segment
    GF: geometric_field instance
    GFEvalMode: string matching a mode for makeGFEvaluator (PCXiGrid, XiGrid, node, PCNodes)
    GFFitMode: string mathcing a mode for makeMeshFit (PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit)
    GD: 2-tuple, xi discretisation
    shapeModel: principleComponent instance or similar, None if not applicable.
    shapeModelModes: list of integers, the mode numbers to use for fitting and or evaluation, None if not applicable.
    segElements: list of element numbers to perform segmentation on, None if not applicable.
    mahalanobisWeight: float, None if not applicable.
    GFInitialRotation: 3-tuple, initial rotations in each axis to apply to the GF
    imageCropPad: number of voxels to pad around mesh when cropping the image around it
    verbose: extra messages.

    returns:
    CLMOutput: dictionary of clm segmentation output
    clm: initialised CLMSegmentation instance
    GF: final segmented GF instance
    croppedScan: cropped scan used for segmentation
    GFCoordEval: GF evaluator function
    GFGetParams: GF parameter getter function
    GFFitter: GF fitting function
    """
    t0 = time.time()
    clm, GFCoordEval, GFGetParams, GFFitter = initialiseGFCLM(CLMParams, GF, GFEvalMode,
                                                              GFFitMode, GD, shapeModel, shapeModelModes, segElements,
                                                              mahalanobisWeight,
                                                              GFInitialRotation, doScale, landmarkTargets,
                                                              landmarkEvaluator, landmarkWeights)
    t1 = time.time()
    CLMOutput, GF, croppedScan = runGFCLM(clm, scan, GF, GFFitMode, GFGetParams, shapeModel,
                                          shapeModelModes, GFInitialRotation,
                                          imageCropPad, filterLandmarks, verbose)
    t2 = time.time()
    CLMOutput['runtimeInit'] = t1 - t0
    CLMOutput['runtimeRun'] = t2 - t1
    CLMOutput['runtimeTotal'] = t2 - t0
    return CLMOutput, clm, GF, croppedScan, GFCoordEval, GFGetParams, GFFitter


# ====================================================#
# Main ASM Segmentation Functions                    #
# ====================================================#
def initialiseGFASM(ASMParams, GF, GFEvalMode, GFFitMode, GD, shapeModel, shapeModelModes,
                    segElements, mahalanobisWeight, GFInitialRotation=None, doScale=False,
                    landmarkTargets=None, landmarkEvaluator=None, landmarkWeights=None):
    """
    function for initialising a ASM with a GF and shape model.

    inputs:
    ASMParams: ASMSegmentationParams instance
    GF: geometric_field instance
    GFEvalMode: string matching a mode for makeGFEvaluator (PCXiGrid, XiGrid, node, PCNodes)
    GFFitMode: string mathcing a mode for makeMeshFit (PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit)
    GD: 2-tuple, xi discretisation
    shapeModel: principleComponent instance or similar, None if not applicable.
    shapeModelModes: list of integers, the mode numbers to use for fitting and or evaluation, None if not applicable.
    segElements: list of element numbers to perform segmentation on, None if not applicable.
    mahalanobisWeight: float, None if not applicable.

    returns:
    asm: initialised ASMSegmentation instance
    GFCoordEval: GF evaluator function
    GFGetParams: GF parameter getter function
    GFFitter: GF fitting function
    """

    # define evaluator functions
    GFCoordEval, GFGetParams = makeGFEvaluator(
        GFEvalMode,
        GF,
        PC=shapeModel,
        PCModes=shapeModelModes,
        GD=GD,
    )

    GFNormalEval = makeGFNormalEvaluator(GFEvalMode,
                                         GF,
                                         PC=shapeModel,
                                         PCModes=shapeModelModes,
                                         GD=GD,
                                         )

    # create PC fitter
    GFFitter = makeMeshFit(
        GFFitMode,
        SSM=shapeModel,
        SSMModes=shapeModelModes,
        GF=GF,
        GD=GD,
        mahalanobisWeight=mahalanobisWeight,
        epIndex=None,
        GFCoordEval=GFCoordEval,
        initRotation=GFInitialRotation,
        doScale=doScale,
        landmarkTargets=landmarkTargets,
        landmarkEvaluator=landmarkEvaluator,
        landmarkWeights=landmarkWeights
    )

    # define xi coordinates for fitting (optional)
    if segElements == None:
        epI = GF.getElementPointIPerTrueElement(ASMParams.GD, list(GF.ensemble_field_function.mesh.elements.keys()))
    elif segElements == 'all':
        epI = GF.getElementPointIPerTrueElement(ASMParams.GD, list(GF.ensemble_field_function.mesh.elements.keys()))
    else:
        epI = GF.getElementPointIPerTrueElement(GD, segElements)

    # instantiate ASM segmenter
    asm = ASM.ASMSegmentation(
        params=ASMParams,
        getMeshCoords=GFCoordEval,
        getMeshNormals=GFNormalEval,
        fitMesh=GFFitter
    )
    log.debug('Loading profile texture models...')
    asm.loadProfilePC()
    log.debug('Loading profile texture models...done.')
    asm.setElementXIndices(epI)

    return asm, GFCoordEval, GFGetParams, GFFitter


def runGFASM(asm, scan, GF, GFFitMode, GFGetParams, shapeModel, shapeModelModes,
             GFInitialRotation, filterLandmarks=True, verbose=0):
    """
    function for running a ASM with a GF and shape model

    inputs:
    asm: ASMSegmentation instance
    scan: Scan instance containing image to segment
    GF: geometric_field instance
    GFFitMode: string mathcing a mode for makeMeshFit (PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit)
    GFGetParams: function for getting GF params. Generated by makeGFEvaluator.
    shapeModel: principleComponent instance or similar, None if not applicable.
    shapeModelModes: list of integers, the mode numbers to use for fitting and or evaluation, None if not applicable.
    GFInitialRotation: 3-tuple, initial rotations in each axis to apply to the GF
    verbose: extra messages.

    returns:
    ASMOutput: dictionary of asm segmentation output
    GF: final segmented GF instance
    croppedScan: cropped scan used for segmentation
    """

    # set image to segment
    asm.setImage(scan)
    asm.filterLandmarks = filterLandmarks

    # get x0
    if GFFitMode in ['PCPointProject', 'pointNoFit', 'PCPointFit']:
        x0 = GF.get_all_point_positions()
    elif GFFitMode in ['PCEPEP', 'PCDPEP', 'PCEPDP']:
        # must get PC mode weights for the current shape, therefore, have to align current mesh
        # must also get rigid transform from mean shape
        x0 = PCA_fitting.fitSSMTo3DPoints(
            GF.get_all_point_positions(),
            shapeModel,
            shapeModelModes,
            mWeight=0.5
        )[0]
        if verbose:
            log.debug('x0:', x0)

    ASMOutput = asm.segment(x0, verbose=verbose, debug=0)
    outputVars = ['segXOpt', 'segData', 'segDataWeight', 'segDataLandmarkMask',
                  'segRMS', 'segSD', 'segPFrac', 'segProfileMatchM', 'segProfileM',
                  'segHistory']
    ASMOutput = dict(list(zip(outputVars, ASMOutput)))
    ASMOutput['segPOpt'] = GFGetParams(ASMOutput['segXOpt'].copy())
    GF.set_field_parameters(ASMOutput['segPOpt'])

    return ASMOutput, GF, scan


def doASM(ASMParams, scan, GF, GFEvalMode, GFFitMode, GD, shapeModel, shapeModelModes,
          segElements, mahalanobisWeight, verbose=0,
          filterLandmarks=True, GFInitialRotation=None, doScale=False,
          landmarkTargets=None, landmarkEvaluator=None, landmarkWeights=None):
    """
    function for initialising then running a ASM with a GF and shape model. Combines 
    initialiseGFASM and runGFASM.

    inputs:
    ASMParams: ASMSegmentationParams instance
    scan: Scan instance containing image to segment
    GF: geometric_field instance
    GFEvalMode: string matching a mode for makeGFEvaluator (PCXiGrid, XiGrid, node, PCNodes)
    GFFitMode: string mathcing a mode for makeMeshFit (PCEPEP, PCDPEP, PCEPDP, PCPointProject, PCPointFit, pointNoFit)
    GD: 2-tuple, xi discretisation
    shapeModel: principleComponent instance or similar, None if not applicable.
    shapeModelModes: list of integers, the mode numbers to use for fitting and or evaluation, None if not applicable.
    segElements: list of element numbers to perform segmentation on, None if not applicable.
    mahalanobisWeight: float, None if not applicable.
    GFInitialRotation: 3-tuple, initial rotations in each axis to apply to the GF
    verbose: extra messages.

    returns:
    ASMOutput: dictionary of asm segmentation output
    asm: initialised ASMSegmentation instance
    GF: final segmented GF instance
    croppedScan: cropped scan used for segmentation
    GFCoordEval: GF evaluator function
    GFGetParams: GF parameter getter function
    GFFitter: GF fitting function
    """
    t0 = time.time()
    asm, GFCoordEval, GFGetParams, GFFitter = initialiseGFASM(ASMParams, GF, GFEvalMode,
                                                              GFFitMode, GD, shapeModel, shapeModelModes, segElements,
                                                              mahalanobisWeight,
                                                              GFInitialRotation, doScale, landmarkTargets,
                                                              landmarkEvaluator, landmarkWeights)
    t1 = time.time()
    ASMOutput, GF, croppedScan = runGFASM(asm, scan, GF, GFFitMode, GFGetParams, shapeModel,
                                          shapeModelModes, GFInitialRotation,
                                          filterLandmarks, verbose)
    t2 = time.time()
    ASMOutput['runtimeInit'] = t1 - t0
    ASMOutput['runtimeRun'] = t2 - t1
    ASMOutput['runtimeTotal'] = t2 - t0
    return ASMOutput, asm, GF, croppedScan, GFCoordEval, GFGetParams, GFFitter
