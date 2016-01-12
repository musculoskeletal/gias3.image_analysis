"""
FILE: image_tools.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: classes and functions for loading, manipulating and visualing image volumes

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import glob
import scipy
from gias2.registration import alignment_analytic
# from gias.common import vtkRender
from scipy.linalg import eigh, inv
from scipy.optimize import leastsq, fmin
from scipy.spatial.distance import euclidean
from scipy.ndimage import rotate, affine_transform, zoom, map_coordinates, gaussian_filter1d, median_filter
from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import linregress
from scipy.misc import imread
from scipy.misc import imsave
import numpy.ma as ma
import re
import dicom
import dicom.contrib.pydicom_series as dicomSeries
import os
import sys

try:
	from matplotlib import pyplot as plot
	from matplotlib import cm
except ImportError:
	print 'Matplotlib not found, 2-D visualisation will be disabled'

class NoScanError(Exception):
	pass

class PhantomError(Exception):
	pass

class ProgressOutput:
    
    def __init__(self, task, total):
        self.task = task
        self.total = total
        self.value = 0
        
    def progress(self, value, comment=''):
        self.value = value        
        percent = 100.*value/self.total
        outcomment = ''
        if len(comment)>0:
            outcomment = ": %s" % comment
        sys.stdout.write("Progress: %s : %2d%% %s  \r" % (self.task, percent, outcomment))
        sys.stdout.flush()
        
    def output(self, comment=''):
        outcomment = ''
        if len(comment)>0:
            outcomment = ": %s" % comment
        sys.stdout.write("\nOutput: %s %s\n" % (self.task, outcomment))
        sys.stdout.flush()
        

def filterDicomPixels(dicomData):
    pixelArray = dicomData.pixel_array

    # Set values < 0 to 0
    ii = pixelArray < 0
    pixelArray[ii] = 0
        
    # Find max/min values.
    minPixel = pixelArray.min()    
    maxPixel = pixelArray.max()

    # Adjust range of pixels
    pixelRange = maxPixel-minPixel
    scalePixel = 255./pixelRange
    newPixelArray = scalePixel*(pixelArray-minPixel)
    return newPixelArray

	
class Scan:
	""" Class for reading and limited manipulation of an image stack
	"""

	def __init__( self, name ):

		self.name = name
		self.sV = {}			# dict of subvolume names to subvolumes
		self.I = None
		self.reader = None
		self.imageType = None
		self.slice0 = None
		self.sliceLast = None
		# self.renderer = vtkRender.VtkImageVolumeRenderer()
		self.mesh = None
		
		self.shape = None
		self.CoM = None			# centre of mass
		self.pAxes = None		# principal axes (vectors)
		self.pAxesMag = None	# magnitude of principal axes
		self.M00 = None			# image mass
		self._iC = None			# centered voxel coordinates
		self.isZeroMass = None
		
		self.roi0 = [0, 0, 0]
		self.subVolumeSize = 64
		self.subVolumeSpacing = 32
		self.spacing = [1.0, 1.0, 1.0]
		
		self.pixelSpacing = None
		self.sliceSpacing = None
		self._previousSliceLocation = None
		self.sliceLocations = None
		self.pixelTolerance = 0.0001
		self.sliceTolerance = 0.0001

		self.isMasked = False
		
	#==================================================================#
	def __del__(self):
		del self.I
		del self.reader
		del self.imageType

	#==================================================================#
	# read / write methods											   #
	#==================================================================#
	def setImageArray( self, I, voxelSpacing=None, voxelOrigin=None ):
		""" set object's image array
		"""
		self.I = I
		self.calculateCoM()
		self.calculatePrincipalAxes()

		if voxelSpacing!=None:
			self.voxelSpacing = scipy.array(voxelSpacing)
		else:
			self.voxelSpacing = scipy.array([1.0,1.0,1.0])

		if voxelOrigin!=None:
			self.voxelOrigin = scipy.array(voxelOrigin)
		else:
			self.voxelOrigin = scipy.array([0.0,0.0,0.0])

		# self.renderer = vtkRender.VtkImageVolumeRenderer( self.I )
		# self.renderer.setCoM( self.CoM, self.pAxes, self.pAxesMag )
		
		return

	def maskImage(self, mask, fillValue=None):
		
		maskedI = ma.masked_array(self.I, mask=mask, fill_value=fillValue)
		self.I = maskedI
		self.isMasked = True

	#==================================================================#
	#~ def LoadScan(self, folder, filenameFormat, fileNumber):
		#~ """ loads a stack of images with numbered filenames in an array
		#~ representation.
		#~ folder: a string of the folder path
		#~ filenameFormat: formatted string of the image filename. e.g.
			#~ 'image_%04d.png'
		#~ fileNumber: list - [start number, end number, increment ]
		#~ """
		#~ 
		#~ self.read_folder = folder
		#~ self.read_filenameFormat = filenameFormat
		#~ self.read_file_0 = fileNumber[0]
		#~ self.read_file_1 = fileNumber[1]
		#~ self.read_file_inc = fileNumber[2]
		#~ self.imageType = itk.Image[itk.UC, 3]
		#~ 
		#~ if self.reader == None:
			#~ self.reader = itk.ImageSeriesReader[self.imageType].New()
			#~ 
		#~ self.read_file_series = itk.NumericSeriesFileNames.New()
		#~ self.read_file_series.SetSeriesFormat(self.read_folder+"/"+self.read_filenameFormat)
		#~ self.read_file_series.SetStartIndex(self.read_file_0)
		#~ self.read_file_series.SetEndIndex(self.read_file_1)
		#~ self.read_file_series.SetIncrementIndex(self.read_file_inc)
#~ 
		#~ self.reader.SetFileNames(self.read_file_series.GetFileNames())
		#~ 
		#~ self.reader.Update()
		#~ self.rawImage = self.reader.GetOutput()
		#~ self.rawImage.SetSpacing(self.spacing)
		#~ print str((self.read_file_1 - self.read_file_0 + 1)/self.read_file_inc)+" image slices loaded."
		#~ 
		#~ self.LoadImageArray()
		#~ self._updateI()
		#~ 
		#~ return

	#==================================================================#
	def LoadScanFolder( self, folder, suffix='.png' ):
		""" Loads all .png files in the folder into an array
		representation
		"""
		
		# add /
		if folder[-1] != '/':
			self.read_folder = folder+'/'
		else:
			self.read_folder = folder
		
		slice = 0
		fileList = glob.glob( self.read_folder+'*'+suffix )
		fileList.sort()
		
		# scipy
		self.I = scipy.array( [ imread( f ) for f in fileList ] ) 
		print 'loaded image shape:', self.I.shape
		
		self._updateI()
		
		return
		
	#==================================================================#
	def LoadDicomFolder(self, folder, filter=False, filePattern='\.dcm$', sliceSpacingOveride=None, nSlice=None, newLoadMethod=True):
		if newLoadMethod:
			return self.LoadDicomFolderNew( folder, filter, filePattern, sliceSpacingOveride, nSlice)
		else:
			return self.LoadDicomFolderOld( folder, filter, filePattern, sliceSpacingOveride, nSlice)
		
	#==================================================================#
	def LoadDicomFolderOld(self, folder, filter=False, filePattern='\.dcm$', sliceSpacingOveride=None, nSlice=None):
		
		# add /
		if folder[-1] != '/':
			self.read_folder = folder+'/'
		else:
			self.read_folder = folder
		
		# Get directory list of files
		directoryList = os.listdir(self.read_folder)
		directoryList.sort()
		
		# Find file pattern in directory list of files
		reFilePattern = re.compile(filePattern, re.IGNORECASE)
		files = []
		for f in directoryList:
			if reFilePattern.search(f):
				files.append(f)
		
		if len(files)==0:
			raise IOError, 'No files found'
			
		if nSlice!=None:
			print 'loading %i slices'%(nSlice)
			files = [files[i] for i in xrange(nSlice)]
		
		# Open a file to get image size
		try:
			slice0 = dicom.read_file(self.read_folder+'/'+files[0], force=True)
		except TypeError:
			slice0 = dicom.read_file(self.read_folder+'/'+files[0])
		self.slice0 = slice0
		
		# Read images into array I
		if filter:
			self.I = scipy.empty([slice0.pixel_array.shape[0], slice0.pixel_array.shape[1], len(files)], dtype='uint8')
		else:
			self.I = scipy.empty([slice0.pixel_array.shape[0], slice0.pixel_array.shape[1], len(files)], dtype='int16')
		
		poLoad = ProgressOutput("Loading scan", len(files))
		self.sliceLocations = scipy.zeros(len(files), dtype=float)
		for sl, f in enumerate(files):
			try:
				slice = dicom.read_file(self.read_folder+'/'+f, force=True)
			except TypeError:
				slice = dicom.read_file(self.read_folder+'/'+f)
			self.testPixelSpacing(slice)
			if filter:
				self.I[:,:,sl] = filterDicomPixels(slice)
			else:
				self.I[:,:,sl] = slice.pixel_array.copy()
			self.sliceLocations[sl] = float(slice.SliceLocation)
			poLoad.progress(sl+1)
		
		self.sliceLast = slice
		poLoad.output("{}: {} slices loaded".format(self.read_folder,sl))

		# reorder slices by slice location
		self.I = self.I[:,:, scipy.argsort(self.sliceLocations)]
		
		if sliceSpacingOveride!=None:
			self.sliceSpacing = sliceSpacingOveride
		
		# set axes to be l-r, a-p, s-i
		self.I = self.I.transpose( [1,0,2] )		# original 
		self.voxelSpacing = scipy.array([self.pixelSpacing[0], self.pixelSpacing[1], self.sliceSpacing])
		self.voxelOrigin = scipy.array(self.slice0.ImagePositionPatient)

	#==================================================================#
	def LoadDicomFolderNew(self, folder, filter=False, filePattern='\.dcm$', sliceSpacingOveride=None, nSlice=None):
		"""
		uses pydicom.contrib.pydicom_series.py
		"""
		
		print 'loading folder '+folder
		
		# add /
		if folder[-1] != '/':
			self.read_folder = folder+'/'
		else:
			self.read_folder = folder
		
		# Get directory list of files
		directoryList = os.listdir(self.read_folder)
		directoryList.sort()
		
		# Find file pattern in directory list of files
		reFilePattern = re.compile(filePattern, re.IGNORECASE)
		files = []
		for f in directoryList:
			if reFilePattern.search(f):
				files.append(self.read_folder+'/'+f)
		
		if len(files)==0:
			raise IOError, 'No files found'
			
		if nSlice!=None:
			print 'loading {} slices'.format(nSlice)
			files = [files[i] for i in xrange(nSlice)]

		# Open a file to get image size
		try:
			slice0 = dicom.read_file(files[0], force=True)
		except TypeError:
			slice0 = dicom.read_file(files[0])
		self.slice0 = slice0
		
		# load stack
		print('Loading {} slices.'.format(len(files)))
		try:
			stack = dicomSeries.read_files( files, showProgress=True, readPixelData=True, force=True)[0]
		except TypeError:
			stack = dicomSeries.read_files( files, showProgress=True, readPixelData=True)[0]
		
		self.I = stack.get_pixel_array().astype(scipy.int16)
		self.info = stack.info
		
		#======================================================#
		# set axes to be l-r, a-p, s-i
		self.I = self.I.transpose( [2,1,0] )		# original
		# self.I = self.I[:,:,::-1]
		self.voxelSpacing = scipy.array(stack.sampling[::-1])	# original
		#======================================================#
		
		if sliceSpacingOveride!=None:
			self.voxelSpacing[2] = sliceSpacingOveride
		self.voxelOrigin = scipy.array(stack.info.ImagePositionPatient)
		# self.voxelOffset = self.voxelOrigin - (-1.0*self.voxelSpacing)*self.I.shape

		self.stack = stack
	#==================================================================#
	def testPixelSpacing(self, slice):
        
		dSlice = 0
        
		if self.pixelSpacing==None:
			self.pixelSpacing = slice.PixelSpacing
			self.pixelTolerance = 0.0001*self.pixelSpacing[0];
			
		if self._previousSliceLocation==None:
			self._previousSliceLocation = slice.SliceLocation
		elif self.sliceSpacing==None:
			self.sliceSpacing = slice.SliceLocation-self._previousSliceLocation
			self.sliceTolerance = 0.0001*self.sliceSpacing;
		
		dPixel = scipy.absolute([slice.PixelSpacing[0]-self.pixelSpacing[0], \
				slice.PixelSpacing[1]-self.pixelSpacing[1]])
		
		if self.sliceSpacing:
			dSlice = scipy.absolute((slice.SliceLocation-self._previousSliceLocation)-self.sliceSpacing)
			
		if dPixel[0]>self.pixelTolerance or dPixel[1]>self.pixelTolerance \
				or dSlice>self.sliceTolerance:
			print 'Warning: Inconsistent pixel or slice spacing ', dPixel, ' ', dSlice
			
		self._previousSliceLocation = slice.SliceLocation
	
	#==================================================================#
	# def index2CoordOld( self, I, negSpacing=False, zShift=False ):
	# 	if negSpacing:
	# 		X = -self.voxelSpacing*I + self.voxelOffset
	# 	else:
	# 		X = self.voxelSpacing*I + self.voxelOffset

	# 	if zShift:
	# 		X[:,2] -= self.I.shape[2]*self.voxelSpacing[2]
	# 	return X	

	def index2Coord( self, I, negSpacing=False, zShift=False ):
		if negSpacing:
			X = -self.voxelSpacing*I + self.voxelOrigin
		else:
			X = self.voxelSpacing*I + self.voxelOrigin

		if zShift:
			if len(X.shape)==2:
				X[:,2] -= self.I.shape[2]*self.voxelSpacing[2]
			elif len(X.shape)==1:
				X[2] -= self.I.shape[2]*self.voxelSpacing[2]
		return X
		
	# def coord2IndexOld( self, X ):
	# 	I = ( (X - self.voxelOffset) / (-1.0*self.voxelSpacing) ).astype(int)
	# 	I[:,2] -= self.I.shape[2]
	# 	return I

	def coord2Index(self, X, zShift=False, negSpacing=False, roundInt=True):
		"""
		converts physical coords p into image index based on self.voxelSpacing
		and self.voxelOrigin
		"""

		# return (p / self.voxelSpacing) + self.voxelOffset
		if negSpacing:
			# print 'neg spacing'
			ind = (X - self.voxelOrigin) / (-self.voxelSpacing)
		else:
			# ind = scipy.around( (X - self.voxelOffset) / (self.voxelSpacing) ).astype(int)	# original
			ind = (X - self.voxelOrigin)/(self.voxelSpacing)
			# ind = scipy.around( (X - self.voxelOrigin)/(self.voxelSpacing) ).astype(int) + [0,0,self.I.shape[2]]
		
		if roundInt:
			ind = scipy.around(ind).astype(int)

		if zShift:
			# print 'z shifting'
			if len(ind.shape)==2:
				ind[:,2] += self.I.shape[2]
				# ind[:,2] += self.I.shape[2]
			elif len(ind.shape)==1:
				ind[2] += self.I.shape[2]
				# ind[2] += self.I.shape[2]
		return ind
	
	#==================================================================#
	def checkIndexInBounds(self, ind):
		"""
		returns True if a voxel coord is in image array bounds.
		"""

		if any(ind<0):
			return False
		elif any(ind>(scipy.array(self.I.shape)-1)):
			return False
		else:
			return True

	def checkIndexIsMasked(self, ind):
		if not self.isMasked:
			return False
		else:
			try:
				return self.I.mask[ind[0], ind[1], ind[2]]
				# return scipy.take(self.I.mask, ind)
			except IndexError:
				return True
	#==================================================================#
	#~ def LoadImageArray(self):
		#~ """ Get a numpy array representation of the loaded image
		#~ """
		#~ 
		#~ itk_py_converter = itk.PyBuffer[self.imageType]
		#~ self.I = itk_py_converter.GetArrayFromImage(self.rawImage)
		#~ self.I = itk_py_converter.GetArrayFromImage( self.reader.GetOutput() )
		#~ self.I = scipy.array( self.I, dtype = int )
		#~ del itk_py_converter
		#~ print "Image array loaded, size = "+str(self.I.shape)
		#~ return
	
	#==================================================================#	
	#~ def GetArrayImage(self, imageArray, OutputImageType = None):
		#~ """ return an ITK image from an input imageArray
		#~ """
		#~ 
		#~ if OutputImageType == None:
			#~ itk_py_converter = itk.PyBuffer[self.imageType]
		#~ else:
			#~ itk_py_converter = itk.PyBuffer[OutputImageType]
		#~ itk_image = itk_py_converter.GetImageFromArray(imageArray)
		#~ return itk_image
		#~ del itk_py_converter
		#~ 
		#~ return itk_py_converter.GetImageFromArray(imageArray)

	#==================================================================#	
	#~ def writeArray2Slices(self, filenameFormat, dir):
		#~ """ Writes self.I to image slices using the given filenameFormat
		#~ into the given directory dir
		#~ """
		#~ self.imageType = itk.Image[itk.UC, 3]
		#~ image = self.GetArrayImage( self.I )
		#~ imageTypeOut = itk.Image[itk.UC, 2]
#~ 
		#~ region = image.GetLargestPossibleRegion()
		#~ start = region.GetIndex()
		#~ size = region.GetSize()
		#~ fileSeriesOut = itk.NumericSeriesFileNames.New()
		#~ fileSeriesOut.SetSeriesFormat(dir+"/"+filenameFormat)
		#~ fileSeriesOut.SetStartIndex(start.GetElement(2))
		#~ fileSeriesOut.SetEndIndex(start.GetElement(2) + size.GetElement(2) - 1)
		#~ fileSeriesOut.SetIncrementIndex(1)
#~ 
		#~ seriesWriter = itk.ImageSeriesWriter[self.imageType, imageTypeOut].New()
		#~ seriesWriter.SetInput(image)
		#~ seriesWriter.SetFileNames( fileSeriesOut.GetFileNames() )
		#~ 
		#~ # renderImage(image)
		#~ print "writing slices..."
		#~ seriesWriter.Update()
		#~ print "writing complete..."
		#~ #seriesWriter.Delete()
		#~ return
		
	#==================================================================#
	def writeSlice( self, prefix ):
		
		writeSlice( self.I, prefix )
		
		#~ numLength = int( scipy.log10( self.I.shape[0] ) ) + 2
		#~ for i in range(self.I.shape[0]):
			#~ filename = prefix + "_%.*d"%(numLength, i ) +'.png'
			#~ imsave( filename, self.I[i,:,:] )	
		
		return
		
	#==================================================================#	
	def _updateI( self ):
		""" recalculates CoM and principal axes and updates renderer
		after I is modified
		"""
		self.shape = self.I.shape
		self.calculateCoM()
		self.calculatePrincipalAxes()
		# self.renderer.setImage( self.I )
		# self.renderer.setCoM( self.CoM, self.pAxes, self.pAxesMag )
		#~ self.renderer.setCoM( self.CoM, self.pAxes )
		
		return		
		
	#==================================================================#
	# sub volume methods										       #
	#==================================================================#
	def createSubVolumes(self, name, origin, dimension ):
		""" instantiate a model object from a subvolume of self.I.
		The subvolume begins at origin = [ x,y,z ] and extends in each
		direction by dimension = [ d0, d1, d2 ]
		
		returns the new subvolume instance
		"""
		
		subI = self.I[ origin[0]: origin[0] + dimension[0],
					   origin[1]: origin[1] + dimension[1],
					   origin[2]: origin[2] + dimension[2] ]
		
		subV = Scan( name )
		subV.setImageArray( subI )
		
		self.sV[ name ] = subV
		
		return subV

	#==================================================================#
	def createSubSlice( self, axis, number ):
		""" instantiates a subslice object
		"""
		
		if number > self.I.shape[axis]:
			print 'ERROR: Scan.createSubSlice: invalid slice number (', str(number), ') in axis', str(axis)
			return 
		elif axis == 0:
			return slice( self.I[number, :, :], number, axis )
		elif axis == 1:
			return slice( self.I[:, number, :], number, axis )
		elif axis == 2:
			return slice( self.I[:, :, number], number, axis )
		else:
			print 'ERROR: Scan.createSubSlice: invalid axis', str(axis)
			return

	#==================================================================#	
	def createSubSliceNormal( self, P, N, sliceShape=(100,100), order=3 ):
		""" returns image slice given a centre point P and normal vector N
		"""
		
		# reverse hack, not sure why 
		#~ P = scipy.array(P)[::-1]
		#~ N = norm(scipy.array(N))[::-1]
		# calc affine matrix going from image normal to slice normal
		sliceV0 = N	# z
		#~ sliceV1 = norm( scipy.subtract( [P[0], P[1], evalPlaneZ( P, N, P[0]+1.0, P[1]+1.0 )], P ) )
		#~ sliceV1 = norm( scipy.subtract( [evalPlaneZ( P, N, P[2]+1.0, P[1]+1.0 ), P[1]+1.0, P[2]+1.0], P ) )	# y
		sliceV1 = norm( scipy.subtract( [evalPlaneZ( P, N, P[2]+1.0, P[1] ), P[1], P[2]+1.0], P ) )	# y
		sliceV2 = scipy.cross( sliceV0, sliceV1 )	# x
		
		flatAxes = [P, scipy.eye(3, dtype=float)]
		#~ sliceAxes = [P, scipy.array([sliceV2, sliceV1, sliceV1]) ] # another reverse hack, not sure why either
		sliceAxes = [P, scipy.array([sliceV0, sliceV1, sliceV2]).transpose() ]
		
		aMatrix = alignment_analytic.calcAffine( flatAxes, sliceAxes )
		#~ aMatrix = alignment.calcAffine( sliceAxes, flatAxes )
		
		# generate grid coords to evaluate image at
		#~ flatX = scipy.linspace( P[0] - sliceShape[0]/2.0, P[0]+sliceShape[0]/2.0, sliceShape[0] )
		#~ flatY = scipy.linspace( P[1] - sliceShape[1]/2.0, P[1]+sliceShape[1]/2.0, sliceShape[1] )
		
		flatX = scipy.linspace( P[2] - sliceShape[1]/2.0, P[2]+sliceShape[1]/2.0, sliceShape[1] )
		flatY = scipy.linspace( P[1] - sliceShape[0]/2.0, P[1]+sliceShape[0]/2.0, sliceShape[0] )
		X,Y = scipy.meshgrid( flatX, flatY )
		xCoords = X.ravel()
		yCoords = Y.ravel()
		zCoords = scipy.ones( xCoords.shape[0] )*P[0]
		
		# image stack indices are z, y, x
		#~ flatCoords = scipy.array( [xCoords, yCoords, zCoords, scipy.ones(xCoords.shape[0])] )
		flatCoords = scipy.array( [zCoords, yCoords, xCoords, scipy.ones(xCoords.shape[0])] )
		sliceCoords = scipy.dot( aMatrix, flatCoords )
		
		sliceArray = map_coordinates( self.I, sliceCoords, order=order )
		sliceImage = scipy.reshape( sliceArray, sliceShape )
		
		return slice( sliceImage, None, None, origin=P, normal=N )
		
	#==================================================================#
	def calculateSubVolumeCoMs(self):
		""" calculates the CoMs of all subvolumes
		"""
		
		for sv in self.sV.values():
			sv.calculateCoM()
			
		return

	#==================================================================#
	# shape descriptor methods									   	   #
	#==================================================================#
	def calculateCoM( self ):
		""" calculate self.I's centre of mass
		"""
		
		# large numbers are involved
		I = self.I.astype( scipy.int64 )
		if I.min() < 0.0:
			I -= I.min()
		
		# mass
		self.M00 = float( I.sum() )
		self.CoM = [ 0.0, 0.0, 0.0 ]					
	
		if self.M00 == 0.0:
			print 'WARNING Section '+self.name+': zero-mass object'
			self.isZeroMass = True
			return
		else:
			i0 = scipy.arange(I.shape[0])
			i1 = scipy.arange(I.shape[1])
			i2 = scipy.arange(I.shape[2])
			# centre of mass		
			self.CoM[0] = scipy.dot(I.sum(1).sum(1), i0)/self.M00	
			self.CoM[1] = scipy.dot(I.sum(0).sum(1), i1)/self.M00		
			self.CoM[2] = scipy.dot(I.sum(0).sum(0), i2)/self.M00
			
			# centered voxel coordinates
			self._iC = ( i0 - self.CoM[0], i1 - self.CoM[1], i2 - self.CoM[2] )
			
			# image has mass	
			self.isZeroMass = False
			
			return self.CoM
	
	#==================================================================#
	def calculatePrincipalAxes( self ):
		""" calculate self.I's principal axes and magnitudes.
		Principal axes are in the columns of self.pAxes
		"""
		if self.isZeroMass:
			print 'ERROR: Scan.calculatePrincipalAxes: zero-mass object'
			return
			
		self.momentMatrix = scipy.zeros( (3,3) )
		
		# populate inertial matrix
		self.momentMatrix[0,0] = self._calcCentralMoment( 2, 0, 0 )
		self.momentMatrix[1,1] = self._calcCentralMoment( 0, 2, 0 )
		self.momentMatrix[2,2] = self._calcCentralMoment( 0, 0, 2 )
		self.momentMatrix[0,1] = self.momentMatrix[1,0] = self._calcCentralMoment( 1, 1, 0 )
		self.momentMatrix[0,2] = self.momentMatrix[2,0] = self._calcCentralMoment( 1, 0, 1 )
		self.momentMatrix[1,2] = self.momentMatrix[2,1] = self._calcCentralMoment( 0, 1, 1 )
		
		# calculate eigenvs of inertia matrix to find principal MoI
		( self.pAxesMag, self.pAxes ) = eigh( self.momentMatrix ) 
		
		return ( self.pAxes, self.pAxesMag )
	
	#==================================================================#
	def _calcCentralMoment(self, p, q, r, scaleNorm = 1):
		""" Calculate central geometric moments of the section image
		p, q, r are the orders in the x, y, z directions
		"""
		
		if self.isZeroMass:
			print 'ERROR: Scan._calcCentralMoment: zero-mass object'
			return

		Itemp = self.I.astype(float)
		if Itemp.min() < 0.0:
			Itemp -= Itemp.min()
		
		if p > 0:
			for i in range(0, Itemp.shape[0]):
				Itemp[i,:,:] = Itemp[i,:,:] * (self._iC[0][i] ** p)
				
		if q > 0:
			for j in range(0, Itemp.shape[1]):
				Itemp[:,j,:] = Itemp[:,j,:] * (self._iC[1][j] ** q)
				
		if r > 0:
			for k in range(0, Itemp.shape[2]):
				Itemp[:,:,k] = Itemp[:,:,k] * (self._iC[2][k] ** r)
		
		# scale normalise	
		if scaleNorm:
			u = Itemp.sum()/(self.M00 ** ( 1.0 + ((p+q+r)/3.0) ))
		else:
			u = Itemp.sum()
		
		return u		
	
	#==================================================================#
	def PCA( self ):
		self.PCA = PCA(self.I)
		self.PCA.ConstructDataArray()
		self.PCA.DoPCA()
		self.IPCA = self.PCA.ConstructNewImageArray()	
	
	
	#==================================================================#
	# geometric transformation methods								   #
	#==================================================================#
	def alignPAxes( self ):
		""" rotate the image to align the principal axes with the
		coordinate system axes.
		
		Aligns largest pAxes with (1,0,0), 2nd largest with (0,1,0)
		"""
		
		if not self.isZeroMass:
			# calculate angle between pMax and the projection of pMax on the x-y plane
			pMax = self.pAxes[ :, self.pAxesMag.argmax() ]
			print pMax
			pMaxYZ = scipy.array( [ 0.0, pMax[1], pMax[2] ])
			theta = scipy.arccos( scipy.inner(pMaxYZ, [0.0,0.0,1.0]) / mag(pMaxYZ) )
			theta = -180.0 * theta/scipy.pi
			if theta > 90.0:
				theta -= 90.0
			# rotate the image in the y-z plane so that pMax lies in XY plane
			self.rotate( theta, (1,2) )
			
			# caculate angle between new pMax and (1,0,0)
			pMax = self.pAxes[ :, self.pAxesMag.argmax() ]
			print pMax
			phi = scipy.arccos( scipy.inner(pMax, [1.0,0.0,0.0]) / mag(pMax) )
			phi = -180.0 * phi/scipy.pi
			if phi > 90.0:
				phi -= 90.0
			# rotate the image in the x-y plane so that pMax lies on [1,0,0]
			self.rotate( phi, (0,2) )
			
			# calculate angle between new 2nd largest pAxes and (0,1,0)
			p2 = self.pAxes[ :, self.pAxesMag.argsort()[-2] ]
			print p2
			p2XZ = scipy.array( [ 0.0, p2[1], p2[2] ])
			ro = scipy.arccos( scipy.inner(p2XZ, [0.0,1.0,0.0]) / mag(p2XZ) )
			ro = -180.0 * ro/scipy.pi
			if ro > 90.0:
				ro -= 90.0
			# rotate the image in the y-z plane so that p2 lies on [0,1,0]
			self.rotate( ro, (1,2) )
			
			return (theta, phi, ro)
		else:
			print 'ERROR: Scan.alignPAxes: zero mass image'
			return None
			
	#==================================================================#
	def rotate( self, angle, axes ):
		""" uses scipy.ndimage.rotate to rotate self.I in the plane
		defined by axes= [ axis1, axis2] by angle in degrees
		"""
		self.I = rotate( self.I, angle, axes, order = 1 )
		self._updateI()
		return
	
	#==================================================================#	
	def zoom( self, scale, order=3 ):
		""" uses scipy.ndimage.zoom to scale self.I isotropically by 
		scale factor scale.
		"""
		self.I = zoom( self.I, scale, order=order )
		print 'New image shape:', self.I.shape
		self._updateI()
		return
	
	#==================================================================#
	def affine( self, matrix, offset=None, order=1 ): 
		""" deforms self.I using a 3x3 symmetric transformation matrix 
		mapping output to input. Will try to predict the output size 
		needed to capture the whole transformed image
		"""
		matrix = scipy.array(matrix)
		#~ S = self.I.shape
		#~ if len( matrix.shape ) > 1:
			#~ box = scipy.array( [[0.0,0.0,0.0,1.0],\
						  #~ [S[0],0.0,0.0,1.0],\
						  #~ [0.0,S[1],0.0,1.0],\
						  #~ [S[0],S[1],0.0,1.0],\
						  #~ [0.0,0.0,S[2],1.0],\
						  #~ [S[0],0.0,S[2],1.0],\
						  #~ [0.0,S[1],S[2],1.0],\
						  #~ [S[0],S[1],S[2],1.0]] ).transpose()
						 #~ 
			#~ newBox = scipy.dot( inv(matrix), box[:3,:] )
			#~ outputShape = scipy.array( [ newBox[0].max() - newBox[0].min(),\
							#~ newBox[1].max() - newBox[1].min(),\
							#~ newBox[2].max() - newBox[2].min() ] ).round()
		#~ else:
			#~ outputShape = scipy.multiply( S, matrix ).round()
		
		outputShape = self.getAffineOutputShape( matrix )
			
		print 'outputShape:', outputShape
		#~ self.I = affine_transform( self.I, matrix, order=order )
		if matrix.shape==(4,4):
			offset = matrix[:3,3]
			matrix = matrix[:3,:3]
			
		self.I = affine_transform( self.I, matrix, output_shape=outputShape, offset=offset, order=order )
		print 'New image shape:', self.I.shape
		self._updateI()
		return
	
	def getAffineOutputShape( self, matrix ): 
		matrix = scipy.array(matrix)
		S = self.I.shape
		if len( matrix.shape ) > 1:
			box = scipy.array( [[0.0,0.0,0.0,1.0],\
						  [S[0],0.0,0.0,1.0],\
						  [0.0,S[1],0.0,1.0],\
						  [S[0],S[1],0.0,1.0],\
						  [0.0,0.0,S[2],1.0],\
						  [S[0],0.0,S[2],1.0],\
						  [0.0,S[1],S[2],1.0],\
						  [S[0],S[1],S[2],1.0]] ).transpose()
						 
			newBox = scipy.dot( inv(matrix), box )
			
			#~ outputShape = scipy.array( [ newBox[0].max() - newBox[0].min(),\
							#~ newBox[1].max() - newBox[1].min(),\
							#~ newBox[2].max() - newBox[2].min() ] ).round()
			
			outputShape = scipy.array( [ newBox[0].max(),\
										 newBox[1].max(),\
										 newBox[2].max() ]).round()
		else:
			outputShape = scipy.multiply( S, matrix ).round()
		
		return outputShape

	def medianFilter(self, size):
		self.I = median_filter(self.I, size=size)
		
	#==================================================================#
	def pad( self, t, padval = 0):
		""" pad self.I by t voxels along each edge or face with 
		value padv
		"""
		
		dim = len(self.I.shape)
		
		if dim > 3:
			print "pad ERROR: invalid dimension"
			return None
			
		newsize = scipy.zeros(dim)
		for i in range(0,dim):
			newsize[i] = self.I.shape[i]+(2*t)
			
		new = scipy.zeros(newsize, self.I.dtype)
		new = new + padval
		
		if dim == 1:
			new[t:t+self.I.shape[0]] = self.I
		elif dim == 2:
			new[t:t+self.I.shape[0], t:t+self.I.shape[1]] = self.I
		elif dim == 3:
			new[t:t+self.I.shape[0], t:t+self.I.shape[1], t:t+self.I.shape[2]] = self.I
			
		self.I = new
		self._updateI()
		return 1
	
	#==================================================================#
	def crop( self, pad, padv = 0.0):
		""" crop self.I to a box bounding non-zero voxels padded by pad
		voxels with value padv
		"""
		nz = scipy.nonzero(self.I)
		xrange = [nz[0].max(), nz[0].min()]
		yrange = [nz[1].max(), nz[1].min()]
		zrange = [nz[2].max(), nz[2].min()]
		
		#~ print "non-zero dimensions:"
		#~ print xrange
		#~ print yrange
		#~ print zrange
		
		cropArray = scipy.zeros([
		xrange[0]-xrange[1]+2*pad+1,
		yrange[0]-yrange[1]+2*pad+1,
		zrange[0]-zrange[1]+2*pad+1], dtype = self.I.dtype)
		
		if pad:
			cropArray += padv
		
		print "precrop shape = ", self.I.shape
		print "crop shape = ", cropArray.shape
		
		if pad == 0:
			cropArray = self.I[
			xrange[1]:xrange[0]+1,
			yrange[1]:yrange[0]+1,
			zrange[1]:zrange[0]+1,]
		else:	
			cropArray[pad:-pad, pad:-pad, pad:-pad] = self.I[
			xrange[1]:xrange[0]+1,
			yrange[1]:yrange[0]+1,
			zrange[1]:zrange[0]+1]
		
		self.I = cropArray
		self._updateI()
		return 1
	
	#==================================================================#
	# ITK image processing methods									   #
	#==================================================================#	
	#~ def ITKThreshold(self, inputImage, lower, outsideValue = 0):
 		#~ filter = itk.ThresholdImageFilter[self.imageType].New()
 		#~ filter.SetOutsideValue(outsideValue)
 		#~ filter.SetLower(lower)
 		#~ self.reader.Update()
 		#~ filter.SetInput(inputImage)
 		#~ filter.Update()
 		#~ self.thresholdImage = filter.GetOutput()
 		#~ self.thresholdArray = self.GetImageArray(self.thresholdImage, self.imageType)
 		#~ del filter
 		#~ return self.thresholdArray
 		#return filter.GetOutput()
 		
	#==================================================================#	
	def threshold(self, lower, outsideValue=0, replaceValue=None, inplace=True):
 		if replaceValue == None:
			replaceValue = self.I
 		if inplace:
			temp = scipy.where( self.I > lower, replaceValue, outsideValue )
			self.setImageArray( temp )
			return 1
		else:
			return where( self.I > lower, self.I, outsideValue )
 	
	#==================================================================#
	def sampleImage( self, samplePoints, maptoindices=0, outputType=float, order=1, zShift=True, negSpacing=False ):
		#~ pdb.set_trace()
		if maptoindices:
			samplePoints = self.coord2Index(samplePoints, zShift=zShift, negSpacing=negSpacing)
		
		s = map_coordinates(self.I, samplePoints.T, output=outputType, order=order)
		return s
					
	#==================================================================#					
	def ResampleImage(self, scaleFactor, inputImage = None):
		
		if inputImage == None:
			self.resample = Resample(self.rawImage, self.imageType, scaleFactor)
		else:
			self.resample = Resample(inputImage, self.imageType, scaleFactor)
			
		self.resampledImage = self.resample.do()
		self.resampledArray = self.GetImageArray(self.resampledImage, self.imageType)
		print "Resampled image size =", self.resampledArray.shape	

	#==================================================================#
	# vtk visualisation												   #
	#==================================================================#	
	# def viewVolume( self ):
	# 	if self.I == None:
	# 		print 'ERROR: Scan.viewVolume: no image loaded'
	# 		return
	# 	else:
	# 		self.renderer.renderVolume()
	
	#==================================================================#
	def getMIP( self, axis, sliceRange=None ):
		if sliceRange==None:
			mip = scipy.fliplr( self.I.max(axis).T )
		else:
			if axis==0:
				mip = scipy.fliplr( self.I[sliceRange[0]:sliceRange[1],:,:].max(axis).T )
			elif axis==1:
				mip = scipy.fliplr( self.I[:,sliceRange[0]:sliceRange[1],:].max(axis).T )
			elif axis==2:
				mip = scipy.fliplr( self.I[:,:,sliceRange[0]:sliceRange[1]].max(axis).T )
		return mip
		
	def viewMIP( self, axis, sliceRange=None, vmin=-200, vmax=2000 ):
		mip = self.getMIP( axis, sliceRange )
		p = plot.imshow( mip, cmap=cm.gray, vmin=vmin, vmax=vmax )
		return p
	#==================================================================#		
	# def viewSurface( self, isoValue = 100 ):
	# 	if self.I == None:
	# 		print 'ERROR: Scan.viewVolume: no image loaded'
	# 		return
	# 	else:
	# 		self.renderer.renderContour( [isoValue] )
	
	# def add_mesh_points_to_renderer( self ):
		
	# 	if self.mesh:
	# 		# get points
	# 		meshPoints = self.mesh.get_all_point_positions()
	# 		for p in meshPoints:
	# 			self.renderer.addNode( p )
	
	# def add_mesh_landmarks_to_renderer( self ):
	# 	if self.mesh:
	# 		for p in self.mesh.named_points_map.values():
	# 			self.renderer.addNode( self.mesh.get_point_position(p), colourStr='green' )
	
	# def restart_renderer( self ):
		
	# 	del self.renderer
	# 	self.renderer = vtkRender.vtkRenderer()
	# 	self.renderer.setImage( self.I )
	# 	self.renderer.setCoM( self.CoM, self.pAxes, self.pAxesMag )
	# 	#~ self.renderer.setCoM( self.CoM, self.pAxes )

	#==================================================================#
	# qCT methods
	def _samplePhantoms( self, useSamples ):
		self.phantom = phantomSampler()
		self.phantom.loadPhantomTemplate()
		self.phantom.setScan( self )
		self.phantomValues = self.phantom.samplePhantoms(useSamples)
		
		if useSamples=='mean':
			print 'average phantom values: '+' '.join( ['%4.1f'%v for v in self.phantomValues] )
		elif useSamples=='all':
			print 'average phantom values: '+' '.join( ['%4.1f'%v.mean() for v in self.phantomValues] )
		
	def calibrateQCT( self, useSamples='all' ):
		
		self._samplePhantoms(useSamples)
		
		self.qCT = qCTLookup()
		self.qCT.setPhantomValues( self.phantomValues )
		self.qCT.calibrate()
		
	def getImageQCT( self, dtype=None ):
		if dtype==None:
			dtype = scipy.int16
		IBMD = self.qCT.int2BMD( self.I ).astype(dtype)
		return IBMD


class qCTLookup( object ):
	sigmaOffset = -0.2174
	betaOffset = +999.6
	
	H2ODensity = scipy.array([1012.2,1057,1103.6,1119.5,923.2])
	K2HPO4Density = scipy.array([-51.8,-53.4,58.9,157,375.8])
	
	def __init__(self):
		self.phantomValues = None
		self.phantomValuesMinusWater = None
		self.sigma = None
		self.beta = None
		self.r = None
		self.stderr = None
		
	def setPhantomValues( self, x ):
		self.phantomValues = scipy.array(x)
		if len(self.phantomValues.shape)==2:
			self.phantomValuesMinusWater = self.phantomValues - self.H2ODensity[:,scipy.newaxis]
		elif len(self.phantomValues.shape)==1:
			self.phantomValuesMinusWater = self.phantomValues - self.H2ODensity
		else:
			raise ValueError, 'phantom values wrong shape'
		
	def calibrate(self):
		# fit straight line to K2HPO4Density(x) vs self.phantomValuesMinusWater(y)
		if self.phantomValuesMinusWater==None:
			raise AttributeError, 'no phantom values set'
			
		# if multiple values for each rod
		if len(self.phantomValuesMinusWater.shape)>1:
			K2HPO4 = self.K2HPO4Density.repeat( [len(v) for v in self.phantomValuesMinusWater] )
		else:
			K2HPO4 = self.K2HPO4Density
		
		#~ pdb.set_trace()
		print 'calibrating qCT'
		sigma, beta, r, p, stderr = linregress(K2HPO4, self.phantomValuesMinusWater.flatten())
		print 'r-value:', r
		print 'p-value:', p
		self.sigma = sigma + self.sigmaOffset
		self.beta = beta + self.betaOffset
		self.r = r
		self.stderr = stderr
		print 'sigma:', self.sigma
		print 'beta:', self.beta
		
		return self.sigma, self.beta
		
	def int2BMD( self, x ):
		bmd = (x - self.beta)/self.sigma
		return bmd
		
class phantomSampler(object):
	phantomImageFilename = '../CT_scans/phantom_template_2008_0909.npy'
	phantomImageSpacing = scipy.array([ 1.19,  1.19,  1.6 ])
	phantomMidY = 16
	nRods = 5
	phantomRodCentres = scipy.array([ 
									  [40,15],
									  [65,13],
									  [88,12],
									  [112,13],
									  [137,15]
									  ])
	
	res = 1.0
	sigma = 1.0
	sampleSlices = [0,10,20,30,40]	# shortest phantom seen is about 50 pixels in z direction
	yRangeLim = [50,200,1]
	yRange = range(yRangeLim[0], yRangeLim[1], yRangeLim[2] ) 
	sampleHalfW = 3
	SSDMax = 5.0
								  
	def __init__(self):
		self.scan = None
		self.phantomImage = None
		self.rodSamples = None
		self.rodValues = None
		self.phantomOrigin = None
		self.bestProfile = None
		self.bestErr = None
	
	def setScan( self, scan ):
		self.scan = scan
		self.scanSampleCoords = scipy.arange(0, self.scan.I.shape[0]-1, self.res/self.scan.voxelSpacing[0])
	
	def loadPhantomTemplate( self ):
		self.phantomImage = scipy.load(self.phantomImageFilename).astype(scipy.int16)
		self.phantomSampleCoords = scipy.arange(0, self.phantomImage.shape[0]-1, self.res/self.phantomImageSpacing[0])
	
	def samplePhantoms( self, useSamples='all' ):
		self.rodSamples = []	# shape: [slice, rods, x, y]
		self.bestProfiles = []
		self.bestErr = []
		print 'locating and sampling phantom'
		for s in self.sampleSlices:
			tempProfile = self.phantomImage[:,self.phantomMidY,s]
			errors, phantomOrigin, [x1,x2], [X1,X2], bestProfileSlice, bestErrSlice = self._findPhantomInSlice( tempProfile, s )
			rodSamplesSlice = self._samplePhantomSlice( self.sampleHalfW, phantomOrigin, s )
			
			#~ pdb.set_trace()
			print 'slice', s, 'SSD', bestErrSlice, 'mean values:', ' '.join( ['%4.1f'%v for v in rodSamplesSlice.mean(1).mean(1) ])
			
			if bestErrSlice < self.SSDMax:
				self.rodSamples.append( rodSamplesSlice )
				self.bestProfiles.append( bestProfileSlice )
				self.bestErr.append( bestErrSlice )
			else:
				print 'dropped'

		self.rodValues = self._calcRodValues(useSamples)
		return self.rodValues
			
	def _calcRodValues( self, mode ):
		self.rodSamples = scipy.array( self.rodSamples )
		if len(self.rodSamples)==0:
			raise PhantomError
		
		# rodSample dims: (slices, rods, y, x)
		
		if mode=='mean':
			# 1 mean for each rod. average across slice, x, and y
			self.rodValues = self.rodSamples.mean(0).mean(1).mean(1)
		elif mode=='all':
			# all values for each rod
			self.rodValues = scipy.array( [self.rodSamples[:,i,:,:].flatten() for i in xrange(self.nRods)] )
		
		return self.rodValues

	def _scanProfile( self, p, func, filterLength ):
		pPadded = scipy.hstack((p,[p[-1]]*(filterLength-1)))
		out = scipy.zeros(len(p))
		for i in xrange(len(p)):
			out[i] = func( pPadded[i:i+filterLength] )
		return out
	
	def _findPhantomInSlice( self, templateProfile, scanSlice ):
		
		tempPRaw = map_coordinates(templateProfile, [self.phantomSampleCoords])
		tempP = gaussian_filter1d( tempPRaw, self.sigma )
		tempPN = tempP/max(abs(tempP))
		
		def NSSD( x ):
			xN = x/max(abs(x))
			return ((xN-tempPN)**2.0).sum()
			
		def SSD( x ):
			return ((x-tempP)**2.0).sum()
		
		errors = []
		bestErr = scipy.inf
		bestProfile = None
		
		for i,y in enumerate(self.yRange):
			dataPRaw = map_coordinates(self.scan.I[:,y,scanSlice], [self.scanSampleCoords])
			dataP = gaussian_filter1d( dataPRaw, self.sigma )
		
			err = self._scanProfile( dataP, NSSD, len(tempPN) )
			errors.append(err)
			errMinArg = err.argmin()
			if err[errMinArg] < bestErr:
				bestErr = err[errMinArg]
				bestProfile = dataPRaw
		
		errors = scipy.array(errors)
		x = errors.argmin()
		# scanned image coords
		x1 = x/errors.shape[1]
		x2 = x - (x1*errors.shape[1]) # scipy.mod(x,matchProfiles.shape[1])
		# main image coords
		X1 = x1*self.yRangeLim[2] + self.yRangeLim[0]
		X2 = float(x2)*self.res/self.scan.voxelSpacing[0]
		
		phantomOrigin = [X2, X1 - (self.phantomMidY * self.phantomImageSpacing[0] / self.scan.voxelSpacing[0])]
		bestProfile = bestProfile
		bestErr = bestErr
		
		return errors, phantomOrigin, [x1,x2], [X1,X2], bestProfile, bestErr
		
	def _samplePhantomSlice( self, halfW, phantomOrigin, scanSlice ):
		#~ phantomOrigin = [X2, X1 - (16*phantomImageSpacing[0]/scan.voxelSpacing[0] )]
		rodSamples = []
		for rod in self.phantomRodCentres:
			r0 = rod[0]*self.phantomImageSpacing[0]/self.scan.voxelSpacing[0]
			r1 = rod[1]*self.phantomImageSpacing[1]/self.scan.voxelSpacing[1]
			rodSamples.append( self.scan.I[ phantomOrigin[0]+r0-halfW : phantomOrigin[0]+r0+halfW,
											phantomOrigin[1]+r1-halfW : phantomOrigin[1]+r1+halfW,
											scanSlice]
							  )
							             
		return scipy.array(rodSamples)
			
#======================================================================#
#======================================================================#
class slice:
	""" 2D image class
	"""
	def __init__(self, image, number, axis, origin=None, normal=None ):
		self.I = image
		self.number = number
		self.axis = axis
		self.origin=origin
		self.normal=normal
		self.calculateCoM()
		self.calculatePrincipalAxes()
		self.polarSpline = None
		self.quadratic = None
		self.R = None
		self.theta = None
				
		if ( self.origin != None ) and ( self.normal != None ):
			self._calc3DAxes()
			
	#==================================================================#
	# shape descriptor methods									   	   #
	#==================================================================#
	def calculateCoM( self ):
		""" calculate self.I's centre of mass
		"""
		# large numbers are involved
		I = self.I.astype( scipy.int64 )
		
		# mass
		self.M00 = float( I.sum() )
		self.CoM = [ 0.0, 0.0 ]					
	
		if self.M00 == 0.0:
			print 'WARNING slice '+str(self.number)+': zero-mass image'
			self.isZeroMass = True
			return
		else:
			i0 = scipy.arange(I.shape[0])
			i1 = scipy.arange(I.shape[1])
			# centre of mass		
			self.CoM[0] = scipy.dot(I.sum(1), i0)/self.M00	
			self.CoM[1] = scipy.dot(I.sum(0), i1)/self.M00		
			
			# centered voxel coordinates
			self._iC = ( i0 - self.CoM[0], i1 - self.CoM[1] )
			
			# image has mass	
			self.isZeroMass = False
			
			return self.CoM
	
	#==================================================================#
	def calculatePrincipalAxes( self ):
		""" calculate self.I's principal axes and magnitudes
		"""
		if self.isZeroMass:
			print 'ERROR: Scan.calculatePrincipalAxes: zero-mass object'
			return
			
		self.momentMatrix = scipy.zeros( (2,2) )
		
		# populate inertial matrix
		self.momentMatrix[0,0] = self._calcCentralMoment( 2, 0, 0 )
		self.momentMatrix[1,1] = self._calcCentralMoment( 0, 2, 0 )
		self.momentMatrix[0,1] = self.momentMatrix[1,0] = self._calcCentralMoment( 1, 1, 0 )
		
		# calculate eigenvs of inertia matrix to find principal MoI
		( self.pAxesMag, self.pAxes ) = eigh( self.momentMatrix ) 
		
		return ( self.pAxes, self.pAxesMag )
	
	#==================================================================#
	def _calcCentralMoment(self, p, q, scaleNorm = 1):
		""" Calculate central geometric moments of the section image
		p, q are the orders in the x, y directions
		"""
		
		if self.isZeroMass:
			print 'ERROR: Scan._calcCentralMoment: zero-mass object'
			return

		Itemp = self.I.astype(float)
		
		if p > 0:
			for i in range(0, Itemp.shape[0]):
				Itemp[i,:] = Itemp[i,:] * (self._iC[0][i] ** p)
				
		if q > 0:
			for j in range(0, Itemp.shape[1]):
				Itemp[:,j] = Itemp[:,j] * (self._iC[1][j] ** q)
				
		
		# scale normalise	
		if scaleNorm:
			u = Itemp.sum()/(self.M00 ** ( 1.0 + ((p+q)/2.0) ))
		else:
			u = Itemp.sum()
		
		return u		
	
	#==================================================================#
	def calculateSlicePolar( self ):
		""" calculates the polar coordinates of pixels in slice centered
		at the CoM
		"""
	
		# get all nonzero point indices
		xi, yi = scipy.nonzero( self.I )
		# center
		xi = xi - self.CoM[0]
		yi = yi - self.CoM[1]
		nTotal = xi.shape[0]
		
		# convert all point indices into radial coordinates
		R = scipy.sqrt( xi**2.0 + yi**2.0 )		# r
		theta = scipy.zeros( nTotal )
		for i in range( nTotal ):
			theta[i] = calcTheta( xi[i], yi[i] )	# theta
		
		self.theta = theta
		self.R = R
			
		return ( self.R, self.theta )	

	#==================================================================#
	def samplePointsRadial( self, n ):
		""" returns the coords of n non-zero pixels evenly distributed 
		radially around the slice CoM
		"""
		
		if (not self.R) or (not self.theta):
			self.calculateSlicePolar()
			
		X,Y = self.I.nonzero()
		tList = scipy.linspace( 0.0, 2*scipy.pi, n+1)[:-1]
		m = 5 # average of 5 closest
		tempX = 0.0
		tempY = 0.0
		
		pointsCart = []
		for t in tList:
			
			# find average coord of closest m points
			iSort = ((self.theta-t)**2.0).argsort()
			tempX = 0.0
			tempY = 0.0
			for i in iSort[:m]:
				tempX += X[i]
				tempY += Y[i]
			
			pointsCart.append( (tempX/m, tempY/m) )
			
			# find data point with closest matching t
			#~ i = ((T-t)**2.0).argmin()
			#~ pointsCart.append( (X[i], Y[i]) )
			
		return pointsCart
	
	#==================================================================#
	def samplePointsInThetaRangeSpline( self, n, coords='stack' ):
		"""returns the cartesian coords of n points sampled evenly 
		on self.polarSpline between self.theta.min() and self.theta.max. 
		coords = 'stack', 'slice', or 'sliceInd'
		"""
		if not self.polarSpline:
			self.fitSplinePolar()
		
		# evaluate n points in polar coords [[theta,r],[theta,r],...]
		theta = scipy.linspace( self.theta.min(), self.theta.max(), n+1 )
		polarPoints = scipy.zeros((n,2))
		for i in range( theta.shape[0]-1 ):
			polarPoints[i,0] = theta[i]
			polarPoints[i,1] = self.polarSpline(theta[i])
			
		# transform to cartesian coords [[x,y],[x,y],...]
		cartPoints = scipy.zeros((n,2))
		cartPoints[:,0] = polarPoints[:,1]* scipy.cos( polarPoints[:,0] )
		cartPoints[:,1] = polarPoints[:,1]* scipy.sin( polarPoints[:,0] )
		
		if coords=='stack':
			stackPoints = self.slice2StackCS( cartCoords )
			return stackPoints
		elif coords=='sliceInd':
			return cartPoints + self.CoM
		elif coords=='slice':
			return cartPoints
		else:
			print 'ERROR: slice.samplePointRadialSpline: unrecognised coords option'
			return 
				 
	#==================================================================#
	def samplePointsRadialSpline( self, n, coords='stack' ):
		""" returns the cartesian coords of n points sampled evenly 
		around 2pi self.polarSpline. coords = 'stack', 'slice', or 'sliceInd'
		"""
		if not self.polarSpline:
			self.fitSplinePolar()
		
		# evaluate n points in polar coords [[theta,r],[theta,r],...]
		theta = scipy.linspace( 0.0, 2*scipy.pi, n+1 )
		polarPoints = scipy.zeros((n,2))
		for i in range( theta.shape[0]-1 ):
			polarPoints[i,0] = theta[i]
			polarPoints[i,1] = self.polarSpline(theta[i])
			
		# transform to cartesian coords [[x,y],[x,y],...]
		cartPoints = scipy.zeros((n,2))
		cartPoints[:,0] = polarPoints[:,1]* scipy.cos( polarPoints[:,0] )
		cartPoints[:,1] = polarPoints[:,1]* scipy.sin( polarPoints[:,0] )
		
		if coords=='stack':
			stackPoints = self.slice2StackCS( cartCoords )
			return stackPoints
		elif coords=='sliceInd':
			return cartPoints + self.CoM
		elif coords=='slice':
			return cartPoints
		else:
			print 'ERROR: slice.samplePointRadialSpline: unrecognised coords option'
			return 
		
	#==================================================================#
	def fitSplinePolar( self, nk=12, order=3 ):
		""" fit a univariate spline to theta and R of the polar-unwrapped
		slice. returns residuals
		"""
		
		if (not self.R) or (not self.theta):
			self.calculateSlicePolar()
		
		theta = self.theta.copy()
		sortI = theta.argsort()
		theta.sort()
		RSort = scipy.zeros(self.R.shape)
		for i in range( sortI.shape[0] ):
			RSort[i] = self.R[sortI[i]]
		
		#~ k = scipy.linspace( 0.0, 2*scipy.pi, nk )
		k = scipy.linspace( theta.min(), theta.max(), nk )
		self.polarSpline = LSQUnivariateSpline( theta, RSort, k[1:-1], k=order )
		
		return self.polarSpline.get_residual()

	#==================================================================#
	def fitQuadratic( self ):
		""" fit quadratic curve to nonzero pixels
		"""
		
		# initialise curve
		self._initQuadratic()
		
		# data
		data = self.I.nonzero()
		data = scipy.array(data).transpose()
		
		fitErr = self.quadratic.fit( data )
		return fitErr
	
	#==================================================================#
	def _initQuadratic( self ):
		# initialise curve 
		p0 = scipy.array( self.CoM ) - self.I.shape[0]*0.5*self.pAxes[:,1]
		p1 = scipy.array( self.CoM )
		p2 = scipy.array( self.CoM ) + self.I.shape[0]*0.5*self.pAxes[:,1]
		initParams = scipy.transpose( [p0, p1, p2] )
		#~ print 'initial quadratic parameters:', initParams
		self.quadratic = quadraticCurve( 2, initParams )
		return self.quadratic
		
	#==================================================================#
	def getQuadraticMidpoint( self ):
		return self.quadratic.getMidpoint()
	#==================================================================#
	def viewSlice( self ):
		plot.figure()
		plot.imshow( self.I, cmap=cm.gray )
		plot.scatter( self.CoM[1], self.CoM[0], 'scatter' )
		
		if self.polarSpline:
			n = 100
			p = self.samplePointsInThetaRangeSpline( n, coords='sliceInd' )
			#~ p = scipy.vstack( [p, p[0]] )
			plot.plot(p[:,1], p[:,0])
		
		if self.quadratic:
			n = 100
			U = scipy.linspace(0.0,1.0,n)
			p=[]
			for u in U:
				p.append( self.quadratic.eval( u ) )
				
			p = scipy.array(p).transpose()
			plot.plot(p[1], p[0])
			centre = self.quadratic.params[:,1]
			plot.plot(centre[1],centre[0], 'scatter')
			
		plot.show()

	#==================================================================#
	def slice2StackCS( self, sliceCoords ):
		""" calculates 3D coord of slice indices: [ ind1, ind2 ]
		"""
		
		if len( sliceCoords.shape ) < 2:
			XOffset = sliceCoords[1] - self.I.shape[1]/2.0
			YOffset = sliceCoords[0] - self.I.shape[0]/2.0
			return self.origin + XOffset * self.sliceV2 + YOffset * self.sliceV1
		else:
			XOffset = sliceCoords[:,1] - self.I.shape[1]/2.0
			YOffset = sliceCoords[:,0] - self.I.shape[0]/2.0
			return self.origin + XOffset * self.sliceV2 + YOffset * self.sliceV1

	#==================================================================#
	def _calc3DAxes( self ):
		P = self.origin
		N = self.normal
		self.sliceV0 = N	# z
		self.sliceV1 = norm( scipy.subtract( [evalPlaneZ( P, N, P[2]+1.0, P[1]+1.0 ), P[1]+1.0, P[2]+1.0 ], P ) )	# y
		#~ sliceV1 = norm( scipy.subtract( [evalPlaneZ( P, N, P[2]+1.0, P[1] ), P[1], P[2]+1.0 ], P ) )	# y
		self.sliceV2 = scipy.cross( self.sliceV0, self.sliceV1 )	# x
		return
			
#======================================================================#
def writeSlice( imageArray, prefix, sliceDim=0 ):
	numLength = int( scipy.log10( imageArray.shape[sliceDim] ) ) + 2
	
	if sliceDim==0:
		for i in range( imageArray.shape[sliceDim]):
			filename = prefix + "_%.*d"%(numLength, i ) +'.png'
			imsave( filename, imageArray[i,:,:] )
	elif sliceDim==1:
		for i in range( imageArray.shape[sliceDim]):
			filename = prefix + "_%.*d"%(numLength, i ) +'.png'
			imsave( filename, imageArray[:,i,:] )
	elif sliceDim==2:
		for i in range( imageArray.shape[sliceDim]):
			filename = prefix + "_%.*d"%(numLength, i ) +'.png'
			imsave( filename, imageArray[:,:,i] )
	else:
		raise InputError, 'invalid sliceDim'
		
	return

#======================================================================#
def pad(array, t, padval = 0):
	""" pad self.I by pad voxels of value padv
	"""
	
	print "Padding..."
	dim = len(array.shape)
	
	if dim > 3:
		print "pad ERROR: invalid dimension"
		return None
		
	newsize = scipy.zeros(dim)
	for i in range(0,dim):
		newsize[i] = array.shape[i]+(2*t)
		
	new = scipy.zeros(newsize, array.dtype)
	new = new + padval
	
	if dim == 1:
		new[t:t+array.shape[0]] = array
	elif dim == 2:
		new[t:t+array.shape[0], t:t+array.shape[1]] = array
	elif dim == 3:
		new[t:t+array.shape[0], t:t+array.shape[1], t:t+array.shape[2]] = array
		
	return new

#======================================================================#
def nz_stats(array, return_val = 0):
	
	print "Non-zero element stats:"
	nz = scipy.nonzero(array)
	values = scipy.zeros([nz[0].shape[0]], dtype = int)
	for i in range(0, nz[0].shape[0]):
		values[i] = array[nz[0][i], nz[1][i], nz[2][i]]
	
	if values.__len__() == 0:
		print "No nonzero elements in array"
	else:
		print "array n = ", values.__len__()
		print "array max = ", values.max()
		print "array min = ", values.min()
		print "array mean = ", values.mean()
		print "array std = ", values.std()
	
	if return_val:
		return values

#======================================================================#
#~ def GetArrayImage(imageArray, imageType):
	#~ itk_py_converter = itk.PyBuffer[imageType]
		#~ 
	#~ return itk_py_converter.GetImageFromArray(imageArray)

#======================================================================#
def evalPlaneZ( P, N, x, y ):
	""" evaluate z coords of a point x,y on a plane defined by P and N.
	"""
	tol = 0.000001
	P = scipy.array(P, dtype=float)
	N = scipy.array(N, dtype=float)
	N = norm(N)
	if abs(N[2]) < tol:
		print 'WARNING: evalPlaneZ: zero normal z component'
		return None
	else:
		x = float(x)
		y = float(y)
		nom = N[2]*(x - P[2]) + N[1]*(y - P[1])
		return P[0] - nom/N[0]
	
#======================================================================#
def norm( v ):
	return scipy.divide( v, scipy.sqrt((v**2.0).sum()) )

#======================================================================#
def mag( v ):
	""" calc |v|
	"""
	return scipy.sqrt( (v**2.0).sum() )

#======================================================================#
def calcTheta( x, y ):
	""" angle anticlockwise from the +ve x axis of a point (x,y)
	"""
	
	s = ( scipy.sign(x), scipy.sign(y) )
	case = { (1.0, 0.0): 0.0,\
			 (1.0, 1.0): scipy.arctan( y/x ),\
			 (0.0, 1.0): scipy.pi/2.0,\
	         (-1.0, 1.0): scipy.pi + scipy.arctan( y/x ),\
			 (-1.0, 0.0): scipy.pi,\
			 (-1.0, -1.0): scipy.pi + scipy.arctan( y/x ),\
			 (0.0, -1.0): scipy.pi * 1.5,\
			 (1.0, -1.0): 2*scipy.pi + scipy.arctan( y/x ) }
			 
	return case[s]

#======================================================================#
class quadraticCurve( object ):
	fitTol = 0.001
	
	def __init__(self, dimension, params=None ):
		""" params = [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]
		"""
		self.params = params
		self.dimension=dimension
		
	def setParams( self, p ):
		self.params = p
		
	def eval( self, t, d=0 ):
		if t<0.0:
			t = 0.0
		elif t>1.0:
			t = 1.0
		
		#~ print 'params:', self.params
		
		ret = []
		if d==0:
			for p in self.params:
				#~ print 'p:', p
				ret.append( scipy.dot( self._basis0(t), p ) )
		elif d==1:
			for p in self.params:
				ret.append( scipy.dot( self._basis1(t), p ) )
		elif d==2:
			for p in self.params:
				ret.append( scipy.dot( self._basis2(t), p ) )
			
		return ret
	
	def _basis0( self, t ):
		t2 = t*t
		p1 = 2.0*t2 - 3.0*t + 1
		p2 = 4.0*t - 4.0*t2
		p3 = 2.0*t2 - t
		return [p1, p2, p3]

	def _basis1( self, t ):
		p1 = 4.0*t - 3.0
		p2 = 4.0 - 8.0*t
		p3 = 4.0*t - 1
		return [p1, p2, p3]
		
	def _basis2( self, t ):
		return [4.0, -8.0, 4.0]
		
	def findClosest( self, p ):
		""" returns coords and parameters u of point on the line closest
		to p
		"""
		self.p = tuple(p)
		u0 = [0.5]
		uMin = leastsq( self._closestObj, u0, xtol=self.fitTol, ftol=self.fitTol )[0]
		pClosest = self.eval(uMin)
		self.p = None
		
		return pClosest, uMin
	
	def _closestObj( self, u ):
		pCurve = self.eval( u[0] )
		d = euclidean( self.p, pCurve )
		return d

	def fit( self, data ):
		""" optimise curve parameters to minimise projected error to 
		data = [[xyz],[xyz],...]
		"""
		print '\nfitting...\n'
		print 'data shape:', data.shape
		self.fitData = data
		x0 = scipy.array(self.params).ravel()
		xOpt = leastsq( self._fitLsqObj, x0, xtol=self.fitTol, ftol=self.fitTol )[0]
		
		print '\nXopt:', xOpt
		finalErr = self._fitLsqObj( xOpt )
		finalRMS = scipy.sqrt(finalErr.mean())
		
		return finalRMS
		
	def _fitLsqObj( self, x ):
		""" x are the coords of the 3 nodes
		"""
		#~ print x
		self.setParams( scipy.reshape(x, (self.dimension,3) ) )
		return self._projectLsqError()	
		
	def _projectLsqError( self ):
		""" calculate RMS error between datapoints and their projection
		on the curve. data = [[xyz],[xyz],...]
		"""
		err = scipy.zeros(self.fitData.shape[0] )
		for i in range(self.fitData.shape[0]):
			err[i] = euclidean( self.fitData[i], self.findClosest(self.fitData[i])[0] )
			
		err = scipy.array( err )
		rms = scipy.sqrt((err**2.0).mean())
		print 'rms error:', rms
		#~ return rms
		
		return err
	
	def getMidpoint( self ):
		return self.params[:,1]

#======================================================================#
def cropImageAroundPoints(points, scan, pad, croppedName=None, transformToIndexSpace=True,
						  zShift=True, negSpacing=False, offsetXYCoeff=1.0):
	
	# offsetXYCoeff = 1 for VIFM, -1 for no negSpacing (ossis)

	if transformToIndexSpace:
		XImage = scan.coord2Index(points, zShift, negSpacing)
	else:
		XImage = points
	# XImage[:,2] -= 190

	# crop image to sampling range
	maxx, maxy, maxz = XImage.max(0) + pad
	minx, miny, minz = XImage.min(0) - pad
	sx,sy,sz = scan.I.shape

	# restrict cropping to be within image size
	maxx = min(sx,maxx)
	maxy = min(sy,maxy)
	maxz = min(sz,maxz)
	minx = max(0,minx)
	miny = max(0,miny)
	minz = max(0,minz)
	print 'cropping max:', maxx,maxy,maxz
	print 'cropping min:', minx,miny,minz
	cropOffset = scipy.array([minx, miny, minz])
	if not negSpacing:
		if zShift:
			zCorrection = scan.voxelSpacing[2] * scipy.array([ 0.0, 0.0, (maxz-minz-scan.I.shape[2]) ])
		else:
			zCorrection = [0,0,0]
	else:
		zCorrection = [0,0,0]
	# zCorrection = scan.voxelSpacing[2] * scipy.array([ 0.0, 0.0, (maxz-minz) ])
	# newScanOffset = scan.voxelOffset-cropOffset*scan.voxelSpacing+zCorrection
	# newScanOffset = scan.voxelOrigin-cropOffset*scan.voxelSpacing+zCorrection
	newScanOrigin = scan.voxelOrigin + offsetXYCoeff*cropOffset*scan.voxelSpacing + zCorrection
	# newScanOrigin = scan.voxelOrigin - (cropOffset + [maxx-minx, maxy-miny,0])*scan.voxelSpacing

	if croppedName==None:
		croppedName = scan.name+'_cropped'
	croppedScan = Scan(croppedName)
	croppedScan.isMasked = scan.isMasked
	croppedScan.setImageArray(scan.I[minx:maxx,
							  		 miny:maxy,
							  		 minz:maxz],
							  voxelSpacing=scan.voxelSpacing,
							  voxelOrigin=newScanOrigin,
							  )

	# croppedScan.setImageArray(scan.I[minx:maxx,
	# 						  		 miny:maxy,
	# 						  		 minz:maxz],
	# 						  voxelSpacing=scan.voxelSpacing,
	# 						  voxelOrigin=scan.voxelOrigin-offset,
	# 						  voxelOffset=scan.voxelOffset-offset)

	return croppedScan, cropOffset
