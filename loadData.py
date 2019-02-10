# Takes in RGB, depth, and mask images, and generates a set of features of
# the given size (numR x numC)
#
# All generated features will be of this given dimension. Bounds are set by
# the mask, the other dimension will be padded so that the image is
# centered. All channels are scaled together
#
# Last argument just tells whether or not to flip the depth channel - for
# some data, depth numbers increase as they get further from us, for some
# it's the other way around, so this lets us choose
#
# Generates:
# I: color image converted to YUV color. Scaled into the [-1 1] range
# (although it'll probably get whitened later anyway)
#
# D: Depth image - to get this, we interpolate then downsample the given
# depth. We also drop the mean and filter out extreme outliers
#
# N: surface normals, 3 channels (X,Y,Z) - averaged from normals computed
# on the interpolated depth image
#
# IMask: Color image mask. Since we don't use the RGBD dataset mask for the
# color image, just masks out the padding needed to fit the target image
# size
#
# DMask: Mask for the depth and normal features. Based on the input mask,
# rescaled, with some additional outliers masked out
#
# Author: Aashiq Muhamed

from util import *
import numpy as np
import os
import pdb
def getNewFeatFromRGBD(I1,D1,mask,numR,numC,negateDepth = 0):
 
    if negateDepth :
        D1 = -D1;

    
    #Get rid of points where we don't have any depth values (probably points
    #where Kinect couldn't get a value for some reason)
    mask = mask*(D1 != 0);
    
    # Interpolate to get better data for downsampling/computing normals
    D1 = D1*mask;
    
    # Do two passes of outlier removal since big outliers (as are present in
    # some cases) can really skew the std, and leave other outliers well within
    # the acceptable range. So, take these out and then recompute the std. If
    # there aren't any huge outliers, std won't shift much and the 2nd pass
    # won't eliminate that much more.
    D1,mask = removeOutliers(D1,mask,4);
    D1,mask = removeOutliers(D1,mask,4);
    D1 = smartInterpMaskedData(D1,mask);
    
    # Get normals from full-res image, then downsample
    #pdb.set_trace()
    #pdb.set_trace()
    N = getSurfNorm(D1);
    N,_ = padImage(N,numR,numC);
    
    # Downsample depth image and get new mask. 
    D,DMask = padMaskedImage2(D1,mask,numR,numC);
    
    I1 = rgb2yuv(I1);
    
    I,IMask = padImage(I1,numR,numC);
    
    # Re-range the YUV image. Just scale the Y channel to the [0,1] range.
    # The version of yuv2rgb used here gives values in the [-128,128] range, so
    # scale that to the [-1,1] range (but keep some values negative since this
    # is more natural for the U and V components)
    I[:,:,0] = I[:,:,0]/255;
    I[:,:,1] = I[:,:,1]/128;
    I[:,:,2] = I[:,:,2]/128;
    return (I,D,N,IMask,DMask)



# Loads features for all rectangles for a given image in the grasping
# rectangle dataset.
# 
# Inputs:
# dataDir: directory containing the dataset
#
# instNum: instance number to load
# 
# numR, numC: number of rows and cols for input features
#
# Outputs:
# *Feat: raw depth-, color-(YUV), and surface normal channel features as 
# flattened NUMR x NUMC images. Non-whitened
#
# class: graspability indicator for each case (1 indicates graspable)
# 
# inst: image that each case comes from
# 
# accepted: whether or not each rectangle was accepted. Rectangles may be
# rejected if not enough data was present inside the rectangle. This will
# contain more cases than the other outputs, since they won't contain
# rejected cases
# 
# depthMask: mask for the depth channel - masks out both areas outside the
# rectangle and points where Kinect failed to return a value. Also used for
# surface normal channels
#
# colorMask: similar, but only masks out areas outside the rectangle
#
# Author: Aashiq Muhamed

def  loadGraspingInstanceImYUVNorm(dataDir,instNum,numR,numC) :
    
    imArea = np.dot(numR,numC);
    
    I = graspPCDToRGBDImage(dataDir,instNum);
    
    #rectFilePos = sprintf('%s/pcd%04dcpos.txt',dataDir,instNum);
    rectFilePos = '{}/pcd{:04d}cpos.txt'.format(dataDir,instNum)
    
    rectPointsPos = np.loadtxt(rectFilePos);
    nRectPos = int(np.size(rectPointsPos,0)/4);
    
    #rectFileNeg = sprintf('%s/pcd%04dcneg.txt',dataDir,0instNum);
    rectFileNeg = '{}/pcd{:04d}cneg.txt'.format(dataDir,instNum);
    
    rectPointsNeg = np.loadtxt(rectFileNeg);
    nRectNeg = int(np.size(rectPointsNeg,0)/4);
    
    
    depthFeat = np.zeros((0,imArea));
    colorFeat = np.zeros((0,imArea*3));
    normFeat = np.zeros((0,imArea*3));
    
    depthMask = np.zeros((0,imArea));
    colorMask = np.zeros((0,imArea));
    
    classes = np.zeros((0,1));
    accepted = np.zeros((nRectPos + nRectNeg,1));
    
    for i in range(nRectPos):
        startInd = (i)*4 ;
        curI = orientedRGBDRectangle(I,rectPointsPos[startInd:startInd+3,:]);
        
        
        if np.size(curI) == 1 :
            val = 1
        else :
            val = np.size(curI,0)
        if  val> 1:
            curI,D,N,IMask,DMask = getNewFeatFromRGBD(curI[:,:,0:3],curI[:,:,3],np.ones((np.size(curI,0),np.size(curI,1)),dtype = bool),numR,numC,0);
            depthFeat = np.vstack((depthFeat,np.ndarray.flatten(D.T)));
            colorFeat = np.vstack((colorFeat, np.ndarray.flatten(curI.T)));
            normFeat = np.vstack((normFeat, np.ndarray.flatten(N.T)));
            colorMask = np.vstack((colorMask, np.ndarray.flatten(IMask.T)));
            depthMask = np.vstack((depthMask, np.ndarray.flatten(DMask.T)));
            classes = np.vstack((classes,1));
            accepted[i] = 1;
      
  
    
    for i in range(nRectNeg):
        startInd = (i)*4 ;
        curI = orientedRGBDRectangle(I,rectPointsNeg[startInd:startInd+3,:]);
        if np.size(curI,0) > 1:
            curI,D,N,IMask,DMask = getNewFeatFromRGBD(curI[:,:,0:3],curI[:,:,3],np.ones((np.size(curI,0),np.size(curI,1)),dtype = bool),numR,numC,0);
            depthFeat = np.vstack((depthFeat,np.ndarray.flatten(D.T)));
            colorFeat = np.vstack((colorFeat, np.ndarray.flatten(curI.T)));
            normFeat = np.vstack((normFeat, np.ndarray.flatten(N.T)));
            colorMask = np.vstack((colorMask, np.ndarray.flatten(IMask.T)));
            depthMask = np.vstack((depthMask, np.ndarray.flatten(DMask.T)));
            classes = np.vstack((classes,0));
            accepted[i + nRectPos] = 1;

    
    return (depthFeat, colorFeat, normFeat, classes, accepted, depthMask, colorMask)


# Load raw input features for the grasping rectangle data in the given
# directory.
#
# Outputs:
# *Feat: raw depth-, color-(YUV), and surface normal channel features as 
# flattened NUMR x NUMC images. Non-whitened
#
# class: graspability indicator for each case (1 indicates graspable)
# 
# inst: image that each case comes from
# 
# accepted: whether or not each rectangle was accepted. Rectangles may be
# rejected if not enough data was present inside the rectangle. This will
# contain more cases than the other outputs, since they won't contain
# rejected cases
# 
# depthMask: mask for the depth channel - masks out both areas outside the
# rectangle and points where Kinect failed to return a value. Also used for
# surface normal channels
#
# colorMask: similar, but only masks out areas outside the rectangle
#
# Author: Aashiq Muhamed

def loadAllGraspingDataImYUVNormals(dataDir) :

# Change these to change input size
    NUMR = 24;
    NUMC = 24;
    
    maxFile = 1100;
    imArea = NUMR*NUMC;
    
    depthFeat = np.zeros((0,imArea));
    colorFeat = np.zeros((0,imArea*3));
    normFeat = np.zeros((0,imArea*3));
    
    depthMask = np.zeros((0,imArea));
    colorMask = np.zeros((0,imArea));
    
    classes = np.zeros((0,1));
    accepted = np.zeros((0,1));
    inst = np.zeros((0,1));
    
    for i in range(maxFile) :
        pcdFile = '{}/pcd{:04d}.txt'.format(dataDir,i)
        #print(pcdFile)
        # Make sure the file exists (some gaps in the dataset)
        if not os.path.exists(pcdFile) :
            continue
            
        print(1,'Loading PC {}\n'.format(i));
        
        #Read and store features
        curDepth, curColor, curNorm, curClass, curAcc, curDMask, curIMask = loadGraspingInstanceImYUVNorm(dataDir,i,NUMR,NUMC);
        
        depthFeat = np.vstack((depthFeat, curDepth));
        colorFeat = np.vstack((colorFeat, curColor));
        normFeat = np.vstack((normFeat, curNorm));
        depthMask = np.vstack((depthMask, curDMask));
        colorMask = np.vstack((colorMask, curIMask));
        
        classes = np.vstack((classes,curClass));
        accepted = np.vstack((accepted,curAcc));
        inst = np.vstack((inst,np.tile(i,(np.size(curClass,0),1))));
        
    return (depthFeat, colorFeat, normFeat, classes, inst, accepted, depthMask, colorMask)
    
    
