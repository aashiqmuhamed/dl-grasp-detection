import numpy as np
import scipy
import scipy.misc
from detectionUtils import *
from util import *
from matplotlib import pyplot
import pdb
import skimage
import heapq
import h5py
#import matlab.engine
# Quick way to run a grasp search over an image. Runs a search for a
# default set of search parameters, loading some other necessary variables.
# 
# Takes the instance number to search for, the directory that the grasping
# data is in, and the directory that the background images are in.
#
# Assumes that the scripts in recTraining have been used to learn a set of
# weights, which are stored in ../weights
# 
# Arguments:
# instNum: instance number to do detection on
# dataDir: directory containing the grasping dataset cases (should have a
#          lot of pcd* files)
# bgDir: directory containing the dataset background files (pdcb*)
#   
# Author: Ian Lenz

# onePassDectionForInstDefaultParams(100,'E:/Data for grasping/data/all','E:/Data for grasping/backgrounds') 
def onePassDectionForInstDefaultParams(instNum,dataDir,bgDir) :
    
      
    temp = scipy.io.loadmat('C:/Users/aashi/Desktop/Python conversion/data/bgNums.mat')
    bgNo = temp['bgNo']
    temp = scipy.io.loadmat('C:/Users/aashi/Desktop/Python conversion/data/graspModes24.mat')
    trainModes = temp['trainModes']
#    featMeans = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses/featMeans.npy')
#    featStds = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses/featStds.npy')
    
    
    

    filepath = 'E:/deepGraspingCode/deepGraspingCode/data/graspWhtParams.mat'
    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    
    featMeans = arrays['featMeans'].T
    featStds = arrays['featStds'].T
    
#    w1= np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWFinal/w1.npy');
#    w2 = np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWFinal/w2.npy');
#    w_class = np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWFinal/2_class.npy');
    
    temp = scipy.io.loadmat('E:/deepGraspingCode/deepGraspingCode/weights/graspWFinal.mat')
    w1 = temp['w1']
    w2 = temp['w2']
    w_class = temp['w_class']
    
    temp = scipy.io.loadmat('E:/deepGraspingCode/deepGraspingCode/weights/graspWFinalsmall.mat')
    w1small = temp['w1']
    w2small = temp['w2']
    w_classsmall = temp['w_class']
    
    # Just use the positive-class weights for grasping.
    w_class = np.expand_dims(w_class[:,0],axis = 1);
    w_classsmall = np.expand_dims(w_classsmall[:,0],axis = 1);
    
    # Run detection with a default set of search parameters.
    [bestRects,bestScores] = onePassDetectionForInst(dataDir,bgDir,bgNo,instNum,w1small,w2small,w_classsmall,w1,w2,w_class,featMeans,featStds,np.arange(0,15*12,15),np.arange(10,100,10),np.arange(10,100,10),10,trainModes);
    
    return [bestRects,bestScores] 


# Quick way to run a grasp search over an image. Runs a search for a
# default set of search parameters, loading some other necessary variables.
# 
# Visualizes the search as it runs. Shows the current rectangle (red/blue)
# and the top-ranked rectangle (yellow/green)
# 
# Takes the instance number to search for, the directory that the grasping
# data is in, and the directory that the background images are in.
#
# Assumes that the scripts in recTraining have been used to learn a set of
# weights, which are stored in ../weights
# 
# Arguments:
# instNum: instance number to do detection on
# dataDir: directory containing the grasping dataset cases (should have a
#          lot of pcd* files)
# bgDir: directory containing the dataset background files (pdcb*)
# 
# Author: Ian Lenz

def onePassDectionForInstDefaultParamsDisplay(instNum,dataDir,bgDir) :
    
    temp = scipy.io.loadmat('../data/bgNums.mat')
    bgNo = temp['bgNo']
    temp = scipy.io.loadmat('../data/graspModes24.mat')
    trainModes = temp['trainModes']
#    featMeans = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses/featMeans.npy')
#    featStds = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses/featStds.npy')
    
    
    

    filepath = '../data/graspWhtParams.mat'
    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    
    featMeans = arrays['featMeans'].T
    featStds = arrays['featStds'].T
    
#    w1= np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWFinal/w1.npy');
#    w2 = np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWFinal/w2.npy');
#    w_class = np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWFinal/2_class.npy');
    
    temp = scipy.io.loadmat('../weights/graspWFinal.mat')
    w1 = temp['w1']
    w2 = temp['w2']
    w_class = temp['w_class']
    
    # Just use the positive-class weights for grasping.
    w_class = np.expand_dims(w_class[:,0],axis = 1);
    
    # Run detection with a default set of search parameters.
    [bestRects,bestScores] = onePassDetectionForInstDisplay(dataDir,bgDir,bgNo,instNum,w1,w2,w_class,featMeans,featStds,np.arange(0,15*12,15),np.arange(10,100,10),np.arange(10,100,10),10,trainModes);
    
    return [bestRects,bestScores] 

# Helper function which reads out the BG file name from a list by instance
# 
# Author: Ian Lenz

def onePassDetectionForInst(dataDir,bgDir,bgNos,instNum,w1,w2,wClass,w1large,w2large,wClasslarge,means,stds,rotAngs,hts,wds,scanStep,modes) :
    #pdb.set_trace()    
    bgFN = '{}/pcdb{:04d}r.png'.format(bgDir,bgNos[instNum][0])
    
    [bestRect,bestScore] = onePassDetectionNormalized(dataDir,bgFN,instNum,w1,w2,wClass,w1large,w2large,wClasslarge,means,stds,rotAngs,hts,wds,scanStep,modes);
    return [bestRect,bestScore] 

# Helper function which reads out the BG file name from a list by instance
# 
# Author: Ian Lenz

def onePassDetectionForInstDisplay(dataDir,bgDir,bgNos,instNum,w1,w2,wClass,means,stds,rotAngs,hts,wds,scanStep,modes) :
    
    bgFN = '{}/pcdb{:04d}r.png'.format(bgDir,bgNos[instNum][0])
    #bgFN = sprintf('%s/pcdb%04dr.png',bgDir,bgNos(instNum));
    
    [bestRect,bestScore] = onePassDetectionNormDisplay(dataDir,bgFN,instNum,w1,w2,wClass,means,stds,rotAngs,hts,wds,scanStep,modes);
    return [bestRect,bestScore] 


# Detection code for grasping (or really anything given different weights)
# using a DBN for scoring
#
# Given an object and background image, finds the object (detected as the
# biggest blob of changed pixels between these) and searches through
# candidate rectangles inside the object's bounding box. 
#
# dataDir, bgFN, and instNum tell the code where to look for the grasping
# data and the background image, and which instance number to load.
#
# w's are the DBN weights. Currently hard-coded for two layers, with a
# linear scoring layer on top
#
# means and stds are used to whiten the data. These should be the same
# whitening parameters used for the training data
#
# roAngs, hts, and wds are vectors of candidate rotations, heights, and
# widths to consider for grasping rectangles. Only rectangles with width >=
# height will be considered.
#
# scanStep determines the step size when sweeping the rectangles across
# image space
# 
# Visualizes the search as it runs. Shows the current rectangle (red/blue)
# and the top-ranked rectangle (yellow/green)
# 
# Author: Ian Lenz

def onePassDetectionNormDisplay(dataDir,bgFN,instNum,w1,w2,wClass,means,stds,rotAngs,hts,wds,scanStep,modes) :
    
    
    #eng = matlab.engine.start_matlab()
    
    PAD_SZ = 20;
    
    # Thresholds to use when transforming masks to convert back to binary
    MASK_ROT_THRESH =  0.75
    MASK_RSZ_THRESH = 0.75
    
    # Fraction of a rectangle which should be masked in as (padded) object for
    # a rectangle to be considered
    OBJ_MASK_THRESH = 0.5;
    
    FEATSZ = 24;
    
    # Make sure heights and widths are in ascending order, since this is a
    # useful property we can exploit to speed some things up
    hts = np.sort(hts);
    wds = np.sort(wds);
    
    # Load grasping data. Loads into a 4-channel image where the first 3
    # channels are RGB and the fourth is depth
    
    I = graspPCDToRGBDImage(dataDir,instNum);
    BG = np.asarray(scipy.misc.imread(bgFN),dtype = np.float64);
    
    # Find the object in the image 
    [M, bbCorners] = maskInObject(I[:,:,0:3],BG,PAD_SZ);
    
    # Do a little processing on the depth data - find points that Kinect
    # couldn't get, and eliminate additional outliers where Kinect gave
    # obviously invalid values
    D = np.copy(I[:,:,3]);
    DMask = D != 0;
    
    [D,DMask] = removeOutliersDet(D,DMask,4);
    [D,DMask] = removeOutliersDet(D,DMask,4);
    
    D = smartInterpMaskedData(D,DMask);
    #D = smartInterpMaskedDataMatlab(D,DMask,eng);
    
    pyplot.figure(1);
    pyplot.imshow(np.asarray(I[:,:,0:3],dtype = np.uint8));
    pyplot.draw()
    
    # Work in YUV color
    I = rgb2yuv(I[:,:,0:3]);
    
    # Pick out the area of the image corresponding to the object's (padded)
    # bounding box to work with
    
    #maskinOject returns entirely in python format so no need to subtract 1 from bbCorners
#    objI = I[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objD = D[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objM = M[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objDM = DMask[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
    
    objI = I[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objD = D[int(bbCorners[0,0]):int(bbCorners[1,0]),int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objM = M[int(bbCorners[0,0]):int(bbCorners[1,0]),int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objDM = DMask[int(bbCorners[0,0]):int(bbCorners[1,0]),int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
    
    objD = D[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1];
    objM = M[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1];
    objDM = DMask[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1];
    
    bestScore = -float('inf');
    
    bestAng = -1;
    bestW = -1;
    bestH = -1;
    bestR = -1;
    bestC = -1;
    
    # Precompute which widths we need to use with each height so we don't have
    # to compute this in an inner loop
    useWdForHt = np.asarray(np.zeros((len(hts),len(wds))),dtype = bool);
    
    for i in range(1,len(hts)+1) :
        useWdForHt[i-1,:] = wds > hts[i-1];

    
    prevLines = [];
    bestLines = [];
    barH = [];
    
    IMask = np.ones((int(bbCorners[1,0])-int(bbCorners[0,0])+1,int(bbCorners[1,1])-int(bbCorners[0,1])+1));
   
    for curAng in rotAngs :
        # Rotate image to match the current angle. Threshold masks to keep them
        # binary
        curI = skimage.transform.rotate(objI,curAng,order = 0,resize = True,preserve_range = True)
        curMask = skimage.transform.rotate(objM,curAng,order = 0,resize = True,preserve_range = True) > MASK_ROT_THRESH;
        curD = skimage.transform.rotate(objD,curAng,order = 0,resize = True,preserve_range = True);
        curDMask = skimage.transform.rotate(objDM,curAng,order = 0,resize = True,preserve_range = True) > MASK_ROT_THRESH;
        curIMask = skimage.transform.rotate(IMask,curAng,order = 0,resize = True,preserve_range = True) > MASK_ROT_THRESH;
        
        # Compute surface normals. Only do this here to avoid having to rotate
        # the normals themselves when we rotate the image (can't just
        # precompute the normals and then rotate the "normal image" since the
        # axes change)
        curN = getSurfNorm(curD);
        
        curRows = np.size(curI,0);
        curCols = np.size(curI,1);
        
        # Going by the r/c dimensions first, then w/h should be more cache
        # efficient since it repeatedly reads from the same locations. Who
        # knows if that actually matters but the ordering's arbitrary anyway
        for r in range(1,curRows-min(hts)+scanStep,scanStep) :
            for c in range(1,curCols-min(wds)+scanStep,scanStep) :
                for i in range(1,len(hts)+1) :
                    
                    h = hts[i-1];
                    
                    # If we ran off the bottom, we can move on to the next col
                    if r + h > curRows :
                        break;
                    
                    
                    # Only run through the widths we need to - anything smaller
                    # than the current height (as precomputed) doesn't need to
                    # be used
                    for w in wds[useWdForHt[i-1,:]] :
                        
                        # If we run off the side, we can move on to
                        # the next height
                        if c + w > curCols :
                            break;
                        
                        # If the rectangle doesn't contain enough of the
                        # object (plus padding), move on because it's probably
                        # not a valid grasp regardless of score
                        if rectMaskFraction(curMask,r,c,h,w) < OBJ_MASK_THRESH or cornerMaskedOut(curIMask,r,c,h,w) :
                            continue;
                        
                        # Have a valid candidate rectangle
                        # Extract features for the current rectangle into the
                        # format the DBN expects
                        
                        [curFeat, curFeatMask] = featForRect(curI,curD,curN,curDMask,r,c,h,w,FEATSZ,MASK_RSZ_THRESH);
                        
                        curFeat = simpleWhiten(curFeat,means,stds);
                        curFeat = scaleFeatForMask(curFeat, curFeatMask, modes);
                        
                        # Run the features through the DBN and get a score.
                        # Might be more efficient to collect features for a
                        # group of rectangles and run them all at once
                        w1Probs = 1/(1+np.exp(-np.hstack((curFeat, np.asarray([[1]]))).dot(w1)));
                        w2Probs = 1/(1+np.exp(-np.hstack((w1Probs, np.asarray([[1]]))).dot(w2)));
                        curScore = np.hstack((w2Probs, np.asarray([[1]]))).dot(wClass);
                        
                        rectPoints = np.asarray([[r, c], [r+h, c], [r+h, c+w], [r, c+w]])
                        curRect = localRectToIm(rectPoints,curAng,bbCorners);
                        
                        #figure(1);
                        removeLines(prevLines);
                        prevLines = plotGraspRect(curRect);
                        #delete(barH);
                        #barH = drawScoreBar(curScore,max(bestScore*1.1,1),20,320,30,300);
                        #figure(2);
                        #bar(w2Probs);
                        #axis([0 51 0 1]);
                        pyplot.draw()
    
                        if curScore > bestScore :
                            bestScore = curScore;
                            bestAng = curAng;
                            bestR = r;
                            bestC = c;
                            bestH = h;
                            bestW = w;
                            
                            #figure(1);
                            removeLines(bestLines);
                            bestLines = plotGraspRect(curRect,'g','y');
                            pyplot.draw()

    
    removeLines(prevLines);
    #eng.quit()
    # Take the best rectangle params we found and convert to image space
    # This is actually a little tricky because the image rotation operation
    # isn't straighforward to invert
    rectPoints = np.asarray([[bestR, bestC], [bestR+bestH, bestC], [bestR+bestH, bestC+bestW], [bestR, bestC+bestW]]);
    
    bestRect = localRectToIm(rectPoints,bestAng,bbCorners);
    
    return [bestRect,bestScore] 


# Detection code for grasping (or really anything given different weights)
# using a DBN for scoring
#
# Given an object and background image, finds the object (detected as the
# biggest blob of changed pixels between these) and searches through
# candidate rectangles inside the object's bounding box. 
#
# dataDir, bgFN, and instNum tell the code where to look for the grasping
# data and the background image, and which instance number to load.
#
# w's are the DBN weights. Currently hard-coded for two layers, with a
# linear scoring layer on top
#
# means and stds are used to whiten the data. These should be the same
# whitening parameters used for the training data
#
# rotAngs, hts, and wds are vectors of candidate rotations, heights, and
# widths to consider for grasping rectangles. Only rectangles with width >=
# height will be considered.
#
# scanStep determines the step size when sweeping the rectangles across
# image space
#
# Mask-scaled version - visible features are scaled based on how much
# is masked in, to eliminate bias towards squares
# 
# Author: Ian Lenz

def onePassDetectionNormalized(dataDir,bgFN,instNum,w1,w2,wClass,w1large,w2large,wClasslarge,means,stds,rotAngs,hts,wds,scanStep,modes) :
    
    PAD_SZ = 20;
    
    # Thresholds to use when transforming masks to convert back to binary
    MASK_ROT_THRESH = 0.75;
    MASK_RSZ_THRESH = 0.75;
    
    # Fraction of a rectangle which should be masked in as (padded) object for
    # a rectangle to be considered
    OBJ_MASK_THRESH = 0.5;
    
    FEATSZ = 24;
    
    # Make sure heights and widths are in ascending order, since this is a
    # useful property we can exploit to speed some things up
    hts = np.sort(hts);
    wds = np.sort(wds);
    
    # Load grasping data. Loads into a 4-channel image where the first 3
    # channels are RGB and the fourth is depth
    I = graspPCDToRGBDImage(dataDir,instNum);
    BG = np.asarray(scipy.misc.imread(bgFN),dtype = np.float64);
    
    # Find the object in the image 
    [M, bbCorners] = maskInObject(I[:,:,:3],BG,PAD_SZ);
    
    # Do a little processing on the depth data - find points that Kinect
    # couldn't get, and eliminate additional outliers where Kinect gave
    # obviously invalid values
    D = np.copy(I[:,:,3]);
    DMask = D != 0;
   
    [D,DMask] = removeOutliersDet(D,DMask,4);
    [D,DMask] = removeOutliersDet(D,DMask,4);
    D = smartInterpMaskedData(D,DMask);
    
#    D = np.expand_dims(D,2)
#    M = np.expand_dims(M,2)
#    DMask = np.expand_dims(DMask,2)
    
    # Work in YUV color
    I = rgb2yuv(I[:,:,:3]);
    
    # Pick out the area of the image corresponding to the object's (padded)
    # bounding box to work with
    # I think these will run off the bottom, so subtract 1 to hopefully make it work
    # Sides should be OK since objects never get near those
    bbCorners[0,0] = max(bbCorners[0,0],0);
    bbCorners[1,0] = min(bbCorners[1,0],np.size(I,0));
    bbCorners[0,1] = max(bbCorners[0,1],0);
    bbCorners[1,1] = min(bbCorners[1,1],np.size(I,1));
    #pdb.set_trace()
    objI = I[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objD = D[int(bbCorners[0,0]):int(bbCorners[1,0]),int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objM = M[int(bbCorners[0,0]):int(bbCorners[1,0]),int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
#    objDM = DMask[int(bbCorners[0,0]):int(bbCorners[1,0]),int(bbCorners[0,1]):int(bbCorners[1,1])+1,:];
    
    objD = D[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1];
    objM = M[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1];
    objDM = DMask[int(bbCorners[0,0]):int(bbCorners[1,0])+1,int(bbCorners[0,1]):int(bbCorners[1,1])+1];
    
    bestScore = -float('inf');
    
    bestAng = -1;
    bestW = -1;
    bestH = -1;
    bestR = -1;
    bestC = -1;
    q = []
    # Precompute which widths we need to use with each height so we don't have
    # to compute this in an inner loop
    useWdForHt = np.asarray(np.zeros((len(hts),len(wds))),dtype = bool);
    
    for i in range(1,len(hts)+1) :
        useWdForHt[i-1,:] = wds >= hts[i-1];
    
    IMask = np.ones((int(bbCorners[1,0])-int(bbCorners[0,0])+1,int(bbCorners[1,1])-int(bbCorners[0,1])+1));
    
    for curAng in rotAngs :
        # Rotate image to match the current angle. Threshold masks to keep them
        # binary
        curI = skimage.transform.rotate(objI,curAng,order = 0, resize= True,preserve_range = True);
        curD = skimage.transform.rotate(objD,curAng,order = 0,resize = True,preserve_range = True);
        curMask = skimage.transform.rotate(objM,curAng,order = 0, resize = True,preserve_range = True) > MASK_ROT_THRESH;
        curDMask =skimage.transform.rotate(objDM,curAng,order = 0, resize = True,preserve_range = True) > MASK_ROT_THRESH;
        curIMask = skimage.transform.rotate(IMask,curAng,order = 0, resize= True,preserve_range = True) > MASK_ROT_THRESH;
        
        #pdb.set_trace()
        # Compute surface normals. Only do this here to avoid having to rotate
        # the normals themselves when we rotate the image (can't just
        # precompute the normals and then rotate the "normal image" since the
        # axes change)
        curN = getSurfNorm(curD);
        
        curRows = np.size(curI,0);
        curCols = np.size(curI,1);
        # Going by the r/c dimensions first, then w/h should be more cache
        # efficient since it repeatedly reads from the same locations. Who
        # knows if that actually matters but the ordering's arbitrary anyway
      
        for r in range(1,curRows-min(hts)+scanStep,scanStep) :
            for c in range(1,curCols-min(wds)+scanStep,scanStep) :
                for i in range(1,len(hts)+1) :
                    
                    h = hts[i-1];
                   
                    # If we ran off the bottom, we can move on to the next col
                    if r + h > curRows :
                        break;
                    
                    # Only run through the widths we need to - anything smaller
                    # than the current height (as precomputed) doesn't need to
                    # be used
                    #pdb.set_trace()
                    for w in wds[useWdForHt[i-1,:]] :
                        
                        # If we run off the side, we can move on to
                        # the next height
                        if c + w > curCols :
                            break;
                        
                        # If the rectangle doesn't contain enough of the
                        # object (plus padding), move on because it's probably
                        # not a valid grasp regardless of score
                        if rectMaskFraction(curMask,r,c,h,w) < OBJ_MASK_THRESH or cornerMaskedOut(curIMask,r,c,h,w) :
                            continue;
                            
                        # Have a valid candidate rectangle
                        # Extract features for the current rectangle into the
                        # format the DBN expects
    
                        [curFeat, curFeatMask] = featForRect(curI,curD,curN,curDMask,r,c,h,w,FEATSZ,MASK_RSZ_THRESH);
                        
                        
                        curFeat = simpleWhiten(curFeat,means,stds);
                        curFeat = scaleFeatForMask(curFeat, curFeatMask, modes);
                        
                       
                        # Run the features through the DBN and get a score.
                        # Might be more efficient to collect features for a
                        
                        w1Probs = 1/(1+np.exp(-np.hstack((curFeat, np.asarray([[1]]))).dot(w1)));
                        w2Probs = 1/(1+np.exp(-np.hstack((w1Probs, np.asarray([[1]]))).dot(w2)));
                        curScore = np.hstack((w2Probs, np.asarray([[1]]))).dot(wClass);
                        #pdb.set_trace()
                        
#                        if curScore > bestScore :
#                            bestScore = curScore;
#                            bestAng = curAng;
#                            bestR = r;
#                            bestC = c;
#                            bestH = h;
#                            bestW = w;
                            
                      
                        if(len(q) <= 20) :
                            heapq.heappush(q, (curScore, [curFeat,curAng,r,c,h,w]))
                        else :
                            heapq.heappush(q, (curScore, [curFeat,curAng,r,c,h,w]))
                            heapq.heappop(q)
    
    print(q)
    for i in range(10) :
        curFeat,curAng,r,c,h,w = heapq.heappop(q)[1]
        w1Probs = 1/(1+np.exp(-np.hstack((curFeat, np.asarray([[1]]))).dot(w1large)));
        w2Probs = 1/(1+np.exp(-np.hstack((w1Probs, np.asarray([[1]]))).dot(w2large)));
        curScore = np.hstack((w2Probs, np.asarray([[1]]))).dot(wClasslarge); 
        
        if curScore > bestScore :
            bestScore = curScore;
            bestAng = curAng;
            bestR = r;
            bestC = c;
            bestH = h;
            bestW = w;                 
   
    # Take the best rectangle params we found and convert to image space
    # This is actually a little tricky because the image rotation operation
    # isn't straighforward to invert
    print(bestR,bestC,bestH,bestW)
    rectPoints = np.asarray([[bestR, bestC], [bestR+bestH, bestC], [bestR+bestH, bestC+bestW], [bestR, bestC+bestW]]);
    print(rectPoints)
    bestRect = localRectToIm(rectPoints,bestAng,bbCorners);
    return [bestRect,bestScore] 
