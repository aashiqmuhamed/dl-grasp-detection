# Expands a set of channel-wise standard deviations to feature-wise
# standard deviations, given the edge size (in image space) for each
# channel (so if each channel is 24x24, chanSz would be given as 24)
#
# This is convenient pre-processing for whitening during detection.
#
# Author: Aashiq Muhamed
import numpy as np
from util import *
from loadData import *
import pdb
def  chanStdsToFeat(chanStds,chanSz) :

    numChan = len(chanStds);
    featPerChan = int(chanSz**2)
    #pdb.set_trace()
    chans = np.asarray(np.ceil((np.arange(1,featPerChan*numChan+1))/featPerChan),dtype = np.int64)
    featStds = np.expand_dims(np.ndarray.flatten(chanStds)[chans-1],axis = 1);
    return featStds

# Combines depth, color, and normal features into a single feature matrix
# for (pre-)training. Uses the same mask for the depth features and each of
# the normal channels.
#
# Author: Aashiq Muhamed

def combineAllFeat(depthFeat, colorFeat, normFeat, depthMask, colorMask) :

    numDepth = np.size(depthFeat,1);
    numColor = np.size(colorFeat,1);
    numNorm = np.size(normFeat,1);
    
    feat = np.zeros((np.size(depthFeat,0),numDepth + numColor + numNorm));
    mask = np.zeros(feat.shape);
    
    feat[:,0:numDepth] = depthFeat;
    mask[:,0:numDepth] = depthMask;
    
    feat[:,numDepth:numDepth+numColor] = colorFeat;
    
    # Repeat color mask for each channel (same for normals)
    mask[:,numDepth:numDepth+numColor] = np.tile(colorMask,(1,3));
    
    feat[:,numDepth+numColor:] = normFeat;
    mask[:,numDepth+numColor:] = np.tile(depthMask,(1,3));
    return (feat,mask) 

# Subtracts mean on a feature-wise basis.
#
# Author: Aashiq Muhamed

def  dropMeanByFeat(feat,mask) :

    # Ignore masked-out values when computing the mean
    feat[mask == 0] = float('nan')
    
    featMean = np.nanmean(feat,0);
    feat = feat-featMean
    
    # Probably don't want NaNs in the final features, convert them back to 0's
    feat[mask == 0] = 0;
    return (feat,featMean)

#Compute the standard deviation for each channel in the given data. 
#
# The input, chanWithNans, is assumed to be formatted as follows:
# 
# NxMxK tensor, where N is the number of cases, M is the number of features
# per channel, and K is the number of channels.
# 
# NaN values for any masked-out data, so that we can use nanstd to ignore
# those values.
#
# Author: Aashiq Muhamed

def getChanStds(chanWithNans) :

    numChan = np.size(chanWithNans,2);
    
    stds = np.zeros((numChan,1));
    
    for i in range(numChan) :
        curChan = chanWithNans[:,:,i];
        stds[i] = np.nanstd(curChan.ravel());

    return stds 


# Randomly split grasping data with the given ratio for test data
# Be a little smart and split positive and negative cases separately so the
# train and test sets will have the same ratio.
#
# Author: Aashiq Muhamed

def getGraspingSplit(classes,ratio) :

    #classes is bool
    numPos = np.sum(classes,0);
    
    isTest = np.zeros(classes.shape);
    
    posSamples = np.random.choice(np.arange(numPos),int(round(numPos*ratio)));
    
    posIsTest = np.zeros((numPos,1));
    posIsTest[posSamples] = 1;
    #pdb.set_trace()
    isTest[classes] = posIsTest.flatten();
    
    classNeg = ~classes;
    numNeg = np.sum(classNeg,0);
    
    negSamples = np.random.choice(np.arange(numNeg),int(round(numNeg*ratio)));
    
    negIsTest = np.zeros((numNeg,1));
    negIsTest[negSamples] = 1;
    
    isTest[classNeg] = negIsTest.flatten();
    
    return isTest 



# Scales each channel by its corresponding standard deviation (or some
# other scaling value, that's just the way it's used here for whitening).
# 
# Assumes data is an NxMxK tensor, where N is the number of cases, M the
# number of features per channel, and K the number of channels. stds is
# then a K-vector.
#
# Author: Aashiq Muhamed

def scaleChannels(data,stds) :

    numChan = len(stds);
    
    for i in range(numChan) :
        data[:,:,i] = data[:,:,i]/stds[i];
    return data



# Scales the given data by the given channel-wise standard deviations (or
# other channel-wise scaling factors). 
# 
#% data is assumed to be an NxM matrix, where N is the number of cases, and
# M is the total number of features (in contrast to some other functions
#% here, where M is the number of features per channel). stds is a K-vector
# of channel-wise scaling factors, which also tells us how many channels
# there are by its length.
#
# Author: Aashiq Muhamed

def scaleDataByChannel(data,stds) :

    numChan = len(stds);
    
    numCases = np.size(data,0);
    featPerChan = int(np.size(data,1)/numChan);
    
    nanChan = np.reshape(data,[numCases, featPerChan, numChan]);
    
    newDat = scaleChannels(nanChan,stds);
    
    newDat = np.reshape(newDat,(numCases,featPerChan*numChan));
    return newDat


# Given some data, its corresponding mask, and the number of channels,
# scales each channel by its standard deviation (accross all features in
# that channel).
# 
# Data is given as an NxM matrix, where N is the number of cases, and M is
# the total number of features (as opposed to the number of features per
# channel, as in some other files here). 
#
# numChan tells us how many channels the data has (7 for the grasping
# code). All channels are assumed to be the same size.
#
# Returns both the scaled data and the channel-wise standard deviations (in
# case we need to scale some other data.)
#
# Author: Aashiq Muhamed

def scaleDataByChannelStds(data,mask,numChan) :

    numCases = np.size(data,0);
    featPerChan = int(np.size(data,1)/numChan);
    
    data[mask==0] = float('nan');
  
    nanChan = np.reshape(data,(numCases,featPerChan,numChan));
    
    stds = getChanStds(nanChan);
    
    newDat = scaleChannels(nanChan,stds);
    newDat[np.isnan(newDat)] = 0;
    
    newDat = np.reshape(newDat,(numCases,featPerChan*numChan));
    return [newDat,stds] 