import numpy as np
from skimage import measure
import scipy.signal
from matplotlib import pyplot
import pdb
import skimage
from util import *
# Checks if one of the corners of the given axis-aligned rectangle
# (parameterized by upper-left corner and height/width) is masked out in
# the given mask
#
# Author: Aashiq Muhamed

def cornerMaskedOut(mask,r,c,h,w) :
    #pdb.set_trace()
    maskedOut = ~mask[r-1,c-1] | ~mask[r+h-1,c-1] | ~mask[r+h-1,c+w-1] | ~mask[r-1,c+w-1];
    return maskedOut

# Converts a rectangle from the format used by the deep learning code to
# the form used by the old Jiang grasping code.
#
# Author: Aashiq Muhamed

def dbnRectToOld(dbnRect) :

    dbnRect = np.fliplr(dbnRect);
    
    fixedRect = dbnRect;
    fixedRect[1,:] = dbnRect[3,:];
    fixedRect[3,:] = dbnRect[1,:];
    return fixedRect

# Extracts features for the given axis-aligned rectangle in the given set
# of color, depth, and normal images.
#
# Author: Aashiq Muhamed

def featForRect(I,D,N,DMask,r,c,h,w,featSz,maskThresh) :
    
    
    # Extract region from each input image
    I = np.copy(I[r-1:r+h-1,c-1:c+w-1,:]);
    #D = D[r-1:r+h-1,c-1:c+w-1,:];
    D = np.copy(D[r-1:r+h-1,c-1:c+w-1]);
    N = np.copy(N[r-1:r+h-1,c-1:c+w-1,:]);
    #DMask = DMask[r-1:r+h-1,c-1:c+w-1,:];
    DMask = np.copy(DMask[r-1:r+h-1,c-1:c+w-1]);
    
    # Scale and pad depth channel and mask
    [D,DMask] = padMaskedImage2(D,DMask,featSz,featSz);
    #[D,DMask] = padMaskedImage2Matlab(D,DMask,featSz,featSz,eng);
    
    DMask = DMask > maskThresh;
    
    # Scale and pad normal channel
    N,_ = padImage(N,featSz,featSz);
    #N,_ = padImageMatlab(N,featSz,featSz,eng);
    
    # Mask depth and normal channels again
    D = D*DMask;
    
    N = N*DMask;
    
    # Scale and pad color channels
    [I,IMask] = padImage(I,featSz,featSz);
    #[I,IMask] = padImageMatlab(I,featSz,featSz,x);
    
    IMask = IMask > maskThresh;
    I = I*np.expand_dims(IMask,2)
    
    # Re-range the YUV image. Just scale the Y channel to the [0,1] range.
    # The version of yuv2rgb used here gives values in the [-128,128] range, so
    # scale that to the [-1,1] range (but keep some values negative since this
    # is more natural for the U and V components)
    I[:,:,0] = I[:,:,0]/255;
    I[:,:,1] = I[:,:,1]/128;
    I[:,:,2] = I[:,:,2]/128;
    
    # Scale the depth data by its std since we do this per-case. Other
    # whitening is assumed to be done elsewhere.
    
    D,_ = caseWiseWhiten(D,DMask);
    
    #if((D== float('nan')).all()) :
    #    return [float('nan'),float('nan')]
    D = D.T; I = I.T; N = N.T;
    feat = np.vstack((np.expand_dims(D.flatten(),axis = 1),np.expand_dims( I.flatten(),axis = 1),np.expand_dims( N.flatten(),axis = 1))).T;
    DMask = DMask.T; IMask = IMask.T;
    mask = np.vstack((np.expand_dims(DMask.flatten(),axis = 1), np.tile(np.expand_dims(IMask.flatten(),axis = 1),(3,1)), np.tile(np.expand_dims(DMask.flatten(),axis = 1),(3,1)))).T;
   
    return [feat, mask]

# Similar to featForRect, but extracts features directly from a 4-channel
# RGB-D image (with channels in that order).
#
# Author: Aashiq Muhamed
 
def featForRectFromIm(I,featSz,maskThresh) :

    D = np.copy(I[:,:,3]);
    DMask = D != 0;
    
    [D,DMask] = removeOutliers(D,DMask,4);
    [D,DMask] = removeOutliers(D,DMask,4);
    D = smartInterpMaskedData(D,DMask);
    
    I = rgb2yuv(I[:,:,0:3]);
    
    N = getSurfNorm(D);
    
    # Scale and pad depth channel and mask
    [D,DMask] = padMaskedImage2(D,DMask,featSz,featSz);
    
    DMask = DMask > maskThresh;
    
    # Scale and pad normal channel
    N,_ = padImage(N,featSz,featSz);
    
    # Mask depth and normal channels again
    D = D*DMask;
    
    N = N*DMask
    
    # Scale and pad color channels
    [I,IMask] = padImage(I,featSz,featSz);
    IMask = IMask > maskThresh;
    I = I*IMask
    
    # Re-range the YUV image. Just scale the Y channel to the [0,1] range.
    # The version of yuv2rgb used here gives values in the [-128,128] range, so
    # scale that to the [-1,1] range (but keep some values negative since this
    # is more natural for the U and V components)
    I[:,:,0] = I[:,:,0]/255;
    I[:,:,1] = I[:,:,1]/128;
    I[:,:,2] = I[:,:,2]/128;
    
    # Scale the depth data by its std since we do this per-case. Other
    # whitening is assumed to be done elsewhere.
    D,_ = caseWiseWhiten(D,DMask);
    
    if((D== float('nan')).all()) :
        return [0,0]
    
    feat = np.vstack((D.flatten(), I.flatten(), N.flatten())).T;
    
    
    mask = np.vstack((DMask.flatten(), np.tile(IMask.flatten(),(3,1)), np.tile(DMask.flatten(),(3,1)))).T;
    
    return [feat, mask] 

# Translates a set of points from an oriented, cropped image to global
# image space.
#
# Author: Aashiq Muhamed

def localRectToIm(rectPoints,rotAng,bbCorners) :

    bbW = int(bbCorners[1,1] - bbCorners[0,1] + 1);
    bbH = int(bbCorners[1,0] - bbCorners[0,0] + 1);
    
    R = np.tile(np.expand_dims(np.arange(1,bbH+1),axis = 1),(1,bbW));
    C = np.tile(np.expand_dims(np.arange(1,bbW+1),axis = 0),(bbH,1));
    
    R = skimage.transform.rotate(R,rotAng,order = 0,resize = True,preserve_range = True);
    C = skimage.transform.rotate(C,rotAng,order = 0,resize = True,preserve_range = True);
    
    imPoints = np.zeros(rectPoints.shape);
  
    for i in range(np.size(rectPoints,0)) :
        imPoints[i,0] = R[rectPoints[i,0]-1,rectPoints[i,1]-1];
        imPoints[i,1] = C[rectPoints[i,0]-1,rectPoints[i,1]-1];

    
    imPoints[:,0] = imPoints[:,0] + bbCorners[0,0];
    imPoints[:,1] = imPoints[:,1] + bbCorners[0,1];
    return imPoints 

# Translates a set of points from an oriented, cropped image to global
# image space, given a mapping between the two.
#
# Author: Aashiq Muhamed

def localRectToImGivenMap(rectPoints,R,C,bbCorners) :

    imPoints = np.zeros(rectPoints.shape);
    
    for i in range(np.size(rectPoints,0)) :
        imPoints[i,0] = R[rectPoints[i,0]-1,rectPoints[i,1]-1];
        imPoints[i,1] = C[rectPoints[i,0]-1,rectPoints[i,1]-1];
    
    imPoints[:,0] = imPoints[:,0] + bbCorners[0,0];
    imPoints[:,1] = imPoints[:,1] + bbCorners[0,1];
    return imPoints 



# Compares the given foreground and background image, finds the largest
# blob that's significantly different from the background, and masks it,
# and a padded area around it in. Also returns the corners of the bounding
# box containing the entire masked-in area.
#
# Author: Aashiq Muhamed

def maskInObject(I, BG, padSz) :
    
    #pdb.set_trace()
    # How far in pixel distance the foreground has to be from the background to
    # be considered different.
    OFF_THRESH = 100;
    
    # Buffer around the edges of the image, to avoid detecting hands/feet/etc.
    # around the edges (since most objects are relatively centered, this
    # shouldn't hurt anything).
    BUF = 100;
    
    mask = np.zeros((np.size(I,0),np.size(I,1)));
    
    # Don't buffer the bottom because some objects get close to/run off it
    mask[BUF-1:,BUF-1:-BUF] = 1;
    
    # Compute and threshold pixel distance
    diff = np.sum(np.abs(I - BG),2);
    diff = diff > OFF_THRESH;

    diff = diff*mask;
    
    
    # Run blob detection

    label = measure.label(diff,8)
    CC = measure.regionprops(label)
    
    maxObj = -1;
    maxSz = -1;
    
    
    # Find the largest blob
    for i in range(len(CC)) :
        curSz = len(CC[i].coords);
        if curSz > maxSz :
            maxObj = i;
            maxSz = curSz;
    
    # Mask in the largest blob, pad, and compute bounding box.
    mask = np.zeros((np.size(I,0),np.size(I,1)));
    
    maskInd = CC[maxObj].coords;
    
    for j in maskInd :
        mask[tuple(j)] = 1 
    
    padFil = np.ones((padSz*2 + 1,padSz*2 + 1));
    mask = scipy.signal.convolve2d(mask,padFil,'same') > 0;
    
    #[R,C] = ind2sub(size(mask),maskInd);
    #maskSub = [R C];
    maskSub = maskInd
    
    bbCorners = np.zeros((2,2));
    
    bbCorners[0,:] = np.min(maskSub,0) - padSz;
    bbCorners[1,:] = np.max(maskSub,0) + padSz;
    return [mask, bbCorners] 

# Makes the masked-out area of the given image "white" (really, sets all
# channels to whatever value BGVAL specifies, slightly off-white
# corresponds well to the grasping dataset). This lets us work with
# different background colors and still have data matching the dataset
# closely.
#
# Author: Aashiq Muhamed

def makeBGWhite(I,mask) :

    BGVAL = 230;
    
    I = I*mask;
    I = I+np.dot((~mask),BGVAL);
    return I 

# Plot all rectangles in the Nx4x2 list of rectPts. Optionally also takes
# the color to plot the vertical and horizontal edges in, otherwise
# defaults to red/blue.
#
# Author: Aashiq Muhamed

def plotAllDBNRects(rectPts,vertColor = 'r',horizColor = 'b') :

    numRects = np.size(rectPts,0);
    
    startInd = 1;
    
    for i in range(1,numRects+1) :
        plotGraspRect(np.squeeze(rectPts[i-1,:,:]),vertColor,horizColor);
        startInd = startInd + 4;

# Returns the fraction of the given mask which is masked-in for the
# rectangle with corner (r,c) and height/width h/w.
#
# Author: Aashiq Muhamed

def rectMaskFraction(mask,r,c,h,w) :

    curM = mask[r-1:r+h-1,c-1:c+w-1];
    
    frac = np.sum(curM.flatten())/(w*h);
    return frac 


# Simple data whitening - just subtract the given feature-wise means, then
# divide by the given feature-wise standard deviations.
#
# Author: Aashiq Muhamed

def simpleWhiten(feat,means,stds) :

    mask = feat != 0;
    
    #feat = feat - np.expand_dims(means,0);
    feat = feat - means;
    #feat = feat/stds.T;
    feat = feat/stds;
    
    feat = feat*mask;
    return feat 


# Returns a scaled version of the input mask, scaled up based on the
# masked-out fraction for each mode. 
#
# Author: Aashiq Muhamed

def scaleMaskByModes2(mask,modes) :

    # Use some minimum scaling factor so we don't over-scale channels with a
    # lot masked out.
    MIN_SCALE = 0.6;
    
    numModes = np.max(modes);
    
    scaledMask = np.zeros(mask.shape);
   
    for mode in range(1,numModes+1) :
        myMask = mask[modes == mode];
        
        maskRatios = np.maximum(np.mean(myMask),MIN_SCALE);
        
        scaledMask[modes == mode] = myMask/maskRatios;
        
    return scaledMask 

# Scales a set of input features based on how much of their mask is
# masked-out for each channel.
#
# Author: Aashiq Muhamed

def scaleFeatForMask(feat,mask,modes) :

    mask = scaleMaskByModes2(mask,modes);
    
    feat = feat*mask;
    return feat 


# Removes outliers for masked data. Returned data will be zero-mean.
#
# Zeros out points with absolute value > stdCutoff * the std of the nonzero
# values in A, and updates the mask to reflect the removed points.
#
# Author: Aashiq Muhamed

def removeOutliersDet(A,mask,stdCutoff) :
    
    A[~mask] = float('nan');
    Amean = np.nanmean(np.expand_dims(A.flatten(),axis = 1));
    A = A - Amean;
    
    Astd = np.nanstd(np.expand_dims(A.ravel(),axis = 1));
    
    mask[np.abs(A) > (stdCutoff * Astd)] = False;
    
    A[~mask] = float('nan');
    A = A - np.nanmean(np.expand_dims(A.ravel(),axis = 1));
    A[np.isnan(A)] = 0;
    return [A,mask] 

## Plotting
    

# Plots a grasping rectangle over the current image, given a 4x2 matrix of
# points. Optionally also takes the color to plot the vertical and 
# horizontal edges in, otherwise defaults to red/blue.
#
# Returns a list of handles to the plotted lines, in case we need to remove
# them from the image later.
#
# Author: Aashiq Muhamed


def plotGraspRect(rectPts,vertColor = 'r',horizColor = 'b') :

    
    h1= pyplot.plot([rectPts[0,1], rectPts[1,1]],[rectPts[0,0], rectPts[1,0]],Color = vertColor);
    h2 = pyplot.plot([rectPts[2,1], rectPts[3,1]],[rectPts[2,0], rectPts[3,0]],Color = vertColor);
    h3 = pyplot.plot([rectPts[1,1], rectPts[2,1]],[rectPts[1,0], rectPts[2,0]],Color = horizColor);
    h4 = pyplot.plot([rectPts[0,1], rectPts[3,1]],[rectPts[0,0], rectPts[3,0]],Color = horizColor);
    pyplot.show()
    h = [h1, h2 , h3, h4]

    return h 

# Removes a set of lines from the current figure (given as a list of
# handles, so really could be anything with a handle to it).
#
# Author: Ian Lenz

def removeLines(h) :
    
    for i in h :
        l = i.pop(0)
        l.remove()
    pyplot.show()
