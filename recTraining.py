import numpy as np
import pdb
def dirtyRegCostL0(W,lseScale,lseEps,lseStepdown,l0Scale) :
    
    [absVal, absGrad] = smoothedAbs(W);
    abs0, _ = smoothedAbs(0);
    
    #Scale the log-sum-exponential to get a good approximation for lower
    # W values - otherwise, doesn't correlate well to the max
    curExp = np.exp(lseScale*absVal);
    
    # To make results easier to interpret, subtract out the minimum value
    # the log-sum-exponential can take from the cost, so the min cost is 0
    #minVal = (size(W,2)/lseScale) * log(size(W,1) + lseEps);
    minVal = np.log(np.exp(lseScale*abs0)*np.size(W,1) + lseEps)/lseScale;
    
    # Include an extra constant value added to the sum of exponents to
    # smooth near 0 (accounting for the absolute value op)
    expSum = np.sum(curExp,1) + lseEps;
    
    logSumExp = np.log(expSum)/lseScale - minVal;
    l0Arg = 1+l0Scale*(logSumExp**2);
    
    rowGrad = (2*l0Scale*logSumExp)/(expSum*l0Arg);
    
    cost = np.sum(np.log(l0Arg));
    
    # LSE derivative is just each individual exp value over the sum
    grad = curExp*rowGrad * absGrad;
    return (cost,grad)

# Scales each row by its L2 norm.
# Code by Quoc Le

def l2rowscaled(x, alpha) :

    normeps = 1e-5;
    epssumsq = np.sum(x**2,1) + normeps;   
    
    l2rows= np.expand_dims(np.sqrt(epssumsq)*alpha,axis = 1);
    y= x/l2rows;
    return y


# Inverse of the sigmoid function
# 
# Author: Aashiq Muhamed

def inverseSigmoid(y) :

    x = -np.log(1/y - 1);
    return x

# Computes the p-norm across the 2nd dim of W and its grad w/r/t each
# element of W.
# 
# Author: Aashiq Muhamed

def pNormGrad(W,p) :

    # Epsilon-smooth near 0
    EPS = 1e-6;
    
    sumPow = np.sum(np.abs(W)**p,1) + EPS;
    
    cost = np.sum(sumPow**(1/p));
    
    grad = (np.abs(W)**(p-1))*np.expand_dims(sumPow**(1/p - 1),axis = 1);
    grad = grad*np.sign(W);
    return (cost,grad)

# Computes an approximation to the L0 of the row-wise max for a set of
# weights W. Uses the log-sum-exponential to approximate the max, and
# log(1+x^2) to approximate L0.
#
# Author: Aashiq Muhamed

def logSumExpL0Cost(W,lseScale,lseEps,l0Scale) :

    # Use a smoothed absolute value. Compute the same for 0 so we can subtract
    # out the min. value, which is important for taking L0
    [absVal, absGrad] = smoothedAbs(W);
    abs0, _ = smoothedAbs(0);
    
    # Scale the log-sum-exponential to get a good approximation for lower
    # W values - otherwise, doesn't correlate well to the max
    curExp = np.exp(lseScale*absVal);
    
    # For the L0 part, we need the min value of the LSE part to actually be 0.
    # Compute the actual minimum and subtract it out (later)
    minVal = np.log(np.exp(lseScale*abs0)*np.size(W,1) + lseEps)/lseScale;
    
    # Include an extra constant value added to the sum of exponents to
    # smooth near 0
    expSum = np.sum(curExp,1) + lseEps;
    
    # Compute some useful values for computing cost & gradients
    logSumExp = np.log(expSum)/lseScale - minVal;
    l0Arg = 1+l0Scale*(logSumExp**2);
    
    # Gradient for each value is the exponential part times this value for the
    # appropriate row:
    rowGrad = (2*l0Scale*logSumExp)/(expSum*l0Arg);
    
    cost = np.sum(np.log(l0Arg));
    #pdb.set_trace()
    grad = curExp*np.expand_dims(rowGrad,axis = 1)*absGrad;
    return (cost,grad)

# Cost and gradient for structured multimodal regularization, taking the L0
# of the max for each modality and feature (as approximated by log(1+x^2)
# and log-sum-exponential, respectively)
# 
# Modes is a vector indicating the mode of each feature. A value of 0 indicates
# that the feature shouldn't be considered for multimodal regularization
# (e.g. a bias)
#
# Author: Aashiq Muhamed

#Params was redefined to be a dictionary
def multimodalRegL0(W,modes,params) :

    numModes = max(modes.ravel());
    
    cost = 0;
    grad = np.zeros(W.shape);
    
    # Compute cost and gradient for each mode
    
    for i in range(1,numModes+1) :
        temp  = (modes == i).flatten()
        
        if(np.size(temp) < W.shape[1]) :
            temp = np.append(temp,np.zeros(W.shape[1]-np.size(temp),dtype = bool))
        myW = W[:,temp];
        
        # Compute the cost and gradient for weights to this modality.
        [myCost,myGrad] = logSumExpL0Cost(myW,params['lseScale'],params['lseEps'],params['l0Scale']);
        
        # Add it to the cost, and set the appropriate gradients for the full
        # weight matrix.
        cost = cost + myCost;
        grad[:,temp] = myGrad;
    return (cost,grad)


# Scale a given set of weights to give a target mean value for sigmoid
# output and a target standard deviation for input to the sigmoid (since
# it's expensive to invert the standard deviation of a sigmoid)
#
# Mean is scaled by setting the bias appropriately, standard deviation is
# scaled by multiplicatively scaling the weights.
#
# This is useful because SAE will often give features with good relative
# weight values, but may not give well-scaled features which use the full
# range of the sigmoid.
# 
# Author: Aashiq Muhamed

def scaleAndBiasWeights(W, data, tgtStd, tgtMean) :

    # Find the target mean value for the input to the sigmoid (argument is
    # assumed to be output, ie in the 0-1 range)
    tgtXMean = inverseSigmoid(tgtMean);
    
    # Compute output and standard deviation. Use this so scale weights
    X = np.dot(data,W);
    
    Xstd = np.std(X,0);
    
    W = W/(Xstd/tgtStd);
    
    # Re-compute outputs and use this to scale mean value
    X = np.dot(data, W);
    meanX = np.mean(X,0);
    
    bias = tgtXMean - meanX;
    
    W = np.vstack((W,bias));
    return W


def scaleMaskByModes2(mask,modes) :

    MIN_SCALE = 0.6;
    
    numModes = np.max(modes);
    
    scaledMask = np.zeros(mask.shape);
    #pdb.set_trace()
    for mode in range(1,numModes+1) :
        temp = (modes == mode).flatten()
        myMask = mask[:,temp];
        
        maskRatios = np.maximum(np.mean(myMask),MIN_SCALE);
       
        
        scaledMask[:,temp] = myMask/np.expand_dims(maskRatios,axis = 1);
    
    
    #scaledMask = bsxfun(@times,scaledMask,size(scaledMask,2)./sum(scaledMask,2));
    return scaledMask 

# Computes the smoothed absolute value for each value in W. This is
# sqrt(w_{i,j}^2 + eps) for each weight, and serves to solve numerical 
# issues around zero.
#
# Author: Aashiq Muhamed
# Partially adapted from code by Quoc Le

def  smoothedAbs(W) :

    EPS = 1e-7;
    
    val = np.sqrt(W**2 + EPS);
    
    grad = W/val;
    return [val,grad]


# Cost and gradient for a smoothed L1 penalty on the weight matrix W. This
# means taking sqrt(w_{i,j}^2 + eps) for each weight w_{i,j}, and serves to
# solve numerical issues around zero by smoothing the penalty function.
# 
# Author: Aashiq Muhamed
# Partially adapted from code by Quoc Le

def smoothedL1Cost(W) :

    EPS = 1e-7;
    
    K = np.sqrt(EPS + W**2);
    
    cost = np.sum(K.ravel());
    grad = W/K;
    return (cost,grad)


