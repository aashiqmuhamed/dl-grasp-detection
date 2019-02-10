#All scripts in this class
# Also groups together training NN functions from Hinton

from util import *
from recTraining import *
import numpy as np
from processData import *
from loadData import *
import pdb
import scipy.optimize
#from minFunc import *
import scipy
class DeepGrasp :
    def __init__(self,depthFeat, colorFeat, normFeat, classes, inst, accepted, depthMask, colorMask) :
        self.params = {}
        self.options = {}
        self.RecordTrain = []
        self.RecordTest = []
        self.multiParams = {}
        self.opttheta = 0
        
        self.depthFeat =  depthFeat
        self.colorFeat =  colorFeat
        self.normFeat =  normFeat
        self.classes = classes
        self.inst = inst 
        self.accepted =  accepted
        self.depthMask =  depthMask
        self.colorMask = colorMask
        
        #self.Ibest = -1
        
        # Need to add a small 'epsilon' to the inputs to softmax to avoid
        # divide-by-zero errors
        self.SOFTMAX_EPS = 1e-6;
        
        # Number of training epochs to run. Each of these will consist of a number
        # of training iterations (set below), followed by a report of the current
        # accuracy
        #
        # Lowering this value will perform "early stopping," which may help combat
        # overfitting for some problems
        self.maxepoch=15;
        
        # Number of epochs to train just the top-level classifier weights, keeping
        # others fixed. This helps to keep the low-level weights from being skewed
        # when the top-level weights are still mostly random.
        self.INIT_EPOCHS = 5;
        
    # Processes grasping data from the workspace. Splits it into training and
    # testing sets, and whitens appropriately. Saves both the split, whitened
    # data and the parameters used for whitening. The later is necessary for
    # detection, where we need to whiten the data in the same way we did when
    # training for recognition.
    #
    # Author: Aashiq Muhamed
    
    #addpath ../util


    def processGraspData(self) :
        # For some purposes, it's useful to have both of these
        self.classes = np.hstack((self.classes, ~self.classes));
        
        # Split data into training and testing sets
        self.isTest = np.asarray(getGraspingSplit(self.classes[:,0],0.2),dtype = bool);
        
        self.splitGraspData();
        
        #Uncomment to save unwhitened split data:
        # save -v7.3 /localdisk/data/graspNewerSplit;
        
        # Whiten the data (critical for deep learning to work)
        
        self.whitenDataCaseWiseDepth() 
        
        # Save the training and testing data to separate files
        np.save('C:/Users/aashi/Desktop/Python conversion/data/train/trainData',self.trainData )
        np.save('C:/Users/aashi/Desktop/Python conversion/data/train/trainMask',self.trainMask)
        np.save('C:/Users/aashi/Desktop/Python conversion/data/train/trainClasses',self.trainClasses)
        
        np.save('C:/Users/aashi/Desktop/Python conversion/data/test/testData',self.testData )
        np.save('C:/Users/aashi/Desktop/Python conversion/data/test/testMask',self.testMask)
        np.save('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses',self.testClasses)
        
        # Save the parameters used for whitening - these will be used for detection
        self.featStds = chanStdsToFeat(self.chanStds,np.sqrt(np.size(self.trainData,1)/7));
        np.save('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses/featMeans', self.featMeans);
        np.save('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses/featStds', self.featStds);
    
    
    # Splits a set of grasping data in the workspace into training and testing
    # data, using a computed binary vector isTest which has one entry for each
    # case, indicating whether or not that case should be sent to the test set.
    #
    # Author: Aashiq Muhamed
    
    def splitGraspData(self) :
        # Split all the data based on isTest
        self.trainDepthData = self.depthFeat[~self.isTest];
        self.testDepthData = self.depthFeat[self.isTest];
        
        self.trainColorData = self.colorFeat[~self.isTest];
        self.testColorData = self.colorFeat[self.isTest];
        
        self.trainNormData = self.normFeat[~self.isTest];
        self.testNormData = self.normFeat[self.isTest];
        
        self.trainDepthMask = self.depthMask[~self.isTest];
        self.testDepthMask = self.depthMask[self.isTest];
        
        self.trainColorMask = self.colorMask[~self.isTest];
        self.testColorMask = self.colorMask[self.isTest];
        
        self.trainClasses = self.classes[~self.isTest];
        self.testClasses = self.classes[self.isTest];
        
        #Clean up after ourselves to save some memory
        #clear depthData colorData normData depthMask colorMask classes;

    
    
    # Whitens a split set of grasping data. This means doing case-wise
    # whitening on the depth channel, then dropping feature-wise means and
    # scaling per channel for all features.
    #
    # Author: Aashiq Muhamed
    
    # Since masks may be resized, need to re-convert them to binary by
    # thresholding
    def whitenDataCaseWiseDepth(self) :
        
        MASK_THRESH = 0.75;
        
        self.trainDepthMask = self.trainDepthMask > MASK_THRESH;
        self.testDepthMask = self.testDepthMask > MASK_THRESH;
        
        self.trainColorMask = self.trainColorMask > MASK_THRESH;
        self.testColorMask = self.testColorMask > MASK_THRESH;
        
        # Re-mask the depth data
        self.trainDepthData = self.trainDepthData*self.trainDepthMask;
        self.testDepthData = self.testDepthData*self.testDepthMask;
        
        # First, drop depth means case-wise - necessary since distance to object
        # might change
        [self.trainDepthData,self.trainScale] = caseWiseWhiten(self.trainDepthData,self.trainDepthMask);
        [self.testDepthData,self.testScale] = caseWiseWhiten(self.testDepthData,self.testDepthMask);
        
        # Now, collect features
        [self.trainData, self.trainMask] = combineAllFeat(self.trainDepthData,self.trainColorData,self.trainNormData,self.trainDepthMask,self.trainColorMask);
        [self.testData, self.testMask] = combineAllFeat(self.testDepthData,self.testColorData,self.testNormData,self.testDepthMask,self.testColorMask);
        
        self.trainData = self.trainData*self.trainMask;
        self.testData = self.testData*self.testMask;
        
        # Drop means by feature
        [self.trainData, self.featMeans] = dropMeanByFeat(self.trainData,self.trainMask);
        self.testData = self.testData-self.featMeans;
        self.testData = self.testData*self.testMask;
        
        # Scale each channel by its standard deviation, preserving relative values
        # within channels so that we don't exaggerate particular features, but
        # giving each channel equal variance
        [self.trainData, self.chanStds] = scaleDataByChannelStds(self.trainData,self.trainMask,7);
        self.testData = scaleDataByChannel(self.testData,self.chanStds);
        


    
    
    # Partially adapted from code by Quoc Le
    # 
    # Pre-train second-layer weights using the sparse autoencoder (SAE)
    # algorithm. Assumes the input features are in the 0-1 range (e.g the
    # output of a previous hidden layer), and so uses a sigmoid to reconstruct 
    # them. 
    # 
    # Takes input as trainFeat1, which is assumed to be the output from the
    # first hidden layer
    #
    # If opttheta is present in your workspace, this script will continue
    # optimizing from those values, otherwise it'll initialize weights to
    # random values (so, be sure to clear it after training the first hidden
    # layer)
    #
    # This code requires minFunc to run, but you can use the included cost
    # functions with the optimizer of your choice. 
    #
    # The following should point to your location for minFunc:
    
    #opttheta should be zero initially
    def runSAEBinary(self) :
        
        # Initialize some parameters based on the training data
        self.params['m']= np.size(self.trainFeat1,0);                 # number of training cases
        self.params['n'] =np.size(self.trainFeat1,1)+1;   # dimensionality of input
        
        # Switch this to toggle the use of a discriminative bias for training SAE.
        # Sometimes, learning without a bias will give better features.
        USE_BIAS = 1;
        
        # Add bias to input data
        self.x = np.vstack((self.trainFeat1.T, USE_BIAS*np.ones((1,np.size(self.trainFeat1,0)))));
        
        #Configure hyperparameters
        
        # Number of features to learn
        self.params['numFeatures'] = 50; #200;
        
        # Weight for the sparsity component of SAE
        self.params['lambda'] = 3;
        
        # Weight for an L1 penalty on weights
        self.params['l1Cost'] = 3e-4;
        
        # Numerical parameter, probably doesn't need to be changed
        self.params['epsilon'] = 1e-5;
        
        # Options for minFunc. L-BFGS typically gives the best results. 
        # MaxIter sets the maximum number of learning iterations to use

        #self. options['Method'] = 'lbfgs';
        self.options['maxfun'] = 15000;
        self.options['maxiter'] = 200;
        
        # If there isn't a set of parameters in the workspace already, initialize
        # them. If there are, we'll keep optimizing from them.
        if (np.size(self.opttheta)) == 1 :
            # initialize with random weights
            self.randTheta = np.random.randn(self.params['numFeatures'],self.params['n'])*0.01;
            self.randTheta = l2rowscaled(self.randTheta,1);
            
            self.opttheta = self.randTheta.ravel();
            
            # This version will also include a generative bias since the features
            # aren't zero-mean. We optimize this even though we don't use it in the
            # final network (it helps in learning good features, since they don't
            # have to compensate for this).
            #pdb.set_trace()
            self.opttheta = np.vstack((np.expand_dims(self.opttheta,axis = 1),np.zeros((self.params['n'],1))))
            
            print('Initializing')
        
        
        # Use minFunc to run optimization
        temp =  scipy.optimize.minimize( lambda theta: self.sparseAECostBinaryGenBias(theta, self.x, self.params), self.opttheta, options = self.options,method = 'L-BFGS-B',jac = True);   # Use x or xw 
        self.opttheta, self.cost, self.exitflag = (temp['x'],temp['fun'],temp['status'])
        # Reshape into a weight matrix for convenience
        self.W = np.reshape(self.opttheta[:self.params['numFeatures']*self.params['n']], (self.params['numFeatures'], self.params['n']));

    
    
    
        # Back-propagates recognition error for a pre-trained two-layer deep net,
        # adding a third classification layer. This layer is trained alone for a 
        # few iterations before the lower-layer weights are unfrozen.
        #
        # Includes both L1 and structured regularization terms on hidden layer 
        # weights, and L1 and L2 penalties on classifier weights.
        # 
        # Runs classification on training and test set every few iterations and
        # reports accuracy, to aid in tuning.
        #
        # Although grasping is a two-class problem, this code uses softmax
        # regression/classification. This doesn't cause any problems for two
        # classes, but would allow it to be used for multiclass problems as well.
        
        
        # Based on code provided by Ruslan Salakhutdinov and Geoff Hinton, which
        # comes with the following notice:
        #
        # Permission is granted for anyone to copy, use, modify, or distribute this
        # program and accompanying programs and documents for any purpose, provided
        # this copyright notice is retained and prominently displayed, along with
        # a note saying that the original programs are available from our
        # web page.
        # The programs and documents are distributed without any warranty, express or
        # implied.  As the programs were written for research purposes only, they have
        # not been tested to the degree that would be advisable in any important
        # application.  All use of these programs is entirely at the user's own risk.
        #
        # Original Salakhutdinov/Hinton code available at: 
        # http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
        
    def runBackpropMultiReg(self) :
        
        self.numclasses=np.size(self.trainClasses,1);
        
        # Configure minFunc. L-BFGS typically gives the best performance. MaxIter
        # will change the number of iterations of L-BFGS run for each training
        # epoch
        #self.options['Method'] = 'lbfgs';
        
        
        self.options['maxfun'] = 15000;
        self.options['maxiter'] = 1;
        self.options['disp'] = True
        
#        self.testbatchdata = self.testData;
#        self.testbatchtargets = self.testClasses;
#        self.batchdata = self.trainData;
#        self.batchtargets = self.trainClasses;
        
        #Edit
        self.testbatchdata = np.expand_dims(self.testData,axis = 2);
        self.testbatchtargets = np.expand_dims(self.testClasses,axis = 2);
        self.batchdata = np.expand_dims(self.trainData,axis = 2);
        self.batchtargets = np.expand_dims(self.trainClasses,axis = 2);
        
        
        self.numdims = np.size(self.trainData,1);
        #self.numbatches = 1;
        self.numcases = np.size(self.batchdata,0);
        self.N= self.numcases; 
        
        # Initialize hidden layer weights from pretrained values
        self.w1 = np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWL1shallow/W.npy')
        
        self.w2 = np.load('C:/Users/aashi/Desktop/Python conversion/data/graspWL2shallow/W.npy') 
        
        # Randomly initialize classifier weights
        self.w_class = 0.01*np.random.randn(np.size(self.w2,1)+1,self.numclasses);
         
        # Record layer sizes and do some other setup
        self.l1=np.size(self.w1,0)-1;
        self.l2=np.size(self.w2,0)-1;
        self.l3=np.size(self.w_class,0)-1;
        self.l4=self.numclasses; 
        self.test_err=[];
        self.train_err=[];
        self.train_crerr = [];
        self.test_crerr = [];
        
        # Main training loop
        #self.Ibest = 0
        for epoch in range(self.maxepoch) :
        
            # Compute training error
            err=0; 
            err_cr=0;
            self.counter=0;
            #pdb.set_trace()
            #self.batchdata = np.expand_dims(self.batchdata,axis = 2)
            self.numcases, self.numdims, self.numbatches=self.batchdata.shape;
            N=self.numcases;
            
            # Add up error for each batch (although the grasping code only makes 1
            # batch since the data is small enough)
            
            for batch in range(self.numbatches) :
                self.data = self.batchdata[:,:,batch];
                self.target = self.batchtargets[:,:,batch];
                
                # Add bias to data
                self.data = np.hstack((self.data, np.ones((N,1))));
                
                # Forward-prop through network
                self.w1probs = 1/(1 + np.exp(-np.dot(self.data,self.w1)));
                print(self.data.shape)
                self.w1probs = np.hstack((self.w1probs, np.ones((N,1))));
                self.w2probs = 1/(1 + np.exp(-np.dot(self.w1probs,self.w2)));
                self.w2probs = np.hstack((self.w2probs, np.ones((N,1))));
                
                # Compute softmax classification
                self.targetout = np.exp(np.dot(self.w2probs,self.w_class));
                self.targetout = self.targetout/np.tile(np.expand_dims(np.sum(self.targetout,1),axis = 1),(1,self.numclasses));
                
                # Compare predicted and ground-truth output
                I =np.max(self.targetout,1); 
                J = np.argmax(self.targetout,1)
                I1=np.max(self.target,1);
                J1 = np.argmax(self.target,1);
                self.counter=self.counter+np.sum(J==J1);
                err_cr = err_cr- np.sum( self.target*np.log(self.targetout)) ;

                ##Store best grasp self.data
#                if (epoch == self.maxepoch-1) :
#                    for i,element in enumerate(I) :
#                        if (element>self.Ibest)  :
#                            self.Ibest = element
#                            self.bestdata = self.data[i,:]
#                            self.Wbest = self.w1[:,i]
          
            
            # Compute and record total training error
            self.train_err.append(self.numcases*self.numbatches-self.counter);
            self.train_crerr.append(err_cr/self.numbatches);
        
            # Now, compute test error, identical to above except for test data
            err=0;
            err_cr=0;
            self.counter=0;
            [self.testnumcases ,self.testnumdims ,self.testnumbatches]=self.testbatchdata.shape;
            N=self.testnumcases;
            
            for batch in range(self.testnumbatches) :
                self.data = self.testbatchdata[:,:,batch];
                self.target = self.testbatchtargets[:,:,batch];
                self.data = np.hstack((self.data, np.ones((N,1))));
                self.w1probs = 1/(1 + np.exp(-np.dot(self.data,self.w1))); 
                self.w1probs = np.hstack((self.w1probs, np.ones((N,1))));
                self.w2probs = 1/(1 + np.exp(-np.dot(self.w1probs,self.w2))); 
                self.w2probs = np.hstack((self.w2probs,np.ones((N,1))));
                self.targetout = np.exp(np.dot(self.w2probs,self.w_class));
                self.targetout = self.targetout/np.tile(np.expand_dims(np.sum(self.targetout,1),axis = 1),(1,self.numclasses));
                
                I =np.max(self.targetout,1); 
                J = np.argmax(self.targetout,1)
                I1=np.max(self.target,1);
                J1 = np.argmax(self.target,1);
                self.counter=self.counter+np.sum(J==J1);
                err_cr = err_cr- np.sum( self.target*np.log(self.targetout)) ;

        
            self.test_err.append(self.testnumcases*self.testnumbatches-self.counter);
            self.test_crerr.append(err_cr/self.testnumbatches);
            
            # Print both training and test error. This gives a good idea of how
            # much the algorithm is overfitting, and when it starts doing so
            print(1,'Before epoch {} Train # misclassified: {} from {} ({}). Test # misclassified: {} from {} ({}) \t \t \n'.format \
                  (epoch,self.train_err[-1],self.numcases*self.numbatches,100*(1-(self.train_err[-1]/(self.numcases*self.numbatches))), \
                   self.test_err[-1],self.testnumcases*self.testnumbatches,100*(1-(self.test_err[-1]/(self.testnumcases*self.testnumbatches)))));
        
            self.RecordTrain.append(self.train_err[-1])
            self.RecordTest.append(self.test_err[-1])
            self.tt=0;
            
            # Loop over batches and optimize training error (w/ regularization) for
            # each
            for batch in range(self.numbatches) :
                print(1,'epoch {} batch {}\r'.format(epoch,batch));
                
                self.tt=self.tt+1; 
                self.data=self.batchdata[:,:,batch];
                self.targets=self.batchtargets[:,:,batch]; 
                
                # For some number of epochs, just update top-level classification
                # weights, keeping others fixed
                if epoch <= self.INIT_EPOCHS :
                    
                    N = np.size(self.data,0);
                    
                    # Forward-prop to get 2nd-layer features
                    self.XX = np.hstack((self.data, np.ones((N,1))));
                    self.w1probs = 1/(1 + np.exp(-np.dot(self.XX,self.w1))); 
                    self.w1probs = np.hstack((self.w1probs ,np.ones((N,1))));
                    self.w2probs = 1/(1 + np.exp(-np.dot(self.w1probs,self.w2)));
                    
                    # Optimize top-layer weights from 2nd-layer features
                    self.VV = self.w_class.ravel();
                    self.Dim = np.vstack((self.l3, self.l4));
                   
                    temp = scipy.optimize.minimize(lambda theta: self.softmaxInitCost(theta,self.Dim,self.w2probs,self.targets), self.VV, options = self.options,method = 'L-BFGS-B',jac = True);
                    self.X, self.fX, exitflag= (temp['x'],temp['fun'],temp['status'])
                    self.w_class = np.reshape(self.X,(self.l3+1,self.l4));
        
                else :
                    # Once top-layer weights have been optimized alone, use
                    # back-prop to optimize all weights together
                    self.VV = np.hstack((self.w1.ravel(), self.w2.ravel(), self.w_class.ravel()));
                    self.Dim = np.vstack((self.l1, self.l2, self.l3, self.l4));
                    
                    temp = scipy.optimize.minimize(lambda theta : self.softmaxBackpropCostMultiReg(theta,self.Dim,self.data,self.targets,self.trainModes), self.VV, method = 'L-BFGS-B',options = self.options,jac = True);
                    self.X, self.fX, exitflag= (temp['x'],temp['fun'],temp['status'])
                    
                    # Re-form weight matrices from the output of minFunc
                    self.w1 = np.reshape(self.X.T.flatten()[:(self.l1+1)*self.l2],(self.l1+1,self.l2));
                    xxx = (self.l1+1)*self.l2;
                    self.w2 = np.reshape(self.X[xxx:xxx+(self.l2+1)*self.l3],(self.l2+1,self.l3));
                    xxx = xxx+(self.l2+1)*self.l3;
                    self.w_class = np.reshape(self.X[xxx:xxx+(self.l3+1)*self.l4],(self.l3+1,self.l4));
        
  
        ###%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

        
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/w1',self.w1)
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/w2',self.w2)
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/w_class',self.w_class)
             
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/test_err',self.test_err)
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/test_crerr',self.test_crerr)
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/test_err',self.test_err)
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/train_crerr',self.train_crerr)
            np.save('C:/Users/aashi/Desktop/Python conversion/data/backpropdata/train_err',self.train_err)
             
        

    
    def runSAEMultiSparse(self) :
        
        # Partially adapted from code by Quoc Le
        # 
        # Pre-train first-layer weights using the sparse autoencoder (SAE)
        # algorithm, with a structured multimodal regularization penalty. Assumes
        # the following variables are present in the workspace:
        #
        # trainData: matrix of training data, where each row represents a case and
        # each column represents a feature
        # 
        # trainMask: mask for the training data, same size as trainData
        # A 0 indicates a masked-out value and 1 is masked-in. Reconstruction 
        # penalties will not be considered for masked-out values. Mask is also used
        # to scale input values and their reconstruction penalties
        #
        # trainModes: vector indicating the modality which each feature belongs to
        # 
        # All of these are prefixed by "train" so you can run:
        # clearvars -except train*
        # to clean up after this script if you want to re-run training
        #
        # If opttheta is present in your workspace, this script will continue
        # optimizing from those values, otherwise it'll initialize weights to
        # random values
        #
        # This code requires minFunc to run, but you can use the included cost
        # functions with the optimizer of your choice. 
        #
        # The following should point to your location for minFunc:
    
        # Switch this to toggle the use of a discriminative bias for training SAE.
        # Sometimes, learning without a bias will give better features.
        USE_BIAS = 0;
        
        # Initialize some parameters based on the training data
    
        self.params['m']=np.size(self.trainData,0);     # number of training cases
        self.params['n']=np.size(self.trainData,1)+1;   # dimensionality of input (+1 for bias)
        
        # Scale the training mask based on fraction of masked-out values
        self.scaledMask = scaleMaskByModes2(self.trainMask,self.trainModes);
        
        # Add bias to training data
        self.x = np.vstack((self.trainData.T, USE_BIAS*np.ones((1,np.size(self.trainData,0)))));
        self.mask = np.vstack((self.scaledMask.T, np.zeros((1,np.size(self.trainData,0)))));
        self.modes = np.hstack((self.trainModes,np.expand_dims(np.asarray([0]),axis = 0)));
        
        # Configure hyperparameters
        
        # Number of features to learn
        self.params['numFeatures'] = 50 #200;
        
        # Weight for the sparsity component of SAE
        self.params['lambda'] = 3;
        
        # Weight for an L1 penalty on weights
        self.params['l1Cost'] = 3e-4;
        
        # Weight for the structured multimodal regularization
        self.params['multiCost'] = 0.01;
        
        # Parameters for numerical approximations
        self.params['epsilon'] = 1e-5;
        self.params['lseScale'] = 200;
        self.params['lseEps'] = 0;
        self.params['lseStepdown'] = 1;
        self.params['l0Scale'] = 1e15;
        self.params['nonDG'] = 0;
        
        # Options for minFunc. L-BFGS typically gives the best results. 
        # MaxIter sets the maximum number of learning iterations to use
        #self.options['Method'] = 'lbfgs';
        self.options['maxfun'] = 15000;
        self.options['maxiter'] = 200;
        
        # If there isn't a set of parameters in the workspace already, initialize
        # them. If there are, we'll keep optimizing from them.
        #pdb.set_trace()
        if np.size(self.opttheta) == 1 :
            # initialize with random weights
            self.randTheta = np.random.randn(self.params['numFeatures'],self.params['n'])*0.01;
            self.randTheta = l2rowscaled(self.randTheta,1);
        
            self.opttheta = self.randTheta.flatten();
            print('Initializing');
    
        
        # Use minFunc to run optimization
                    
        temp = scipy.optimize.minimize( lambda theta : self.sparseAECostMultiRegWeighted(theta, self.x, self.mask, self.modes, self.params), self.opttheta, options = self.options,method = 'L-BFGS-B',jac = True);
        [self.opttheta, self.cost, exitflag] = (temp['x'],temp['fun'],temp['status'])
        #pdb.set_trace()
        # Reshape into a weight matrix for convenience
        self.W = np.reshape(self.opttheta, (self.params['numFeatures'], self.params['n']));
    


    
    # Computes softmax classification error (as the cross-entropy between the
    # estimated and target distributions) and a set of regularization
    # penalties, along with their gradients for optimization.
    #
    # Back-propagates gradients through classifier weights and two hidden
    # layers
    # 
    # Inputs:
    # VV: vector of hidden layer and classifier weights, flattened for minFunc
    # Dim: vector of hidden layer sizes
    # XX: training data
    # target: target output distribution (usually, just ground-truth classes)
    # modes: modality index for each input feature
    
    # GPU-enabled: if you don't add nogpu to MATLAB's path, matrices will be
    # loaded onto the GPU. This can speed up computation, but might overload
    # GPU memory
    # Adding the nogpu directory to the path will turn GPU operations into
    # no-ops, so everything will run on the CPU as normal
    
    # Based on code provided by Ruslan Salakhutdinov and Geoff Hinton, which
    # comes with the following notice:
    #
    # Permission is granted for anyone to copy, use, modify, or distribute this
    # program and accompanying programs and documents for any purpose, provided
    # this copyright notice is retained and prominently displayed, along with
    # a note saying that the original programs are available from our
    # web page.
    # The programs and documents are distributed without any warranty, express or
    # implied.  As the programs were written for research purposes only, they have
    # not been tested to the degree that would be advisable in any important
    # application.  All use of these programs is entirely at the user's own risk.
    #
    # Original Salakhutdinov/Hinton code available at: 
    # http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
    
    
    def softmaxBackpropCostMultiReg(self,VV,Dim,XX,target,modes) :
        
        N = np.size(XX,0);
        
        # Weight for class error
        CLASS_W = 1;
        
        # Weight for class-weight L2 regularization
        L2_REG_W = N*1e-6;
        
        # Weight for L1 regularization of other weights
        L1_REG_W = N*1e-6;
        
        # Weight for multimodal sparsity
        MULTI_W = N*1e-6;
        
        # Numerical parameters for multimodal regularization
        self.multiParams['lseScale'] = 300;
        self.multiParams['lseEps'] = 0;
        self.multiParams['l0Scale'] = 5e4;
        
        numclasses = np.size(target,1);
        
        ## Hidden layer dimensions
        l1 = int(Dim[0]);
        l2 = int(Dim[1]);
        l3= int(Dim[2]);
        l4= int(Dim[3]);
        
        #target = gpuArray(target);
        
        # Re-form weight matrices from parameter vector from minFunc
        #w1 = gpuArray(reshape(VV(1:(l1+1)*l2),l1+1,l2));
        w1 = (np.reshape(VV[:(l1+1)*l2],(l1+1,l2)));
        
        xxx = (l1+1)*l2;
        #w2 = gpuArray(reshape(VV(xxx+1:xxx+(l2+1)*l3),l2+1,l3));
        w2 = (np.reshape(VV[xxx:xxx+(l2+1)*l3],(l2+1,l3)));
        xxx = xxx+(l2+1)*l3;
        #w_class = gpuArray(reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4));
        w_class = (np.reshape(VV[xxx:xxx+(l3+1)*l4],(l3+1,l4)));
        
        # Forward-propagate and compute softmax output
        #XX = gpuArray([XX ones(N,1)]);
        XX = np.hstack((XX,np.ones((N,1))));
        w1probs = 1/(1 + np.exp(-np.dot(XX,w1))); w1probs = np.hstack((w1probs,np.ones((N,1))));
        w2probs = 1/(1 + np.exp(-np.dot(w1probs,w2))); w2probs = np.hstack((w2probs,np.ones((N,1))));
        
        targetout = np.exp(np.dot(w2probs,w_class));
        targetout = targetout/np.tile(np.expand_dims(np.sum(targetout,1),axis = 1),(1,numclasses));
        
        # Compute regularization penalties and gradients
        l2Cost,l2Grad = pNormGrad(w_class[:-1,:].T,2);
        
        [w1L1Cost,w1L1Grad] = smoothedL1Cost(w1[:-1,:]);
        [w2L1Cost,w2L1Grad] = smoothedL1Cost(w2[:-1,:]);
        
        #pdb.set_trace()
        [multiCost, multiGrad] = multimodalRegL0(w1.T,modes,self.multiParams);
        multiGrad = multiGrad.T;
        
        #gather
        f = (-np.sum( target*np.log(targetout))*CLASS_W + l2Cost * L2_REG_W + (w1L1Cost + w2L1Cost)*L1_REG_W + MULTI_W * multiCost);
        
        deriv1 = w1probs*(1-w1probs);
        deriv2 = w2probs*(1-w2probs);
        
        # Compute error gradient for classifier weights
        IO = (targetout-target)*CLASS_W;
        Ix_class=IO; 
        dw_class =  np.dot(w2probs.T,Ix_class); 
        
        # Back-propagate erorr gradient to second-layer hiddens
        Ix2 = np.dot(Ix_class,w_class.T)*deriv2; 
        Ix2 = Ix2[:,:-1];
        dw2 =  np.dot(w1probs.T,Ix2);
        
        # Back-propagate to first-layer hiddens
        Ix1 = (np.dot(Ix2,w2.T))*deriv1; 
        Ix1 = Ix1[:,:-1];
        dw1 =  np.dot(XX.T,Ix1);
        
        # Add regularization to gradients and collect and flatten them for minFunc
        #gather
        dw1 = (dw1 + np.vstack((w1L1Grad, np.zeros((1,np.size(w1,1)))))*L1_REG_W + MULTI_W * multiGrad);
        dw2 = (dw2 + np.vstack((w2L1Grad, np.zeros((1,np.size(w2,1)))))*L1_REG_W);
        dw_class = dw_class + np.vstack((l2Grad.T, np.zeros((1,np.size(w_class,1)))))*L2_REG_W
        
        df = np.expand_dims(np.hstack((dw1.flatten(), dw2.flatten(), dw_class.flatten())),axis = 1); 
        
        return (f, df)
    
    # Cost and gradient for initializing a softmax classifier. Uses the given
    # input features to classify for the given target classes, and computes the
    # cross-entropy between the estimated and target distributions.
    
    # Based on code provided by Ruslan Salakhutdinov and Geoff Hinton, which
    # comes with the following notice:
    #
    # Permission is granted for anyone to copy, use, modify, or distribute this
    # program and accompanying programs and documents for any purpose, provided
    # this copyright notice is retained and prominently displayed, along with
    # a note saying that the original programs are available from our
    # web page.
    # The programs and documents are distributed without any warranty, express or
    # implied.  As the programs were written for research purposes only, they have
    # not been tested to the degree that would be advisable in any important
    # application.  All use of these programs is entirely at the user's own risk.
    #
    # Original Salakhutdinov/Hinton code available at: 
    # http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html


    def softmaxInitCost(self,VV,Dim,feat,target) :
    
        # Some small values to avoid divide-by-zeros
        SOFTMAX_EPS = 1e-6;
        EPS2 = 1e-10;
        
        numclasses = np.size(target,1);
        N = np.size(feat,0);
        
        # Weights for the regularization and misclassification penalties in the
        # cost function
        W_L1 = N*1e-8;
        W_L2 = N*1e-6;
        W_CLASS = 1;
        
        l1 = int(Dim[0]);
        l2 = int(Dim[1]);
        #pdb.set_trace()
        # Convert the vector input from minFunc into a weight matrix
        w_class = np.reshape(np.expand_dims(VV,axis = 1),(l1+1,l2));
        
        # Add a bias to input features
        feat = np.hstack((feat,np.ones((N,1)))) 
        
        # Compute classification estimates
        targetout = np.exp(np.dot(feat,w_class));
        targetout = targetout/np.tile(np.expand_dims(np.sum(targetout,1),axis = 1)+SOFTMAX_EPS,(1,numclasses));
        
        # Compute L2-regularization cost and gradient
        [l2Cost,l2Grad] = pNormGrad(w_class[:-1,:].T,2);
        l2Grad = l2Grad.T;
        
        # Use a smoothed L1 penalty to avoid "ringing"
        [l1Cost,l1Grad] = smoothedL1Cost(w_class[:-1,:]);
        
        # Compute total cost 
        f = -np.sum( target*np.log(targetout+EPS2))*W_CLASS + l2Cost*W_L2 + l1Cost*W_L1;
        
        # Compute gradient for classification cost for the classifier weights
        IO = (targetout-target)*W_CLASS;
        Ix_class=IO; 
        #pdb.set_trace()
        dw_class =  (feat.T.dot(Ix_class)) + np.vstack((l2Grad, np.zeros((1,np.size(w_class,1))))).dot(W_L2) + np.vstack((l1Grad, np.zeros((1,np.size(w_class,1))))).dot(W_L1); 
        
        # Convert gradient back into a vector for minFunc
        df = np.expand_dims(dw_class.flatten(),axis = 1); 
        return (f, df)

    
    # Sparse autoencoder cost for binary (sigmoidal) inputs. These are assumed
    # to be in the 0-1 range, probably the outputs of a previous hidden layer. 
    #
    # Since these features will not be zero-mean, we also learn a generative
    # bias for each, which is added to the input to the sigmoid that tries to
    # reconstruct the feature. Even though these biases will not be used in the
    # final network, learning them here improves the quality of the learned
    # features.
    # 
    # Author: Aashiq Muhamed
    
    def sparseAECostBinaryGenBias(self,theta, x, params) :
        
        N = np.size(x,1);
        
        # Unpack weights and gen. biases from parameters. Weights are the first
        # part, gen. biases the second
        W = np.reshape(theta[:params['numFeatures']*params['n']], (params['numFeatures'], params['n']));
        genBias = theta[params['numFeatures']*params['n']:];
        
        # Compute and scale L1 regularization cost
        [l1Cost, l1Grad] = smoothedL1Cost(W[:,:-1]);
        l1Cost = l1Cost*params['l1Cost'];
        l1Grad = np.hstack((l1Grad, np.zeros((np.size(W,0),1))))*params['l1Cost'];
       
        # Forward propagation through autoencoder
        h = 1/(1+np.exp(-np.dot(W,x)));
        r = 1/(1+np.exp(-(np.dot(W.T,h)+ np.expand_dims(genBias,axis = 1))));
        
        # Sparsity cost (smoothed L1)
        K = np.sqrt(params['epsilon'] + h**2);
        sparsity_cost = params['lambda']* np.sum(K);
        K = 1/K;
        
        # Compute reconstruction cost and back-propagate
        diff = (r - x);
        
        # Assume last term is bias - don't reconstruct it
        diff[-1:,:] = 0;
        
        reconstruction_cost = 0.5 * np.sum(diff**2);
        outderv = diff*r*(1-r);
        
        # Sum up cost terms
        # Scale up reg. term based on # of training cases to keep them even with
        # data-based costs
        cost = sparsity_cost + reconstruction_cost + l1Cost*N;
        
        # Backprop output layer
        W2grad = np.dot(outderv,h.T);
        genBiasGrad = np.sum(outderv,1);
        
        # Backprop hidden Layer
        outderv = np.dot(W,outderv);
        outderv = outderv + params['lambda']* (h * K);
        outderv = outderv* h * (1-h);
        
        W1grad = np.dot(outderv, x.T);
        Wgrad = W1grad + W2grad.T + l1Grad*N;
        #pdb.set_trace()
        # Unproject gradient for minFunc
        grad = np.vstack((np.expand_dims(Wgrad.ravel(),axis = 1), np.expand_dims(genBiasGrad.ravel(),axis =1)))
        return (cost,grad)

    
    # Cost function and gradients for sparse autoencoder (SAE) with L1 and
    # multimodal regularization. Weights network input and reconstruction
    # penalties based on the given mask, ignoring reconstruction for masked-out
    # values.
    #
    # Author: Aashiq Muhamed
    # Partially adapted from code by Quoc Le
    
    def sparseAECostMultiRegWeighted(self,theta, x, mask, modes, params) :
    
        N = np.size(x,1);
        
        # unpack weight matrix
        W = np.reshape(theta, (params['numFeatures'], params['n']));
        
        # Compute and scale regularization penalties
        [l1Cost, l1Grad] = smoothedL1Cost(W[:,:-1]);
        l1Cost = l1Cost*params['l1Cost'];
        l1Grad = np.hstack((l1Grad, np.zeros((np.size(W,0),1))))*params['l1Cost'];
        
        [multiCost,multiGrad] = multimodalRegL0(W,modes,params);
        multiCost = multiCost*params['multiCost'];
        multiGrad = multiGrad*params['multiCost'];
        
        # Scale inputs based on mask
        xScaled = x*mask;
        
        # Forward propagation through autoencoder
        h = 1/(1+np.exp(-np.dot(W,xScaled)));
        r = np.dot(W.T,h);
        
        # Sparsity cost (smoothed L1)
        K = np.sqrt(params['epsilon'] + h**2);
        sparsity_cost = params['lambda']* np.sum(K);
        K = 1/K;
        
        # Compute reconstruction cost and back-propagate
        # Mask out non-visible pixels - don't care about their recon
        diff = (r - x);
        
        reconstruction_cost = 0.5 * np.sum(mask*(diff**2));
        outderv = diff*mask;
        
        
        #Sum up cost terms
        # Scale up reg. terms based on # of training cases to keep them even with
        # data-based costs
        cost = sparsity_cost + reconstruction_cost + (l1Cost + multiCost)*N;
        
        # Backprop output layer
        W2grad = np.dot(outderv, h.T);
        
        # Backprop hidden Layer
        outderv = np.dot(W,outderv);
        outderv = outderv + params['lambda'] * (h * K);
        outderv = outderv* h * (1-h);
        
        W1grad = np.dot(outderv,xScaled.T);
        Wgrad = W1grad + W2grad.T + (l1Grad + multiGrad)*N;
        
        # Unproject gradient for minFunc
        grad = np.expand_dims(Wgrad.ravel(),axis = 1);
        return  (cost,grad)



    # Main script for training a network for grasp recognition (which can then
    # also be used for detection.) Assumes data has already been loaded and
    # processed. 
    # 
    # Pre-trains two hidden layers using the sparse autoencoder algorithm.
    # Training for the first layer will include our multimodal regularization
    # function (doesn't apply to the second layer since there's no distinction
    # in modalities after the first layer). 
    #
    # Then, trains a classifier on top of the second hidden layer's features
    # and back-propagates through the network to arrive at a final network
    # trained for recognition. Reports results on both the training and testing
    # sets.
    #
    # Note that minFunc may give you some warnings like:
    # "Matrix is close to singular or badly scaled." 
    # for the first few iterations. This is OK, and will go away quickly.
    #
    # Author: Aashiq Muhamed
 
    def trainGraspRecMultiSparse(self) :
        
        # Switch this to your path to minFunc
        #addpath ../minFunc
        
        # Comment this out if you want to use MATLAB's GPU functionality for
        # backprop
        #addpath nogpu/
        
        # Load processed data
        
        self.trainData = np.load('C:/Users/aashi/Desktop/Python conversion/data/train/trainData.npy')
        self.trainMask = np.load('C:/Users/aashi/Desktop/Python conversion/data/train/trainMask.npy')
        self.trainClasses = np.load('C:/Users/aashi/Desktop/Python conversion/data/train/trainClasses.npy')
        
        self.testData = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testData.npy' )
        self.testMask = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testMask.npy')
        self.testClasses = np.load('C:/Users/aashi/Desktop/Python conversion/data/test/testClasses.npy')
        
        #load ../data/graspTrainData
        #load ../data/graspTestData
        #load ../data/graspModes24
        temp = scipy.io.loadmat('C:/Users/aashi/Desktop/Python conversion/data/graspModes24.mat')
        self.trainModes = temp['trainModes']
        
            
        # Train layer 1 with sparse autoencoder
        self.opttheta = 0
        self.runSAEMultiSparse()
        
        # Have to scale the input features based on mask (runSAEMultiSparse does
        # this for pre-training, but we also need it for computing features and
        # backprop.)
        self.trainMask = scaleMaskByModes2(self.trainMask,self.trainModes);
        self.trainData = self.trainData*self.trainMask;
        self.testMask = scaleMaskByModes2(self.testMask,self.trainModes);
        self.testData = self.testData*self.testMask;
        
        # Scale the first-layer weights to get some desirable statistics in the
        # first-layer features. 
        self.W = scaleAndBiasWeights(self.W[:,:np.size(self.W,1)-1].T,self.trainData,1.5,0.15);
        
        # Compute first-layer features.
        self.trainFeat1 = 1/(1+np.exp(-np.hstack((self.trainData, np.ones((np.size(self.trainData,0),1)))).dot(self.W)));
        
        # Save first-layer weights.
        np.save('C:/Users/aashi/Desktop/Python conversion/data/graspWL1shallow/W', self.W);
        
        # Train second layer similarly based on first-layer features.
        self.opttheta = 0;
        self.runSAEBinary();
        
        self.W = scaleAndBiasWeights(self.W[:,:np.size(self.W,1)-1].T,self.trainFeat1,1.5,0.15);
        np.save('C:/Users/aashi/Desktop/Python conversion/data/graspWL2shallow/W', self.W);
        self.trainFeat1 = 0;
        
        # Run backpropagation
        self.runBackpropMultiReg();
        
        # Save post-backprop weights
        np.save('C:/Users/aashi/Desktop/Python conversion/data/graspWFinalshallow/w1', self.w1);
        np.save('C:/Users/aashi/Desktop/Python conversion/data/graspWFinalshallow/w2', self.w2);
        np.save('C:/Users/aashi/Desktop/Python conversion/data/graspWFinalshallow/2_class', self.w_class);
