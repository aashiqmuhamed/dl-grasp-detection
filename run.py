# Set this to the path to your folder containing the Cornell Grasping
# Dataset (this folder should contain a lot of pcd_* files)
from util import *
from recTraining import *
import numpy as np
from processData import *
from loadData import *
import DeepGrasp as DG
import pdb; 
#dataDir = 'E:/Data for grasping/data/sample1';
dataDir = 'E:/Data for grasping/data/all';
#'~/data/rawDataSet'
# Load the grasping dataset

[depthFeat, colorFeat, normFeat, classes, inst, accepted, depthMask, colorMask]  = loadAllGraspingDataImYUVNormals(dataDir);
classes = np.asarray(classes,dtype = bool);


# Process data, splitting into train/test sets and whitening
Obj = DG.DeepGrasp(depthFeat, colorFeat, normFeat, classes, inst, accepted, depthMask, colorMask)
Obj.processGraspData()


Obj.trainGraspRecMultiSparse()
#print(Obj.RecordTrain)
#print(Obj.RecordTest)
#print(Obj.bestdata)
print(Obj.w1)

# Workspace will be pretty messy here, but I don't like putting a clear all
# in this script, since you might have your own stuff there that you don't
# want to lose.