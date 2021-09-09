from tree import decisionTree
from createDataset import *
from tqdm import tqdm
from Declare import checkCostraints
from Temporal import time, timeConstraint
import sys


# python3 training_dataset.py data/BPIChallenge2012A.xes data/BPISimulated.xes /Users/frameneghello/Desktop/TEST_DA_ESEGUIRE/BPI1/constraint.csv

print("Input Argument", str(sys.argv))



data= createDataset(sys.argv[1], sys.argv[2])
real, sim= data.dataset("/Users/frameneghello/Desktop/codiceTesi/github/real.csv", "/Users/frameneghello/Desktop/codiceTesi/github/sim.csv")


conR=checkCostraints(sys.argv[3], real, data.getReal())
data1=conR.insertCostraints()
#con.write_csv("/Users/frameneghello/Desktop/codiceTesi/github/real.csv")

conS=checkCostraints(sys.argv[3],sim, data.getSim())
data2=conS.insertCostraints()
#con.write_csv("/Users/frameneghello/Desktop/codiceTesi/github/sim.csv")


timeR=time(data.getReal(), data1, 'R') #### cambiare
data1=timeR.timeFeatures()
print(data1)

timeS=time(data.getSim(), data2, 'S') #### cambiare
data2=timeS.timeFeatures()
print(data2)

#path_costraints, df, log, type
timeConstraintReal=timeConstraint(sys.argv[3], data1, sys.argv[1], 'R')
data1=timeConstraintReal.addTimeConstraint()
print(data1)
data1.to_csv("real.csv", index=False)

timeConstraintSim=timeConstraint(sys.argv[3], data2, sys.argv[2], 'S')
data2=timeConstraintSim.addTimeConstraint()
print(data2)
data2.to_csv("sim.csv", index=False)