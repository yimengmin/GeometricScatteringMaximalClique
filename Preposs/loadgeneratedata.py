import pickle
#load graph data
with open("../datasets/mygraph.p", 'rb') as f:
    dataload = pickle.load(f)
#print(len(dataload))
#load results
import numpy as np
with open("../Gtruthclique/mygraphcliqno.txt", "rb") as fp:   # Unpickling
    MCresults = pickle.load(fp)
#print(len(MCresults))
#load erdos data
with open("../datasets/cliques_test_set_solved.p", "rb") as fp:   # Unpickling
    ERdosrsults = pickle.load(fp)
#print(ERdosrsults)
#for i in range(len(ERdosrsults)):
#    print(ERdosrsults[i].clique_number)
with open("../RUN-CSP/TrainingRBRUNCSPsolved.p", "rb") as fp:   # Unpickling
    TrainRBsrsults = pickle.load(fp)
#for i in range(len(TrainRBsrsults)):
for i in range(500):
    print(TrainRBsrsults[i])
