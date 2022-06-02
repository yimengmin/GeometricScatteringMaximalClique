from itertools import product
import time
from math import ceil
import networkx as nx
import myfuncs
from cut_utils import get_diracs
import scipy
import scipy.io
from networkx.algorithms.approximation import max_clique
import pickle
from random import shuffle
import numpy as np
#import matplotlib.pyplot as plt
from  cut_utils import solve_gurobi_maxclique
import gurobipy as gp
from gurobipy import GRB


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tlimit', type=float, default=500,
                    help='time limit')
args = parser.parse_args()
print(args.tlimit)
dataset_name = "COLLAB"

if dataset_name == "TWITTER_SNAP":
    stored_dataset = open('datasets/TWITTER_SNAP.p', 'rb')
    dataset = pickle.load(stored_dataset)
elif dataset_name == "COLLAB":
    dataset = TUDataset('datasets/'+dataset_name, name=dataset_name)
    #stored_dataset = open('datasets/dataset_shuffle_1.p', 'rb')
elif dataset_name == "IMDB-BINARY":
    dataset = TUDataset('datasets/'+dataset_name, name=dataset_name)
print('Len of dataset')
print(len(dataset))
listclique = []
import pickle
data_clique = []
time_clique = []
index = 0
indexlist = []
for data in dataset:
#for i in range(4000,5000):
    data = dataset[i]
    my_graph = to_networkx(data)
    print('num of nodes: %d'%my_graph.number_of_nodes())
    my_graph.remove_edges_from(nx.selfloop_edges(my_graph))
    t_start = time.time()
    cliqno, _ = solve_gurobi_maxclique(my_graph,time_limit=args.tlimit)
    t_1 = time.time()
    data_clique += [cliqno]
    total_time = t_1 - t_start
    time_clique += [total_time]
#    if (cliqno>=5)and(cliqno<=25):
#        indexlist += [index]
#        print('Index: %d,clique: %d'%(index,cliqno))
    indexlist += [index]
    print('Index: %d,clique: %d,time: %.6f'%(index,cliqno,total_time))
    index+=1
print('Len')
print(len(data_clique))

with open("Gtruthclique/"+dataset_name+"cliqnolimit%.3f.txt"%(args.tlimit), "wb") as fp:   #Pickling
    pickle.dump(data_clique, fp)
with open("Gtruthclique/"+dataset_name+"timecliqnolimit%.3f.txt"%(args.tlimit), "wb") as fp:   #Pickling
    pickle.dump(time_clique, fp)
with open("Gtruthclique/"+dataset_name+"indexcliqnolimit%.3f.txt"%(args.tlimit), "wb") as fp:   #Pickling
    pickle.dump(indexlist, fp)
