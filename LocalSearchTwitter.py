import pickle
import random
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj,to_undirected

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--restart', type=int, default=20, help='Restart')
parser.add_argument('--max_selections', type=int, default=20, help='max selection')
parser.add_argument('--sample_idx', type=int, default=0,
                    help='sample_idx')
args = parser.parse_args()
np.random.seed(args.seed)



dataset_name = "TWITTER_SNAP"
dataset_scale = 1
LENGDATA = 973
total_samples = int(np.floor(LENGDATA*dataset_scale))
num_trainpoints = int(np.floor(0.6*total_samples))
num_valpoints = int(np.floor(num_trainpoints/3))
num_testpoints = total_samples - (num_trainpoints + num_valpoints)
#print(num_testpoints) #196
with open("Gtruthclique/"+dataset_name+"cliqno.txt","rb") as fp:
    loadedclique  = pickle.load(fp)
loadedclique =  loadedclique[:total_samples]
testloadedclique = loadedclique[num_trainpoints + num_valpoints:]
valloadedclique = loadedclique[num_trainpoints:num_trainpoints + num_valpoints]
trainloadedclique = loadedclique[0:num_trainpoints]

#https://link.springer.com/content/pdf/10.1007/s10732-007-9055-x.pdf
edge_index_ll = pickle.load(open("Preposs/"+dataset_name+"/edge_index.pkl",'rb'))
sample_idx = args.sample_idx 
graph_index = sample_idx + num_trainpoints + num_valpoints
edge_index=to_undirected(torch.transpose(torch.tensor(edge_index_ll[graph_index],dtype=torch.long),0,1))
edge_indexed = edge_index 
adjmatrix = to_dense_adj(edge_index = edge_indexed)[0] #torch tensor
adjmatrix = adjmatrix.numpy()











def CpClique(p,adjmatrix,clique_set = [0,1]):
    #expensive: need to calculate all nodes not adjacent to exactly p nodes in K.
    # p is an interger
    neighbornodes = []
    clique_set = set(clique_set)
    for items in range(adjmatrix.shape[0]):
        nonzreos = np.where(adjmatrix[items]==1)[0].tolist()
        nonzreos = set(nonzreos)
        set_minus =   clique_set - nonzreos #k \ N(i)  
        if len(set_minus) == p:
            neighbornodes.append(items)
    return set(neighbornodes)

#print(CpClique(p=1,adjmatrix = adjmatrix))



max_selections = args.max_selections
Kbest = set()
K = set()
selections = 0
start_node = np.random.choice(adjmatrix.shape[0])
#print({start_node})
start_node_neigbs = set(np.where(adjmatrix[start_node]==1)[0].tolist())
#print(start_node_neigbs)
K = (K.intersection(start_node_neigbs)).union({start_node})
U = set()

def run_selection():
    global K
    global Kprime
    global U
    global Kbest
    global selections
    if not (CpClique(p=0,adjmatrix = adjmatrix,clique_set=K) - U ) ==set():
        select_i = random.choice(tuple( CpClique(p=0,adjmatrix = adjmatrix,clique_set=K) - U ))
        K = K.union({select_i})
        if (U==set()):
            Kprime = K
        else:
            pass
#        print(select_i)
    elif not ((CpClique(p=1,adjmatrix = adjmatrix,clique_set=K) - U ) ==set()):
        select_i = random.choice(tuple( CpClique(p=1,adjmatrix = adjmatrix,clique_set=K) - U ))
        select_j = K - set(np.where(adjmatrix[select_i]==1)[0].tolist())
#        print('select_j')
#        print(select_j)
        K = K.union({select_i}) - select_j 
        U = U.union(select_j)
    elif not ( CpClique(p=1,adjmatrix = adjmatrix,clique_set=K).intersection(U) == set()):
        select_i = CpClique(p=1,adjmatrix = adjmatrix,clique_set=K).intersection(U)
        K = K.union({select_i})
    else:
        pass
    selections = selections + 1
#    print('Iteration: %d'%selections)


#################################
####Start calculate the time#####
#################################
import time

def run_local_search():
    global Kbest
    global K
    starter_time = time.time()
    run_selection()
    #note there is no do while loop in python so we run the do-loop at least one
    while ( (CpClique(p=0,adjmatrix = adjmatrix,clique_set=K) != set()) or \
            ((CpClique(p=1,adjmatrix = adjmatrix,clique_set=K) - U) != set()) ) and\
            (Kprime.intersection(K) != set()) and\
            (selections<max_selections):
        run_selection()
        if len(Kbest)<len(K):
            Kbest = K
    t1 = time.time() - starter_time
    return len(Kbest),t1


maxclique = -1
t_list = []
for restart_time in range(args.restart):


    max_selections = args.max_selections
    Kbest = set()
    K = set()
    selections = 0
    start_node = np.random.choice(adjmatrix.shape[0])
    #print({start_node})
    start_node_neigbs = set(np.where(adjmatrix[start_node]==1)[0].tolist())
    #print(start_node_neigbs)
    K = (K.intersection(start_node_neigbs)).union({start_node})
    U = set()




    _clique, _t = run_local_search()
    if maxclique<_clique:
        maxclique = _clique
    t_list.append(_t)


#print('The truth is')
#print(testloadedclique[sample_idx])
print('Sample id: %d, Takes: %.4f seconds, the ratio is: %.3f'%(args.sample_idx,np.sum(t_list),maxclique/testloadedclique[sample_idx]))


with open('accu_restart_%d_selection_%d.txt'%(args.restart,args.max_selections), 'a') as file_object:
    file_object.write('%.4f'%(maxclique/testloadedclique[sample_idx]))
    file_object.write('\n')

with open('time_restart_%d_selection_%d.txt'%(args.restart,args.max_selections), 'a') as file_object:
    file_object.write('%.4f'%(np.sum(t_list)))
    file_object.write('\n')

