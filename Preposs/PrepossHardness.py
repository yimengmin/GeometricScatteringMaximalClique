import torch
import networkx as nx
import pickle
from torch_geometric.datasets import TUDataset
from utils import *
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.convert import to_networkx
import json
import numpy as np
datasets = ["Easy","Medium","Hard"]
dataset_name = datasets[2]

stored_dataset = open('../Datasets/Cliques_%s.p'%dataset_name,'rb')
dataset = pickle.load(stored_dataset)
processedfeature = () # create tuples
processedlabel = ()
processedA_tilde = ()
processedadj_p = ()
edge_indexll = ()
from A2Edge_idx import G2edgeindex


print('Length')
print(len(dataset))
for i in range(len(dataset)):
    print("======%d======"%i)
    data = dataset[i]
    #G = to_networkx(data)
    G = data.to_undirected()
    n_nodes = len(G)
    feature_vector = []
    contg = nx.is_connected(G)
    if contg:
        '''
        Connected Graph
        '''
        for j in range(n_nodes):
            try:
                _eccen = nx.eccentricity(G,j)
                _deg = nx.degree(G,j)
                _deg = np.log(_deg)
#                _deg = 0.1*_deg
#                _deg = np.sqrt(_deg)
                _cluter = nx.clustering(G,j)
                feature_vector.append([_eccen,_deg,_cluter])
            except:
                print("Sth is Wrong")
                feature_vector.append([0,0,0])
    else:
        '''
        Unconnected Graph
        '''
#        for j in range(n_nodes):
#            H = G.subgraph([j])  
#            try:
#                _eccen = nx.eccentricity(H,j)
#                _deg = nx.degree(H,j)
#                _cluter = nx.clustering(H,j)
#                feature_vector.append([_eccen,_deg,_cluter])
#            except:
#                print("Sth is Wrong")
#                feature_vector.append([0,0,0])
        graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for gsub in graphs:
            for j in gsub.nodes:
                try:
                    _eccen = nx.eccentricity(gsub,j)
                    _deg = nx.degree(gsub,j)
                    _deg = np.log(_deg)
#                    _deg = 0.1*_deg
#                    _deg = np.sqrt(_deg)
                    #_deg = np.log(nx.degree(gsub,j))
                    _cluter = nx.clustering(gsub,j)
                    feature_vector.append([_eccen,_deg,_cluter])
                except:
                    print("Sth is Wrong")
                    feature_vector.append([0,0,0])
    feature_vector = np.array(feature_vector)
    edge_index =  G2edgeindex(data)# use G2edgeindex
    adj = nx.adjacency_matrix(data) 
    adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    A_tilde = normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))
    adj_p = normalizemx(adj)
    data.x =   feature_vector
    processedfeature += (data.x.tolist(),)
    processedA_tilde += (A_tilde,)
    processedadj_p += (adj_p,)
    edge_indexll += (edge_index,)
import os
name = dataset_name
if not os.path.exists(name):
    os.makedirs(name)
with open(name+'/psdfeature.json', 'w') as fjson:
    json.dump(processedfeature, fjson)
import pickle
pickle.dump(processedA_tilde, open(name+'/psdA_tilde.pkl','wb'))
pickle.dump(processedadj_p, open(name+'/psdadj_p.pkl','wb'))
pickle.dump(edge_indexll, open(name+'/edge_index.pkl','wb'))
