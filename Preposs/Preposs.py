import torch
import networkx as nx
import pickle
from torch_geometric.datasets import TUDataset
from utils import *
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.convert import to_networkx
import json
import numpy as np
datasets = ["TWITTER_SNAP", "COLLAB", "IMDB-BINARY"]
dataset_name = datasets[1]
if dataset_name == "TWITTER_SNAP":
    stored_dataset = open('../datasets/TWITTER_SNAP.p', 'rb')
    dataset = pickle.load(stored_dataset)
elif dataset_name == "COLLAB":
    dataset = TUDataset(root='/tmp/'+dataset_name, name=dataset_name)
elif dataset_name == "IMDB-BINARY":
    dataset = TUDataset(root='/tmp/'+dataset_name, name=dataset_name)
#init
processedfeature = () # create tuples
processedlabel = ()
processedA_tilde = ()
processedadj_p = ()
print('Length')
print(len(dataset))
for i in range(len(dataset)):
    print("======%d======"%i)
    data = dataset[i]
    G = to_networkx(data)
    G = G.to_undirected()
    n_nodes = len(G)
    feature_vector = []
    contg = nx.is_connected(G)
    print(contg)
    if contg:
        '''
        Connected Graph
        '''
        for j in range(n_nodes):
            try:
                _eccen = nx.eccentricity(G,j)
                _deg = nx.degree(G,j)
                _deg = np.log(_deg)
                _cluter = nx.clustering(G,j)
                feature_vector.append([_eccen,_deg,_cluter])
            except:
                print("Sth is Wrong")
                feature_vector.append([0,0,0])
    else:
        '''
        Unconnected Graph
        '''
        graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for gsub in graphs:
            for j in gsub.nodes:
                try:
                    _eccen = nx.eccentricity(gsub,j)
                    _deg = nx.degree(gsub,j)
                    _deg = np.log(_deg)
                    _cluter = nx.clustering(gsub,j)
                    feature_vector.append([_eccen,_deg,_cluter])
                except:
                    print("Sth is Wrong")
                    feature_vector.append([0,0,0])
    feature_vector = np.array(feature_vector)
    print(feature_vector)
    adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
    adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print('dataset_index :%d, with size %d'%(i,adj.shape[0]))
    A_tilde = normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))
    adj_p = normalizemx(adj)
    data.x =   feature_vector
    processedfeature += (data.x.tolist(),)
    processedA_tilde += (A_tilde,)
    processedadj_p += (adj_p,)
import os
name = dataset_name
if not os.path.exists(name):
    os.makedirs(name)
with open(name+'/psdfeature.json', 'w') as fjson:
    json.dump(processedfeature, fjson)
import pickle
pickle.dump(processedA_tilde, open(name+'/psdA_tilde.pkl','wb'))
pickle.dump(processedadj_p, open(name+'/psdadj_p.pkl','wb'))
