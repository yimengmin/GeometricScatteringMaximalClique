import torch
import networkx as nx
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.convert import to_networkx
import json
import numpy as np
def G2edgeindex(data):
    Glist = []
    '''
    data: a Graph in networkx from
    '''
    G = data
    G = list(G.edges)
    for i in range(data.number_of_edges()):
        Glist.append([G[i][0],G[i][1]])
#        Glist.append([G[i][1],G[i][0]]) #repeat this because pyG requires [[0,1],[1,0]] to represent an undirected edge
         # use TORCH_GEOMETRIC.UTILS.UNDIRECTED and we don't need repeat 
    return Glist
