from torch_geometric.utils.convert import to_scipy_sparse_matrix
from utils import sparse_mx_to_torch_sparse_tensor
import networkx as nx
import scipy.sparse as sp
import torch
import  pickle
import numpy as np

def getclicnum(adjmatrix,dis,walkerstart = 0,thresholdloopnodes = 50):
    '''
    adjmatrix: adj matrix of the graph
    dis: distribution on the nodes, higher ->better
    '''
    I_n = torch.eye(adjmatrix.size(0)).cuda()
    Fullm = torch.ones(I_n.size(0),I_n.size(1)).cuda() - I_n #(N,N)
    ComplementedgeM = (Fullm - adjmatrix)*1.
    _sorted, indices = torch.sort(dis.squeeze(),descending=True)#flatten, elements are sorted in descending order by value.
    initiaprd = 0.*indices  # torch zeros
    for walker in range(min(thresholdloopnodes,I_n.size(0))):
        if walker < walkerstart:
            initiaprd[indices[walker]] = 0.
        else:
            pass
    initiaprd[indices[walkerstart]] = 1. # the one with walkerstart'th largest prob is in the clique, start with walkerstart
    for clq in range(walkerstart+1,min(thresholdloopnodes,I_n.size(0))): # loop the 50 high prob nodes
        initiaprd[indices[clq]] = 1.
        ZorO = torch.sum(initiaprd.unsqueeze(1)*torch.mm(ComplementedgeM,initiaprd.unsqueeze(1)))
        if ZorO < 0.0001: # same as ZorO == 0
            pass
        else:
            initiaprd[indices[clq]] = 0.
    return torch.sum(initiaprd)


