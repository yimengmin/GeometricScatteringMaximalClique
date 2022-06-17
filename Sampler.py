from torch_geometric.utils.convert import to_scipy_sparse_matrix
from utils import sparse_mx_to_torch_sparse_tensor
import networkx as nx
import scipy.sparse as sp
import torch
import  pickle
import numpy as np

def getclicnum(ComplementedgeM,dis,walkerstart = 0,thresholdloopnodes = 50):
    '''
    ComplementedgeM: complement matrix of adj 
    dis: distribution on the nodes, higher ->better
    cpu: cpu is usually better for small model
    '''
    _sorted, indices = torch.sort(dis.squeeze(),descending=True)#flatten, elements are sorted in descending order by value.
    initiaprd = 0.*indices  # torch zeros
    for walker in range(min(thresholdloopnodes,ComplementedgeM.size(0))):
        if walker < walkerstart:
            initiaprd[indices[walker]] = 0.
        else:
            pass
    initiaprd[indices[walkerstart]] = 1. # the one with walkerstart'th largest prob is in the clique, start with walkerstart
    initiaprd = initiaprd.cpu()
    for clq in range(walkerstart+1,min(thresholdloopnodes,ComplementedgeM.size(0))): # loop the 50 high prob nodes
        initiaprd[indices[clq]] = 1.
        binary_vec = initiaprd.unsqueeze(1)
        ZorO = torch.sum(initiaprd.unsqueeze(1)*torch.mm(ComplementedgeM,initiaprd.unsqueeze(1)))
        if ZorO < 0.0001: # same as ZorO == 0
            pass
        else:
            initiaprd[indices[clq]] = 0.
    return torch.sum(initiaprd)
