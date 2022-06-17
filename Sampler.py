from torch_geometric.utils.convert import to_scipy_sparse_matrix
from utils import sparse_mx_to_torch_sparse_tensor
import networkx as nx
import scipy.sparse as sp
import torch
import  pickle
import numpy as np
def getclicnum(adjmatrix,dis,walkerstart = 0,thresholdloopnodes = 50):
    '''
    ComplementedgeM: complement matrix of adj 
    dis: distribution on the nodes, higher ->better
    cpu: cpu is usually better for small model
    '''
    _sorted, indices = torch.sort(dis.squeeze(),descending=True)#flatten, elements are sorted in descending order by value.
    initiaprd = 0.*indices  # torch zeros
    initiaprd = initiaprd.cpu().numpy() 
    for walker in range(min(thresholdloopnodes,adjmatrix.get_shape()[0])):
        if walker < walkerstart:
            initiaprd[indices[walker]] = 0.
        else:
            pass
    initiaprd[indices[walkerstart]] = 1. # the one with walkerstart'th largest prob is in the clique, start with walkerstart
    for clq in range(walkerstart+1,min(thresholdloopnodes,adjmatrix.get_shape()[0])): # loop the 50 high prob nodes
        initiaprd[indices[clq]] = 1.
        binary_vec = np.reshape(initiaprd, (-1,1)) 
        ZorO = np.sum(binary_vec)**2  - np.sum(binary_vec) - np.sum(binary_vec*(adjmatrix.dot(binary_vec)))
        if ZorO < 0.0001: # same as ZorO == 0
            pass
        else:
            initiaprd[indices[clq]] = 0.
    return np.sum(initiaprd)
