'''
adapted from Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks
'''
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from time import perf_counter
from torch_geometric.utils.convert import to_scipy_sparse_matrix 
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def exploss(edge_index,output_dis,penalty_coefficient=0.005,device = 'cuda'):
    adjmatrix = to_scipy_sparse_matrix(edge_index)
    adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix).to(device) 
    I_n = sp.eye(adjmatrix.size(0))
    I_n = sparse_mx_to_torch_sparse_tensor(I_n).cuda()
    Fullm = torch.ones(I_n.size(0),I_n.size(1)).cuda() - I_n #(N,N) 
    diffusionprob = torch.mm(Fullm - adjmatrix,output_dis)
    elewiseloss = output_dis * diffusionprob
    lossComplE = penalty_coefficient * torch.sum(elewiseloss) # loss on compl of Edges
    lossE = torch.sum(output_dis*torch.mm(adjmatrix,output_dis))
    loss = -lossE + lossComplE
    retdict = {}
    retdict["loss"] = [loss,lossComplE] #final loss
    return retdict
