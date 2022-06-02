'''
adapted from Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks
'''
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops, dropout_adj
from torch_geometric.utils.convert import to_scipy_sparse_matrix 
def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalizemx(mx):
    degrees = mx.sum(axis=0)[0].tolist()
#    print(degrees)
    D = sp.diags(degrees, [0])
    D = D.power(-1)
    mx = mx.dot(D)
    return mx
def scattering1st(spmx,order):
    I_n = sp.eye(spmx.shape[0])
    adj_sct = 0.5*(spmx+I_n)
    adj_power = adj_sct
    adj_power = sparse_mx_to_torch_sparse_tensor(adj_power).cuda()
    adj_sct = sparse_mx_to_torch_sparse_tensor(adj_sct).cuda()
    I_n = sparse_mx_to_torch_sparse_tensor(I_n)
    if order>1:
        for i in range(order-1):
            adj_power = torch.spmm(adj_power,adj_sct.to_dense())
        adj_int = torch.spmm((adj_power-I_n.cuda()),adj_power)
    else:
        adj_int = torch.spmm((adj_power-I_n.cuda()),adj_power.to_dense())
    return -1.0*adj_int
def tensorscattering1st(sptensor,order):
    I_n = sp.eye(sptensor.size(0))
    I_n = sparse_mx_to_torch_sparse_tensor(I_n).cuda()
    adj_sct = 0.5*(sptensor+I_n)
    adj_power = adj_sct
    adj_power = adj_power
    adj_sct = adj_sct
    if order>1:
        for i in range(order-1):
            adj_power = torch.spmm(adj_power,adj_sct.to_dense())
        adj_int = torch.spmm((adj_power-I_n),adj_power)
    else:
        adj_int = torch.spmm((adj_power-I_n),adj_power.to_dense())
    return -1.0*adj_int

def scattering2nd(m1,m2):
    _m2 = m2
    _m2.data=abs(_m2.data)
    m3 =  torch.spmm(m1,_m2)
    return m3

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def exploss(edge_index,output_dis,penalty_coefficient=0.005):
    adjmatrix = to_scipy_sparse_matrix(edge_index)
    adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix).cuda() 
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
