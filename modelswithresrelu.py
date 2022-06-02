##############################
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import tensor
import torch.nn
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from torch.nn import Parameter
###########
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from layers import GC_withres,GC
import torch.nn as nn
class SCTConv(torch.nn.Module):
    def __init__(self, hidden_dim, smooth, dropout,Withgres=False):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2*hidden_dim, 1)))
        self.smoothlayer = Withgres #turn on graph residual layer or not
        self.gres = GC_withres(hidden_dim,hidden_dim,smooth = smooth)
        self.dropout = dropout
    def forward(self,X,A_nor,P_sct,P_sct1,P_sct2,P_sct3,moment = 1):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        support0 = X
        N = support0.size()[0]
        h = support0
        h_A =  torch.spmm(A_nor, support0)
        h_A2 =  torch.spmm(A_nor, torch.spmm(A_nor, support0)) 
        h_A3 =  torch.spmm(A_nor, torch.spmm(A_nor, torch.spmm(A_nor, support0))) 
#        h_sct1 = torch.nn.functional.relu(torch.spmm(P_sct1, support0))
        h_sct1 = torch.abs(torch.spmm(P_sct1, support0))**moment
#        h_sct2 = torch.nn.functional.relu(torch.spmm(P_sct2, support0))
        h_sct2 = torch.abs(torch.spmm(P_sct2, support0))**moment
#        h_sct3 = torch.nn.functional.relu(torch.spmm(P_sct3, support0))
        h_sct3 = torch.abs(torch.spmm(P_sct3, support0))**moment
        a_input_A = torch.cat([h,h_A]).view(N, -1, 2 * self.hid)
        a_input_A2 = torch.cat([h,h_A2]).view(N, -1, 2 * self.hid)
        a_input_A3 = torch.cat([h,h_A3]).view(N, -1, 2 * self.hid)
        a_input_sct1 = torch.cat([h,h_sct1]).view(N, -1, 2 * self.hid)
        a_input_sct2 = torch.cat([h,h_sct2]).view(N, -1, 2 * self.hid)
        a_input_sct3 = torch.cat([h,h_sct3]).view(N, -1, 2 * self.hid)
        a_input =  torch.cat((a_input_A,a_input_A2,a_input_A3,a_input_sct1,a_input_sct2,a_input_sct3),1).view(N,6,-1)
        #GATV2
        e = torch.matmul(torch.nn.functional.relu(a_input),self.a).squeeze(2)
        attention = F.softmax(e, dim=1).view(N, 6, -1)
        h_all = torch.cat((h_A.unsqueeze(dim=2),h_A2.unsqueeze(dim=2),h_A3.unsqueeze(dim=2),\
                h_sct1.unsqueeze(dim=2),h_sct2.unsqueeze(dim=2),h_sct3.unsqueeze(dim=2)),dim=2).view(N,6, -1)
        h_prime = torch.mul(attention, h_all) # element eise product
        h_prime = torch.mean(h_prime,1)
        #grpah residual layer
        if self.smoothlayer:
            h_prime = self.gres(h_prime,P_sct)
        else:
            pass
        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,dropout,Withgres=False,smooth=0.):
        super().__init__()
        self.dropout = dropout
        self.smooth = smooth
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(hidden_dim,self.smooth,self.dropout,Withgres))
        self.mlp1 = Linear(hidden_dim*(1+n_layers), hidden_dim)
        self.mlp2 = Linear(hidden_dim,output_dim)
    def forward(self,X,A,P_sct,s1_sct,s2_sct,s3_sct,moment = 1):
        numnodes = X.size(0)
        scale = np.sqrt(numnodes) # for graph norm
        X = X/scale
        X = self.in_proj(X)
        hidden_states = X
        for layer in self.convs:
            X = layer(X,A,P_sct,s1_sct,s2_sct,s3_sct,moment = moment)
            # normalize
            X = X/scale
            hidden_states = torch.cat([hidden_states, X], dim=1)
        X = hidden_states 
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X) 
        maxval = torch.max(X)
        minval = torch.min(X)
        X = (X-minval)/(maxval+1e-6-minval)
        return X

class GCN(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,dropout = 0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GC(input_dim, hidden_dim)
        self.gc2 = GC(hidden_dim,output_dim)
    def forward(self,X,A):
        X = F.leaky_relu(self.gc1(X, A))
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.gc2(X, A)
        maxval = torch.max(X)
        minval = torch.min(X)
        X = (X-minval)/(maxval+1e-6-minval)
        return X
