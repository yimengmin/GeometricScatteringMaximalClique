import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
from matplotlib import pyplot as plt
from torch_geometric.data import Data
import networkx as nx
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.data import Batch 
import pickle
from torch.utils.data import  DataLoader# use pytorch dataloader
from random import shuffle
from torch_geometric.datasets import TUDataset
#import visdom 
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dropout', type=float, default=0.0,help='probability of an element to be zeroed:')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='smoo')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=2,
                    help='num of layers')
parser.add_argument('--EPOCHS', type=int, default=10,
                    help='epochs to train')
parser.add_argument('--penalty_coefficient', type=float, default=.1,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--Numofwalkers', type=int, default=3,
                    help='number of walkers in decoder')
parser.add_argument('--SampLength', type=int, default=300,
                    help='Sampling Length in decoder')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
dataset_name = "IMDB-BINARY"
dataset = TUDataset(root='/tmp/'+dataset_name, name=dataset_name)
dataset_scale = 1
total_samples = int(np.floor(len(dataset)*dataset_scale))
dataset = dataset[:total_samples]
#load scattering matrix and feature
import json
with open("Preposs/"+dataset_name+"/psdfeature.json") as jfile:
    psdfeature = json.load(jfile)
psdadj_p = pickle.load(open("Preposs/"+dataset_name+"/psdadj_p.pkl",'rb'))
psdA_tilte = pickle.load(open("Preposs/"+dataset_name+"/psdA_tilde.pkl",'rb'))
#finish loading 
print("Loading Preposs data")
#construct my dataset
from torch.utils.data import Dataset
#preposs dataset
from utils import sparse_mx_to_torch_sparse_tensor,tensorscattering1st,exploss
sctdataset = []
for index in range(total_samples):
    Pmat = psdadj_p[index]
    adjp = sparse_mx_to_torch_sparse_tensor(Pmat).cuda()
    adj_sct1 = tensorscattering1st(adjp,1)
    adj_sct2 = tensorscattering1st(adjp,2)
    adj_sct4 = tensorscattering1st(adjp,4)
    data = Data(x=psdfeature[index],edge_index=dataset[index].edge_index,
            Pmat = psdadj_p[index],Amat = psdA_tilte[index],\
                    adj_sct1=adj_sct1,adj_sct2=adj_sct2,adj_sct4=adj_sct4)
    sctdataset += [data]
def my_collate(batch):
    data = [item for item in batch]
    return data
print('Creating my own dataset')
num_trainpoints = int(np.floor(0.6*total_samples))
num_valpoints = int(np.floor(num_trainpoints/3))
num_testpoints = total_samples - (num_trainpoints + num_valpoints)
traindata= sctdataset[0:num_trainpoints]
valdata = sctdataset[num_trainpoints:num_trainpoints + num_valpoints]
testdata = sctdataset[num_trainpoints + num_valpoints:]
batch_size = args.batch_size
train_loader = DataLoader(traindata, batch_size, shuffle=True,collate_fn=my_collate)
test_loader = DataLoader(testdata, batch_size, shuffle=False,collate_fn=my_collate)
val_loader =  DataLoader(valdata, batch_size, shuffle=False,collate_fn=my_collate)
#set up random seeds
torch.manual_seed(1)
np.random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from modelswithresrelu import GNN
model = GNN(input_dim=3, hidden_dim=args.hidden, output_dim=1, n_layers=args.nlayers,dropout=args.dropout,Withgres=False,smooth=args.smoo)
model.cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.wdecay)
def train(epoch):
    model.train()
    print('Epoch:-----%d'%epoch)
    for i, batch in enumerate(train_loader):
        batchloss = 0.0
        for j in range(len(batch)): # len(batch) len of the batch
            features = torch.FloatTensor(batch[j].x).cuda()
            A_tilte = sparse_mx_to_torch_sparse_tensor(batch[j].Amat).cuda()
            P_sct = sparse_mx_to_torch_sparse_tensor(batch[j].Pmat).cuda()
            adj_sct1 = batch[j].adj_sct1
            adj_sct2 = batch[j].adj_sct2
            adj_sct4 = batch[j].adj_sct4
            output = model(features,A_tilte,P_sct,adj_sct1,adj_sct2,adj_sct4,moment = args.moment)
            retdict = exploss(batch[j].edge_index,output,penalty_coefficient=args.penalty_coefficient)
            batchloss += retdict["loss"][0]
        batchloss = batchloss/len(batch)
        print("Length of batch:%d"%len(batch))
        print('Loss: %.5f'%batchloss)
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
from Sampler import getclicnum
import time
def test(loader):
    index = 0
    clilist = []
    timelist = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            for j in range(len(batch)): # len(batch[0]) len of the batch
                t_0 = time.time()
                features = torch.FloatTensor(batch[j].x).cuda()
                P_sct = sparse_mx_to_torch_sparse_tensor(batch[j].Pmat).cuda()
                A_tilte = sparse_mx_to_torch_sparse_tensor(batch[j].Amat).cuda()
                adj_sct1 = batch[j].adj_sct1
                adj_sct2 = batch[j].adj_sct2
                adj_sct4 = batch[j].adj_sct4
                edge_index = batch[j].edge_index
                adjmatrix = to_scipy_sparse_matrix(edge_index)
                adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix)
                adjmatrix = adjmatrix.cpu()
                I_n = torch.eye(adjmatrix.size(0)).cpu()
                Fullm = torch.ones(I_n.size(0),I_n.size(1)).cpu() - I_n #(N,N)
                ComplementedgeM = (Fullm - adjmatrix)*1.
                output = model(features,A_tilte,P_sct,adj_sct1,adj_sct2,adj_sct4,moment = args.moment)
                predC = []
# my decoder
                for walkerS in range(0,min(args.Numofwalkers,adjmatrix.size(0))): # with Numofwalkers walkers
                    predC += [getclicnum(ComplementedgeM,output,walkerstart=walkerS,thresholdloopnodes=args.SampLength).item()]
                cliques = max(predC)
                index += 1
                clilist += [cliques]
                t_pred = time.time() - t_0 #calculate  time
                timelist += [t_pred]
    return clilist,timelist
import pickle
with open("Gtruthclique/"+dataset_name+"cliqno.txt","rb") as fp:
    loadedclique  = pickle.load(fp)
loadedclique =  loadedclique[:total_samples]
testloadedclique = loadedclique[num_trainpoints + num_valpoints:]
valloadedclique = loadedclique[num_trainpoints:num_trainpoints + num_valpoints] 
trainloadedclique = loadedclique[0:num_trainpoints]
for i in range(args.EPOCHS):
    train(i)
    if i%5 == 4:
        clilist,timelist = test(val_loader)
        ratio = []
        for testidx in range(len(valloadedclique)):
            ratio += [clilist[testidx]/valloadedclique[testidx]]
            print('Graph  %d: Pred: %d, Truth %d'%(testidx,clilist[testidx],valloadedclique[testidx]))
        print("Vali Accu:%.6f"%np.mean(np.array(ratio)))
        print("Val Var:%.6f"%np.var(np.array(ratio)))
        print("Average Processing Time on Val data:%.6f"%np.mean(np.array(timelist)))
clilist,timelist = test(test_loader)
ratio = []
for i in range(len(testloadedclique)):
    print('Test Graph %d, Pred: %d Gtruth: %d'%(i,clilist[i],testloadedclique[i]))
for testidx in range(len(testloadedclique)):
    ratio += [clilist[testidx]/testloadedclique[testidx]]
print('Test Results: %.5f'%np.mean(np.array(ratio)))
print("Test Var:%.6f"%np.var(np.array(ratio)))
print("Average Processing Time on test data:%.6f"%np.mean(np.array(timelist)))
print('args: dropout: %.4f'%args.dropout)
print('args: penalty_coefficient: %.4f'%args.penalty_coefficient)
print('Bsize: %.3d'%args.batch_size)
print('hidden: %.3d'%args.hidden)
print('nlayers: %.3d'%args.nlayers)
print('moment: %.3d'%args.moment)
print('smoo: %.3f'%args.smoo) #no resial layer
#save
np.savetxt('clilist.csv',np.array(clilist), delimiter=',', fmt='%s')
np.savetxt('testloadedclique.csv',np.array(testloadedclique), delimiter=',', fmt='%s')
