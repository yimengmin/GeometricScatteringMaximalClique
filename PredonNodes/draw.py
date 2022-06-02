import torch
import matplotlib.pyplot as plt
import  pickle
import numpy as np
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
edge_index_ll = pickle.load(open("edge_index.pkl",'rb'))
dataset = edge_index_ll
dataset_scale = 1
total_samples = int(np.floor(len(dataset)*dataset_scale))
dataset = dataset[:total_samples]
num_trainpoints = int(np.floor(0.6*len(dataset)))
num_valpoints = int(np.floor(num_trainpoints/3))
num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)
traindata= dataset[0:num_trainpoints]
valdata = dataset[num_trainpoints:num_trainpoints + num_valpoints]
testdata = dataset[num_trainpoints + num_valpoints:]
from torch_geometric.utils.convert import to_networkx
import networkx as nx
for index in range(num_testpoints):
    data = dataset[index+num_trainpoints+num_valpoints]
    data = to_undirected(torch.transpose(torch.tensor(data,dtype=torch.long),0,1))
    data = data.numpy()
    G = nx.Graph()
    for i in range(len(data[0])):
        G.add_edge(data[:,i][0], data[:,i][1])
    g = G.to_undirected()
    pos = nx.spring_layout(g)
    nodes = g.nodes()
    dissct = torch.load("SCT_file%d.pt"%index)
    dissct = dissct.cpu().numpy()
    if g.number_of_nodes() == len(dissct):
        plt.figure(figsize =(8, 4))
        plt.subplot(121)
        ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=dissct[[nodes]],\
                                node_size=20, cmap=plt.cm.jet)
        plt.colorbar(nc)
        plt.axis('off')

        plt.subplot(122)
        dis = torch.load("GCN_file%d.pt"%index)
        dis = dis.cpu().numpy()
        ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=dis[[nodes]],\
                node_size=20, cmap=plt.cm.jet)


        print(index)
        plt.colorbar(nc)
        plt.axis('off')
        plt.savefig('Figs/Visualize_Dis%d.png'%index)
        plt.clf()
    else:
        pass
