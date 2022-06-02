import pickle
import numpy as np
import networkx as nx
edge_indexll = ()
from A2Edge_idx import G2edgeindex
scatp_dict = pickle.load(open('psdadj_p.pkl','rb'))
adj_dict = pickle.load(open('psdA_tilde.pkl','rb'))
for i in range(len(adj_dict)):
    G = nx.from_scipy_sparse_matrix(adj_dict[i], create_using=nx.MultiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    edge_index =  G2edgeindex(G)# use G2edgeindex
    edge_indexll += (edge_index,)
    print('Running at Sample %3d'%i)


pickle.dump(edge_indexll, open('edge_index.pkl','wb'))




