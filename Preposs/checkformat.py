import pickle
dataset_name = 'Medium'
edge_index_ll = pickle.load(open(dataset_name+"/edge_index.pkl",'rb'))
adj = pickle.load(open(dataset_name+"/psdA_tilde.pkl",'rb'))
for i in range(1000):
    if adj[i].shape[0] == max(max(edge_index_ll[i]))+1:
        pass
    else:
        print(i)
        print('Wrong--')
        print(adj[i].shape[0])
        print(max(max(edge_index_ll[i]))+1)
