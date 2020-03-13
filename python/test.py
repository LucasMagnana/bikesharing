import pickle
import data
import torch
import torch.nn
import learning
from neuronalNetwork import *
from sklearn.decomposition import PCA

with open("../files/gpx_pathfindind_cycling.df",'rb') as infile:
    df_pathfinding = pickle.load(infile)
with open("../files/dict_cluster",'rb') as infile:
    dict_cluster = pickle.load(infile)

df_pathfinding = data.clean_dataframe(df_pathfinding)

size_routes = 100

network = NN(size_routes, len(dict_cluster)-1).cuda()
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
loss = nn.NLLLoss()

learning.learn_full_connected(df_pathfinding, dict_cluster, loss, optimizer, network, size_routes)
for i in range(50):
    learning.learn_full_connected(df_pathfinding, dict_cluster, loss, optimizer, network, size_routes)


pca = PCA(n_components=1)
df = df_pathfinding
for key in dict_cluster:
    pca = PCA(n_components=1)
    if(key != -1):
        for num_route in dict_cluster[key]:
            route = data.dataframe_to_array(df[df["route_num"]==num_route])
            data.normalize_route(route, size_routes)
            route = pca.fit_transform(route)
            output = network(torch.Tensor(route).squeeze(1).cuda())
            pred = output.argmax(dim=0, keepdim=True)
            print(key, pred.item())