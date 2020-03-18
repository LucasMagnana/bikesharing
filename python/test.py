import pickle
import data
import torch
import torch.nn
import learning
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from NN import *
from RNN import *

cuda = False

with open("../files/gpx_pathfindind_cycling.df",'rb') as infile:
    df_pathfinding = pickle.load(infile)
with open("../files/gpx_matched_simplified.df",'rb') as infile:
    df_simplified = pickle.load(infile)
with open("../files/dict_cluster",'rb') as infile:
    dict_cluster = pickle.load(infile)
with open("../files/cluster_dbscan_custom.tab",'rb') as infile:
    tab_clusters = pickle.load(infile)

df = df_pathfinding

df = data.clean_dataframe(df)

size_routes = 100
learning_rate = 5e-4


network = NN(size_routes, len(dict_cluster)-1)
network = RNN(2, len(dict_cluster)-1)
if(cuda):
    network = network.cuda()

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
loss = nn.NLLLoss()

tab_loss = learning.learn_recursive(df, tab_clusters, loss, optimizer, network, size_routes, cuda)


good_predict = 0
nb_predict = 0
for key in dict_cluster:
    if(key != -1):
        for num_route in dict_cluster[key]:
            route = data.dataframe_to_array(df[df["route_num"]==num_route])
            input = torch.Tensor(route).unsqueeze(1)
            if(cuda):
                input = input.cuda()

            hidden = network.initHidden()
            for i in range(input.shape[0]):
                output, hidden = network(input[i], hidden)
            target = torch.Tensor([key]).long()
            if(cuda):
                output = output.cuda()
                target = target.cuda()
            pred = output.argmax(dim=1, keepdim=True)
            if(key == pred.item()):
                good_predict += 1
            nb_predict += 1

print("Good predict:", good_predict/nb_predict*100, "%")

plt.plot(tab_loss)
plt.ylabel('Error')
plt.show()