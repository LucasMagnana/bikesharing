
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import data as data
import numpy as np
from random import sample

import matplotlib.pyplot as plt


def learn_full_connected(df, dict_cluster, loss, optimizer, network, size_routes):
    loss_tab = []
    pca = PCA(n_components=1)
    for key in dict_cluster:
        if(key != -1):
            input = []
            for i in range(30):
                num_route = sample(dict_cluster[key], 1)
                route = data.dataframe_to_array(df[df["route_num"]==num_route[0]])
                data.normalize_route(route, size_routes)
                route = pca.fit_transform(route)
                input.append((route))
            input = torch.Tensor(input).squeeze(2).cuda()
            optimizer.zero_grad()
            output = network(input)
            target = torch.Tensor([key]*len(input)).long().cuda()
            output = output.cuda()
            l = loss(output, target)
            loss_tab.append(l.item())
            l.backward()
            optimizer.step()
    plt.plot(loss_tab)
    plt.ylabel('Error')
    plt.show()
    