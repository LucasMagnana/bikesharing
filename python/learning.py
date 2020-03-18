
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import data as data
import numpy as np
import random


def learn_full_connected(df, dict_cluster, loss, optimizer, network, size_routes, cuda):
    loss_tab = []
    pca = PCA(n_components=1)
    for key in dict_cluster:
        if(key != -1):
            input = []
            for i in range(30):
                num_route = random.sample(dict_cluster[key], 1)
                route = data.dataframe_to_array(df[df["route_num"]==num_route[0]])
                data.normalize_route(route, size_routes)
                route = pca.fit_transform(route)
                input.append((route))
            input = torch.Tensor(input).squeeze(2)
            if(cuda):
                input = input.cuda()
            optimizer.zero_grad()
            output = network(input)
            target = torch.Tensor([key]*len(input)).long()
            if(cuda):
                output = output.cuda()
                target = target.cuda()
            l = loss(output, target)
            loss_tab.append(l.item())
            l.backward()
            optimizer.step()
    return(loss_tab)

def learn_recursive(df, tab_clusters, loss, optimizer, network, size_routes, cuda):
    loss_tab = []
    for _ in range(15000):
        num_route = random.randint(0, len(tab_clusters)-1)
        key = tab_clusters[num_route]
        if(key != -1):
            route = data.dataframe_to_array(df[df["route_num"]==num_route+1])
            input = torch.Tensor(route).unsqueeze(1)
            if(cuda):
                input = input.cuda()

            hidden = network.initHidden()
            #network.zero_grad()
            optimizer.zero_grad()
            for i in range(input.shape[0]):
                output, hidden = network(input[i], hidden)
            target = torch.Tensor([key]).long()
            if(cuda):
                output = output.cuda()
                target = target.cuda()
            l = loss(output, target)
            loss_tab.append(l.item())
            l.backward()
            optimizer.step()
            '''for p in network.parameters():
                p.data.add_(-0.005, p.grad.data)'''
    return(loss_tab)
    