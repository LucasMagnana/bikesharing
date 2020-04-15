import pickle
import data
import torch
import torch.nn
import learning
import voxels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from NN import *
from RNN import *


def main(file1, file2, file3, file4):

    cuda = False

    with open(file1,'rb') as infile:
        df_pathfinding = pickle.load(infile)
    with open(file2,'rb') as infile:
        df_simplified = pickle.load(infile)
    with open(file3,'rb') as infile:
        tab_clusters = pickle.load(infile)
    with open(file4,'rb') as infile:
        dict_voxels = pickle.load(infile)

    df = df_pathfinding

    tab_routes_voxels, _ = voxels.create_dict_vox(df, df.iloc[0]["route_num"], df.iloc[-1]["route_num"])

    tab_routes_voxels_int = []

    df_voxels = pd.DataFrame()


    for i in range(len(tab_routes_voxels)):
        max_cycl_coeff = 0 
        nb_vox = 0
        tab_routes_voxels_int.append([])
        route = tab_routes_voxels[i]
        for v in route:
            vox = dict_voxels[v]
            if(vox["cyclability_coeff"]>max_cycl_coeff):
                max_cycl_coeff = vox["cyclability_coeff"]
        for vox in route:
            if(nb_vox%4==0):
                vox_str = vox.split(";")
                vox_int = [int(vox_str[0]), int(vox_str[1])]
                tab_points = voxels.get_voxel_points(vox_int)
                #points = tab_points[0][:2]+tab_points[1][:2]+tab_points[2][:2]+tab_points[3][:2]
                points = dict_voxels[vox]["cluster"]
                tab_routes_voxels_int[i].append(points)
            nb_vox += 1

        '''tab_routes_voxels_int[i] = sorted(tab_routes_voxels_int[i], key=lambda k: dict_voxels[str(voxels.find_voxel_int([k[0],k[1]])[0])+";"+str(voxels.find_voxel_int([k[0],k[1]])[1])]["cyclability_coeff"])
        
        if(len(tab_routes_voxels_int[i])>50):
            tab_routes_voxels_int[i] = tab_routes_voxels_int[i][:50]
        
        print(len(tab_routes_voxels_int[i]))'''

        df_temp = pd.DataFrame(tab_routes_voxels_int[i])
        df_temp["route_num"] = i+1
        df_voxels = df_voxels.append(df_temp)

    #print(tab_clusters)

    df = df_voxels

    size_data = 1

    learning_rate = 5e-4


    fc = NN(size_data, max(tab_clusters)+1)
    rnn = RNN(size_data, max(tab_clusters)+1)
    lstm = RNN_LSTM(size_data, max(tab_clusters)+1)


    network = lstm

    if(cuda):
        network = network.cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    tab_loss = learning.train(df, tab_clusters, loss, optimizer, network, size_data, cuda, 30000)


    g_predict = learning.test(df, None, tab_clusters, size_data, cuda)
    print("Random:", g_predict*100, "%")

    g_predict = learning.test(df, network, tab_clusters, size_data, cuda)
    print("Good predict:", g_predict*100, "%")

    plt.plot(tab_loss)
    plt.ylabel('Error')
    plt.show()

    '''import torch
    import torch.nn as nn

    lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
            torch.randn(1, 1, 3))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)
    print(out)

    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out[4])'''

if __name__ == "__main__": 
    main("./files/gpx_pathfindind_cycling.df", "./files/gpx_matched_simplified.df", "./files/cluster_dbscan_custom.tab", "./files/voxels_pathfinding.dict")