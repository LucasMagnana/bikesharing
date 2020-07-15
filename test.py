import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
import json
import torch
import torch.nn as nn
from math import sin, cos, sqrt, atan2, radians
import copy
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import geopy.distance
from scipy.spatial.distance import *
import random
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KDTree
import folium
import matplotlib.pyplot as plt

import python.data as data
import python.display as dp
import python.voxels as voxel
import python.metric as metric
import python.clustering as cl
import python.RNN as RNN
import python.validation as validation
#import python.learning as learning
#from python.NN import *

with open("files/gpx_matched_simplified.df",'rb') as infile:
    df_simplified = pickle.load(infile)
tab_routes_voxels_simplified, dict_voxels_simplified = voxel.create_dict_vox(df_simplified, df_simplified.iloc[0]["route_num"], df_simplified.iloc[-1]["route_num"])
tab_routes_voxels_simplified_global = voxel.get_tab_routes_voxels_global(dict_voxels_simplified, df_simplified.iloc[-1]["route_num"])

with open("files/gpx_pathfindind_cycling.df",'rb') as infile:
    df_pathfinding = pickle.load(infile)       
tab_routes_voxels_pathfinding, dict_voxels_pathfinding = voxel.create_dict_vox(df_pathfinding, df_pathfinding.iloc[0]["route_num"], df_pathfinding.iloc[-1]["route_num"])
tab_routes_voxels_pathfinding_global = voxel.get_tab_routes_voxels_global(dict_voxels_pathfinding, df_pathfinding.iloc[-1]["route_num"])

with open("files/lyon.ox", "rb") as infile:
    G_lyon = pickle.load(infile)
with open("files/st_etienne.ox", "rb") as infile:
    G_stetienne = pickle.load(infile)
    
nodes_lyon, _ = ox.graph_to_gdfs(G_lyon)
tree_lyon = KDTree(nodes_lyon[['y', 'x']], metric='euclidean')

nodes_stetienne, _ = ox.graph_to_gdfs(G_stetienne)
tree_stetienne = KDTree(nodes_stetienne[['y', 'x']], metric='euclidean')

G = G_lyon
tree = tree_lyon

with open("files/cluster_dbscan_custom.tab",'rb') as infile:
    tab_clusters = pickle.load(infile)
with open("files/voxels_pathfinding.dict",'rb') as infile:
    dict_voxels = pickle.load(infile)
with open("files/kmeans.sk",'rb') as infile:
    kmeans = pickle.load(infile)
with open("files/dict_cluster",'rb') as infile:
    dict_cluster = pickle.load(infile)
    
df = df_pathfinding
    
size_data = 1
hidden_size = 256
num_layers = 2
voxels_frequency = 4

network = RNN.RNN_LSTM(size_data, max(tab_clusters)+1, hidden_size, num_layers)
network.load_state_dict(torch.load("files/network_osmnx.pt"))
network.eval()

nb_good_predict = 0
nb_predict = 0

deviation = 0 #5e-3

tab_predict = []

for i in range(9, 10): #len(tab_clusters)):
    print(i)
    if(tab_clusters[i] != -1 and i != 675):
        df_temp = df[df["route_num"]==i+1]
        d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
        f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
        rand = random.uniform(-deviation, deviation)
        d_point[0] += rand
        rand = random.uniform(-deviation, deviation)
        d_point[1] += rand
        rand = random.uniform(-deviation, deviation)
        f_point[0] += rand
        rand = random.uniform(-deviation, deviation)
        f_point[1] += rand
        
        if(d_point[0] < 45.5):
            tree = tree_stetienne
            G = G_stetienne
        else:
            tree = tree_lyon
            G = G_lyon
        df_route, cl, nb_new_cluster = validation.find_cluster(d_point, f_point, network, voxels_frequency, df_pathfinding, dict_voxels, 
                                     kmeans, tree, G, False)
        if(cl == tab_clusters[i]):
            nb_good_predict += 1
        nb_predict += 1
if(nb_predict > 0):
    tab_predict.append(nb_good_predict/nb_predict)

tot_predict = 0
for predict in tab_predict:
    tot_predict += predict
print(tot_predict/len(tab_predict))

dp.display(df_route)
dp.display_cluster_heatmap(df_simplified, dict_cluster[cl])


################################################################################_




nodes, _ = ox.graph_to_gdfs(G)

for p in range(len(df_route)-1):
    if(p == 0):
        point = [df_route.iloc[p]["lat"], df_route.iloc[p]["lon"]]
    d_idx = tree.query([point], k=1, return_distance=False)[0]
    node = nodes.iloc[d_idx].index.values[0]
    max_coeffs = 0
    next_node = -1
    for n in G.neighbors(node):
        df_seg = pd.DataFrame([[point[0],point[1], 1]], columns=["lat", "lon", "route_num"])
        df_seg = df_seg.append(pd.DataFrame([[G.nodes[n]['y'], G.nodes[n]['x'], 1]], columns=["lat", "lon", "route_num"]))

        tab_routes_voxels, _ = voxel.create_dict_vox(df_seg, 1, 1)
        coeffs = 0
        for vox in tab_routes_voxels[0] :
            if vox in dict_voxels_simplified :
                coeffs += dict_voxels_simplified[vox]["cyclability_coeff"]
        coeffs /= len(tab_routes_voxels[0])
        if(coeffs > max_coeffs):
            max_coeffs = coeffs
            next_node = n

    point = [df_route.iloc[p+1]["lat"], df_route.iloc[p+1]["lon"]]

    if(next_node != -1):
        next_point = [G.nodes[next_node]['y'], G.nodes[next_node]['x']]
        if(point[0] != next_point[0] or point[1] != next_point[1]):
            print(p, point, next_point, max_coeffs)
        else: 
            print("same")
    else:
        print("same")


