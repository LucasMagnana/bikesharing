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

coeff_min = 0.5


def test_deviation(vox, point, dist_prev, dict_voxels):
    vox = vox.split(";")
    vox = [int(vox[0]), int(vox[1])]
    tab_vox_adj = []
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, 0, 1))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, 0, -1))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, 1, 0))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, -1, 0))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, 1, 1))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, -1, 1))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, 1, -1))
    tab_vox_adj.append(voxel.get_adjacent_voxel(vox, -1, -1))

    max_dist = 999999999999999999999
    next_vox = None
    for vox_adj in tab_vox_adj:
        key_adj = str(int(vox_adj[0]))+";"+str(int(vox_adj[1]))
        point_adj = voxel.get_voxel_points(vox_adj)
        point_adj = [point_adj[0][0], point_adj[0][1]]
        if(key_adj in dict_voxels and dict_voxels[key_adj]["cyclability_coeff"]>coeff_min):
            dist = data.distance_between_points(point, point_adj)
            if(dist<dist_prev and dist<max_dist):
                max_dist = dist
                next_vox = key_adj
    return next_vox, max_dist







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

i = 11
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

df_cluster = pd.DataFrame(columns=["lat", "lon", "route_num"])
for i in range(len(dict_cluster[cl])):
    df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[cl][i]+1]
    df_temp["num_route"] = i+1
    df_cluster = df_cluster.append(df_temp)
_, dict_voxels_cluster = voxel.create_dict_vox(df_cluster, 1, df_cluster.iloc[-1]["route_num"])

dist_start = 999999999
dist_end = 99999999
start_point = [df_route.iloc[0]["lat"], df_route.iloc[0]["lon"]]
end_point = [df_route.iloc[-1]["lat"], df_route.iloc[-1]["lon"]]
for vox in dict_voxels_cluster:
    if(dict_voxels_cluster[vox]["cyclability_coeff"]>coeff_min):
        vox_str = vox.split(";")
        vox_int = [int(vox_str[0]), int(vox_str[1])]
        point_vox = voxel.get_voxel_points(vox_int)
        point_vox = [point_vox[0][0], point_vox[0][1]]
        dist = data.distance_between_points(start_point, point_vox)
        if(dist < dist_start):
            dist_start = data.distance_between_points(end_point, point_vox)
            vox_start = vox
            vox_point_start = point_vox
        dist = data.distance_between_points(end_point, point_vox)
        if(dist < dist_end):
            dist_end = data.distance_between_points(start_point, point_vox)
            vox_end = vox
            vox_point_end = point_vox





vox = vox_start

distance = data.distance_between_points(end_point, vox_point_end)
tab_vox_start = []
while(vox != None):
    tab_vox_start.append(vox)
    vox, distance = test_deviation(vox, end_point, distance, dict_voxels_cluster)

vox_str = tab_vox_start[-1].split(";")
vox_int = [int(vox_str[0]), int(vox_str[1])]
point_vox = voxel.get_voxel_points(vox_int)

route = data.pathfind_route_osmnx(end_point, [point_vox[0][0], point_vox[0][1]], tree, G)
route_coord = [[G.nodes[x]["x"], G.nodes[x]["y"]] for x in route]
route_coord = [x + [1, 1] for x in route_coord]

df_route_modified = pd.DataFrame(route_coord, columns=["lon", "lat", "route_num", "type"])  




vox = vox_end
distance = dist_end
tab_vox_end = []
while(vox != None):
    tab_vox_end.append(vox)
    vox, distance = test_deviation(vox, start_point, distance, dict_voxels_cluster)

vox_str = tab_vox_end[-1].split(";")
vox_int = [int(vox_str[0]), int(vox_str[1])]
point_vox = voxel.get_voxel_points(vox_int)

route = data.pathfind_route_osmnx(start_point, [point_vox[0][0], point_vox[0][1]], tree, G)
route_coord = [[G.nodes[x]["x"], G.nodes[x]["y"]] for x in route]
route_coord = [x + [2, 1] for x in route_coord]
df_route_modified = df_route_modified.append(pd.DataFrame(route_coord, columns=["lon", "lat", "route_num", "type"])  )


if(tab_vox_end[0] != tab_vox_start[0]):
    vox_str = tab_vox_end[0].split(";")
    vox_int = [int(vox_str[0]), int(vox_str[1])]
    point_vox_end = voxel.get_voxel_points(vox_int)

    vox_str = tab_vox_start[0].split(";")
    vox_int = [int(vox_str[0]), int(vox_str[1])]
    point_vox_start = voxel.get_voxel_points(vox_int)

    route = data.pathfind_route_osmnx([point_vox_start[0][0], point_vox_start[0][1]], [point_vox_end[0][0], point_vox_end[0][1]], tree, G)
    route_coord = [[G.nodes[x]["x"], G.nodes[x]["y"]] for x in route]
    route_coord = [x + [3, 1] for x in route_coord]
    df_route_modified = df_route_modified.append(pd.DataFrame(route_coord, columns=["lon", "lat", "route_num", "type"])  )




#AFFICHAGE

tab_points_vox_start = []
for i in range(len(tab_vox_start)):
    key_vox = tab_vox_start[i]
    vox_str = key_vox.split(";")
    vox_int = [int(vox_str[0]), int(vox_str[1])]
    vox_points = voxel.get_voxel_points(vox_int, -1-i)
    tab_points_vox_start.extend(vox_points)


df = pd.DataFrame(tab_points_vox_start, columns=["lat", "lon", "route_num", "type"])

tab_points_vox_end = []
for i in range(len(tab_vox_end)):
    key_vox = tab_vox_end[i]
    vox_str = key_vox.split(";")
    vox_int = [int(vox_str[0]), int(vox_str[1])]
    vox_points = voxel.get_voxel_points(vox_int, -len(tab_points_vox_start)-1-i)
    tab_points_vox_end.extend(vox_points)

df_end = pd.DataFrame(tab_points_vox_end, columns=["lat", "lon", "route_num", "type"])
df = df.append(df_end)
df = df.append(df_route_modified)

dp.display(df)





"""nodes, _ = ox.graph_to_gdfs(G)
last_node = None

for p in range(len(df_route)-1):
    if(p == 0):
        point = [df_route.iloc[p]["lat"], df_route.iloc[p]["lon"]]
        df_route_modified = pd.DataFrame([[point[0], point[1], 1]], columns=["lat", "lon", "route_num"])
    else:
        df_route_modified = df_route_modified.append(pd.DataFrame([[point[0], point[1], 1]], columns=["lat", "lon", "route_num"]))
    d_idx = tree.query([point], k=1, return_distance=False)[0]
    node = nodes.iloc[d_idx].index.values[0]
    max_coeffs = 0
    next_node = -1
    for n in G.neighbors(node):
        if(n != last_node and ((df_route_modified['lat'] == point[0]) & (df_route_modified['lon'] == point[1])).any()):
            df_seg = pd.DataFrame([[point[0],point[1], 1]], columns=["lat", "lon", "route_num"])
            df_seg = df_seg.append(pd.DataFrame([[G.nodes[n]['y'], G.nodes[n]['x'], 1]], columns=["lat", "lon", "route_num"]))

            tab_routes_voxels, _ = voxel.create_dict_vox(df_seg, 1, 1)
            coeffs = 0
            for vox in tab_routes_voxels[0] :
                if vox in dict_voxels_cluster :
                    coeffs += dict_voxels_cluster[vox]["cyclability_coeff"]
            coeffs /= len(tab_routes_voxels[0])
            if(coeffs > max_coeffs):
                max_coeffs = coeffs
                next_node = n

    next_point = [df_route.iloc[p+1]["lat"], df_route.iloc[p+1]["lon"]]
    last_node = node

    if(next_node != -1):
        next_point_cluster = [G.nodes[next_node]['y'], G.nodes[next_node]['x']]
        if(next_point_cluster[0] != next_point[0] or next_point_cluster[1] != next_point[1]):
            point = next_point_cluster
        else: 
            point = next_point
    else:
        point = next_point

print(df_route_modified)
dp.display(df_route_modified)

"""
