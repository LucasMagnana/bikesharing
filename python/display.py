
import plotly.express as px
import pandas as pd

import python.voxels as voxel

token = "pk.eyJ1IjoibG1hZ25hbmEiLCJhIjoiY2s2N3hmNzgwMGNnODNqcGJ1N2l2ZXZpdiJ9.-aOxDLM8KbEQnJfXegtl7A"
px.set_mapbox_access_token(token)


def display(dfdisplay, n=75, line_group="route_num", color=None, filename=None):
    """
    Display a dataframe of gps points on a mapbox map.
    Parameters
    ----------
    df or str : pandas' DataFrame with columns=['lat', 'lon', 'route_num'] or the name of a file containing one
        Dataframe to display or the file where it is located
    n : int, optional
        Number of routes to display
    line_group : str, optional
        Dataframe's attribute used to differenciate routes
    color : str, optional
        Dataframe's attribute used to color routes
    """
    if(type(dfdisplay) == str): #if df is a file location
        with open(dfdisplay,'rb') as infile:
            n+=1
            dfdisplay = pickle.load(infile) #open the file to load the dataframe
            dfdisplay = dfdisplay[dfdisplay[line_group]<n]
    fig = px.line_mapbox(dfdisplay, lat="lat", lon="lon", line_group=line_group, color=color, zoom=11)
    fig.show()
    if(filename != None):
        fig.write_image(filename)


def display_routes(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for i in range(len(tab_routes)):
        dfdisplay = dfdisplay.append(df[df["route_num"]==tab_routes[i]+1])
    display(dfdisplay, len(tab_routes), line_group, color)


def display_cluster_heatmap(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for i in range(len(tab_routes)):
        df_temp = df[df["route_num"]==tab_routes[i]+1]
        df_temp["num_route"] = i+1
        dfdisplay = dfdisplay.append(df_temp)
    _, dict_voxels = voxel.create_dict_vox(dfdisplay, 1, dfdisplay.iloc[-1]["route_num"])
    tab = []
    for key in dict_voxels:
        tab_routes = dict_voxels[key]["tab_routes_real"]+dict_voxels[key]["tab_routes_extended"]
        vox_str = key.split(";")
        vox_int = [int(vox_str[0]), int(vox_str[1])]
        vox_pos = voxel.get_voxel_points(vox_int, 0)
        if(dict_voxels[key]["cyclability_coeff"]):
            tab.append([vox_pos[0][0], vox_pos[0][1], dict_voxels[key]["cyclability_coeff"]])

    dfdisplay = pd.DataFrame(tab, columns=["lat", "lon", "value"])
    fig = px.scatter_mapbox(dfdisplay, lat="lat", lon="lon",  color="value", size="value", zoom=10)
    fig.show()