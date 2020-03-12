import json 
import pandas as pd
import pickle
import sys
from rdp import *
import requests
import numpy as np
import xml.etree.ElementTree as ET
import os
from math import sin, cos, sqrt, atan2, radians

def check_file(file, content):
    if(not(os.path.isfile(file))):
        print("Warning: creating", file)
        open(file, "x")
        with open(file,'wb') as infile:
            pickle.dump(content, infile)



def load_gpx():
    with open('gpx.df','rb') as infile:
        df = pickle.load(infile)
    begin = int(df.iloc[-1]["route_num"])
    print(begin)
    for i in range(begin+1, begin+1109+1):
        tree = ET.parse('Datas/GPS/GPX/data'+str(i)+'.gpx')
        if(len(tree.getroot()) > 1):
            root = tree.getroot()[1][0]
            df_temp = pd.DataFrame(columns=['lat', 'lon'])
            j=0
            for child in root:
                coord = child.attrib
                coord['lat'] = float(coord['lat'])
                coord['lon'] = float(coord['lon'])
                df_temp = df_temp.append(pd.DataFrame(coord, index=[j]))
                j+=1
            df_temp["route_num"] = i
            df = df.append(df_temp)
    with open('gpx.df', 'wb') as outfile:
        pickle.dump(df, outfile)
        

def request_map_matching(df_route):
    route = df_route.to_numpy()
    coord=""
    tab_requests = []
    i=0
    for i in range(len(route)):
        coord += str(route[i][1])+","+str(route[i][0])+";"
        if(i!=0 and i%99 == 0):
            coord = coord[:-1]
            tab_requests.append(requests.get("https://api.mapbox.com/matching/v5/mapbox/cycling/"+coord+"?access_token="+token))
            coord = ""
    if(i!=0 and i%99 != 0):
        coord = coord[:-1]
        tab_requests.append(requests.get("https://api.mapbox.com/matching/v5/mapbox/cycling/"+coord+"?access_token="+token))
    return tab_requests


def dataframe_to_array(df):
    tab = []
    for i in range(len(df)):
        tab.append([df.iloc[i]["lat"], df.iloc[i]["lon"]])
    return tab


def clean_dataframe(df):
    nb_empty = 0
    df_final = pd.DataFrame(columns=['lat', 'lon', 'route_num'])
    for i in range(1, df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        if(len(df_temp)==0):
            nb_empty += 1
        else:
            df_temp["route_num"] = i-nb_empty
            df_final = df_final.append(df_temp)
    return df_final


def load_bikepath(file):
    with open(file) as infile:
        data = json.load(infile)        
    df_bikepath = pd.DataFrame(columns=['lat','lon', 'route_num'])
    for i in range(len(data["features"])):
        route = data["features"][i]["geometry"]["coordinates"]
        while(len(route[0]) != 2 or not(isinstance(route[0][0], float))):
            route = route[0]
        df_temp = pd.DataFrame(route, columns=['lon','lat'])
        df_temp
        df_temp['route_num'] = i
        df_bikepath = df_bikepath.append(df_temp)
    return df_bikepath


def rd_compression(df, nb_routes=sys.maxsize, eps=1e-4):
    """
    Compress a dataframe with douglas-peucker's algorithm.

    Parameters
    ----------
    df : pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        Dataframe to compress
    eps : int in [0, 1[ , optional
        Precision of the compression (high value = few points)
    nb_routes : int
        Number of routes to compress

    Returns
    -------
    pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        the compressed DataFrame
    """
    
    df_simplified = pd.DataFrame(columns=['lat', 'lon', 'route_num'])
    nb_routes = min(df.iloc[-1]["route_num"]+1, nb_routes)
    for i in range(1, nb_routes):
        route = df[df['route_num']==i].values
        if(len(route)>0):
            simplified = rdp(np.delete(route, 2, 1), epsilon=eps)
            simplified = np.insert(simplified, 2, route[0][2], axis=1) #add the route_number to the compressed route
            df_temp = pd.DataFrame(simplified, columns=['lat', 'lon', 'route_num'])
            df_simplified = df_simplified.append(df_temp)
    return df_simplified


def mapmatching(infile, outfile, nb_routes=sys.maxsize):
    with open(infile,'rb') as infile:
        df = pickle.load(infile)
    check_file(outfile, pd.DataFrame(columns=['lon', 'lat', 'route_num']))
    with open(outfile,'rb') as infile:
        df_map_matched = pickle.load(infile)

    if(df_map_matched.empty):
        begin = 0
    else:
        begin = df_map_matched.iloc[-1]["route_num"]+1
    for i in range(begin, min(begin+1+nb_routes, df.iloc[-1]["route_num"]+1)):
        distance = 0
        df_temp = df[df["route_num"]==i]
        tab_requests = request_map_matching(df_temp)
        tab_points = []
        for req in tab_requests:
            response = req.json()
            if("tracepoints" in response):
                route = response["tracepoints"]
                for point in route:
                    if(point != None):
                        tab_points.append([point['location'][1], point['location'][0], i])
                        distance += point['distance']
        df_map_matched = df_map_matched.append(pd.DataFrame(tab_points, columns=["lat", "lon", "route_num"]))
        with open(outfile, 'wb') as outfile:
            pickle.dump(df_map_matched, outfile)


def request_route(lat1, long1, lat2, long2, mode="cycling"):
    coord = str(long1)+","+str(lat1)+";"+str(long2)+","+str(lat2)
    return requests.get("https://api.mapbox.com/directions/v5/mapbox/"+mode+"/"+coord, 
                            params={"alternatives": "true", "geometries": "geojson", "steps": "true", "access_token": token}) 


def pathfinding(infile, outfile, nb_routes=sys.maxsize):
    with open(infile,'rb') as infile:
        df_map_matched_simplified = pickle.load(infile)
    check_file(outfile, pd.DataFrame(columns=['lon', 'lat', 'route_num']))
    with open(outfile,'rb') as infile:
        df_pathfinding = pickle.load(infile)
    if(df_pathfinding.empty):
        begin = 0
    else:
        begin = df_pathfinding.iloc[-1]["route_num"]+1
    for i in range(begin, min(begin+1+nb_routes, df_map_matched_simplified.iloc[-1]["route_num"]+1)): #df_map_matched_simplified.iloc[-1]["route_num"]+1):
        save_route = True
        df_temp = df_map_matched_simplified[df_map_matched_simplified["route_num"]==i]
        if(not(df_temp.empty)):
            req = request_route(df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"],
                                df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]) #mapbox request to find a route between the stations
            response = req.json()
            if(response['code']=='Ok'): #if a route have been found
                steps = response['routes'][0]['legs'][0]['steps'] #we browse all the steps of the route
                for step in steps:
                    if(step['maneuver']['instruction'].find("Wharf") != -1):
                        save_route = False #if the route is not good (using a boat) we don't save it
                        break
                if(save_route): #if we save the route
                    df_temp = pd.DataFrame.from_records(response['routes'][0]['geometry']['coordinates'], 
                                            columns=['lon', 'lat']) #create a DF from the route (nparray)
                    df_temp["route_num"] = i
                    df_pathfinding = df_pathfinding.append(df_temp) #save the DF in dict_trips
                else: #if we don't save we store an empty DF in dict_trips
                    df_pathfinding = df_pathfinding.append(pd.DataFrame(columns=['lon', 'lat', 'route_num']))
                with open(outfile, 'wb') as outfile:
                    pickle.dump(df_pathfinding, outfile)

def simplify_gps(infile, outfile, nb_routes=sys.maxsize):
    with open(infile,'rb') as infile:
        df_map_matched = pickle.load(infile)
    check_file(outfile, pd.DataFrame(columns=['lon', 'lat', 'route_num']))
    with open(outfile,'rb') as infile:
        df_map_matched_simplified = pickle.load(infile)
    nb_routes = min(df_map_matched.iloc[-1]["route_num"] - df_map_matched_simplified.iloc[-1]["route_num"], nb_routes)
    df_map_matched_simplified = df_map_matched_simplified.append(rd_compression(df_map_matched, nb_routes))
    df_map_matched_simplified
    with open(outfile, 'wb') as outfile:
        pickle.dump(df_map_matched_simplified, outfile)


def distance_between_points(p1, p2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


def compute_distance(infile, outfile):
    with open(infile,'rb') as infile:
        df = pickle.load(infile)
    check_file(outfile, [])
    if(os.stat(outfile).st_size != 0):
        with open(outfile,'rb') as infile:
            tab_distances = pickle.load(infile)
    else:
        tab_distances = []
    for i in range(len(tab_distances), df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        dist = 0
        if(df_temp.shape[0] >= 2):
            for j in range(df_temp.shape[0]-1):
                dist += distance_between_points(df_temp.iloc[j], df_temp.iloc[j+1])
            tab_distances.append(dist)
        else:
            tab_distances.append(0)
    with open(outfile, 'wb') as outfile:
        pickle.dump(tab_distances, outfile)