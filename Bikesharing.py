import pandas as pd
import requests
import plotly.express as px
import pickle
token = "pk.eyJ1IjoibG1hZ25hbmEiLCJhIjoiY2s2N3hmNzgwMGNnODNqcGJ1N2l2ZXZpdiJ9.-aOxDLM8KbEQnJfXegtl7A"

def request_route(lat1, long1, lat2, long2):
    coord = str(long1)+","+str(lat1)+";"+str(long2)+","+str(lat2)
    return requests.get("https://api.mapbox.com/directions/v5/mapbox/cycling/"+coord, 
                            params={"alternatives": "true", "geometries": "geojson", "steps": "true", "access_token": token})
    
def fill_dict(file):
    file_already_used = False
    with open("files_done.txt",'rb') as infile:
        files_done = pickle.load(infile)
        for f in files_done:
            if(f == file):
                file_already_used = True
                break
                
    if(file_already_used):
        return
    
    with open('trips_washington.dict', 'rb') as infile:
        dict_trips = pickle.load(infile)
        
    trips = pd.read_csv("Datas/America/Washington/"+file)
    stations = pd.read_csv("Datas/America/Washington/Capital_Bike_Share_Locations.csv")
    
    #Remplissage hors requêtes mapbox
    '''for index, row in trips.iterrows():
        key = str(row["Start station number"])+','+str(row["End station number"])
        if(key in dict_trips):
            dict_trips[key]['durations'].append(row['Duration'])
            dict_trips[key]['number_of_trip']+=1
        else:
            dict_trips[key] = {'durations': [row['Duration']], 'routes': [], 'number_of_trip': 1}
            
    with open('trips_washington.dict', 'wb') as outfile:
        pickle.dump(dict_trips, outfile)'''
            
    #Remplissage avec requêtes mapbox
    for key in dict_trips:
        if(dict_trips[key]['routes'] == []):
            save_route = True
            tab_stations = key.split(',')
            st1 = stations[stations['TERMINAL_NUMBER']==int(tab_stations[0])]
            st2 = stations[stations['TERMINAL_NUMBER']==int(tab_stations[1])]
            if(len(st1)>0 and len(st2)>0):
                lon1 = st1['LONGITUDE'].values[0]
                lat1 = st1['LATITUDE'].values[0]
                lon2 = st2['LONGITUDE'].values[0]
                lat2 = st2['LATITUDE'].values[0]
                req = request_route(lat1, lon1, lat2, lon2)
                response = req.json()
                print(key)
                if(response['code']=='Ok'):
                    steps = response['routes'][0]['legs'][0]['steps']
                    for step in steps:
                        if(step['maneuver']['instruction'].find("Wharf") != -1):
                            save_route = False
                            break
                    if(save_route):
                        df_temp = pd.DataFrame.from_records(response['routes'][0]['geometry']['coordinates'], 
                                                   columns=['lon', 'lat'])
                        dict_trips[key]['routes'].append(df_temp)
                    else:
                        dict_trips[key]['routes'].append(None)
                    with open('trips_washington.dict', 'wb') as outfile:
                        pickle.dump(dict_trips, outfile)
                
    with open('trips_washington.dict', 'wb') as outfile:
        pickle.dump(dict_trips, outfile)
        
    files_done.append(file)
    with open('files_done.txt.dict', 'wb') as outfile:
        pickle.dump(files_done, outfile)
                
                

        
'''with open('trips_washington.dict', 'wb') as outfile:
        pickle.dump({}, outfile)'''

#fill_dict("2012Q1-capitalbikeshare-tripdata.csv")


with open('trips_washington.dict', 'rb') as infile:
    dict_trips = pickle.load(infile)
route_num = 0
df = pd.DataFrame(columns=['lon', 'lat', 'route_num']) #creation of an empty dataframe
for key in dict_trips:
    if(len(dict_trips[key]['routes'])!=0):
        df_temp = dict_trips[key]['routes'][0]
        df_temp['route_num'] = route_num
        route_num+=1
        df = df.append(df_temp)

px.set_mapbox_access_token(token)
fig = px.line_mapbox(df, lat="lat", lon="lon", line_group="route_num", zoom=11)
fig.show()