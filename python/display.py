
import plotly.express as px
import pandas as pd

token = "pk.eyJ1IjoibG1hZ25hbmEiLCJhIjoiY2s2N3hmNzgwMGNnODNqcGJ1N2l2ZXZpdiJ9.-aOxDLM8KbEQnJfXegtl7A"
px.set_mapbox_access_token(token)


def display(dfdisplay, n=75, line_group="route_num", color=None):
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


def display_routes(df, tab_routes, tab_voxels=[], line_group="route_num", color=None):
    dfdisplay = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for i in range(len(tab_routes)):
        dfdisplay = dfdisplay.append(df[df["route_num"]==tab_routes[i]+1])
    display(dfdisplay, len(tab_routes), line_group, color)