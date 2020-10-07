from OSMPythonTools.overpass import overpassQueryBuilder
from OSMPythonTools.overpass import Overpass
from OSMPythonTools.data import Data, dictRangeYears, ALL

from OSMPythonTools.nominatim import Nominatim

from collections import OrderedDict

overpass = Overpass()
nominatim = Nominatim()

dimensions = OrderedDict([
  ('typeOfRoad', OrderedDict({
    0: '"highway"="cycleway"',
    1: '"cycleway"',
    2: '"cycleway:left"',
    3: '"cycleway:right"',
    4: '"cycleway:lane"'
  }))
])

def fetch(typeOfRoad):
    areaId = nominatim.query("Lyon").areaId()
    query = overpassQueryBuilder(area=areaId, elementType='way', selector=typeOfRoad, out='body', includeGeometry=True)
    return overpass.query(query)
data = Data(fetch, dimensions)

for i in range(len(dimensions['typeOfRoad'])):
    print(data.select(typeOfRoad=i).countElements())
    json = data.select(typeOfRoad=i).toJSON()
    for el in json["elements"] :
        if(len(el['nodes']) != len(el['geometry'])):
            print("faux")
