# Introduction

Introduce me.

# Modules 

## GranuleInfo: Simple Dict like Python Object with Spatial Information about Sentinel-2 MSI Granules

Data was derived from here: https://sentinel.esa.int/web/sentinel/missions/sentinel-2/data-products

There was only limited testing of the data.

```python
from S2MSI.GranuleInfo import GranuleInfo
Ginfo = GranuleInfo(version="lite")
print(Ginfo["32UPV"])
>>>{'tr': {'lat': 49.6162737214, 'lon': 11.9045727629}, 'll': {'lat': 48.6570271781, 'lon': 10.3579107698}}
Ginfo = GranuleInfo(version="full")
print(Ginfo["32UPV"])
>>>{'pos': {'tl': {'x': 600000.0000025682, 'lat': 49.644436702, 'lon': 10.3851737332, 'y': 5500020.000361709}, 'tr': {'x': 709800.0000165974, 'lat': 49.6162737214, 'lon': 11.9045727629, 'y': 5500020.000351718}, 'lr': {'x': 709800.0000132157, 'lat': 48.6298215752, 'lon': 11.8474784519, 'y': 5390220.000321694}, 'll': {'x': 599999.9999970878, 'lat': 48.6570271781, 'lon': 10.3579107698, 'y': 5390220.000326163}}, 'name': '32UPV', 'zone': 32, 'epsg': 32632}
```

Data was created with this [notebook](https://git.gfz-potsdam.de/hollstei/S2MSI/tree/master/S2MSI/GranuleInfo/s2_kml_to_dict.ipynb).
