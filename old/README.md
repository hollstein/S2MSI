# Introduction

The S2MSI python module is intended to allow to perform basic tasks with [Sentinel-2 MSI](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) images, such as:

* reading tiles into numpy arrays (using either glymur, gdal, or kakadu)
* getting spectral response functions for a Sentinel-2
* manage digital elevation models for Sentinel-2 tiles
* getting basic data about Sentinel-2 tiles 
* general masking capabilities (e.g. clouds and cirrus)
* application of machine learning tools like the classical Bayesian classifier
* incorporating the functionality of GRASS GIS for Sentinel-2

It is also work in progress since I add functionality as I need it.

# Install

For now, installing is a little peculiar. I try to include functionality from GRASS GIS, for which currently only **python2.7** bindings exists. Apart from that, all code is **python3.4+**. My solution is to have everything in one module (one **setup.py**) which should be called with a **python3.4+** and a **python2.7** interpreter. This is automated with a bash script:


```
bash ./setup.sh install 
```

This script expects a **python2.7** environment which can be activated with ```source activate ${py27env} ``` where ```py27env``` is an environment variable which can be set in ```setup.sh``` and is set to ```py27``` as a default. To get everything right, the usual way of executing `python setup.py install` is not recommended as long as the GRASS GIS parts are on **python2.7**.

If you want to get in touch, [try this]( http://www.gfz-potsdam.de/en/section/remote-sensing/staff/profil/andre-hollstein/). 

# Removal

For now, un-installing is only possible by manually removing the folders of the module. To get a list of to-be-removed folder, call:

```
bash ./setup.sh uninstall
```

# Modules 

High level documentation is mostly missing for now, but docstrings are in place.

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


# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">S2MSI</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/hollstein/S2MSI" property="cc:attributionName" rel="cc:attributionURL">S2MSI</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/hollstein/S2MSI" rel="dct:source">https://github.com/hollstein/S2MSI</a>.


