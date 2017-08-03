import json
from pkg_resources import resource_filename, Requirement, DistributionNotFound
from os.path import dirname, join, isfile

__author__ = "Andre Hollstein"


class GranuleInfo(dict):
    def __init__(self, version="lite"):
        """ Dict like object with basic information's about Sentinel2-MSI granules.

        :param version: should be either "lite" or "full"
        :type version: string with granule name, see keys() method for available names
        :return: dict like object, keys are S2 granule names (e.g. '32UVX'), data can be accessed via () and [] methods

        :Example

        >>> from S2MSI import GranuleInfo as GranuleInfo
        >>> S2gi = GranuleInfo(version="full")
        >>> S2gi["32UPV"]
            {u'country': [u'Germany'],
             u'epsg': 32632,
             u'name': u'32UPV',
             u'pos': {u'll': {u'lat': 48.6570271781,
               u'lon': 10.3579107698,
               u'x': 599999.9999970878,
               u'y': 5390220.000326163},
              u'lr': {u'lat': 48.6298215752,
               u'lon': 11.8474784519,
               u'x': 709800.0000132157,
               u'y': 5390220.000321694},
              u'tl': {u'lat': 49.644436702,
               u'lon': 10.3851737332,
               u'x': 600000.0000025682,
               u'y': 5500020.000361709},
              u'tr': {u'lat': 49.6162737214,
               u'lon': 11.9045727629,
               u'x': 709800.0000165974,
               u'y': 5500020.000351718}},
             u'region': [u'Europe'],
             u'zone': 32}


        """
        files = {"lite": "data/S2_tile_data_lite.json",
                 "full": "data/S2_tile_data_full.json"}
        try:
            fn_base = files[version]
        except KeyError:
            raise ValueError("Version should be: %s" % str(list(files.keys())))

        try:
            fn = resource_filename(Requirement.parse("S2MSI"), fn_base)
        except DistributionNotFound:
            fn = join(dirname(__file__), fn_base)
            if isfile(fn) is False:
                raise FileNotFoundError(files[version])
        else:
            if isfile(fn) is False:
                fn = join(dirname(__file__), fn_base)
                if isfile(fn) is False:
                    raise FileNotFoundError(files[version])

        with open(fn, 'r') as fl:
            S2_tile_data = json.load(fl)
        self.update(S2_tile_data)

    def __call__(self, arg):
        return self[arg]
