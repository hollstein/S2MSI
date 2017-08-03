import numpy as np
from . import grassbin
import sys
from os.path import join
from subprocess import check_output


gisbase = check_output("%s --config path" % grassbin,shell=True).strip('\n')
sys.path.append(join(gisbase, "etc", "python"))

import grass.script.array as garray

class GGar(garray.array):
    def __init__(self):
        """Helper Class to exchange numpy arrays with Grass Gis and vice verse

        Need to create a valid region first, e.g.:
        grass.run_command("g.region",rows=,cols=,n=,s=,e=,w=)
        :return: instance derived from garray.array wirh two extra methods:
            - self.toGG
            - self.fromGG
        """
        super(garray.array, self).__init__()

    def toGG(self,data,name):
        self[:] = data[:]
        self.write(mapname=name,overwrite=True)

    def fromGG(self,name):
        self.read(name)
        return np.copy(self[:])