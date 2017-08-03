import os
import psutil
import resource
from psutil import virtual_memory



class RAM(object):
    def __init__(self,unit="GB"):
        """ helper util to manage RAM usage of this process

        :param unit: string as coded in self.units
        """
        self.units = {"GB":1024**3}
        self.unit = unit
        self.to_byte = self.units[self.unit]

    def used(self):
        """ currently used ram by process """
        return psutil.Process(os.getpid()).memory_info().rss / self.to_byte

    def get_limit(self):
        """ limit ram usage of process, either by rlimit or hardware memory """
        lim = resource.getrlimit(resource.RLIMIT_AS)[0]
        if lim == -1:  # unlimited
            lim = virtual_memory().total
        return lim / self.to_byte

    def free(self):
        """ return free memory """
        return self.get_limit() - self.used()

    def set_limit(self, limit):
        """ set soft limit for ram usage
        :param limit:
        """
        resource.setrlimit(resource.RLIMIT_AS, (limit * self.to_byte, resource.RLIM_INFINITY))
