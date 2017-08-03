import pandas
import numpy as np
from pkg_resources import resource_filename, Requirement, DistributionNotFound
from os.path import dirname, join, isfile


class S2SRF(object):
    def __init__(self,srf="S2A",fn_sheet='Spectral Responses'):

        srf_files = {
            "S2A":"data/Sentinel-2A MSI Spectral Responses.xlsx"
        }
        if srf in srf_files:
            fn_srf = srf_files[srf]
            try:
                fn = resource_filename(Requirement.parse("S2MSI"), fn_srf)
            except DistributionNotFound:
                fn = join(dirname(__file__), fn_srf)
                if isfile(fn) is False:
                    raise FileNotFoundError(fn_srf)
            else:
                if isfile(fn) is False:
                    fn = join(dirname(__file__), fn_srf)
                    if isfile(fn) is False:
                        raise FileNotFoundError(fn_srf)
            self.fn_srf = fn
        else:
            self.fn_srf = srf

        self.fn_sheet = fn_sheet
        assert fn_sheet in pandas.ExcelFile(self.fn_srf).sheet_names
        s2srf_excel = pandas.read_excel(self.fn_srf,sheetname=self.fn_sheet)

        band_map = {'B1':'B01',
                    'B2':'B02',
                    'B3':'B03',
                    'B4':'B04',
                    'B5':'B05',
                    'B6':'B06',
                    'B7':'B07',
                    'B8':'B08',
                    'B8A':'B8A',
                    'B9':'B09',
                    'B10':'B10',
                    'B11':'B11',
                    'B12':'B12'}

        srfs = {}
        for item in (s2srf_excel.items()):
            if item[0] == "SR_WL":
                wvl = np.array(item[1],dtype=np.float)
            else:
                band = item[0].split("SR_AV_")[-1]
                srf = np.array(item[1],dtype=np.float)
                srfs[band_map[band]] = srf / np.trapz(x=wvl,y=srf)

        self.srfs_wvl = wvl
        self.srfs = srfs
        self.bands = sorted(list(self.srfs.keys()))
        self.wvl = [int(np.trapz(x=self.srfs_wvl,
                                 y=self.srfs_wvl * self.srfs[band])) for band in self.bands]
        self.conv = {}
        self.conv.update({key:value for key,value in zip(self.bands,self.wvl)})
        self.conv.update({value:key for key,value in zip(self.bands,self.wvl)})

    def instrument(self,bands):
        return {
            'rspf':np.vstack([self[band] for band in bands]),
            'wvl_rsp':np.copy(self.srfs_wvl),
            'wvl_inst':np.copy(self.wvl),
            'sol_irr':None
        }

    def __call__(self,band):
        return self.srfs[band]

    def __getitem__(self,band):
        return self.srfs[band]
