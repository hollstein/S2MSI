import zipfile
from tempfile import TemporaryDirectory
from os.path import isdir,join,abspath,basename,dirname
from os import remove
from glob import glob
import numpy as np
from ..Tools import mkdir_p
import h5py
from scipy.ndimage import zoom
import json
from ..Tools import write_image_gdal, read_image_gdal
from ..Tools.inpaint import fill_nan

class GranuleDEM(object):
    def __init__(self,fn,target_resolution=60,
                 sampling_to_shape={10.0: (10980, 10980), 20.0: (5490, 5490), 60.0: (1830, 1830)},
                 zoom_order=2,sliceX=slice(None),sliceY=slice(None),**kwargs):
        """ get digital elevation models for S2 MSI Granules from archive

        :param fn: filename of DEM archive (zip file or hdf5 file) or path to dem folder structure

        :type fn: string

        :return: Dict like object which returns DEM for given granules if get_dem, [],or () are called
        """

        self.fn = fn
        self.ext = "tif"
        self.badvalue = -32768
        self.dtype = np.float32
        self.target_resolution = target_resolution
        self.sampling_to_shape = sampling_to_shape
        self.zoom_order = zoom_order
        assert type(sliceX) == slice
        self.sliceX = sliceX
        assert type(sliceY) == slice
        self.sliceY = sliceY

        if self.fn.split(".")[-1] == "zip":
            self.mode = "zip"
            self.zf = zipfile.ZipFile(fn)
            self.get_dem = self._get_dem_zip
            self.tiles = sorted([bf.filename.split(".%s" % self.ext)[0] for bf in self.zf.filelist])

        elif self.fn.split(".")[-1] == "h5":
            self.mode = "h5"
            self.get_dem = None
            self.get_dem = self._get_dem_hdf5
            self.h5f = h5py.File(name=fn,mode="r")
            self.tiles = sorted(list(self.h5f.keys()))
        elif isdir(fn) is True:
            self.mode = "dir"
            self.get_dem = self._get_dem_dir
            self.dems = {basename(fn).split(".%s" % self.ext)[0]:abspath(fn) for fn
                         in glob(join(self.fn,"**/*.%s" % self.ext),recursive=True)}
            self.tiles = sorted(list(self.dems.keys()))
        else:
            raise ValueError("fn=%s is neither zip file or directory" % self.fn)

    def _to_arr(self,arr):
        return self._to_slice(self._to_target_resolution(self._to_dtype(arr)))

    def _to_target_resolution(self,arr):
        zf = self.sampling_to_shape[self.target_resolution] / np.array(arr.shape,dtype=np.float)
        if zf[0] == 1.0 and zf[1] == 1.0:
            return arr
        else:
            return zoom(input=arr,zoom=zf,order=self.zoom_order)

    def _to_dtype(self,arr):
        if arr.dtype == self.dtype:
            return arr
        else:
            bad = arr == self.badvalue
            arr = np.array(arr,dtype=self.dtype)
            arr[bad] = np.nan
            fill_nan(arr)
            return arr

    def _to_slice(self,arr):
        if self.sliceX == slice(None) and self.sliceY == slice(None):
            return arr
        else:
            return arr[self.sliceX,self.sliceY]

    def _get_dem_hdf5(self, tile):
        try:
            return self._to_arr(self.h5f[tile].value)
        except KeyError:
            raise ValueError("The tile:%s is missing in this archive. Included are: %s" % (tile,str(list(self.tiles))))

    def _get_dem_zip(self, tile):
        with TemporaryDirectory() as tmp_dirname:
            try:
                fn = self.zf.extract(self.zf.getinfo("%s.%s" % (tile,self.ext)), path=tmp_dirname)
            except:
                raise ValueError("The tile:%s if missing in this archive. Included are: %s" % (tile, str(self.tiles)))
            else:
                dat = self._to_arr(read_image_gdal(fn))
                return dat

    def _get_dem_dir(self,tile):
        try:
            return self._to_arr(read_image_gdal(self.dems[tile]))
        except KeyError:
            raise ValueError("The tile:%s is missing in this archive. Included are: %s" % (tile,str(list(self.tiles))))


    def dem_to_file(self,tile,filename,driver_map = {"tif":"gtiff"},lat_lon=None,extent=None):
        """ Write digital elevation data to file

        :param tile: S2 MSI tile name e.g. '32UMU'
        :param filename: filename, string
        :return:None
        """

        dem = np.array(self.get_dem(tile),dtype=np.uint16)
        mkdir_p(dirname(filename))
        try:
            remove(filename)
        except FileNotFoundError:
            pass
        file_ext = filename.split(".")[-1]
        driver=driver_map[file_ext]
        write_image_gdal(data=dem,filename=filename,driver=driver)

        if extent is not None:
            extent = (lambda x:x["pos"] if "pos" in x else x)(extent)
            fn = filename.replace("." + file_ext,"_extent.json")
            with open(fn,"w") as fl:
                json.dump({"n":extent["tr"]["lat"],
                           "s":extent["ll"]["lat"],
                           "w":extent["ll"]["lon"],
                           "e":extent["tr"]["lon"]},fl)

        if lat_lon is not None:
            lat_lon = (lambda x:x["pos"] if "pos" in x else x)(lat_lon)
            fnpat = filename.replace("." + file_ext,"_%s." + file_ext)
            zoom_fac = np.array(dem.shape)/2.0
            ll = {ii:zoom(np.array(
                    [[lat_lon["tl"][ii],lat_lon["tr"][ii]],[lat_lon["ll"][ii],lat_lon["lr"][ii]]],dtype=np.float32)
                    ,zoom_fac) for ii in ["lat","lon"]}

            for ii in ["lat","lon"]:
                fn = fnpat % ii
                write_image_gdal(data=ll[ii],filename=fn)

    def __call__(self,tile):
        """ Wrapper for get_dem """
        return self.get_dem(tile)
