# standard imports
import glymur
from glob import glob
import numpy as np
from xml.etree.ElementTree import QName
import xml.etree.ElementTree
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import bisplrep
from scipy.interpolate import bisplev
from time import time
from os import path
from psutil import virtual_memory
import gdal
from gdalconst import GA_ReadOnly
import pygrib
from skimage.exposure import rescale_intensity, adjust_gamma
from uuid import uuid1
from subprocess import call
from os.path import abspath
from PIL import Image
import warnings
import re
from tempfile import TemporaryFile
import logging

# non standard imports
from ..GranuleInfo import GranuleInfo

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


class S2Image(object):
    def __init__(self, S2_MSI_granule_path, import_bands="all",
                 namespace="https://psd-12.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd",
                 target_resolution=None, dtype_float=np.float16, dtype_int=np.int16,unit="reflectance",
                 interpolation_order=1, data_mode="dense", driver="OpenJpeg2000",
                 call_cmd=None, logger=None,sliceX=slice(None),sliceY=slice(None),
                 aux_fields=None,
                 **kwarg):
        """
        Reads Sentinel-2 MSI data into numpy array. Images for different channels are resampled to a common sampling
        :param S2_MSI_granule_path: path to granule folder, folder should contain IMG_DATA folder and S2A_[..].xml file
        :param import_bands: list of bands to import, default: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                                                                                    'B8A', 'B09', 'B10', 'B11', 'B12',]
        :param namespace: for XML file
        :param target_resolution: spatial resolution in meter, or None for data not interpolated
        :param interpolation_order: integer for interpolation 1,2,3
        :param data_mode: either "dense" or "sparse"
        :param aux_fields: either None or dict auf to be loaded aus fields, e.g. {"cwv":[ss],"spr":[ss],"ozo":[ss]}
                           with ss beeing "mean" or iterable of spatial samplings, e.g. [20.0] or [20.0,60.0]
        :param driver: should be "OpenJpeg2000" -> glymur, "kdu_expand"->shell command, "gdal_[driver] e.g. gdal_JP2ECW
                       gdal_JPEG2000, gdal_JP2OpenJPEG,gdal_JP2KAK


        todo: Real nodata mask

        """

        t0_init = time()

        self.full_band_list = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11',
                               'B12']
        self.logger = logger or logging.getLogger(__name__)
        self.metadata = {}
        self.namespace = namespace
        self.call_cmd = call_cmd
        assert type(sliceX) == slice
        self.sliceX = sliceX
        assert type(sliceY) == slice
        self.sliceY = sliceY


        self.driver = str(driver)
        if unit in ["reflectance", "dn"]:
            self.unit = unit
        else:
            raise ValueError("Unit not implemented: %s" % str(unit))

        self.dtype_float = dtype_float
        self.dtype_int = dtype_int

        if self.unit == "dn":
            self.dtype_return = self.dtype_int
            self.bad_data_value = -999
        else:
            self.dtype_return = self.dtype_float
            self.bad_data_value = np.nan

        self.target_resolution = target_resolution
        self.S2_MSI_granule_path = path.abspath(S2_MSI_granule_path)
        self.tile_name = self.S2_MSI_granule_path.split("_")[-2][1:]

        # find XML files for granule (granule folder) and product (two folders above granule folder)
        self.granule_xml_file_name = glob(path.join(self.S2_MSI_granule_path, "S2*.xml"))[0]
        self.product_xml_file_name = glob(path.join(path.dirname(path.dirname(self.S2_MSI_granule_path)), "S2*.xml"))[0]

        self.logger.info("Load S2 MSI data from:%s" % self.S2_MSI_granule_path)

        # parse product xml files
        self.metadata.update(S2Image.parse_S2_product_xml(self.product_xml_file_name))
        self.metadata.update(S2Image.parse_S2_granule_xml(self.granule_xml_file_name, self.namespace))
        self.metadata["aux_data"] = S2Image.read_aux_data(self.S2_MSI_granule_path)
        # select bands
        if import_bands == "all":
            self.band_list = self.metadata["bandNames"]
        else:
            self.band_list = list(import_bands)
        # search for jp2 files which contain the data
        self.band_fns = {self._band_name(fn): fn for fn in
                         glob(path.join(self.S2_MSI_granule_path, "IMG_DATA", "S2A_*.jp2"))}
        # set final sizes
        if self.target_resolution is None:
            self.final_shape = None
            self.data = {}
        elif self.target_resolution in self.metadata["spatial_samplings"].keys():
            ss = self.metadata["spatial_samplings"]
            self.full_shape = [ss[self.target_resolution][ii] for ii in ("NROWS", "NCOLS")]
            self.final_shape = list(np.empty(self.full_shape)[self.sliceX,self.sliceY].shape)

            self.logger.info("Final shape for each channel: %s" % str(self.final_shape))
            if data_mode == "dense":
                self.data = self.__zeros__(shape=self.final_shape + [len(self.band_list)],
                                           dtype=self.dtype_return,logger=self.logger)
            elif data_mode == "sparse":
                self.data = self.__zeros__(shape=list(self.final_shape) + [len(self.metadata["bandNames"])],
                                           dtype=self.dtype_return,logger=self.logger)
            else:
                raise ValueError("data_mode=%s not implemented" % data_mode)
        else:
            raise ValueError("target_resolution should be None or in: %s" %
                             str(list(self.metadata["spatial_samplings"].keys())))


        # simple nodata mask -> should be improved
        self.nodata = {10.0:self.__read_img(self.band_fns["B02"]) == 0}
        self.nodata[20.0] = zoom(self.nodata[10.0],1 / 2,order=0)
        self.nodata[60.0] = zoom(self.nodata[10.0],1 / 6,order=0)

        self.cnv = {}
        for iband, band in enumerate(self.band_list):
            t0 = time()
            image_data_raw = self.__read_img(self.band_fns[band])

            if self.unit == "reflectance":
                # make image conversion on high precision, then convert to final type
                image_data = np.array(np.array(image_data_raw[:, :], dtype=np.float64) / self.metadata["dn2rfl"],
                                      dtype=self.dtype_return)

            elif self.unit == "dn":
                image_data = np.array(image_data_raw[:, :], dtype=self.dtype_return)

            image_data[self.bad_data_mask(image_data,band)] = self.bad_data_value

            if self.target_resolution is None:
                self.data[band] = np.array(image_data,dtype=self.dtype_float)
                t2 = time()
                self.logger.info("Read band %s in %.2fs." % (band, t2 - t0))
            else:
                zoom_fac = self.metadata["shape_to_resolution"][image_data.shape] / self.target_resolution
                if data_mode == "dense":
                    ii = iband
                elif data_mode == "sparse":
                    ii = self.full_band_list.index(band)
                else:
                    raise ValueError("data_mode=%s not implemented" % data_mode)
                t1 = time()
                self.data[:, :, ii] = zoom(input=np.array(image_data,dtype=np.float32),
                                           zoom=zoom_fac,
                                           order=interpolation_order)[self.sliceX,self.sliceY]
                self.cnv[ii] = band
                self.cnv[band] = ii
                t2 = time()
                self.logger.info("""Read band %s in %.2fs, pure load time:%.2fs, resample time: %.2fs, zoom: %.3f,
final shape: %s, index: %i""" % (band, t2 - t0, t1 - t0, t2 - t1, zoom_fac, str(self.final_shape), ii))

        if self.target_resolution is None:
            self.band_spatial_sampling = {band:self.metadata['shape_to_resolution'][data.shape]
                                          for band,data in self.data.items()}


        if aux_fields is not None and self.metadata["aux_data"]:
            self.logger.info("Read aux-data from level 1C Product.")
            ti = GranuleInfo(version="lite")[self.tile_name]
            zm = 2 * np.ceil(np.mean(self.metadata["aux_data"]["lons"].shape))

            lats_S2 = zoom(np.array([[ti["tl"]["lat"], ti["tr"]["lat"]], [ti["ll"]["lat"], ti["lr"]["lat"]]]), zoom=zm)
            lons_S2 = zoom(np.array([[ti["tl"]["lon"], ti["tr"]["lon"]], [ti["ll"]["lon"], ti["lr"]["lon"]]]), zoom=zm)

            fields_func = {field: self.get_tck(x=self.metadata["aux_data"]["lons"].flatten(),
                                               y=self.metadata["aux_data"]["lats"].flatten(),
                                               z=self.metadata["aux_data"][field].flatten()
                                               ) for field in aux_fields.keys()}

            self.fields_func = fields_func
            self.aux_fields = {}
            self.aux_fields["lats_S2"] = lats_S2
            self.aux_fields["lons_S2"] = lons_S2

            for field_name in aux_fields.keys():
                self.logger.info("Interpolate: %s" % field_name)
                bf = np.zeros(lats_S2.shape)
                for ii0 in range(lats_S2.shape[0]):
                    for ii1 in range(lats_S2.shape[1]):
                        bf[ii0, ii1] = bisplev(x=lons_S2[ii0, ii1],
                                               y=lats_S2[ii0, ii1],
                                               tck=fields_func[field_name])


                if self.final_shape is not None:
                    zoom_fac = (self.full_shape[0] / bf.shape[0],
                                self.full_shape[1] / bf.shape[1])
                    bf_zoom = zoom(bf, zoom_fac, order=3)
                    self.aux_fields[field_name] = np.array(bf_zoom[self.sliceX,self.sliceY],dtype=np.float32)
                else:
                    if aux_fields[field_name] == "mean":
                        self.aux_fields[field_name] = np.mean(bf)
                    else:
                        self.aux_fields[field_name] = {}
                        for spatial_sampling, res in self.metadata["spatial_samplings"].items():
                            if spatial_sampling in aux_fields[field_name]:
                                zoom_fac = (res["NCOLS"] / bf.shape[0],res["NROWS"] / bf.shape[0])
                                bf_zoom = zoom(bf, zoom_fac, order=3)
                                self.aux_fields[field_name][spatial_sampling] = np.array(bf_zoom,dtype=np.float32)
                                self.logger.info("Interpolate: %s, zoom: %s, sampling: %f" %
                                                 (field_name, "(%.1f,%.1f)" % zoom_fac, spatial_sampling))

        self.logger.info("Total runtime: %.2fs" % (time() - t0_init))

    @staticmethod
    def get_tck(x,y,z,kk_default=3,logger=None):
        logger = logger or logging.getLogger(__name__)

        # estimate max number of allowed spline order
        mm = len(x)
        kk_max = np.int(np.floor(np.sqrt(mm/2.0)-1))
        kk = np.min([kk_default,kk_max])

        result = bisplrep(x=x,y=y,z=z,kx=kk,ky=kk,full_output=1)
        if result[2]>0:
            logger.info("Interpolation problem:%s" % result[-1])
            logger.info("Now, try to adjust s")
            result = bisplrep(x=x,y=y,z=z,kx=kk,ky=kk,s=result[1],full_output=1)
            if result[2]>0:
                raise ValueError("Interpolation problem:%s" % result[-1])
        return result[0]

    @staticmethod
    def __zeros__(shape, dtype, max_mem_frac=0.3,logger=None):
        logger = logger or logging.getLogger(__name__)

        def in_memory_array(shape, dtype):
            return np.zeros(shape, dtype)

        def out_memory_array(shape, dtype):
            logger.warning("Not enough memory to keep full image -> fall back to memorymap.")
            dat = np.memmap(filename=TemporaryFile(mode="w+b"), dtype=dtype, shape=tuple(shape))
            dat[:] = 0.0
            return dat

        to_gb = 1.0 / 1024.0 ** 3
        mem = virtual_memory().total * to_gb
        arr = np.int(np.prod(np.array(shape, dtype=np.int64)) * np.zeros(1, dtype=dtype).nbytes * to_gb)

        if arr < max_mem_frac * mem:
            try:
                return in_memory_array(shape, dtype)
            except MemoryError as err:
                return out_memory_array(shape, dtype)
        else:
            logger.info(
                "Try to create array of size %.2fGB on a box with %.2fGB memory -> fall back to memorymap." % (arr, mem))
            return out_memory_array(shape, dtype)

    @staticmethod
    def parse_S2_product_xml(fn):
        """
        S2 XML helper function to parse product xml file
        :param fn: file name of xml file
        :return: metadata dictionary
        """
        metadata = {}
        xml_root = xml.etree.ElementTree.parse(fn).getroot()
        metadata["dn2rfl"] = (int(xml_root.find(".//QUANTIFICATION_VALUE").text))
        metadata["solar_irradiance"] = {int(ele.get('bandId')): float(ele.text) for ele in
                                        xml_root.find(".//Solar_Irradiance_List").findall("SOLAR_IRRADIANCE")}
        metadata["physical_gains"] = {int(ele.get('bandId')): float(ele.text) for ele in
                                      xml_root.findall(".//PHYSICAL_GAINS")}
        metadata["U"] = float(xml_root.find(".//U").text)
        return metadata

    @staticmethod
    def read_aux_data(fn):
        """
        Read grib file with aux data, return dictionary
        :param fn: path to granule
        :return: dict with data
        """
        metadata = {}
        try:
            fn_aux_data = glob(path.join(fn, "AUX_DATA", "S2A_*"))[0]
            aux_data = pygrib.open(fn_aux_data)

            for var_name, index in zip(["cwv", "spr", "ozo"], [1, 2, 3]):
                metadata[var_name] = aux_data.message(index).data()[0]
                metadata["%s_unit" % var_name] = aux_data.message(index)["parameterUnits"]

            metadata["lats"], metadata["lons"] = aux_data.message(1).latlons()
            return metadata
        except:
            return metadata

    @staticmethod
    def parse_S2_granule_xml(fn, namespace):
        """
        parse XML file in granule folder and return metadata
        :param fn: full filename ti xml file
        :param namespace: xml namespace
        :return: dictionary with metadata
        """

        def stack_detectors(inp):
            warnings.filterwarnings(action='ignore', message=r'Mean of empty slice')
            res = {bandId: np.nanmean(np.dstack(tuple(inp[bandId].values())), axis=2) for bandId, dat in inp.items()}
            warnings.filterwarnings(action='default', message=r'Mean of empty slice')
            return res

        xml_root = xml.etree.ElementTree.parse(fn).getroot()
        metadata = {}
        geo_codings = S2Image.find_in_xml_root(namespace, xml_root, 'Geometric_Info', "Tile_Geocoding")
        metadata["HORIZONTAL_CS_NAME"] = geo_codings.find("HORIZONTAL_CS_NAME").text
        metadata["HORIZONTAL_CS_CODE"] = geo_codings.find("HORIZONTAL_CS_CODE").text
        metadata["bandId2bandName"] = {int(ele.get("bandId")): ele.text.split("_")[-2] for ele in
                                       xml_root.findall(".//MASK_FILENAME") if ele.get("bandId") is not None}
        metadata["bandName2bandId"] = {bandName: bandId for bandId, bandName in metadata["bandId2bandName"].items()}
        metadata["bandIds"] = sorted(list(metadata["bandId2bandName"].keys()))
        metadata["bandNames"] = sorted(list(metadata["bandName2bandId"].keys()))
        metadata["sun_zenith"] = S2Image.get_values_from_xml(S2Image.find_in_xml_root(namespace, xml_root, *(
            "Geometric_Info", "Tile_Angles", "Sun_Angles_Grid", "Zenith", "Values_List")))
        metadata["sun_azimuth"] = S2Image.get_values_from_xml(S2Image.find_in_xml_root(namespace, xml_root, *(
            "Geometric_Info", "Tile_Angles", "Sun_Angles_Grid", "Azimuth", "Values_List")))
        metadata["sun_mean_zenith"] = float(S2Image.find_in_xml_root(namespace, xml_root, *(
            "Geometric_Info", "Tile_Angles", "Mean_Sun_Angle", "ZENITH_ANGLE")).text)
        metadata["sun_mean_azimuth"] = float(S2Image.find_in_xml_root(namespace, xml_root, *(
            "Geometric_Info", "Tile_Angles", "Mean_Sun_Angle", "AZIMUTH_ANGLE")).text)

        branch = S2Image.find_in_xml_root(namespace, xml_root, *("Geometric_Info", "Tile_Angles"))
        metadata["viewing_zenith_detectors"] = {
            bandId: {bf.get("detectorId"): S2Image.get_values_from_xml(
                    S2Image.find_in_xml(bf, *("Zenith", "Values_List"))) for bf in
                     branch.findall("Viewing_Incidence_Angles_Grids[@bandId='%i']" % bandId)} for bandId in
            metadata["bandIds"]}
        metadata["viewing_zenith"] = stack_detectors(metadata["viewing_zenith_detectors"])

        metadata["viewing_azimuth_detectors"] = {bandId: {bf.get("detectorId"): S2Image.get_values_from_xml(
                S2Image.find_in_xml(bf, *("Azimuth", "Values_List"))) for bf in branch.findall(
                "Viewing_Incidence_Angles_Grids[@bandId='%i']" % bandId)} for bandId in metadata["bandIds"]}
        metadata["viewing_azimuth"] = stack_detectors(metadata["viewing_azimuth_detectors"])

        metadata["spatial_samplings"] = {
            float(size.get("resolution")): {key: int(size.find(key).text) for key in ["NROWS", "NCOLS"]} for size in
            geo_codings.findall("Size")}
        for geo in geo_codings.findall("Geoposition"):
            metadata["spatial_samplings"][float(geo.get("resolution"))].update(
                    {key: int(geo.find(key).text) for key in ["ULX", "ULY", "XDIM", "YDIM"]})
        metadata["shape_to_resolution"] = {(values["NCOLS"], values["NROWS"]): spatial_sampling for
                                           spatial_sampling, values in metadata["spatial_samplings"].items()}
        return metadata

    @staticmethod
    def find_in_xml_root(namespace, xml_root, branch, *branches, findall=None):
        """
        S2 xml helper function, search from root
        :param namespace:
        :param xml_root:
        :param branch: first branch, is combined with namespace
        :param branches: repeated find's along these parameters
        :param findall: if given, at final a findall
        :return: found xml object, None if nothing was found
        """
        buf = xml_root.find(str(QName(namespace, branch)))
        for br in branches:
            buf = buf.find(br)
        if findall is not None:
            buf = buf.findall(findall)
        return buf

    @staticmethod
    def find_in_xml(xml, *branch):
        """
        S2 xml helper function
        :param xml: xml object
        :param branch: iterate to branches using find
        :return: xml object, None if nothing was found
        """
        buf = xml
        for br in branch:
            buf = buf.find(br)
        return buf

    @staticmethod
    def get_values_from_xml(leaf, dtype=np.float):
        """
        S2 xml helper function
        :param leaf: xml object which is searched for VALUES tag which are then composed into a numpy array
        :param dtype: dtype of returned numpy array
        :return: numpy array
        """
        return np.array([ele.text.split(" ") for ele in leaf.findall("VALUES")], dtype=dtype)

    @staticmethod
    def _band_name(fn):
        """
        S2 helper function, parse band name from jp2 file name
        :param fn: filename
        :return: string with band name
        """
        return fn.split(".jp2")[0].split("_")[-1]

    def bad_data_mask(self,image_data,band):
        """ Return mask of bad data

        :param image_data: numpy array
        :param band: band name, e.g. 'B11'
        :return: 2D boolean array

        .. todo:: Check included masks to derive bad data mask
        """

        self.logger.warning("Masks are computed on value basis, provider mask are not jet implemented")

        nodata = self.nodata[self.metadata['shape_to_resolution'][image_data.shape]]
        assert nodata.shape == image_data.shape

        if self.unit == "reflectance":
            return nodata
        elif self.unit == "dn":
            return nodata
        else:
            raise ValueError("Unit not implemented: %s" % self.unit)

    def S2_image_to_rgb(self, rgb_bands=("B11", "B08", "B03"), rgb_gamma=(1.0, 1.0, 1.0),hist_chop_off_fraction=0.01,
                        output_size=None,max_hist_pixel=1000**2,resample_order=3):

        if output_size is None:
            if self.target_resolution is None:
                raise ValueError("output_size=None is only allowed for target_resolution != None")
            else:
                output_shape = list(self.final_shape)
        else:
            output_shape = [output_size,output_size]

        rgb_type = np.uint8
        S2_rgb = np.zeros(output_shape + [len(rgb_bands),],dtype=rgb_type)

        if self.unit == "reflectance":
            bins = np.linspace(0.0,1.0,100 / 2.0)
        elif self.unit == "dn":
            bins = np.linspace(0,10000,100 / 2.0)

        for i_rgb, (band, gamma) in enumerate(zip(rgb_bands, rgb_gamma)):
            if self.target_resolution is None:
                data = self.data[band]
            else:
                i_band = self.band_list.index(band)
                data = self.data[:,:,i_band]

            if self.bad_data_value is np.NAN:
                bf = data[:,:][np.isfinite(data[:,:])]
            else:
                bf = data[:,:][data[:,:] == self.bad_data_value]

            pixel_skip = np.int(np.floor(bf.shape[0] / max_hist_pixel) + 1)
            bf = bf[::pixel_skip]
            hh, xx = np.histogram(bf, bins=bins,normed=False)
            bb = 0.5 * (xx[1:] + xx[:-1])
            hist_chop_off = hist_chop_off_fraction * np.sum(hh) / len(bins)
            lim = (lambda x: (np.min(x), np.max(x)))(bb[hh > hist_chop_off])
            zoom_factor = np.array(output_shape) / np.array(data[:,:].shape)

            zm = np.nan_to_num(np.array(data[:, :],dtype=np.float32))
            if (zoom_factor != [1.0,1.0]).all():
                self.logger.info("Resample band for RGB image: %i,%s,zoom:%.2f" % (i_rgb, band,zoom_factor[0]))
                zm = zoom(input=zm,zoom=zoom_factor,order=resample_order)

            bf = rescale_intensity(image=zm,in_range=lim,out_range=(0.0, 255.0))
            S2_rgb[:, :, i_rgb] = np.array(bf,dtype=rgb_type)

            self.logger.info("Rescale band for RGB image: %i,%s,(%.2f,%.2f)->(0,256), zoom:%.2f" %
                             (i_rgb, band, lim[0], lim[1],zoom_factor[0]))

            if gamma != 0.0:
                S2_rgb[:, :, i_rgb] = np.array(
                        adjust_gamma(np.array(S2_rgb[:, :, i_rgb], dtype=np.float32),gamma),dtype=rgb_type)
        return S2_rgb

    def __read_img(self, fn):
        self.logger.info("Reading: %s" % fn)
        if self.driver == "OpenJpeg2000":
            img = np.array(glymur.Jp2k(fn)[:,:],dtype=np.int16)
        elif self.driver == "kdu_expand":
            img = self.__read_jp2_kdu_app(fn, call_cmd=self.call_cmd)
        elif re.search("gdal[_]", self.driver) is not None:
            img = S2Image.gdal_read(fn, self.driver.split("_")[-1])
        else:
            raise ValueError("Driver not supported: %s" % self.driver)

        return img

    @staticmethod
    def __read_jp2_kdu_app(fn, call_cmd=None):
        fn_tmp = "/dev/shm/%s.tif" % uuid1()
        cmd_kdu = "kdu_expand -i %s -o %s -fprec 16" % (abspath(fn), fn_tmp)
        cmd_rm = "rm %s" % fn_tmp

        if call_cmd is None:
            _ = call(cmd_kdu, shell=True)
        else:
            _ = call_cmd(cmd_kdu)

        dat = np.array(Image.open(fn_tmp), dtype=np.int16)

        if call_cmd is None:
            _ = call(cmd_rm, shell=True)
        else:
            _ = call_cmd(cmd_rm)

        return dat

    @staticmethod
    def gdal_read(fnjp, driver_name):
        gdal.AllRegister()
        gdal_drivers = [gdal.GetDriver(ii).GetDescription() for ii in range(gdal.GetDriverCount())]
        if driver_name not in gdal_drivers:
            raise ValueError("Selected driver seems missing in gdal, available are: %s" % str(gdal_drivers))
        else:
            while gdal.GetDriverCount() > 1:
                for ii in range(gdal.GetDriverCount()):
                    drv = gdal.GetDriver(ii)
                    if drv is not None:
                        if drv.GetDescription() != driver_name:
                            try:
                                drv.Deregister()
                            except AttributeError:
                                pass

        ds = gdal.Open(fnjp, GA_ReadOnly)
        img = ds.GetRasterBand(1).ReadAsArray()
        return img

    @staticmethod
    def save_rgb_image(rgb_img, fn, dpi=100.0):
        fig = plt.figure(figsize=np.array(rgb_img.shape[:2]) / dpi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax = plt.subplot()
        ax.imshow(rgb_img, interpolation="none")
        ax.set_axis_off()
        plt.savefig(fn, dpi=dpi)
        fig.clear()
        plt.close(fig)

    def image_subsample(self,channels,target_resolution,order=3):
        """

        :param channels: list of strings with channel names
        :param target_resolution: float
        :param order: interpolation order, integer
        :return: data as desired
        """
        assert self.target_resolution is None

        if target_resolution is None:
            shape = list(self.data[channels[0]].shape)
        else:
            shape = [self.metadata["spatial_samplings"][target_resolution][ii] for ii in ["NCOLS","NROWS"]]
        shape.append(len(channels))

        dtype_internal = np.float32
        data = np.zeros(shape,dtype=dtype_internal)
        for ich,ch in enumerate(channels):
            zoom_fac = [shape[0] / self.data[ch].shape[0],
                        shape[1] / self.data[ch].shape[1]
                        ]

            bf = np.array(self.data[ch],dtype=dtype_internal)
            bf_nan = np.isnan(bf)
            bf[bf_nan] = 0.0
            data[:,:,ich] = zoom(input=bf,zoom=zoom_fac,order=order)
            bf_nan = zoom(input=np.array(bf_nan,dtype=np.float32),zoom=zoom_fac,order=0)
            data[:,:,ich][bf_nan > 0.0] = np.NaN

        return np.array(data,dtype=self.dtype_float)


if __name__ == "__main__":
    import zipfile

    with zipfile.ZipFile('./test_data.zip', "r") as zp:
        zp.extractall()
    test_granule = "test_data/S2A_OPER_PRD_MSIL1C_PDMC_20151231T175559_R022_V20151231T102248_20151231T102248.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20151231T141323_A002736_T33UXT_N02.01"
    s2Image = S2Image(S2_MSI_granule_path=test_granule, driver="kdu_expand")
    s2Image = S2Image(S2_MSI_granule_path=test_granule, driver="OpenJpeg2000")
    print(s2Image.metadata.keys())
    print("Done -> EEooFF")
