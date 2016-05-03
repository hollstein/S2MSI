from ..S2Image import S2Image
from ..S2Mask import S2Mask
import numpy as np
import h5py
import json
from time import time
import logging
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

def get_clf_functions():
    """
    this is just an example on how one could define classification
    functions, this is an argument to the ToClassifier Classes
    """
    return {
        "ratio": lambda d1, d2: save_divide(d1, d2),
        "index": lambda d1, d2: save_divide(d1 - d2, d1 + d2),
        "difference": lambda d1, d2: to_clf(d1) - to_clf(d2),
        "channel": lambda d: to_clf(d),
        "depth": lambda d1, d2, d3: save_divide(to_clf(d1) + to_clf(d2), d3),
        "index_free_diff": lambda d1, d2, d3, d4: save_divide(to_clf(d1) - to_clf(d2), to_clf(d3) - to_clf(d4)),
        "index_free_add": lambda d1, d2, d3, d4: save_divide(to_clf(d1) + to_clf(d2), to_clf(d3) + to_clf(d4)),
    }


def save_divide(d1, d2, mx=100.0):
    """ save division without introducing NaN's
    :param d1:
    :param d2:
    :param mx: absolute maximum allows value which from which on the result is chopped
    :return: d1/d2
    """
    dd1 = to_clf(d1)
    dd2 = to_clf(d2)
    dd2[dd2 == 0.0] = 1e-6
    dd1 /= dd2
    dd1 = np.nan_to_num(dd1)
    dd1[dd1 > mx] = mx
    dd1[dd1 < -mx] = -mx
    return dd1


def to_clf(inp):
    """ helper function which sets the type of features and assures numeric values
    :param inp:
    :return: np.float array without NaN's or INF's
    """
    return np.nan_to_num(np.array(inp, dtype=np.float))


class _ToClassifierBase(object):
    def __init__(self,logger=None):
        """ internal base class for generation of classifiers, only to use common __call__

        dummy __init__ which sets all basic needed attributes to none,
        need derived classes to implement proper __init__
        :return:
        """

        self.logger = logger or logging.getLogger(__name__)
        self.n_classifiers = None
        self.classifiers_fk = None
        self.classifiers_id = None
        self.clf_functions = None
        self.classifiers_id_full = None

    def adjust_classifier_ids(self, full_bands, band_lists):
        self.classifiers_id = [np.array([band_lists.index(full_bands[ii]) for ii in clf], dtype=np.int)
                               for clf in self.classifiers_id_full]
        self.logger.info("""Adjusting classifier channel list indices to actual image, convert from:
%s to \n %s. \n This results in a changed classifier index array from:""" % (str(full_bands), str(band_lists)))
        for func, old, new in zip(self.classifiers_fk, self.classifiers_id_full, self.classifiers_id):
            self.logger.info("%s : %s -> %s" % (func, old, new))

    @staticmethod
    def list_np(arr):
        """
        This is fixing a numpy annoyance where scalar arrays and vectors arrays are treated differently,
        namely one can not iterate over a scalar, this function fixes this in that a python list is
        returned for both scalar and vector arrays
        :param arr: numpy array or numpy scalar
        :return: list with values of arr
        """
        try:
            return list(arr)
        except TypeError:  # will fail if arr is numpy scalar array
            return list(arr.reshape(1))

    def __call__(self, data):
        """
        Secret sauce of the Classical Bayesian approach in python, here the input data->([n_samples,n_data_channels])
        are transformed into ret->([n_samples,n_classifiers])
        Iteration is performed over classifiers_fk (name found in clf_functions) and classifiers_id
        (channel selection from data for this function)
        :param data: n_samples x n_data_channels
        :return: res: n_samples x n_classifiers
        """

        # ret = np.zeros((data.shape[0], self.n_classifiers))  # initialize result
        ret = S2Image.__zeros__(shape=(data.shape[0], self.n_classifiers), dtype=np.float32)  # initialize result
        for ii, (fn, idx_clf) in enumerate(zip(self.classifiers_fk, self.classifiers_id)):
            # note that that input of clf_function[fn] is a generator expression where
            # iteration is performed over the selected classifiers_id's
            ret[:, ii] = self.clf_functions[fn](*(data[:, ii] for ii in self.list_np(idx_clf)))
        return ret


class ToClassifierDef(_ToClassifierBase):

    def __init__(self, classifiers_id, classifiers_fk, clf_functions,id2name=None,logger=None):
        """ Most simple case of a usable ToClassifier instance, everything is fixed

        classifiers_id: list of lists/np.arrays with indices which are inputs for classifier functions
        classifiers_fk: list of names for the functions to be used
        clf_functions: dictionary for key, value pairs of function names as used in classifiers_fk
        """

        self.logger = logger or logging.getLogger(__name__)
        self.n_classifiers = len(classifiers_fk)
        self.clf_functions = clf_functions
        self.classifiers_id_full = classifiers_id
        self.classifiers_id = classifiers_id
        self.classifiers_fk = classifiers_fk
        self.id2name = id2name

        # assert equal length
        assert len(self.classifiers_id) == self.n_classifiers
        assert len(self.classifiers_fk) == self.n_classifiers
        # assert that used functions are in self.clf_functions
        for cl_fk in self.classifiers_fk:
            assert cl_fk in self.clf_functions
        # assert that each value in the dict is a callable
        for name, func in self.clf_functions.items():
            if hasattr(func, "__call__") is False:
                raise ValueError("Each value in clf_functions should be a callable, error for: %s" % name)


class ClassicalBayesian(object):
    def __init__(self, mk_clf, bns, hh_full, hh, hh_n, n_bins, classes, n_classes, bb_full,logger=None):
        """

        :param mk_clf:
        :param bns:
        :param hh_full:
        :param hh:
        :param hh_n:
        :param n_bins:
        :param classes:
        :param n_classes:
        :param bb_full:
        :return:
        """
        self.logger = logger or logging.getLogger(__name__)
        self.mk_clf = mk_clf
        self.bns = bns
        self.hh_full = hh_full
        self.hh = hh
        self.hh_n = hh_n
        self.n_bins = n_bins
        self.classes = classes
        self.n_classes = n_classes
        self.bb_full = bb_full
        self.zm = 0.5
        self.gs = 0.5
        self.ar_hh_full = {}
        self.ar_hh = {cl:{} for cl in self.classes}

    def __in_bounds__(self, ids):
        ids[ids > self.n_bins - 1] = self.n_bins - 1

    def __predict__(self, xx):
        ids = [np.digitize(ff, bb) - 1 for ff, bb in zip(self.mk_clf(xx).transpose(), self.bb_full)]
        # ids = [c_digitize(ff, bb) - 1 for ff, bb in zip(self.mk_clf(xx).transpose(), self.bb_full)]

        tt=0

        for ii in ids:
            self.__in_bounds__(ii)
        pp = np.zeros((self.n_classes, len(ids[0])), dtype=np.float)
        for ii, cl in enumerate(self.classes):
            hh = self.hh[cl][ids]
            hh_full = self.hh_full[ids]
            hh_valid = hh_full > 0.0
            pp[ii, hh_valid] = hh[hh_valid] / hh_full[hh_valid] / self.n_classes

            hh_invalid = hh_valid == False

            t0 = time()
            if np.sum(hh_invalid) > 0:
                #ar_hh_full = np.copy(self.hh_full)
                #ar_hh = np.copy(self.hh[cl])

                ar_hh_full,ar_hh = None,None

                iw = -1
                while np.sum(hh_invalid) > 0:
                    iw += 1

                    #ar_hh_full = gaussian_filter(zoom(ar_hh_full,order=1,zoom=self.zm),sigma=self.gs)
                    #"""
                    try:
                        ar_hh_full = self.ar_hh_full[iw]
                    except KeyError:
                        if ar_hh_full is None and iw == 0:
                            ar_hh_full = np.copy(self.hh_full)
                        self.ar_hh_full[iw] = gaussian_filter(zoom(ar_hh_full,order=1,zoom=self.zm),sigma=self.gs)
                        ar_hh_full = self.ar_hh_full[iw]
                    #"""

                    #ar_hh = gaussian_filter(zoom(ar_hh,order=1,zoom=self.zm),self.gs)
                    #"""
                    try:
                        ar_hh = self.ar_hh[cl][iw]
                    except KeyError:
                        if ar_hh is None and iw == 0:
                            ar_hh = np.copy(self.hh[cl])
                        self.ar_hh[cl][iw] = gaussian_filter(zoom(ar_hh,order=1,zoom=self.zm),self.gs)
                        ar_hh = self.ar_hh[cl][iw]
                    #"""

                    n_bins = ar_hh_full.shape[0]

                    ids_bf = [np.array(id_bf[hh_invalid] * (n_bins / self.n_bins),dtype=np.int) for id_bf in ids]
                    for ii_bf in ids_bf:
                        ii_bf[ii_bf > n_bins - 1] = n_bins

                    hh_full = ar_hh_full[ids_bf]
                    hh = ar_hh[ids_bf]
                    good = hh_full != 0.0

                    hh_ok = np.copy(hh_invalid)
                    hh_ok[hh_ok == True] = good

                    hh_invalid[hh_invalid == True] = np.logical_not(good)
                    pp[ii, hh_ok] = hh[good] / hh_full[good] / self.n_classes
                    self.logger.info("class: %s, bins: %i->%i,curr ok: %i, still bad: %i" %
                          (str(cl),self.n_bins,n_bins,np.sum(hh_ok),np.sum(hh_invalid)))
            tt += time() - t0
        self.logger.info("Time spend in reduced form: %.2f" % tt)
        return pp

    def predict_proba(self, xx):
        pr = self.__predict__(xx.reshape((-1, xx.shape[-1]))).transpose()
        return pr.reshape(list(xx.shape[:-1]) + [pr.shape[-1], ])

    def predict(self, xx):
        pr = self.classes[np.argmax(self.__predict__(xx.reshape((-1, xx.shape[-1]))), axis=0)]
        return pr.reshape(xx.shape[:-1])

    def conf(self, xx):
        proba = self.predict_proba(xx)
        conf = np.nan_to_num(np.max(proba, axis=1) / np.sum(proba, axis=1))
        return conf.reshape(xx.shape[:-1])

    def predict_and_conf(self, xx):
        proba = self.__predict__(xx.reshape((-1, xx.shape[-1]))).transpose()
        tot = np.sum(proba, axis=1)
        conf = np.max(proba, axis=1) / tot
        pr = self.classes[np.argmax(proba, axis=1)]

        pr[tot == 0.0] = 0.0
        conf[tot == 0.0] = 0.0

        return pr.reshape(xx.shape[:-1]), conf.reshape(xx.shape[:-1])


def read_classical_Bayesian_persistence_file(filename):
    """ loads persistence data for classical Bayesian classifier from hdf5 file

    :param filename:
    :return: dictionary needed data
    """
    h5f = h5py.File(filename, 'r')

    with h5py.File(filename, 'r') as h5f:
        kwargs_mk_clf = {name:json.loads(h5f[name].value) for name in ["classifiers_fk","classifiers_id"]}
        kwargs_mk_clf["id2name"] = json.loads(h5f["band_names"].value)

        kwargs_cB = {}
        kwargs_cB["bns"] = [h5f[name].value for name in json.loads(h5f["bns_names"].value)]
        kwargs_cB["hh_full"] = h5f["hh_full"].value
        kwargs_cB["hh"] = {h5f["%s_key" % name].value:h5f["%s_value" % name].value for name in json.loads(h5f["hh_names"].value)}
        kwargs_cB["hh_n"] = {key:value for key,value in zip(h5f["hh_n_keys"].value,h5f["hh_n_values"].value)}
        kwargs_cB["n_bins"] = json.loads(h5f["n_bins"].value)
        kwargs_cB["classes"] = h5f["classes"].value
        kwargs_cB["n_classes"] = json.loads(h5f["n_classes"].value)
        kwargs_cB["bb_full"] = [h5f[name].value for name in json.loads(h5f["bb_full_names"].value)]

        try:
            mask_legend = {int(key):value for key,value in json.loads(h5f["mask_legend"].value).items()}
            mask_legend.update({value:key for key,value in mask_legend.items()})
        except KeyError:
            mask_legend = None

        try:
            clf_to_col = {int(key):value for key,value in json.loads(h5f["clf_to_col"].value).items()}
        except KeyError:
            clf_to_col = None

    return {"kwargs_cB":kwargs_cB,
            "kwargs_mk_clf":kwargs_mk_clf,
            "mask_legend":mask_legend,
            "clf_to_col":clf_to_col
           }


class S2cB(object):
    """

    """
    def __init__(self, cb_clf, mask_legend, clf_to_col, processing_tiles=11,logger=None):

        self.logger = logger or logging.getLogger(__name__)

        self.cb_clf = cb_clf
        self.mask_legend = mask_legend
        self.clf_to_col = clf_to_col
        self.processing_tiles = processing_tiles

        self.S2_MSI_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11',
                                'B12']

        unique_channel_list = []
        for clf_ids in cb_clf.mk_clf.classifiers_id:
            unique_channel_list += list(clf_ids)
        self.unique_channel_ids = np.unique(unique_channel_list)
        self.unique_channel_str = [self.S2_MSI_channels[ii] for ii in self.unique_channel_ids]

    def __call__(self, S2_img,target_resolution=None):
        if S2_img.target_resolution is None:
            #if target_resolution is None:
            #    raise ValueError("target_resolution should be given.")

            channel_ids = np.unique([item for sublist in self.cb_clf.mk_clf.classifiers_id_full for item in sublist])
            cb_channels = [self.cb_clf.mk_clf.id2name[channel_id] for channel_id in channel_ids]
            self.cb_clf.mk_clf.adjust_classifier_ids(full_bands=self.cb_clf.mk_clf.id2name,
                                                     band_lists=cb_channels)
            data = S2_img.image_subsample(channels=cb_channels,target_resolution=target_resolution)
            good_data = S2_img.nodata[target_resolution] == False

            mask_shape = [S2_img.metadata["spatial_samplings"][target_resolution][ii] for ii in ["NCOLS","NROWS"]]
            mask_array = np.zeros(mask_shape, dtype=np.float32)
            mask_conf = np.zeros(mask_shape, dtype=np.float32)

            if self.processing_tiles == 0:
                mask_array[good_data], mask_conf[good_data] = self.cb_clf.predict_and_conf(data[good_data,:])
            else:
                line_segs = np.linspace(0, mask_shape[0], self.processing_tiles, dtype=np.int)
                for ii, (i1, i2) in enumerate(zip(line_segs[:-1], line_segs[1:])):
                    self.logger.info("Processing lines segment %i of %i -> %i:%i" %
                                     (ii + 1, self.processing_tiles, i1, i2))


                    ma,mc = self.cb_clf.predict_and_conf(data[i1:i2,:,:][good_data[i1:i2,:],:])
                    maf = np.zeros(good_data[i1:i2,:].shape,dtype=np.float32)
                    mcf = np.zeros(good_data[i1:i2,:].shape,dtype=np.float32)
                    maf[good_data[i1:i2,:]],mcf[good_data[i1:i2,:]] = ma,mc
                    mask_array[i1:i2, :], mask_conf[i1:i2, :] = maf,mcf


                    #mask_array[i1:i2, :], mask_conf[i1:i2, :] = self.cb_clf.predict_and_conf(data[i1:i2,:,:])


                    #number_of_nans = np.sum(np.isnan(data[i1:i2,:,:]),axis=2)
                    #badv = number_of_nans > 0
                    #nodata = number_of_nans == data.shape[2]
                    #mask_array[i1:i2, :][badv] = -1 * number_of_nans[badv]
                    #mask_array[i1:i2, :][nodata] = np.NaN

            gc = S2_img.metadata["spatial_samplings"][target_resolution] if target_resolution is not None else None
            return S2Mask(S2_img=S2_img, mask_array=mask_array,clf_to_col=self.clf_to_col,
                          mask_legend=self.mask_legend, mask_confidence_array=mask_conf,
                          geo_coding=gc)

        else:
            if target_resolution is not None:
                raise ValueError("target_resolution should only be given if target_resolution=None for the S2 image.")

            self.cb_clf.mk_clf.adjust_classifier_ids(full_bands=S2_img.full_band_list,
                                                     band_lists=S2_img.band_list)
            if self.processing_tiles == 0:
                mask_array, mask_conf = self.cb_clf.predict_and_conf(S2_img.data)
            else:
                mask_array = np.zeros(S2_img.data.shape[:2], dtype=np.int)
                mask_conf = np.zeros(S2_img.data.shape[:2], dtype=np.float16)

                line_segs = np.linspace(0, S2_img.data.shape[0], self.processing_tiles, dtype=np.int)
                for ii, (i1, i2) in enumerate(zip(line_segs[:-1], line_segs[1:])):
                    self.logger.info("Processing lines segment %i of %i -> %i:%i" % (ii + 1, self.processing_tiles, i1, i2))
                    mask_array[i1:i2, :], mask_conf[i1:i2, :] = self.cb_clf.predict_and_conf(S2_img.data[i1:i2, :, :])


            bad_values = np.sum(np.isnan(S2_img.data),axis=2) != 0
            mask_array = np.array(mask_array,dtype=np.float32)
            mask_array[bad_values] = np.NaN
            mask_conf[bad_values] = np.NaN

            return S2Mask(S2_img=S2_img, mask_array=mask_array,clf_to_col=self.clf_to_col,
                          mask_legend=self.mask_legend, mask_confidence_array=mask_conf)

