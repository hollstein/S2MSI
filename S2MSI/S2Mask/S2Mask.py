from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
import glymur
from scipy.ndimage import zoom
from PIL import Image


class S2Mask(object):
    def __init__(self, S2_img, mask_array, mask_legend,clf_to_col=None,mask_confidence_array=None,
                 geo_coding=None):
        """
        Sentinel-2 MSI masking object

        :param S2_img:instance of S2_MSI_Image
        :param mask_array: masking result, numpy array
        :param mask_legend: dictionary of mask_id:mask_name
        """
        self.metadata = deepcopy(S2_img.metadata)
        self.clf_to_col = clf_to_col
        if geo_coding is None and S2_img.target_resolution is not None:
            self.geo_coding = self.metadata["spatial_samplings"][S2_img.target_resolution]
        else:
            self.geo_coding = geo_coding
        self.mask_array = mask_array
        self.mask_confidence_array = mask_confidence_array
        self.mask_legend = {key: value for key, value in mask_legend.items()}
        self.mask_legend_inv = {value: key for key, value in mask_legend.items()}


    def mk_mask_at_spatial_scales(self,flags,samplings):
        """ use s2msi mask object to create boolean mask at desired spatial sampling

        :param options:
        :param flags: list of flag names in s2msk which are True for mask
        :param samplings: list of desired spatial scales
        :param s2msk:
        :return:
        """

        assert abs(self.geo_coding["XDIM"]) == abs(self.geo_coding["YDIM"])
        tr = float(abs(self.geo_coding["XDIM"]))

        mask = {tr: np.logical_or.reduce([self.mask_array == self.mask_legend[flag] for flag in flags])}

        for key in samplings:
            if key not in mask:
                zoom_fac = tr/key
                mask[key] = zoom(mask[tr], zoom=zoom_fac, order=0)
        return mask

    def mask_rgb_array(self,dtype=np.float16):
        mask_rgb = np.zeros(list(self.mask_array.shape) + [3], dtype=dtype)
        for key, col in self.clf_to_col.items():
            if np.issubdtype(dtype,np.uint8):
                col = np.array(np.array(self.clf_to_col[key])*255,dtype=np.uint8)
            else:
                col = self.clf_to_col[key]

            mask_rgb[self.mask_array == key, :] = col

        return mask_rgb

    def export_mask_rgb(self, fn_img, rgb_img):
        mask_rgb = self.mask_rgb_array()

        dpi = 100.0
        fig = plt.figure(figsize=np.array(rgb_img.shape[:2]) / dpi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax = plt.subplot()
        ax.imshow(mask_rgb, interpolation="none")
        ax.set_axis_off()
        plt.savefig(fn_img, dpi=dpi)
        fig.clear()
        plt.close(fig)

    def export_mask_blend(self, fn_img, rgb_img,alpha=0.6,plot_method="PIL",quality=60):

        if alpha > 0.0:
            mask_rgb = self.mask_rgb_array(dtype=np.uint8)
            zoom_fac = np.array([s1 / s2 for s1,s2 in zip(rgb_img.shape,mask_rgb.shape)])

            if (zoom_fac != [1.0,1.0,1.0]).all():
                mask_rgb = zoom(input=mask_rgb,order=0,zoom=zoom_fac)

        if plot_method == "PIL":
            if alpha > 0.0:
                img_rgb = Image.fromarray(rgb_img)
                img_msk = Image.fromarray(mask_rgb)
                img = Image.blend(img_rgb,img_msk,alpha)
                img.save(fn_img,quality=quality)
            else:
                img_rgb = Image.fromarray(rgb_img)
                img_rgb.save(fn_img,quality=quality)

        elif plot_method == "mpl":
            dpi = 100.0
            fig = plt.figure(figsize=np.array(rgb_img.shape[:2]) / dpi)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            ax = plt.subplot()
            # RGB image of scene as background
            ax.imshow(rgb_img, interpolation="none")
            # mask colors above, but transparent
            ax.imshow(mask_rgb, interpolation="none", alpha=alpha)
            ax.set_axis_off()
            plt.savefig(fn_img, dpi=dpi)
            fig.clear()
            plt.close(fig)
        else:
            raise ValueError("Plot method: %s not implemented." % str(plot_method))

    def export_confidence_to_jpeg2000(self, fn_img):
        if self.mask_confidence_array is not None:

            mask_confidence_array = np.copy(self.mask_confidence_array)
            mask_confidence_array -= np.nanmin(mask_confidence_array)
            mask_confidence_array /= np.nanmax(mask_confidence_array)
            mask_confidence_array *= 100
            _ = glymur.Jp2k(fn_img, data=np.array(mask_confidence_array, dtype=np.uint8))

    def export_to_jpeg200(self, fn_img, fn_metadata=None, delimiter=","):
        if fn_img is not None:
            _ = glymur.Jp2k(fn_img, data=np.array(self.mask_array, dtype=np.uint8))

        if fn_metadata is not None:
            with open(fn_metadata, 'w') as outfile:
                for key, value in sorted(self.metadata.items()):
                    value_str = " ".join(str(value).replace("\n","").split())
                    if len(value_str)<100:
                        outfile.write(str(key) + delimiter + value_str + '\n')
                for key, value in sorted(self.geo_coding.items()):
                    value_str = " ".join(str(value).replace("\n","").split())
                    if len(value_str)<100:
                        outfile.write(str(key) + delimiter + value_str + '\n')

