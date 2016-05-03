import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
import logging
from time import time
from itertools import product
from numba import jit
from scipy.interpolate import bisplrep,bisplev


class SL(object):
    def __init__(self, mx, extend=1.1):
        """ returns slices for given center Y and radius D but makes sure borders are within [0,mx]

        :param mx: maximum allowed index
        :param extend: multiply D by extend to over- oder under-sample the slice
        :return: function like object, to be called like f((Y,D))
        """
        self.mx = mx
        self.extend = extend

    def __call__(self, yd):
        """ to be called like (yy,dd)

        :param yd: tuple,list with two elements (yy,dd)
        :return: slice
        """
        yy, dd = yd
        i0 = int(yy - self.extend * dd)
        i1 = int(yy + self.extend * dd)
        if i0 < 0:
            i0 = 0
        if i1 > self.mx:
            i1 = self.mx
        return slice(i0, i1)


@jit(nopython=True)
def fill_nan(arr):
    for ix in range(arr.shape[0]):
        bf = np.nan
        for iy in range(arr.shape[1]):
            if np.isfinite(arr[ix,iy]):
                bf = arr[ix,iy]
            if np.isfinite(bf) and np.isnan(arr[ix,iy]):
                arr[ix,iy] = bf

        bf = np.nan
        for iy in range(arr.shape[1] - 1,-1,-1):
            if np.isfinite(arr[ix,iy]):
                bf = arr[ix,iy]
            if np.isfinite(bf) and np.isnan(arr[ix,iy]):
                arr[ix,iy] = bf

    for iy in range(arr.shape[1]):
        bf = np.nan
        for ix in range(arr.shape[0]):
            if np.isfinite(arr[ix,iy]):
                bf = arr[ix,iy]
            if np.isfinite(bf) and np.isnan(arr[ix,iy]):
                arr[ix,iy] = bf

        bf = np.nan
        for ix in range(arr.shape[0] - 1,-1,-1):
            if np.isfinite(arr[ix,iy]):
                bf = arr[ix,iy]
            if np.isfinite(bf) and np.isnan(arr[ix,iy]):
                arr[ix,iy] = bf


def itt2d(shape, dd, extend=1.00):
    """Separate 2d array into 2D blocks based in iterator returning slice tuple

    returns an iterator with ((slice,slice),...,(slice,slice)) which separates an 2D array with [shape]
    into dd^2 smaller blocks. Blocks can over- and under-sampled using the extend factor

    :param shape: shape of 2D array
    :param dd: number of desired blocks in each axis
    :param extend: over-sampling factor
    :return: iterator with ((slice,slice),...,(slice,slice))
    """

    slx = SL(shape[0], extend=extend)
    sly = SL(shape[1], extend=extend)

    vv = np.linspace(0, shape[0], dd + 1)
    Vx = np.array(0.5 * (vv[:-1] + vv[1:]), dtype=np.int)
    Dx = np.array(0.5 * (vv[1:] - vv[:-1]), dtype=np.int)
    vv = np.linspace(0, shape[1], dd + 1)
    Vy = np.array(0.5 * (vv[:-1] + vv[1:]), dtype=np.int)
    Dy = np.array(0.5 * (vv[1:] - vv[:-1]), dtype=np.int)

    return product(map(slx, zip(Vx, Dx)), map(sly, zip(Vy, Dy)))


def inpaint(array, sigma=None, logger=None, update_in_place=False, fill_corners=True, fill_remaining=True,
            extend=1.15, max_allowed_good_points=50000,mask=None,max_interpolation_points=1500000,
            interpolation_method="griddata",post_processing=None):
    """ Fill nan's in array with meaningful data.

    :param array: array to be inpainted
    :param sigma: if not None, the final result will be smoothed by a gaussian kernel with [sigma]
    :param logger: logger for logging
    :param update_in_place: if True, perform inpainting directly on array, if False, return result
    :param fill_corners: True / False, assign values to corner blocks to increase number of pixels which can be
                         interpolated wel
    :param fill_remaining: True / False, if True, all remaining nan's after interpolation are replaced with nearest
                           neighbour median value, if set, no nan's remain in the image, is so, a ValueError is raised
    :param extend: oversampling parameter for block splitting of array, should be chosen between 0.8 and 1.2
    :param max_allowed_good_points: max input points for griddata, determines performance and number of block splits
    :param max_interpolation_points: either None or integer, max number of points which are interpolated by griddata
                                     if too much, might eat up all memory
    :param mask, either None or boolean array of shape array, only those nan's in array where mask is True will
           filled with meaningful data
    :param interpolation_method, None,"griddata","spline"
    :param post_processing: None,"spline","gaussian_filter"
    :return: array filled with meaningful data if update_in_place=False, otherwise None
    """

    t0 = time()
    logger = logger or logging.getLogger(__name__)

    # arr is the working array, either get copy or use the given one
    if update_in_place is True:
        arr = array
    else:
        logger.info("Create new array of shape: %s" % str(array.shape))
        arr = np.copy(array)

    shape = arr.shape
    # meshgrid to get array indices right for
    yv, xv = np.meshgrid(np.arange(arr.shape[0],dtype=np.int16),  # this way memory is saved
                         np.arange(arr.shape[1],dtype=np.int16),copy=False)

    good_results = np.isfinite(arr)  # usable points for interpolation
    bad_results = np.isnan(arr) if mask is None else np.logical_and(np.isnan(arr),mask)

    logger.info("bad/total ratio: %.5f" % (np.sum(bad_results) / np.prod(shape)))

    # flat index tables for good and bad data
    xvb = xv[bad_results]
    yvb = yv[bad_results]
    """
    griddata should not get too many input points, otherwise the memory use and performance
    will be horrible, the main idea behind the following mess is to intelligently reduce the number
    of input points while maintaining good performance, the algorithm starts to 2d split the working array
    until the max number of good values over all splits is lower then max_allowed_good_points
    """
    dd_split = 1  # initialize number of splits
    if interpolation_method is not None:
        for ii in range(1, np.min(shape)):
            max_good_points = np.max([np.sum(good_results[sx, sy]) for sx, sy in itt2d(shape, ii)])
            logger.info((ii,max_good_points))
            if max_good_points < max_allowed_good_points:
                dd_split = ii
                break
        logger.info("Splitting array into %i^2 parts." % dd_split)
    """
    now loop over all 2d segments and interpolate using griddata
    """
    for sx, sy in itt2d(good_results.shape, dd_split, extend=extend):
        xvg = xv[sx, sy][good_results[sx, sy]]  # good x indices for block
        yvg = yv[sx, sy][good_results[sx, sy]]  # good y indices for block
        values = arr[xvg, yvg]  # construct input fields for griddata

        if interpolation_method is None:
            pass
        elif interpolation_method == "splines":
            tck = bisplrep(x=xvg,y=yvg,z=values,kx=5, ky=5)
            xmm = np.min(xvg),np.max(xvg)
            ymm = np.min(yvg),np.max(yvg)
            rr = bisplev(x=np.arange(xmm[0],xmm[1]),
                         y=np.arange(ymm[0],ymm[1]),
                         tck=tck)
            arr[xmm[0]:xmm[1],
                ymm[0]:ymm[1]] = rr

        elif interpolation_method == "griddata":
            if len(values) < 1:
                pass
            else:
                points = np.squeeze(np.dstack((xvg, yvg)))  # construct input fields for griddata
                """
                attempt for interpolation should only be for bad values which lie in the current block
                """
                bb = ((np.min(xvg) < xvb) * (xvb < np.max(xvg)) * (np.min(yvg) < yvb) * (yvb < np.max(yvg)))
                xi = np.squeeze(np.dstack((xvb[bb], yvb[bb])))
                """
                attempt interpolation for block only if some bad values lie in this block
                """
                if len(xi) > 0:
                    """
                    if within a block the corners are not filled with data, then fewer points can be interpolated
                    since for each interpolation point, sufficiently neighbours need to be found, if fill_corners is
                    True we check the corners for nans, if they are nan we search the increasing 2d area for valid
                    points until we found something which is not nan -> the corner values are then replaced with the
                    nanmedian of the actual search window
                    """
                    if fill_corners is True:
                        sub_shape = arr[sx, sy].shape
                        nx, ny = sub_shape
                        points_corners = []
                        values_corners = []
                        # suppress warnings from nanmedian if only nan's are there, which is ok here
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            box_x = [np.min(points[:,0]),np.max(points[:,0])]
                            box_y = [np.min(points[:,1]),np.max(points[:,1])]
                            for ix, iy in product(box_x,box_y):
                                aa = arr[ix, iy]

                                if np.isnan(aa):  # only if corner is empty
                                    for jj in range(2, np.min(sub_shape), 2):  #incease search window fast to find stuff
                                        rbf = arr[sx, sy][np.max([ix - jj, 0]):np.min([ix + jj, nx]),
                                                          np.max([iy - jj, 0]):np.min([iy + jj, ny])]
                                        rpl = np.nanmean(rbf)
                                        if np.isfinite(rpl):
                                            points_corners.append([ix, iy])
                                            values_corners.append(rpl)
                                            break  # we found something -> stop increasing search block

                        if len(values_corners) > 0:  # if we found new values->modify griddata input values accordingly
                            logger.info("Adding corners for %s,%s, pt: %s, values: %s" %
                                        (sx, sy, points_corners, values_corners))
                            points = np.vstack((points, points_corners))
                            values = np.hstack((values, values_corners))
                            # not only inputs, also new bad values which lie in the new hull should be added
                            xvg = xv[sx, sy]
                            yvg = yv[sx, sy]
                            bb = ((np.min(xvg) < xvb) * (xvb < np.max(xvg)) * (np.min(yvg) < yvb) * (yvb < np.max(yvg)))
                            xi = np.squeeze(np.dstack((xvb[bb], yvb[bb])))

                    # finally, interpolate stuff

                    if max_interpolation_points is None:
                        grd = griddata(points=points, values=values, xi=xi, fill_value=np.nan,method="linear")
                        good_grd = np.isfinite(grd)  # points for which interpolation succeeded
                        arr[xvb[bb][good_grd], yvb[bb][good_grd]] = grd[good_grd]  # copy interpolated results
                    else:
                        nn = xi.shape[0]
                        max_interpolation_points = 1500000
                        nn_blocks = np.int(np.ceil(nn / max_interpolation_points))
                        ss = np.linspace(0,nn,nn_blocks,dtype=np.int,endpoint=True)
                        for ii,(j1,j2) in enumerate(zip(ss[:-1],ss[1:])):
                            logger.info("%i / %i ,%i,%i" % (ii,nn_blocks,j1,j2))
                            grd = griddata(points=points, values=values, xi=xi[j1:j2], fill_value=np.nan,method="linear")
                            good_grd = np.isfinite(grd)  # points for which interpolation succeeded
                            arr[xvb[bb][j1:j2][good_grd], yvb[bb][j1:j2][good_grd]] = grd[good_grd]  # copy results
        else:
            raise ValueError("interpolation_method:%s not implemented" % interpolation_method)

    # count still remaining bad points
    bad_results = np.isnan(arr) if mask is None else np.logical_and(np.isnan(arr),mask)
    logger.info("bad/total ratio: %.5f" % (np.sum(bad_results) / np.prod(shape)))

    if fill_remaining == "nearest_neighbour":
        """
        remaining nan's are filled by searching for every point the neighbourhood vor valid points and
        replace the actual nan with the median over the search window, this is slow
        """
        nx, ny = shape
        # suppress warnings from nanmedian if only nan's are there, which is ok here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ix, iy in zip(xv[bad_results], yv[bad_results]):
                for jj in range(2, np.min([nx, ny]), 2):
                    rbf = arr[np.max([ix - jj, 0]):np.min([ix + jj, nx]),
                              np.max([iy - jj, 0]):np.min([iy + jj, ny])]
                    rpl = np.nanmedian(rbf)
                    if np.isfinite(rpl):
                        arr[ix, iy] = rpl
                        break  # we found something -> stop increasing search block
    elif fill_remaining == "broom":
        fill_nan(arr)
    elif fill_remaining == "median":
        bf = np.nanmean(arr)
        bad_results = np.isnan(arr) if mask is None else np.logical_and(np.isnan(arr),mask)
        arr[bad_results] = bf
        print(bf)
    elif fill_remaining is None:
        pass
    else:
        raise ValueError("fill_remaining=%s not implemented" % str(fill_remaining))

    if fill_remaining is not None:
        bad_results = np.isnan(arr) if mask is None else np.logical_and(np.isnan(arr),mask)
        n_bad = np.sum(bad_results)
        logger.info("bad/total ratio: %.5f" % (n_bad / np.prod(shape)))
        if n_bad > 0:
            raise ValueError("Field still contains bad %i results." % n_bad)

    if post_processing is None:
        pass
    elif post_processing == "gaussian_filter":
        logger.info("Post processing using gaussian filter.")
        if sigma is not None:
            """
            is sigma is not none, we can apply some gaussian smoothing to make things more smoth
            """
            arr[:] = gaussian_filter(input=np.array(arr,dtype=np.float32), sigma=sigma)
        else:
            raise ValueError("When using gaussian_filter, sigma should be supplied")
    elif post_processing == "spline":
        logger.info("Post processing using splines.")
        idx_val = np.logical_and(mask == True,np.isfinite(arr))  # need '==' here
        idx_sel = np.random.choice(np.arange(0,np.sum(idx_val)),
                                   np.min([max_allowed_good_points,np.sum(idx_val)]),replace=False)
        tck = bisplrep(x=xv[idx_val][idx_sel],
                       y=yv[idx_val][idx_sel],
                       z=arr[idx_val][idx_sel],kx=5, ky=5)
        arr[:] = bisplev(x=np.arange(0,arr.shape[0]),y=np.arange(0,arr.shape[0]),tck=tck)
    else:
        raise ValueError("post_processing=%s not implemented" % post_processing)

    if mask is not None:
        logger.info("Apply mask.")
        arr[mask == False] = np.nan  # need '==' here !

    logger.info("Inpainting runtime: %.2fs" % (time() - t0))
    if update_in_place is False:
        return arr




if __name__ == "__main__":

    shape = (10980,10980)
    #shape = (1800,1800)
    s2aot = np.zeros(shape,dtype=np.float32)
    s2aot[:,:] = np.NAN

    for ii in range(600000):
        i1,i2 = np.random.randint(shape[0]),np.random.randint(shape[1])
        s2aot[i1,i2] = 0.5+0.1*np.sin(i1/1500)+0.1*np.cos(i2/150)

    s2aot[s2aot<0.0] = np.nan

    mask = None
    #mask = np.empty(shape,dtype=np.bool)
    #mask[:] = False
    #mask[500:9000,1000:6000] = True

    #imshow(s2aot[::10,::10],vmin=np.nanmin(s2aot),vmax=np.nanmax(s2aot))
    #colorbar()

    arr = inpaint(s2aot,update_in_place=False,
                  interpolation_method=["splines","griddata",None][1],
                  post_processing = ["spline",None,"gaussian_filter"][1],
                  fill_remaining=["broom",None][0],
                  sigma=10,
                  extend=1.01,
                  max_allowed_good_points=80000,
                  max_interpolation_points=2500000,
                  mask=mask)
