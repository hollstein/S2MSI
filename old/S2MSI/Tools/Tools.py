import os
import errno
from os.path import dirname, exists,isfile
from resource import getrusage, RUSAGE_SELF
import tkinter as tk
import numpy as np
import gdal
import glymur
from gdalconst import GA_ReadOnly


def fl(inp, dec):
    """ Floor inp by number of decimals as given by dec

    :param inp: input, float
    :param dec: integer, number of decimals
    :return: floored value
    """
    return np.floor(inp*10**dec)/10**dec


def cl(inp, dec):
    """ Ceil inp by number of decimals as given by dec

    :param inp: input, float
    :param dec: integer, number of decimals
    :return: ceiled value
    """
    return np.ceil(inp*10**dec)/10**dec


def write_image_gdal(data,filename,numpy_type=np.int16,gdal_type=gdal.GDT_Int16,driver="GTIFF"):
    """ write numpy array to image file

    :param data: numpy array
    :param filename: destination filename
    :param numpy_type: either None or numpy type, if None data is written as given, if given, numpy
                       array conversion is performed before writing, should be consistent with gdal_type
    :param gdal_type: GDAL type, e.g. gdal.GDT_Int16
    :param driver: string with internal gdal name for file writing
    :return: None
    """
    driver = gdal.GetDriverByName(driver)
    dset = driver.Create(filename, data.shape[0], data.shape[1], 1,gdal_type)
    bnd = dset.GetRasterBand(1)
    if numpy_type is not None:
        bnd.WriteArray(np.array(data,dtype=numpy_type))
    else:
        bnd.WriteArray(data)
    bnd.FlushCache()
    del dset


def write_jp2_image(fn_out,data,scale=None,dtype=np.uint16):
    """ write numpy array to jp2 file, apply scaling and type conversion

    :param fn_out: str, filename
    :param data: numpy array
    :param scale:None or scalar type
    :param dtype:output type
    :return:fn_out is passed back
    """
    if scale is None:
        out = np.array(data, dtype=dtype)
    else:
        out = np.array(data * scale, dtype=dtype)

    _ = glymur.Jp2k(fn_out,data=out)

    return fn_out


def read_image_gdal(filename):
    """ read image file using gdal, driver is determined by gdal

    :param filename:
    :return: numpy array with image data
    """
    if isfile(filename):
        ds = gdal.Open(filename,GA_ReadOnly)
    else:
        raise FileNotFoundError(filename)
    if ds is not None:
        return np.array(ds.GetRasterBand(1).ReadAsArray())
    else:
        raise ValueError("GDAL was unable to open file:%s" % filename)


def mkdir_p(path_inp):
    """ same as mkdir -p path on GNU Linux

    :param path_inp:
    :return: None
    """
    try:
        os.makedirs(path_inp)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path_inp):
            pass
        else:
            raise


class StdRedirector(object):
    def __init__(self, widget, gui=None,logfile=None):
        self.widget = widget
        self.counter = 0
        self.gui = gui
        if logfile is not None:
            mkdir_p(dirname(logfile))
            mode = "w" if exists(logfile) is False else "a"
            self.logfile = open(logfile,mode)
            self.logging = True
        else:
            self.logging = False

    def write(self, string):
        self.counter += 1
        rr = getrusage(RUSAGE_SELF).ru_maxrss / 1024. ** 2
        ss = str(string).rstrip()

        if len(ss)>0:
            ss = "mem: %.2f GB : %s" % (rr,ss)
            self.widget.insert(tk.END,ss+"\n")
            self.widget.see(tk.END)
            if self.logging:
                self.logfile.write(ss+"\n")
                self.logfile.flush()

        if self.gui.update is not None:
            pass
            #self.gui.update()update()

    def __close_logfile__(self):
        if self.logging:
            self.logfile.close()

    def __del__(self):
        if self.logging:
            self.__close_logfile__()

    def flush(self):
        pass


class StdoutToList(object):
    def __init__(self, ):
        self.lines = []

    def write(self, string):
        self.lines.append(string)

    def flush(self):
        pass


def interpolate_nans(data,mode="return"):
    mask = np.isnan(data)
    if mode == "inplace":
        if np.sum(~mask)>0:
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    elif mode == "return":
        data_loc = np.copy(data)
        if np.sum(~mask)>0:
            data_loc[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        return data_loc
    else:
        raise ValueError("Mode [%s] not implemented." % mode)
    return data


class ToolTip:
    def __init__(self, master, text='Your text here', delay=150, **opts):
        self.master = master
        self._opts = {
            'anchor': 'center',
            'bd': 1,
            'bg': 'lightyellow',
            'delay': delay,
            'fg': 'black',
            'follow_mouse': 0,
            'font': None,
            'justify': 'left',
            'padx': 4,
            'pady': 2,
            'relief': 'solid',
            'state': 'normal',
            'text': text,
            'textvariable': None,
            'width': 0,
            'wraplength': 500
        }
        self.configure(**opts)
        self._tipwindow = None
        self._id = None
        self._id1 = self.master.bind("<Enter>", self.enter, '+')
        self._id2 = self.master.bind("<Leave>", self.leave, '+')
        self._id3 = self.master.bind("<ButtonPress>", self.leave, '+')
        self._follow_mouse = 0
        if self._opts['follow_mouse']:
            self._id4 = self.master.bind("<Motion>", self.motion, '+')
            self._follow_mouse = 1

    def configure(self, **opts):
        for key in opts:
            if key in self._opts:
                self._opts[key] = opts[key]
            else:
                keyerror = 'KeyError: Unknown option: "%s"' % key
                raise keyerror

    def enter(self, event=None):  # handles <Enter> event
        self._schedule()

    def leave(self, event=None):  # handles <Leave> event
        self._unschedule()
        self._hide()

    def motion(self, event=None):  # handles <Motion> event
        if self._tipwindow and self._follow_mouse:
            x, y = self.coords()
            self._tipwindow.wm_geometry("+%d+%d" % (x, y))

    def _schedule(self):
        self._unschedule()
        if self._opts['state'] == 'disabled':
            return
        self._id = self.master.after(self._opts['delay'], self._show)

    def _unschedule(self):
        idd = self._id
        self._id = None
        if idd:
            self.master.after_cancel(idd)

    def _show(self):
        if self._opts['state'] == 'disabled':
            self._unschedule()
            return
        if not self._tipwindow:
            self._tipwindow = tw = tk.Toplevel(self.master)
            # hide the window until we know the geometry
            tw.withdraw()
            tw.wm_overrideredirect(1)

            if tw.tk.call("tk", "windowingsystem") == 'aqua':
                tw.tk.call("::tk::unsupported::MacWindowStyle", "style", tw._w, "help", "none")

            self.create_contents()
            tw.update_idletasks()
            x, y = self.coords()
            tw.wm_geometry("+%d+%d" % (x, y))
            tw.deiconify()

    def _hide(self):
        tw = self._tipwindow
        self._tipwindow = None
        if tw:
            tw.destroy()

    def coords(self):
        # The tip window must be completely outside the master widget;
        # otherwise when the mouse enters the tip window we get
        # a leave event and it disappears, and then we get an enter
        # event and it reappears, and so on forever :-(
        # or we take care that the mouse pointer is always outside the tipwindow :-)
        tw = self._tipwindow
        twx, twy = tw.winfo_reqwidth(), tw.winfo_reqheight()
        w, h = tw.winfo_screenwidth(), tw.winfo_screenheight()
        # calculate the y coordinate:
        if self._follow_mouse:
            y = tw.winfo_pointery() + 20
            # make sure the tipwindow is never outside the screen:
            if y + twy > h:
                y = y - twy - 30
        else:
            y = self.master.winfo_rooty() + self.master.winfo_height() + 3
            if y + twy > h:
                y = self.master.winfo_rooty() - twy - 3
        # we can use the same x coord in both cases:
        x = tw.winfo_pointerx() - twx / 2
        if x < 0:
            x = 0
        elif x + twx > w:
            x = w - twx
        return x, y

    def create_contents(self):
        opts = self._opts.copy()
        for opt in ('delay', 'follow_mouse', 'state'):
            del opts[opt]
        label = tk.Label(self._tipwindow, **opts)
        label.pack()


