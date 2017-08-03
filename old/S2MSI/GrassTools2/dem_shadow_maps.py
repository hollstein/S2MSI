#! /usr/bin/env python

import os
import sys
from subprocess import check_output
from os.path import basename
import gdal
import numpy as np
import json
from tempfile import mkdtemp
import argparse
from shutil import rmtree


def run_r_sun(dem,extent,day,time,return_data=["incidout"]):
    grass.run_command("g.region",rows=dem.shape[0],cols=dem.shape[1],**extent)

    gg_dem = garray.array()
    gg_dem[:,:] = dem[:,:]
    gg_dem.write(mapname="dem",overwrite=True)

    grass.run_command("r.slope.aspect",overwrite=True, elevation="dem",aspect="aspect",slope="slope")

    grass.run_command("r.sun",overwrite=True,verbose=True,day=day,time=time,step=1,dist=1.0,
                      elevation="dem",incidout="incidout",aspect="aspect",slope="slope",
                      beam_rad="beam_rad",diff_rad="diff_rad",refl_rad="refl_rad",
                      )

    res = {"day":day,"time":time}
    bf = garray.array()
    for mp in ["incidout","dem","aspect","slope","beam_rad","diff_rad","refl_rad"]:
        if mp in return_data:
            bf.read(mp)
            res[mp] = np.copy(bf[:,:])
    return res


if __name__ == "__main__":
    sv = sys.version_info
    major, minor = sv.major,sv.minor    
    if (major, minor) != (2, 7):
        print("This program requires at python 2.7, this version is %i.%i. -> quit here." % (major, minor))
        sys.exit(2)
        
    parser = argparse.ArgumentParser(prog='',description='')
    
    parser.add_argument("-e", "--epsg", help="asd", action="store", type=str,required=True)
    parser.add_argument("-g", "--grass_bin", help="asd", action="store", type=str,default='grass70',required=False)
    parser.add_argument("-d", "--day", help="asd", action="store", type=int,required=True)
    parser.add_argument("-t", "--time", help="asd", action="store", type=float,required=True)
    parser.add_argument("-i", "--input", help="asd", action="store", type=str,required=True)

    parser.add_argument("-o", "--outputs", help="asd", action="store", type=str,required=False,
                        default="slope,incidout")

    args = parser.parse_args()
    args.outputs = args.outputs.split(",")

    gisbase = check_output("%s --config path" % args.grass_bin,shell=True).strip('\n')
    sys.path.append(os.path.join(gisbase, "etc", "python"))
    gisdb = mkdtemp()
    mapset,location = 'PERMANENT',"LOC"
    location_path = os.path.join(gisdb, location)
    os.environ.update({'LANG':"en_US",'LOCALE':"C",'GISBASE':gisbase,"GISDBASE":gisdb,
                       'LD_LIBRARY_PATH':os.getenv('LD_LIBRARY_PATH')})
    check_output("%s -c epsg:%s -e %s" % (args.grass_bin,args.epsg,location_path),shell=True)


    import grass.script as grass
    import grass.script.setup as gsetup
    import grass.script.array as garray

    tmp_dir_grass = gsetup.init(gisbase, gisdb, location, mapset)

    ds = gdal.Open(args.input)
    dem = np.array(ds.GetRasterBand(1).ReadAsArray())

    fl_end = basename(args.input).split(".")[-1]

    ds = gdal.Open(args.input)
    dem = np.array(ds.GetRasterBand(1).ReadAsArray())

    with open(args.input.replace("." + fl_end,"_extent.json"),"r") as fl:
        extent = json.load(fl)

    res = run_r_sun(dem=dem,extent=extent,day=args.day,time=args.time,return_data=args.outputs)

    fl_pat = args.input.replace("." + fl_end,"_%s." + fl_end)
    driver = gdal.GetDriverByName("gtiff")
    for rr,data in res.items():
        if rr in args.outputs:
            fn = fl_pat % rr
            print("Write: %s" % fn)
            dset = driver.Create(fn, dem.shape[0], dem.shape[0], 1, gdal.GDT_Float32)
            bnd = dset.GetRasterBand(1)
            bnd.WriteArray(np.array(data,dtype=np.float32))
            bnd.FlushCache()
            del dset

    rmtree(tmp_dir_grass,ignore_errors=True)
    rmtree(gisdb,ignore_errors=True)
