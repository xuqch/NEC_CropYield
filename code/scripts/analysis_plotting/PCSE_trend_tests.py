import numpy as np
import matplotlib.pyplot as plt
import os, sys
import xarray as xr
import dask.array as da
from scipy.stats import linregress
from numba import jit  # Speedup for python functions
from matplotlib import colors
import cartopy.crs as ccrs
from pylab import rcParams
import matplotlib
import pymannkendall as mk
from joblib import Parallel, delayed
import multiprocessing

### Plot settings
font = {'family': 'DejaVu Sans'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

import pymannkendall as mk


def mk_test_s(cropland, ilon, ilat, j, i):
    """
    MK检验
    """
    crop = cropland.sel(lon=ilon, lat=ilat).to_array()
    slope = crop.values[0, 0]
    # p = crop.values[0, 0]
    if crop.values[0, 0] > -9999:
        slope = mk.original_test(crop.values[0, :], alpha=0.01).slope
        # p = mk.original_test(crop.values[0,:],alpha=0.01).p
        # print(p)
    return slope


def trend_s(cropland, trend_land):
    max_cpu = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count()
    num_cores = multiprocessing.cpu_count()
    p_land = trend_land
    for i, lon in enumerate(cropland.lon.values):
        # print("now at location: lon-->%s" % (cropland.lon.values[i]))
        crop = Parallel(n_jobs=num_cores)(
            delayed(mk_test_s)(cropland, lon, lat, j, i) for j, lat in enumerate(cropland.lat.values))
        for j, lat in enumerate(cropland.lat.values):
            if crop[j] > -9999:
                # print(crop[j])
                trend_land['TAGP'][j, i] = crop[j]
    return trend_land


def mk_test_p(cropland, ilon, ilat, j, i):
    """
    MK检验
    """
    crop = cropland.sel(lon=ilon, lat=ilat).to_array()
    # slope = crop.values[0, 0]
    p = crop.values[0, 0]
    if crop.values[0, 0] > -9999:
        # slope = mk.original_test(crop.values[0, :], alpha=0.01).slope
        p = mk.original_test(crop.values[0, :], alpha=0.01).p
        # print(p)
    return p


def trend_p(cropland, trend_land):
    max_cpu = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count()
    num_cores = multiprocessing.cpu_count()
    p_land = trend_land
    for i, lon in enumerate(cropland.lon.values):
        # print("now at location: lon-->%s" % (cropland.lon.values[i]))
        crop = Parallel(n_jobs=num_cores)(
            delayed(mk_test_p)(cropland, lon, lat, j, i) for j, lat in enumerate(cropland.lat.values))
        for j, lat in enumerate(cropland.lat.values):
            if crop[j] > -9999:
                # print(crop[j])
                trend_land['TAGP'][j, i] = crop[j]
    return trend_land


def save_xarray(outpath, arr, rid):
    """
    Save output
    """
    arr["lat"].attrs["units"] = "degrees_north"
    arr["lon"].attrs["units"] = "degrees_east"
    arr.to_netcdf(os.path.join(outpath, '%s.nc' % (rid)),
                  encoding={'TAGP': {'dtype': 'float32',
                                     'zlib': True,
                                     'complevel': 6,
                                     '_FillValue': -9999.0}})


if __name__ == '__main__':
    import glob, os, shutil

    #
    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop)
    trend_land = xr.where(crop >= 0, np.nan, crop)
    trend_land = trend_land.rename({'crop': 'TAGP'})

    '''
    rice:0, maize:1, soybean:2
    -'''
    pathin = "/tera01/xuqch3/PCSE/test_1/data/output/"
    # tests = ['test1', 'test2', 'test3', 'test4', 'test5']
    tests = ['test_1']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    names = ['rice', 'maize', 'soybean']

    years = np.arange(2015, 2101)
    outputshape = (len(years), len(crop.lat), len(crop.lon))
    for name in names:
        print(name, ">>")
        for ssp in ssps:
            print(ssp)
            filename = ("/tera04/zhwei/PCSE/data/output/sensitivity/default/%s_output_%s_default.nc" % (name, ssp))
            ds = xr.open_dataset(filename)
            TAGP = xr.Dataset({'TAGP': (('time', 'lat', 'lon'), np.full(outputshape, np.nan))},
                              coords={'time': (('time'), years),
                                      'lat': (('lat'), ds.lat.values),
                                      'lon': (('lon'), ds.lon.values),
                                      })
            for i, year in enumerate(years):
                Start = "20150101"
                Sdate = str(str(year) + "0415")
                S = len(ds.sel(time=slice(Start, Sdate)).time) - 1
                m = S + 130
                TAGP["TAGP"][i, :, :] = ds[TAGP][m, :, :]
                TAGP.to_netcdf(os.path.join(pathin, "%s_output_%s_default_max.nc" % (name, ssp)))

    for name in names:
        print(name, ">>")
        for ssp in ssps:
            print(ssp)
            filename = ("%s/%s_output_%s_default_max.nc" % (pathin, name, ssp))
            pathout = ("%s" % (pathin))
            File = xr.open_dataset(filename).TAGP
            crops = File.where(crop >= 0, np.nan)
            trend_TAGP = trend_s(crops, trend_land)  # 计算趋势
            trend_name = ("%s_%s_default_trend" % (name, ssp))
            save_xarray(pathout, trend_TAGP, trend_name)

            p = trend_p(crop, trend_land)  # 计算p<0.01
            p_TAGP = xr.where(p < 0.01, 1, np.nan)
            p_name = ("%s_%s_default_p" % (name, ssp))
            save_xarray(pathout, p_TAGP, p_name)
