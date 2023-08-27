import numpy as np
import matplotlib.pyplot as plt
import os, sys
import xarray as xr
import dask.array as da
from scipy.stats import linregress
#from numba import jit  # Speedup for python functions
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

#import pymannkendall as mk


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
                trend_land['ta'][j, i] = crop[j]
    return trend_land
    # for j, lat in enumerate(cropland.lat.values[:10]):
    #     crop = mk_test(cropland, lon, lat)
    #     if crop >= -9999:
    #         trend_land['crop'][j,i] = crop


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
                trend_land['ta'][j, i] = crop[j]
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
    import time
    import datetime
    #
    Inpath = "/tera01/xuqch3/PCSE/PCSE_input"
    Figout = '/tera04/zhwei/PCSE/Fig/'
    maskfile_Crop = "/tera04/zhwei/PCSE/data/crop_distribution/crop.nc"
    crop = xr.open_dataset(maskfile_Crop)
    trend_land = xr.where(crop >= 0, np.nan, crop)
    trend_land = trend_land.rename({'crop': 'ta'})


    '''
    rice:0, maize:1, soybean:2
    -'''

    t0 = time.strftime('%H:%M:%S')
    
    VarFile = f'{Inpath}/PCSE_input_ssp585.nc'
    print(VarFile)

    with xr.open_dataset(VarFile) as ds1:
        ds1['ta']=(ds1['tasmax'] + ds1['tasmin'])/2
    File =ds1.ta
    crop = File.where(crop >= 0, np.nan)


    trend_ta = trend_s(crop, trend_land)  # 计算趋势
    print(trend_ta)
    trend_ta.to_netcdf(f'{Figout}/test.nc')
    p = trend_p(crop, trend_land)  # 计算p<0.01
    p_ta = xr.where(p < 0.01, 1, np.nan)
   

'''      
        ds1=ds1['ta'].resample(time='1Y').mean()#.squeeze().values
        ds_a1 = ds1.where(crop >= 0.0, drop=True)
    



                filename = ("%s%s/%s_output_2100_%s_test_%s_yearmean.nc" % (pathin, test, name, ssp, i + 1))
                pathout = ("%s%s/" % (pathin, test))
                File = xr.open_dataset(filename).TAGP
                crop = File.where(crop >= 0, np.nan)
                trend_TAGP = trend_s(crop, trend_land)  # 计算趋势
                trend_name = ("%s_%s_test_%s_trend" % (name, ssp, i + 1))
                save_xarray(pathout, trend_TAGP, trend_name)

                p = trend_p(crop, trend_land)  # 计算p<0.01
                p_TAGP = xr.where(p < 0.01, 1, np.nan)
                p_name = ("%s_%s_test_%s_p" % (name, ssp, i + 1))
                save_xarray(pathout, p_TAGP, p_name)
















    pathin = "/Volumes/xuqch/PCSE/"
    tests = ['test1', 'test2', 'test3', 'test4', 'test5']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    names = ['rice', 'maize', 'soybean']
    for i, test in enumerate(tests):
        print(test, ">>>>")
        for name in names:
            print(name, ">>")
            for ssp in ssps:
                print(ssp)
                filename = ("%s%s/%s_output_2100_%s_test_%s_yearmean.nc" % (pathin, test, name, ssp, i + 1))
                pathout = ("%s%s/" % (pathin, test))
                File = xr.open_dataset(filename).TAGP
                crop = File.where(crop >= 0, np.nan)
                trend_TAGP = trend_s(crop, trend_land)  # 计算趋势
                trend_name = ("%s_%s_test_%s_trend" % (name, ssp, i + 1))
                save_xarray(pathout, trend_TAGP, trend_name)

                p = trend_p(crop, trend_land)  # 计算p<0.01
                p_TAGP = xr.where(p < 0.01, 1, np.nan)
                p_name = ("%s_%s_test_%s_p" % (name, ssp, i + 1))
                save_xarray(pathout, p_TAGP, p_name)
'''  