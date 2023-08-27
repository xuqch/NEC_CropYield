from email.policy import default
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
from pylab import rcParams
import time
import datetime

### Plot settings
font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 8,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

if __name__ == '__main__':
    import glob, os, shutil

    path = "/stu01/xuqch3/PCSE/NEX-GDDP-CMIP6/output/adaptation/sowing/"
    maskfile_Crop = "/tera04/zhwei/PCSE/data/crop_distribution/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''
    names = ['rice', 'maize', 'soybean']


    idxs = [0, 1, 2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    scenarios = ['default', 'co2', 'precipitation', 'temperature']
    scenarios = ['default']
    ssps = ['ssp585']

    df = pd.DataFrame()
    veg = []
    sspx = []
    scenariox = []
    Yieldmean = []
    Yieldstd = []
    # Apr 15-->105

    for scenario in scenarios:
        for name, idx, colr in zip(names, idxs, colors):
            for ssp in ssps:
                print(name, scenario, ssp)
                lons = xr.open_dataset('/stu01/xuqch3/PCSE/NEX-GDDP-CMIP6/pr_t/pr_t_Growthday_ssp585.nc').lon.values
                lats = xr.open_dataset('/stu01/xuqch3/PCSE/NEX-GDDP-CMIP6/pr_t/pr_t_Growthday_ssp585.nc').lat.values
                # fixtime=Start_time*0+105
                ds1 = xr.open_dataset(f'/stu01/xuqch3/PCSE/NEX-GDDP-CMIP6/output/sensitivity/default/{name}_output_{ssp}_default.nc')
                ds1 = ds1.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100), drop=True)
                ds1 = ds1["TAGP"]
                Max_Yield_default = ds1.groupby("time.year").max("time")
                Max_Yield_default.to_netcdf(f'{path}/{name}_output_{ssp}_sowing_Max_Yield_default.nc')
                Max_Yield = Max_Yield_default  # ds4.groupby("time.year").max("time")

                ds5 = xr.open_dataset(f'/stu01/xuqch3/PCSE/NEX-GDDP-CMIP6/pr_t/pr_t_Growthday_{ssp}.nc')
                ds5 = ds5.where((ds5.time > 2014) & (ds5.time < 2100), drop=True)
                ds5 = ds5.rename({'time': 'year'})
                ds5.to_netcdf("test.nc")
                # ds5.coords.rename({'time': 'year'})

                Start_time = ds5['start']
                Start_time1 = Start_time
                # Start_time=ds5.groupby("time.year").max("time")

                # ds4   = xr.open_dataset(f'{path}/{name}_output_{ssp}_sowing_0.nc')
                # ds4 = ds4.where((ds4.time.dt.month>4)&(ds4.time.dt.month<12)&(ds4.time.dt.year > 2014)&(ds4.time.dt.year < 2100), drop=True)
                # ds4 = ds4["TAGP"]

                for i in np.arange(0, 77, 7):
                    t0 = time.strftime('%H:%M:%S')
                    VarFile = f'{path}/{name}_output_{ssp}_sowing_{i}.nc'
                    ds4 = xr.open_dataset(VarFile)
                    ds4 = ds4.where((ds4.time.dt.month > 4) & (ds4.time.dt.month < 12) & (ds4.time.dt.year > 2014) & (ds4.time.dt.year < 2100),
                                    drop=True)
                    ds4 = ds4["TAGP"]
                    Max_Yield_i = ds4.groupby("time.year").max("time")
                    Max_Yield_i.to_netcdf(f'/tera04/zhwei/PCSE/data/output/adaptation/sowing/{name}_output_{ssp}_sowing_Max_Yield_{i}.nc')

                    Max_Yield = xr.where(Max_Yield <= Max_Yield_i, Max_Yield_i, Max_Yield)

                    Start_time1 = xr.where(Max_Yield <= Max_Yield_i, Start_time + i, Start_time1)
                    # Start_time1= xr.where(Max_Yield<=Max_Yield_default,105.0,Start_time1)

                    '''
                    for j in np.arange(0,85):
                        Max_Yield.where(Max_Yield)
                        for ilon in np.arange(len(lons)):
                            for ilat in np.arange(len(lats)):
                                if (Max_Yield[j,ilat,ilon]<Max_Yield_i[j,ilat,ilon]):
                                    Max_Yield[j,ilat,ilon]=Max_Yield_i[j,ilat,ilon]
                                    Start_time[j,ilat,ilon]=Start_time[j,ilat,ilon]+i
                    '''
                Max_Yield = xr.where(Max_Yield <= Max_Yield_default, Max_Yield_default, Max_Yield)
                Max_Yield.to_netcdf(f'/tera04/zhwei/PCSE/data/output/adaptation/sowing/{name}_output_{ssp}_sowing_Max_Yield_final.nc')
                Start_time1.to_netcdf(f'/tera04/zhwei/PCSE/data/output/adaptation/sowing/{name}_output_{ssp}_sowing_Growthday_final.nc')
