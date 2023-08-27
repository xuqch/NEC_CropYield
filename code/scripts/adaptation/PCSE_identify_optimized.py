from ast import IsNot
from cmath import isnan
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
    path = '/stu01/xuqch3/finished/data/'
    pathin = f"{path}/PCSE/output/adaptation/sowing/"
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''

    names = ['rice', 'maize', 'soybean']


    idxs = [0, 1, 2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    scenarios = ['default']
    ssps = ['ssp585']
    run_name='TAGP'
    df = pd.DataFrame()

    veg = []
    sspx = []
    scenariox = []
    Yieldmean = []
    Yieldstd = []
    # Apr 15-->105
    ll = xr.open_dataset(f'{pathin}/rice_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')
    default_ssp585_rice = xr.open_dataset(f'{pathin}/rice_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"].where(crop > -1)[0, :, :]
    default_ssp585_maize = xr.open_dataset(f'{pathin}/maize_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"].where(crop > -1)[0, :, :]
    default_ssp585_soybean = xr.open_dataset(f'{pathin}/soybean_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"].where(crop > -1)[0, :, :]

    rice_sowing_Max_Yield = xr.open_dataset(f'{pathin}/rice_{run_name}_output_ssp585_sowing_Max_Yield_final.nc')[f"{run_name}"].where(
        crop > -1) / default_ssp585_rice * 100.0
    maize_sowing_Max_Yield = xr.open_dataset(f'{pathin}/maize_{run_name}_output_ssp585_sowing_Max_Yield_final.nc')[f"{run_name}"].where(
        crop > -1) / default_ssp585_maize * 100.0
    soybean_sowing_Max_Yield = xr.open_dataset(f'{pathin}/soybean_{run_name}_output_ssp585_sowing_Max_Yield_final.nc')[f"{run_name}"].where(
        crop > -1) / default_ssp585_soybean * 100.0
    Max_Yield = rice_sowing_Max_Yield
    crop_dis = rice_sowing_Max_Yield * 0.0

    Max_Yield = xr.where(rice_sowing_Max_Yield <= maize_sowing_Max_Yield, maize_sowing_Max_Yield, rice_sowing_Max_Yield)
    Max_Yield = xr.where(Max_Yield <= soybean_sowing_Max_Yield, soybean_sowing_Max_Yield, Max_Yield)

    crop_dis = xr.where(rice_sowing_Max_Yield <= maize_sowing_Max_Yield, 1.0, 0.0)
    crop_dis = xr.where(Max_Yield <= soybean_sowing_Max_Yield, 2.0, crop_dis)

    # crop_dis  = crop_dis['TWSO']
    print(crop_dis)

    # ll=crop_dis.values

    Max_Yield = xr.where(crop_dis == 0., Max_Yield * default_ssp585_rice / 100.0, Max_Yield)
    Max_Yield = xr.where(crop_dis == 1., Max_Yield * default_ssp585_maize / 100.0, Max_Yield)
    Max_Yield = xr.where(crop_dis == 2., Max_Yield * default_ssp585_soybean / 100.0, Max_Yield)

    Max_Yield.to_netcdf(f"{pathin}/../optimized/optimized_Yield_{run_name}.nc")

    crop_dis = xr.where(Max_Yield > 100.0, crop_dis, np.nan)

    crop_dis.to_netcdf(f"{pathin}/../optimized/optimized_distribution_{run_name}.nc")
