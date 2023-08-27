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

    df = pd.DataFrame()

    veg = []
    sspx = []
    scenariox = []
    Yieldmean = []
    Yieldstd = []
    run_name = 'TAGP'

    # ===================================
    # optimized_Yield change
    names = ['rice', 'maize', 'soybean']
    idxs = [0.0, 1.0, 2.0]

    path = '/stu01/xuqch3/finished/data/'
    pathin = f"{path}/PCSE/output/adaptation/sowing/"
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop

    for name, idx in zip(names, idxs):
        scenario = 'optimized'# optimized CR
        optimized_distribution = f"{path}/PCSE/output/adaptation/{scenario}/{scenario}_distribution_{run_name}.nc"

        VarFile = f"{path}/PCSE/output/adaptation/{scenario}/{scenario}_Yield_{run_name}.nc"
        distribution = xr.open_dataset(optimized_distribution)[f"{run_name}"]

        default_ssp585 = xr.open_dataset(f'{pathin}/{name}_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"]
        default_ssp585 = default_ssp585.where(distribution == idx)
        default_ssp585 = default_ssp585.groupby('year').mean(...)
        default_ssp585 = default_ssp585.where(default_ssp585 > -999, drop=True)[0]


        with xr.open_dataset(VarFile) as ds1:
            ds1 = ds1[f"{run_name}"]
            ds_a4 = ds1.where(distribution == idx)
            ssp585 = ds_a4.groupby('year').mean(...)
            ssp585 = ssp585.where(ssp585> -999,drop=True)

            ssp585_land = (ssp585 - default_ssp585) / default_ssp585 * 100
            # print(ssp585_land)
            print(f'{scenario} ' + f'{name}' + f' {idx} ' + 'SSP585 mean: ' + str(ssp585_land.mean(...).values))
            print(f'{scenario} ' + f'{name}' + f' {idx} ' + 'SSP585 std: ' + str(ssp585_land.std(...).values))
            veg.append(str(name))
            scenariox.append(str(scenario))
            sspx.append('ssp585')
            Yieldmean.append(ssp585_land.mean(...).values)
            Yieldstd.append(ssp585_land.std(...).values)
            print('======================================\n')

    df['veg'] = pd.Series(veg)
    df['scenario'] = pd.Series(scenariox)
    df['sspx'] = pd.Series(sspx)
    df['Yieldmean'] = pd.Series(Yieldmean)
    df['Yieldstd'] = pd.Series(Yieldstd)
    df.to_csv(f'{run_name}_Yield_{scenario}.csv')
    # exit()
