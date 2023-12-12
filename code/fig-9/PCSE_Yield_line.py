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

    run = 'TAGP'
    scenario = 'CR'# optimized CR
    # ========================================================================================================================
    path = "/stu01/xuqch3/finished/data/PCSE/output/adaptation/sowing/"

    CR_distribution = f"/stu01/xuqch3/finished/data/PCSE/output/adaptation/{scenario}/{scenario}_distribution_{run}.nc"
    distribution = xr.open_dataset(CR_distribution)[f'{run}']
    df = pd.DataFrame()

    Figout = './'
    maskfile_Crop = "/tera04/zhwei/PCSE/data/crop_distribution/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    names = ['rice', 'maize', 'soybean']

    idxs = [0, 1, 2]
    # colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']

    VarFile1 = f"/stu01/xuqch3/finished/data/PCSE/output/adaptation/{scenario}/{scenario}_Yield_{run}.nc"
    print(VarFile1)
    with xr.open_dataset(VarFile1) as ds1:
        ds1 = ds1[f"{run}"]
        ds_a1 = ds1.where(distribution == 0, drop=True)

        default_rice_input = xr.open_dataset(f'{path}/rice_{run}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run}"]
        default_rice = default_rice_input.where(distribution == 0, drop=True)[0, :, :]
        default_rice = default_rice.mean(...)

        rice = ds_a1.groupby('year').mean(...)
        rice_land = (rice - default_rice) / default_rice * 100
        print(f'{scenario} ' + 'rice mean: ' + str(rice_land.mean(...).values))
        print(f'{scenario} ' + 'rice std: ' + str(rice_land.std(...).values))
        rice_land = rice_land.reindex(year=default_rice_input.year)

    VarFile2 = f"/stu01/xuqch3/finished/data/PCSE/output/adaptation/{scenario}/{scenario}_Yield_{run}.nc"
    print(VarFile2)
    with xr.open_dataset(VarFile2) as ds2:
        ds2 = ds2[f"{run}"]
        ds_a2 = ds2.where(distribution == 1, drop=True)

        default_maize_input = xr.open_dataset(f'{path}/maize_{run}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run}"]
        default_maize = default_maize_input.where(distribution == 1, drop=True)[0, :, :]
        default_maize = default_maize.mean(...)

        maize = ds_a2.groupby('year').mean(...)
        maize_land = (maize - default_maize) / default_maize * 100
        print(f'{scenario} ' + 'maize mean: ' + str(maize_land.mean(...).values))
        print(f'{scenario} ' + 'maize std: ' + str(maize_land.std(...).values))
        maize_land = maize_land.reindex(year=default_maize_input.year)

    VarFile3 = f"/stu01/xuqch3/finished/data/PCSE/output/adaptation/{scenario}/{scenario}_Yield_{run}.nc"
    print(VarFile3)
    with xr.open_dataset(VarFile3) as ds3:
        ds3 = ds3[f"{run}"]
        ds_a3 = ds3.where(distribution == 2, drop=True)

        default_soybean_input = xr.open_dataset(f'{path}/soybean_{run}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run}"]
        default_soybean = default_soybean_input.where(distribution == 2, drop=True)[0, :, :]
        default_soybean = default_soybean.mean(...)

        soybean = ds_a3.groupby('year').mean(...)
        soybean_land = (soybean - default_soybean) / default_soybean * 100
        print(f'{scenario} ' + 'soybean mean: ' + str(soybean_land.mean(...).values))
        print(f'{scenario} ' + 'soybean std: ' + str(soybean_land.std(...).values))
        soybean_land = soybean_land.reindex(year=default_soybean_input.year)

        print('plotting now')
    markers = ['*', 'x', '+']
    lines = [1.5, 1.5, 1.5, 1.5]
    alphas = [1., 1., 1., 1.]
    linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rice_land.plot.line(x='year', label='Rice', linewidth=lines[1], linestyle=linestyles[1],
                        alpha=alphas[1], color=colors[0], marker='D')  # ,color = 'blue'
    maize_land.plot.line(x='year', label='Maize', linewidth=lines[2], linestyle=linestyles[2],
                         alpha=alphas[2], color=colors[1], marker='D')  # ,color = 'green
    soybean_land.plot.line(x='year', label='Soybean', linewidth=lines[0], linestyle=linestyles[0],
                           alpha=alphas[0], color=colors[2], marker='D')  # ,color = 'orangered'

    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('Yield Change (%)', fontsize=20)
    ax.set_xlabel('Year', fontsize=20)
    ax.tick_params(axis='both', top='off', labelsize=18)
    ax.legend(loc='best', shadow=False, fontsize=18)
    # ax.set_title('%s_%s' % (name,fnames[i]))
    plt.tight_layout()
    plt.savefig(f'{Figout}/{scenario}_ssp585_output_{run}_Yield_change.eps', format='eps', dpi=800)  # timeseries_lines
    plt.savefig(f'{Figout}/{scenario}_ssp585_output_{run}_Yield_change.png', format='png', dpi=800)  # timeseries_lines

    print('plot end')

    df['time'] = pd.Series(soybean_land.year)
    df['rice'] = pd.Series(rice_land)
    df['maize'] = pd.Series(maize_land)
    df['soybean'] = pd.Series(soybean_land)
    df.to_csv(f'./Yield_{scenario}_line.csv')
