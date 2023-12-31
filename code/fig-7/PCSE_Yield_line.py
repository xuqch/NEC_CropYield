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

#=================================================================================================================
    path = '/stu01/xuqch3/finished/data/'
    pathin = f"{path}/PCSE/output/adaptation/sowing/"
    Figout = './'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    names = ['rice', 'maize', 'soybean']

    run_name = 'TAGP'
    idxs=[0,1,2]
    colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
    df = pd.DataFrame()

    VarFile1 = f'{pathin}/rice_{run_name}_output_ssp585_sowing_Max_Yield_final.nc'
    print(VarFile1)
    with xr.open_dataset(VarFile1) as ds1:
        ds1 = ds1[f"{run_name}"]
        ds_a1 = ds1.where(crop == 0, drop=True)

        default_rice = xr.open_dataset(f'{path}/rice_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"]
        default_rice = default_rice.where(crop == 0, drop=True)
        default_rice = default_rice.groupby('year').mean(...)[0]
        rice = ds_a1.groupby('year').mean(...)
        rice_land = (rice - default_rice) / default_rice * 100
        print(f'sowing '+'rice mean: '+str(rice_land.mean(...).values))
        print(f'sowing '+'rice std: '+str(rice_land.std(...).values))

    VarFile2 = f'{pathin}/maize_{run_name}_output_ssp585_sowing_Max_Yield_final.nc'
    print(VarFile2)
    with xr.open_dataset(VarFile2) as ds2:
        ds2 = ds2[f"{run_name}"]
        ds_a2 = ds2.where(crop == 1, drop=True)

        default_maize = xr.open_dataset(f'{path}/maize_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"]
        default_maize = default_maize.where(crop == 1, drop=True)
        default_maize = default_maize.groupby('year').mean(...)[0]
        maize = ds_a2.groupby('year').mean(...)
        maize_land = (maize - default_maize) / default_maize * 100
        print(f'sowing '+'maize mean: '+str(maize_land.mean(...).values))
        print(f'sowing '+'maize std: '+str(maize_land.std(...).values))

    VarFile3 = f'{pathin}/soybean_{run_name}_output_ssp585_sowing_Max_Yield_final.nc'
    print(VarFile3)
    with xr.open_dataset(VarFile3) as ds3:
        ds3 = ds3[f"{run_name}"]
        ds_a3 = ds3.where(crop == 2, drop=True)

        default_soybean = xr.open_dataset(f'{path}/soybean_{run_name}_output_ssp585_sowing_Max_Yield_default.nc')[f"{run_name}"]
        default_soybean = default_soybean.where(crop == 2, drop=True)
        default_soybean = default_soybean.groupby('year').mean(...)[0]
        soybean = ds_a3.groupby('year').mean(...)
        soybean_land = (soybean - default_soybean) / default_soybean * 100
        print(f'sowing '+'soybean mean: '+str(soybean_land.mean(...).values))
        print(f'sowing '+'soybean std: '+str(soybean_land.std(...).values))
        print('plotting now')
    markers = ['*', 'x', '+']
    lines = [1.5, 1.5, 1.5, 1.5]
    alphas = [1., 1., 1., 1.]
    linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rice_land.plot.line(x='year', label='Rice', linewidth=lines[1], linestyle=linestyles[1],
                          alpha=alphas[1], color=colors[0])  # ,color = 'blue'
    maize_land.plot.line(x='year', label='Maize', linewidth=lines[2], linestyle=linestyles[2],
                          alpha=alphas[2], color=colors[1])  # ,color = 'green
    soybean_land.plot.line(x='year', label='Soybean', linewidth=lines[0], linestyle=linestyles[0],
                          alpha=alphas[0], color=colors[2])  # ,color = 'orangered'


    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('Yield Change (%)', fontsize=18)
    ax.set_xlabel('Year', fontsize=20)
    ax.tick_params(axis='both', top='off', labelsize=16)
    ax.legend(loc='best', shadow=False, fontsize=16)

    plt.tight_layout()
    plt.savefig(f'{Figout}/sowing_ssp585_output_{run_name}_Yield_change.eps', format='eps', dpi=300)  # timeseries_lines
    plt.savefig(f'{Figout}/sowing_ssp585_output_{run_name}_Yield_change.png', format='png', dpi=300)  # timeseries_lines

    print('plot end')
    df['time'] = pd.Series(soybean.year)
    df['rice'] = pd.Series(rice_land)
    df['maize'] = pd.Series(maize_land)
    df['soybean'] = pd.Series(soybean_land)
    df.to_csv(f'./{run_name}_Yield_sowing_line.csv' )






