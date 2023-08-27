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

    Inpath = "/tera01/xuqch3/PCSE/PCSE_input"
    Figout = '/tera01/xuqch3/PCSE/sensitivity/Fig/'
    maskfile_Crop = "/tera04/zhwei/PCSE/data/crop_distribution/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''
    df = pd.DataFrame()
    year = []
    ssp_126 = []
    ssp_245 = []
    ssp_370 = []
    ssp_585 = []

    names = ['rice', 'maize', 'soybean']
    idxs=[0,1,2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    scenarios=['default','temperature','precipitation','co2']
    scenarios=['precipitation']
    
    for scenario in scenarios:
        t0 = time.strftime('%H:%M:%S')
        VarFile = f'{Inpath}/ssp126/PCSE_input.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds1: 
            ds1['ta']=(ds1['tasmax'] + ds1['tasmin'])*0.5
            ds1 = ds1.where((ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
            ds1=ds1['ta'].resample(time='1Y').mean() #.squeeze().values
            ds_a1 = ds1.where(crop >= 0.0, drop=True)
            ssp126 = ds_a1.groupby('time').mean(...)#*86400*365

        VarFile = f'{Inpath}/ssp245/PCSE_input.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds1:
            ds1['ta']=(ds1['tasmax'] + ds1['tasmin'])*0.5
            ds1 = ds1.where((ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
            ds1=ds1['ta'].resample(time='1Y').mean() #.squeeze().values
            ds_a1 = ds1.where(crop >= 0.0, drop=True)
            ssp245 = ds_a1.groupby('time').mean(...)#*86400*365

        VarFile = f'{Inpath}/ssp370/PCSE_input.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds1:
            ds1['ta']=(ds1['tasmax'] + ds1['tasmin'])*0.5
            ds1 = ds1.where((ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
            ds1=ds1['ta'].resample(time='1Y').mean()#.squeeze().values
            ds_a1 = ds1.where(crop >= 0.0, drop=True)
            ssp370 = ds_a1.groupby('time').mean(...)#*86400*365

        VarFile = f'{Inpath}/ssp585/PCSE_input.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds1:
            ds1['ta']=(ds1['tasmax'] + ds1['tasmin'])*0.5
            ds1 = ds1.where((ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
            ds1=ds1['ta'].resample(time='1Y').mean()#.squeeze().values
            ds_a1 = ds1.where(crop >= 0.0, drop=True)
            ssp585 = ds_a1.groupby('time').mean(...)#*86400*365
        # print('plotting now')
        # # ds_line5 =np.zeros(86).to_array()
        # markers = ['*', 'x', '+']
        # lines = [1.5, 1.5, 1.5, 1.5]
        # alphas = [1., 1., 1., 1.]
        # linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
        # with plt.style.context(['science','no-latex']):
        #     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        #     ssp126.plot.line(x='time', label='ssp126', linewidth=lines[1], linestyle=linestyles[1],
        #                           alpha=alphas[1], color=colors[0])  # ,color = 'blue'
        #     ssp245.plot.line(x='time', label='ssp245', linewidth=lines[2], linestyle=linestyles[2],
        #                           alpha=alphas[2], color=colors[1])  # ,color = 'green
        #     ssp370.plot.line(x='time', label='ssp370', linewidth=lines[0], linestyle=linestyles[0],
        #                           alpha=alphas[0], color=colors[2])  # ,color = 'orangered'
        #     ssp585.plot.line(x='time', label='ssp585', linewidth=lines[3], linestyle=linestyles[3],
        #                           alpha=alphas[3], color=colors[3])  # ,color = 'red'

        #     #ax.axhline(y=0, color='gray', linestyle='--')
        #     ax.set_ylabel('Temperature (K)', fontsize=18)
        #     ax.set_xlabel('Year', fontsize=20)
        #     ax.set_title(None)

        #     #plt.yticks(np.arange(-80, 60, 20), np.arange(-80, 60, 20))
        #     # plt.yticks(np.arange(-10, 50, 10), np.arange(-10, 50, 10))
        #     ax.tick_params(axis='both', top='off', labelsize=16)
        #     ax.legend(loc='best', shadow=False, fontsize=12)
        #     #ax.set_title('%s_%s' % (name,fnames[i]))
        #     plt.tight_layout()
        #     plt.savefig(f'{Figout}/Temperature_year.eps', format='eps',dpi=800)  # timeseries_lines
        #     plt.show()

        t1 = time.strftime('%H:%M:%S')
        start_date = t0
        end_date = t1
        start_time = datetime.datetime.strptime(start_date, '%H:%M:%S')
        end_time = datetime.datetime.strptime(end_date, '%H:%M:%S')
        during_time = end_time - start_time
        print(during_time)
        df['year'] = pd.Series(ssp126.time)
        df['ssp_126'] = pd.Series(ssp126)
        df['ssp_245'] = pd.Series(ssp245)
        df['ssp_370'] = pd.Series(ssp370)
        df['ssp_585'] = pd.Series(ssp585)
        df.to_csv('tas.csv')
    print('end')
