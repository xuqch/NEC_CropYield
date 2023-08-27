import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from scipy import stats
import numpy as np
import pandas as pd
import os, sys
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.stats import linregress
# from numba import jit #  Speedup for python functions
import cartopy.crs as ccrs
from pylab import rcParams
import time
import datetime

Syear = 2015
Eyear = 2030
Sdate = str(str(Syear) + "0101")
Edate = str(str(Eyear) + "1231")

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

    # define the directory
    # set variables
    Figout = '/tera01/xuqch3/PCSE/sensitivity/Fig/TAGP_line/'
    Vars = [ 'TAGP']
    # Vars = ['DVS', 'LAI', 'RD', 'SM', 'TAGP', 'TRA', 'TWLV', 'TWRT', 'TWSO', 'TWST', 'WWLOW']
    # Co = ['', '', '(kg/ha)', '(kg/ha)', '(kg/ha)', '(kg/ha)', '(kg/ha)', '(cm/day)', '(cm)', '', '(cm)']
    Co = ['(kg/ha)']
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    names = [ 'rice', 'maize', 'soybean']
    idxs = [0,1,2]
    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop


    for name,idx in zip(names,idxs):
        for var in Vars:
            print(var,idx)
            VarFile = ('/tera01/xuqch3/PCSE/sensitivity/harvest_date/default/%s_output_ssp126_default_max.nc' % (name))
            with xr.open_dataset(VarFile) as ds1:
                # ds = ds.sel(time=slice(Sdate, Edate))
                ds1 = ds1["%s" % (var)]
                ds_a1 = ds1.where(crop == idx, drop=True)
                ssp126 = ds_a1.groupby('year').mean(...)
                # print(ssp126)
                # exit(0)
  

            VarFile = ('/tera01/xuqch3/PCSE/sensitivity/harvest_date/default/%s_output_ssp245_default_max.nc' % (name))
            # print(VarFile)
            with xr.open_dataset(VarFile) as ds2:
                # ds = ds.sel(time=slice(Sdate, Edate))
                ds2 = ds2["%s" % (var)]
                ds_a2 = ds2.where(crop == idx, drop=True)
                ssp245 = ds_a2.groupby('year').mean(...)
                # ds2 = ds_a.resample(time='1Y').mean()

            VarFile = ('/tera01/xuqch3/PCSE/sensitivity/harvest_date/default/%s_output_ssp370_default_max.nc' % (name))
            # print(VarFile)
            with xr.open_dataset(VarFile) as ds3:
                # ds = ds.sel(time=slice(Sdate, Edate))
                ds3 = ds3["%s" % (var)]
                ds_a3 = ds3.where(crop == idx, drop=True)
                ssp370 = ds_a3.groupby('year').mean(...)
                # ds3 = ds_a.resample(time='1Y').mean()

            VarFile = ('/tera01/xuqch3/PCSE/sensitivity/harvest_date/default/%s_output_ssp585_default_max.nc' % (name))
            # print(VarFile)
            with xr.open_dataset(VarFile) as ds4:
                # ds = ds.sel(time=slice(Sdate, Edate))
                ds4 = ds4["%s" % (var)]
                ds_a4 = ds4.where(crop == idx, drop=True)
                ssp585 = ds_a4.groupby('year').mean(...)
                # ds4 = ds_a.resample(time='1Y').mean()

            # ds_line5 =np.zeros(86).to_array()
            markers = ['*', 'x', '+']
            # legs =['ssp126','ssp245','ssp370','ssp585']
            lines = [1.5, 1.5, 1.5, 1.5]
            alphas = [1., 1., 1., 1.]
            linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            # # x = np.arange(time)
            ssp126.plot.line(x='year', label='ssp126', linewidth=lines[1], linestyle=linestyles[1],
                          alpha=alphas[1],color = colors[0])  # ,color = 'blue'
            ssp245.plot.line(x='year', label='ssp245', linewidth=lines[2], linestyle=linestyles[2],
                          alpha=alphas[2],color = colors[1])  # ,color = 'green
            ssp370.plot.line(x='year', label='ssp370', linewidth=lines[0], linestyle=linestyles[0],
                          alpha=alphas[0],color = colors[2])  # ,color = 'orangered'
            ssp585.plot.line(x='year', label='ssp585', linewidth=lines[3], linestyle=linestyles[3],
                          alpha=alphas[3],color = colors[3])  # ,color = 'red'
            # ds_line5.plot.line (x='year', linewidth=lines[3], linestyle=linestyles[4], alpha=alphas[3],color = 'black') #,color = 'red'
            # ax.axhline(y=f , color='black' , linestyle='--')

            # ax.legend(fontsize=18, loc=1)
            ax.set_ylabel('%s%s' % (var, Co[0]), fontsize=18)
            ax.set_xlabel('Year', fontsize=20)
            ax.tick_params(axis='both', top='off', labelsize=16)
            ax.legend(loc='best', shadow=False, fontsize=12)
            ax.set_title('%s' % (name))
            plt.tight_layout()
            plt.savefig('%s%s_%s.png' % (Figout, name, var),dpi = 800)  # timeseries_lines
            # plt.show()

        print(name)
    print('end')
