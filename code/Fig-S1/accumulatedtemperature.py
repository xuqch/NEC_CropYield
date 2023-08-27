import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os, sys
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.stats import linregress
# from numba import jit #  Speedup for python functions
from matplotlib import colors
import cartopy.crs as ccrs
from pylab import rcParams
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)
'''
Rice 2300~4800, Corn 2300~3900, Soybean 2700~3600
Rice 2700~4800, Corn 2300~3000, Soybean 2800~3200
'''
if __name__ == '__main__':
    import glob, os, shutil

    path = '/stu01/xuqch3/finished/data/'
    pathout = (f'{path}/code/Fig-S1/')
    maskfile_Crop = f"{path}/crop/crop.nc"
    cropland = xr.open_dataset(maskfile_Crop)
    crops = ['Rice', 'Maize', 'Soybean']
    ssps_1 = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    markers = ['*', 'x', '+']
    lines = [1.5, 1.5, 1.5, 1.5]
    alphas = [1., 1., 1., 1.]
    linestyles = ['solid', 'solid', 'solid', 'dashed', 'dashdot', 'solid']
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']  ##蓝，绿，黄，红
    fillcolor = ['#D6D6D4']

    up_line = [4800, 3300, 3200]
    down_line = [2700, 2300, 2800]

    for i, crop in enumerate(crops):
        data = np.full((4, len(range(2015, 2101))), np.nan)
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        print(crop)
        year = []
        for j, ssp in enumerate(ssps_1):
            ds = xr.open_dataset(f"{path}/met/{ssp}/tas_day_ensmean_{ssp}_2015-2100.nc")
            ds1 = ds.tas
            ds1 = ds1 - 273.15
            ds0 = ds1.pipe(lambda x: x).where(ds1 > 10.0, drop=True)
            dssum = ds0.groupby("time.year").sum()
            crop_l = dssum.where(cropland == i, np.nan)
            crop_line = crop_l.groupby('year').mean(...).to_array()
            # ===================================
            crop_line.plot.line(x='year', label=ssp, linestyle=linestyles[1], linewidth=lines[0],
                                alpha=alphas[0], color=colors[j])  # ,color = 'blue', linewidth=lines[0]
            plt.fill_between(x=crop_line.year.values, y1=down_line[i], y2=up_line[i] , alpha=0.2, color=fillcolor[0])
            data[j, :] = crop_line

        ax.set_ylabel('Accumulated temperature (K)', fontsize=20)
        ax.set_xlabel('Year', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        plt.xticks(np.arange(2020, 2110, 10), np.arange(2020, 2110, 10))
        plt.yticks(np.arange(2300, 5100, 400), np.arange(2300, 5100, 400))
        ax.legend(loc='best', shadow=False, fontsize=16, title='%s' % (crop))
        ax.set_title('')
        plt.tight_layout()
        plt.savefig('%s/acctas_%s_lines_fill.eps' % (pathout_Dongbei, crop), dpi=800)  # timeseries_lines
        plt.savefig('%s/acctas_%s_lines_fill.png' % (pathout_Dongbei, crop), dpi=800)


        df = pd.DataFrame({'year': range(2015, 2101),
                                   'ssp_126': data[0, :],
                                   'ssp_245': data[1, :],
                                   'ssp_370': data[2, :],
                                   'ssp_585': data[3, :]})  # 'R': R,'KGE': KGE,
        df.to_excel('%s/%s_acca.xlsx' % (pathout_Dongbei, crop), sheet_name=f'{crop}', index=True)
    #
    #
    data = np.full((4, len(range(2015, 2101))), np.nan)
    year = []
    for j, ssp in enumerate(ssps_1):
        ds = xr.open_dataset(f"{path}/met/{ssp}/tas_day_ensmean_{ssp}_2015-2100.nc")
        ds1 = ds.tas
        ds1 = ds1 - 273.15
        ds0 = ds1.pipe(lambda x: x).where(ds1 > 10.0, drop=True)
        dssum = ds0.groupby("time.year").sum()
        crop_l = dssum.where(cropland >=0, np.nan)
        crop_line = crop_l.groupby('year').mean(...).to_array()
        # ===================================
        data[j, :] = crop_line

    df = pd.DataFrame({'year': range(2015, 2101),
                               'ssp_126': data[0, :],
                               'ssp_245': data[1, :],
                               'ssp_370': data[2, :],
                               'ssp_585': data[3, :]})  # 'R': R,'KGE': KGE,
    df.to_excel('%s/acca.xlsx' % (pathout), sheet_name=f'regional', index=True)