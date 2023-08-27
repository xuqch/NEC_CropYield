import matplotlib.pyplot as plt
from email.policy import default
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import multiprocessing
import cartopy.crs as ccrs
from pylab import rcParams
import time
import datetime

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


def box(data_crop, Figout):
    fig = plt.figure(figsize=(13, 8))
    rice = [data_crop[0, 0, :], data_crop[0, 1, :], data_crop[0, 2, :], data_crop[0, 3, :]]
    maize = [data_crop[1, 0, :], data_crop[1, 1, :], data_crop[1, 2, :], data_crop[1, 3, :]]
    soybean = [data_crop[2, 0, :], data_crop[2, 1, :], data_crop[2, 2, :], data_crop[2, 3, :]]

    labels = ["default", "co2", "precipitation", "temperature"]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    bplot1 = plt.boxplot(rice, patch_artist=True, labels=labels, positions=(1, 1.3, 1.6, 1.9), widths=0.2, medianprops = {'color':'red','linewidth':'2'})
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(maize, patch_artist=True, labels=labels, positions=(2.5, 2.8, 3.1, 3.4), widths=0.2, medianprops = {'color':'red','linewidth':'2'})
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(soybean, patch_artist=True, labels=labels, positions=(4, 4.3, 4.6, 4.9), widths=0.2, medianprops = {'color':'red','linewidth':'2'})
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2.5, 4]
    x_position_fmt = ["Rice", "Maize", "Soybean"]
    plt.xticks([i + 0.9 / 2 for i in x_position], x_position_fmt, fontsize=18)
    plt.ylabel('TAGP change (%)', fontsize=18)
    # plt.title('SSPs',fontsize=18)
    # plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot2['boxes'], labels, loc='best',fontsize=16)  # 绘制表示框，右下角绘制
    plt.savefig(f'{Figout}/TAGP_change_ssp.eps', format='eps', dpi=800)  # timeseries_lines
    plt.savefig(f'{Figout}/TAGP_change_ssp.png', format='png', dpi=800)  # timeseries_lines
    plt.show()


def save_data(path, name, ssp, i, scenario):
    print(scenario)
    VarFile = f'{path}/{scenario}/{name}_output_{ssp}_{scenario}.nc'
    print(VarFile)
    with xr.open_dataset(VarFile) as ds1:  
        ds1 = ds1.where((ds1.time.dt.month>4)&(ds1.time.dt.month<12)&(ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
        ds1 = ds1["TAGP"]
        ds_a1 = ds1.where(crop == i, drop=True)
        ssp126 = ds_a1.groupby("time.year").max("time").groupby('year').mean(...)
        default_land = (ssp126 - ssp126[0]) / ssp126[0] * 100

    return default_land


if __name__ == '__main__':
    import glob, os, shutil

    # path = '/tera01/xuqch3/PCSE/sensitivity/harvest_date'
    path = "/tera04/zhwei/PCSE/data/output/sensitivity"
    Figout = '/tera01/xuqch3/PCSE/sensitivity/Fig'
    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"  # 'F:/PCSE/crop/crop.nc'
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    # ssps = ['ssp585']
    names = ['rice', 'maize', 'soybean']
    idx = [0, 1, 2]

    scenarios = ['default', 'co2', 'precipitation', 'temperature']
    data_crop = np.empty((3, 4, 84 * len(ssps)))

    max_cpu = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count()
    num_cores = multiprocessing.cpu_count()
    for i, name in enumerate(names):
        default = Parallel(n_jobs=num_cores)(delayed(save_data)(path, name, ssp, i, "default") for ssp in ssps)
        co2 = Parallel(n_jobs=num_cores)(delayed(save_data)(path, name, ssp, i, "co2") for ssp in ssps)
        precipitation = Parallel(n_jobs=num_cores)(delayed(save_data)(path, name, ssp, i, "precipitation") for ssp in ssps)
        temperature = Parallel(n_jobs=num_cores)(delayed(save_data)(path, name, ssp, i, "temperature") for ssp in ssps)

        data_crop[i, 0, :] = np.concatenate([default] ,axis = 0)
        data_crop[i, 1, :] = np.concatenate([co2],axis = 0)
        data_crop[i, 2, :] = np.concatenate([precipitation],axis = 0)
        data_crop[i, 3, :] = np.concatenate([temperature],axis = 0)

    # exit(0)
    print("ploting now")
    box(data_crop,Figout)
