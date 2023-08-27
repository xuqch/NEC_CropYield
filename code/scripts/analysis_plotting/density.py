import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import multiprocessing
from pylab import rcParams

font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'grid.linewidth': 1.5,
          'font.size': 20,
          'legend.fontsize': 20,
          'legend.frameon': False,
          'xtick.labelsize': 20,
          'xtick.direction': 'out',
          'ytick.labelsize': 20,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def density(default, co2, precipitation, temperature, Figout, name):
    fig = plt.figure(figsize=(10, 10))

    labels = ["Default", "CO2 constraint", "Precipitation constraint", "Temperature constraint"]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    sns.kdeplot(default,        clip=(-100, 140),label=labels[0], shade=True, color=colors[0])
    sns.kdeplot(co2,            clip=(-100, 140),label=labels[1], shade=True, color=colors[1])
    sns.kdeplot(precipitation,  clip=(-100, 140),label=labels[2], shade=True, color=colors[2])
    sns.kdeplot(temperature,    clip=(-100, 140),label=labels[3], shade=True, color=colors[3])

    plt.ylabel('Density', fontsize=28)
    plt.xlabel('Yield Change (%)', fontsize=28)
    plt.xticks(np.arange(-100, 160, 20), np.arange(-100, 160, 20),fontsize=18)
    plt.legend(loc='best',fontsize=14,title = f'{name}')
    # plt.title(f'{name}',fontsize=14)
    plt.savefig(f'{Figout}/{name}_TAGP_change_density.eps', format='eps', dpi=600)  # timeseries_lines
    plt.savefig(f'{Figout}/{name}_TAGP_change_density.png', format='png', dpi=600)  # timeseries_lines
    # plt.show()


def save_data(path, name, ssp, i, scenario):
    print(scenario)
    VarFile = f'{path}/{scenario}/{name}_output_{ssp}_{scenario}_max.nc'
    # print(VarFile)
    with xr.open_dataset(VarFile) as ds1:
        # ds1 = ds1.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2015) & (ds1.time.dt.year < 2100),drop=True)
        ds1 = ds1["TAGP"]
        ssp126 = ds1.where(crop == i, drop=True)
        ssp126 = ssp126.groupby('year').mean(...)
        # ssp126 = ssp126.groupby("time.year").max("time").groupby('year').mean(...)
        default_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
        #default_land = (ssp126[:,:,:] - ssp126[0,:,:]) / ssp126[0,:,:] * 100
        #default_land = default_land.where(default_land < 200, drop=True)
        #default_land = default_land.where(default_land > -100, drop=True)

# 
    return default_land



if __name__ == '__main__':
    import glob, os, shutil

    path = '/tera01/xuqch3/PCSE/sensitivity/harvest_date'
    # path = "/tera04/zhwei/PCSE/data/output/sensitivity"
    Figout = '/tera01/xuqch3/PCSE/sensitivity/Fig/'
    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"  # 'F:/PCSE/crop/crop.nc'
    crop = xr.open_dataset(maskfile_Crop).crop
    # ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    ssps = ['ssp585']
    names = ['rice', 'maize', 'soybean']
    #names = ['rice']
    idx = [0, 1, 2]

    scenarios = ['default', 'co2', 'precipitation', 'temperature']
    # data_crop = np.empty((3, 4, 84))

    for i, name in enumerate(names):
        print(name)
        for ssp in ssps:
            default = save_data(path, name, ssp, i, "default") 
            co2 = save_data(path, name, ssp, i, "co2") 
            precipitation = save_data(path, name, ssp, i, "precipitation") 
            temperature = save_data(path, name, ssp, i, "temperature") 
            print("ploting now")

            default_data = default.values.reshape(-1)[~np.isnan(default.values.reshape(-1))]
            co2_data = co2.values.reshape(-1)[~np.isnan(co2.values.reshape(-1))]
            precipitation_data = precipitation.values.reshape(-1)[~np.isnan(precipitation.values.reshape(-1))]
            temperature_data = temperature.values.reshape(-1)[~np.isnan(temperature.values.reshape(-1))]
            print(max(temperature_data))
            print(min(temperature_data))
            density(default_data,co2_data,precipitation_data,temperature_data , Figout, name)

