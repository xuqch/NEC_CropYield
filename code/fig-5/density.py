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
          'axes.labelsize': 45,
          'grid.linewidth': 0.2,
          'font.size': 25,
          'legend.fontsize': 25,
          'legend.frameon': False,
          'xtick.labelsize': 45,
          'xtick.direction': 'out',
          'ytick.labelsize': 45,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)



def save_data(path, name, ssp, i, run, scenario):
    print(scenario)
    VarFile = f'{path}/{scenario}/{name}_output_{ssp}_{scenario}.nc'
    # print(VarFile)
    with xr.open_dataset(VarFile) as ds1:
        ds1 = ds1.where(
            (ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100),
            drop=True)
        ds1 = ds1[f"{run}"]
        ds_a1 = ds1.where(crop == i, drop=True)
        ssp126 = ds_a1.groupby('time.year').mean(...)
        default_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
    return default_land


if __name__ == '__main__':
    import glob, os, shutil, sys

    # run = 'TAGP'
    argv = sys.argv
    run = str(argv[1])
    path = '/stu01/xuqch3/finished/data'
    pathin = f"{path}/PCSE/output/sensitivity/"
    Figout = './'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop

    ssps = ['ssp585']
    names = ['rice', 'maize', 'soybean']
    idx = [0, 1, 2]
    scenarios = ['default', 'co2', 'precipitation', 'temperature']
    labels = ["Default", "Co2", "Precipitation", "Temperature"]
    colors_list = sns.color_palette("Set3", n_colors=6, desat=.7).as_hex()  #
    colors = [colors_list[2], colors_list[0], colors_list[5], colors_list[3]]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(27, 8), sharey=True)
    for i, name in enumerate(names):
        print(name)
        ssp = ssps[0]
        default = save_data(pathin, name, ssp, i, run, "default")
        co2 = save_data(pathin, name, ssp, i, run, "co2")
        precipitation = save_data(pathin, name, ssp, i, run, "precipitation")
        temperature = save_data(pathin, name, ssp, i, run, "temperature")
        print("ploting now")

        default_data = np.delete(default.values, np.argwhere(np.isnan(default.values))).reshape(-1)
        co2_data = np.delete(co2.values, np.argwhere(np.isnan(co2.values))).reshape(-1)
        precipitation_data = np.delete(precipitation.values, np.argwhere(np.isnan(precipitation.values))).reshape(-1)
        temperature_data = np.delete(temperature.values, np.argwhere(np.isnan(temperature.values))).reshape(-1)
        print(max(temperature_data))
        print(min(temperature_data))

        sns.kdeplot(data=default_data, ax=axes[i], fill=True, color=colors[0])
        sns.kdeplot(data=co2_data, ax=axes[i], fill=True, color=colors[1])
        sns.kdeplot(data=precipitation_data, ax=axes[i], fill=True, color=colors[2])
        sns.kdeplot(data=temperature_data, ax=axes[i], fill=True, color=colors[3])
        axes[i].set_xlabel(f'Yield Change (%)', fontsize=40)

    axes[0].set_ylabel('Density', fontsize=45)
    # fig.legend(bplot1['boxes'], label, loc=8, borderaxespad=2, ncol=n, shadow=False, fontsize=30)
    axes[0].legend(labels, loc='best', shadow=False, fontsize=30)
    axes[0].set_xticks(range(-100, 250, 100))
    axes[1].set_xticks(range(-100, 100, 50))
    axes[2].set_xticks(range(-100, 100, 50))
    plt.legend(loc='best')
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(f'{Figout}/{run}_change_density.eps', format='eps', dpi=300)  # timeseries_lines
    plt.savefig(f'{Figout}/{run}_change_density.png', format='png', dpi=300)  # timeseries_lines
