import matplotlib.pyplot as plt
from email.policy import default
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import cartopy.crs as ccrs
from pylab import rcParams
import time
import datetime

font = {'family': 'Times New Roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 30,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 25,
          'xtick.direction': 'out',
          'ytick.labelsize': 25,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def box(data_crop, Figout, run):
    # fig = plt.figure(figsize=(20, 10))
    fig, axes = plt.subplots(figsize=(20, 10))
    rice = [data_crop[0, 0, :], data_crop[0, 1, :], data_crop[0, 2, :], data_crop[0, 3, :]]
    maize = [data_crop[1, 0, :], data_crop[1, 1, :], data_crop[1, 2, :], data_crop[1, 3, :]]
    soybean = [data_crop[2, 0, :], data_crop[2, 1, :], data_crop[2, 2, :], data_crop[2, 3, :]]
    labels = ["Default", "CO2", "Percipitation", "Temperature"]
    # colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    colors_list = sns.color_palette("Set3", n_colors=6, desat=.9).as_hex()  #
    colors = [colors_list[2], colors_list[0], colors_list[5], colors_list[3]]
    line_color = sns.color_palette().as_hex()

    bplot1 = plt.boxplot(rice, patch_artist=True, labels=labels, positions=(1, 1.3, 1.6, 1.9), widths=0.2,
                         medianprops={'color': f'{line_color[3]}', 'linewidth': '2.0'},
                         showmeans=True, meanline=True, meanprops={'color': 'black'})
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(maize, patch_artist=True, labels=labels, positions=(2.5, 2.8, 3.1, 3.4), widths=0.2,
                         medianprops={'color': f'{line_color[3]}', 'linewidth': '2.0'},
                         showmeans=True, meanline=True, meanprops={'color': 'black'})
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(soybean, patch_artist=True, labels=labels, positions=(4, 4.3, 4.6, 4.9), widths=0.2,
                         medianprops={'color': f'{line_color[3]}', 'linewidth': '2.0'},
                         showmeans=True, meanline=True, meanprops={'color': 'black'})
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    positions = [[1, 1.3, 1.6, 1.9], [2.5, 2.8, 3.1, 3.4], [4, 4.3, 4.6, 4.9]]
    for i in range(3):
        position = positions[i]
        for j in range(len(labels)):
            data = data_crop[i, j, :]
            y = np.max(data) + 3
            if np.median(data) > 50:
                y = np.min(data) - 10
            axes.text(position[j] - 0.1, y, f"{np.mean(data):.3f}", fontsize=20, c='black')

    x_position = [1, 2.5, 4]
    x_position_fmt = ["Rice", "Maize", "Soybean"]
    plt.xticks([i + 0.9 / 2 for i in x_position], x_position_fmt)
    plt.yticks(range(-80, 160, 40))
    plt.ylabel('Yield change (%)', fontsize=22)
    plt.axhline(y=0, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885

    plt.legend(bplot2['boxes'], labels, loc='best', fontsize=20)  # 绘制表示框，右下角绘制
    plt.savefig(f'{Figout}/{run}_change_ssp585_box.eps', format='eps', dpi=300)  # timeseries_lines
    plt.savefig(f'{Figout}/{run}_change_ssp585_box.png', format='png', dpi=300)  # timeseries_lines
    # plt.show()


def get_data(path, Figout, crop, run):
    names = ['rice', 'maize', 'soybean']
    idx = [0, 1, 2]
    scenarios = ['default', 'co2', 'precipitation', 'temperature']
    data_crop = np.empty((3, 4, 85))
    df = pd.DataFrame()
    veg = []
    scenariox = []
    media = []
    mean = []
    media_up = []
    media_down = []
    data_max = []
    data_min = []

    for i, name in enumerate(names):
        VarFile = f'{path}/default/{name}_output_ssp585_default.nc'
        with xr.open_dataset(VarFile) as ds1:
            ds1 = ds1.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100), drop=True)
            ds1 = ds1[f"{run}"]
            # ds1 = ds1["TAGP"]
            ds_a1 = ds1.where(crop == i, drop=True)
            ssp126 = ds_a1.groupby("time.year").max("time").groupby('year').mean(...)
            default_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
            veg.append(str(name))
            scenariox.append("default")
            media.append(np.median(default_land.values))
            mean.append(np.mean(default_land.values))
            media_up.append(np.percentile(default_land.values, 75))
            media_down.append(np.percentile(default_land.values, 25))
            data_max.append(default_land.max(...).values)
            data_min.append(default_land.min(...).values)

        VarFile = f'{path}/co2/{name}_output_ssp585_co2.nc'
        # print(VarFile)
        with xr.open_dataset(VarFile) as ds2:
            ds2 = ds2.where((ds2.time.dt.month > 4) & (ds2.time.dt.month < 12) & (ds2.time.dt.year > 2014) & (ds2.time.dt.year < 2100), drop=True)
            ds2 = ds2[f"{run}"]
            # ds2 = ds2["TAGP"]
            ds_a2 = ds2.where(crop == i, drop=True)
            ssp245 = ds_a2.groupby("time.year").max("time").groupby('year').mean(...)
            co2_land = (ssp245 - ssp245[0]) / ssp245[0] * 100
            veg.append(str(name))
            scenariox.append("co2")
            media.append(np.median(co2_land.values))
            mean.append(np.mean(co2_land.values))
            media_up.append(np.percentile(co2_land.values, 75))
            media_down.append(np.percentile(co2_land.values, 25))
            data_max.append(co2_land.max(...).values)
            data_min.append(co2_land.min(...).values)

        VarFile = f'{path}/precipitation/{name}_output_ssp585_precipitation.nc'
        # print(VarFile)
        with xr.open_dataset(VarFile) as ds3:
            ds3 = ds3.where((ds3.time.dt.month > 4) & (ds3.time.dt.month < 12) & (ds3.time.dt.year > 2014) & (ds3.time.dt.year < 2100), drop=True)
            ds3 = ds3[f"{run}"]
            # ds3 = ds3["TAGP"]
            ds_a3 = ds3.where(crop == i, drop=True)
            ssp370 = ds_a3.groupby("time.year").max("time").groupby('year').mean(...)
            precipitation_land = (ssp370 - ssp370[0]) / ssp370[0] * 100
            veg.append(str(name))
            scenariox.append("precipitation")
            media.append(np.median(precipitation_land.values))
            mean.append(np.mean(precipitation_land.values))
            media_up.append(np.percentile(precipitation_land.values, 75))
            media_down.append(np.percentile(precipitation_land.values, 25))
            data_max.append(precipitation_land.max(...).values)
            data_min.append(precipitation_land.min(...).values)

        VarFile = f'{path}/temperature/{name}_output_ssp585_temperature.nc'
        # print(VarFile)
        with xr.open_dataset(VarFile) as ds4:
            ds4 = ds4.where((ds4.time.dt.month > 4) & (ds4.time.dt.month < 12) & (ds4.time.dt.year > 2014) & (ds4.time.dt.year < 2100), drop=True)
            ds4 = ds4[f"{run}"]
            # ds4 = ds4["TAGP"]
            ds_a4 = ds4.where(crop == i, drop=True)
            ssp585 = ds_a4.groupby("time.year").max("time").groupby('year').mean(...)
            temperature_land = (ssp585 - ssp585[0]) / ssp585[0] * 100
            veg.append(str(name))
            scenariox.append("temperature")
            media.append(np.median(temperature_land.values))
            mean.append(np.mean(temperature_land.values))
            media_up.append(np.percentile(temperature_land.values, 75))
            media_down.append(np.percentile(temperature_land.values, 25))
            data_max.append(temperature_land.max(...).values)
            data_min.append(temperature_land.min(...).values)
        # crop = [[default_land.values],[co2_land.values],[precipitation_land.values],[temperature_land.values]]

        data_crop[i, 0, :] = default_land.values
        data_crop[i, 1, :] = co2_land.values
        data_crop[i, 2, :] = precipitation_land.values
        data_crop[i, 3, :] = temperature_land.values

    df['veg'] = pd.Series(veg)
    df['scenario'] = pd.Series(scenariox)
    df['mean'] = pd.Series(mean)
    df['media'] = pd.Series(media)
    df['media_up'] = pd.Series(media_up)
    df['media_down'] = pd.Series(media_down)
    df['data_max'] = pd.Series(data_max)
    df['data_min'] = pd.Series(data_min)

    df.to_excel(f"./box_{run}.xlsx", sheet_name=f'{run}', index=True)
    box(data_crop, Figout, run)


if __name__ == '__main__':
    import glob, os, shutil, sys


    argv = sys.argv
    run = str(argv[1])

    path = "/stu01/xuqch3/finished/data/"
    pathin = f"{path}/PCSE/output/"
    Figout = './'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''
    get_data(pathin, Figout, crop, run)
