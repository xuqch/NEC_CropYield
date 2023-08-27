import matplotlib.pyplot as plt
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


def box(data_crop,Figout):
    fig = plt.figure(figsize=(14, 8))
    rice = [data_crop[0,0,:],data_crop[0,1,:],data_crop[0,2,:],data_crop[0,3,:]]
    maize = [data_crop[1,0,:], data_crop[1,1,:],data_crop[1,2,:],data_crop[1,3,:]]
    soybean = [data_crop[2,0,:],data_crop[2,1,:],data_crop[2,2,:],data_crop[2,3,:]]
    labels = ["Default", "CO2", "Percipitation", "Temperature"]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    bplot1 = plt.boxplot(rice, patch_artist=True, labels=labels, positions=(1, 1.3, 1.6, 1.9), widths=0.2, medianprops = {'color':'red','linewidth': '2.0'})
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(maize, patch_artist=True, labels=labels, positions=(2.5, 2.8, 3.1, 3.4), widths=0.2, medianprops = {'color':'red','linewidth': '2.0'})
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(soybean, patch_artist=True, labels=labels, positions=(4, 4.3, 4.6, 4.9), widths=0.2, medianprops = {'color':'red','linewidth': '2.0'})
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2.5, 4]
    x_position_fmt = ["Rice", "Maize", "Soybean"]
    plt.xticks([i + 0.9 / 2 for i in x_position], x_position_fmt,fontsize=18)
    plt.ylabel('TAGP change (%)',fontsize=18)
    # plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    # plt.title('SSP585',fontsize=18)
    plt.legend(bplot2['boxes'], labels, loc='best',fontsize=16)  # 绘制表示框，右下角绘制
    plt.savefig(f'{Figout}/TAGP_change_ssp5851.eps', format='eps', dpi=800)  # timeseries_lines
    plt.savefig(f'{Figout}/TAGP_change_ssp5851.png', format='png', dpi=800)  # timeseries_lines
    # plt.show()


if __name__ == '__main__':
    import glob, os, shutil

    path = "/tera04/zhwei/PCSE/data/output/sensitivity/"
    Figout = '/tera01/xuqch3/PCSE/sensitivity/Fig'
    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"  # 'F:/PCSE/crop/crop.nc'
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''
    ssps = ['ssp126','ssp245', 'ssp370', 'ssp585']
    names = ['rice', 'maize', 'soybean']
    idx = [0, 1, 2]

    scenarios = ['default', 'co2', 'precipitation', 'temperature']
    data_crop = np.empty((3,4,84))
    # ssps=['ssp126','ssp245','ssp370','ssp585']
    df = pd.DataFrame()
    veg = []
    scenariox = []
    media = []
    media_up = []
    media_down = []
    data_max = []
    data_min = []
    # print(data_crop.shape)
    for i,name in enumerate(names):
        VarFile = f'{path}/default/{name}_output_ssp585_default.nc'
        with xr.open_dataset(VarFile) as ds1:
            ds1 = ds1.where((ds1.time.dt.month>4)&(ds1.time.dt.month<12)&(ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
            ds1 = ds1["TAGP"]
            #ds1 = ds1["TWSO"] 
            ds_a1 = ds1.where(crop == i, drop=True)
            ssp126 = ds_a1.groupby("time.year").max("time").groupby('year').mean(...)
            default_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
            veg.append(str(name))
            scenariox.append("default")
            media.append(np.median(default_land.values))
            media_up.append(np.percentile(default_land.values, 75))
            media_down.append(np.percentile(default_land.values, 25))
            data_max.append(default_land.max(...).values)
            data_min.append(default_land.min(...).values)

        VarFile = f'{path}/co2/{name}_output_ssp585_co2.nc'
        # print(VarFile)
        with xr.open_dataset(VarFile) as ds2:
            ds2 = ds2.where((ds2.time.dt.month>4)&(ds2.time.dt.month<12)&(ds2.time.dt.year > 2015)&(ds2.time.dt.year < 2100), drop=True)
            ds2 = ds2["TAGP"]
            #ds2 = ds2["TWSO"] 
            ds_a2 = ds2.where(crop == i, drop=True)
            ssp245 = ds_a2.groupby("time.year").max("time").groupby('year').mean(...)
            co2_land = (ssp245 - ssp245[0]) / ssp245[0] * 100
            veg.append(str(name))
            scenariox.append("co2")
            media.append(np.median(co2_land.values))
            media_up.append(np.percentile(co2_land.values, 75))
            media_down.append(np.percentile(co2_land.values, 25))
            data_max.append(co2_land.max(...).values)
            data_min.append(co2_land.min(...).values)

        VarFile = f'{path}/precipitation/{name}_output_ssp585_precipitation.nc'
        # print(VarFile)
        with xr.open_dataset(VarFile) as ds3:
            ds3 = ds3.where((ds3.time.dt.month>4)&(ds3.time.dt.month<12)&(ds3.time.dt.year > 2015)&(ds3.time.dt.year < 2100), drop=True)
            ds3 = ds3["TAGP"]
            #ds3 = ds3["TWSO"] 
            ds_a3 = ds3.where(crop == i, drop=True)
            ssp370 = ds_a3.groupby("time.year").max("time").groupby('year').mean(...)
            precipitation_land = (ssp370 - ssp370[0]) / ssp370[0] * 100
            veg.append(str(name))
            scenariox.append("precipitation")
            media.append(np.median(precipitation_land.values))
            media_up.append(np.percentile(precipitation_land.values, 75))
            media_down.append(np.percentile(precipitation_land.values, 25))
            data_max.append(precipitation_land.max(...).values)
            data_min.append(precipitation_land.min(...).values)

        VarFile = f'{path}/temperature/{name}_output_ssp585_temperature.nc'
        # print(VarFile)
        with xr.open_dataset(VarFile) as ds4:
            ds4 = ds4.where((ds4.time.dt.month>4)&(ds4.time.dt.month<12)&(ds4.time.dt.year > 2015)&(ds4.time.dt.year < 2100), drop=True)
            ds4 = ds4["TAGP"] 
            #ds4 = ds4["TWSO"] 
            ds_a4 = ds4.where(crop == i, drop=True)
            ssp585 = ds_a4.groupby("time.year").max("time").groupby('year').mean(...)
            temperature_land = (ssp585 - ssp585[0]) / ssp585[0] * 100
            veg.append(str(name))
            scenariox.append("temperature")
            media.append(np.median(temperature_land.values))
            media_up.append(np.percentile(temperature_land.values, 75))
            media_down.append(np.percentile(temperature_land.values, 25))
            data_max.append(temperature_land.max(...).values)
            data_min.append(temperature_land.min(...).values)
        # crop = [[default_land.values],[co2_land.values],[precipitation_land.values],[temperature_land.values]]

        data_crop[i,0,:] = default_land.values
        data_crop[i,1,:] = co2_land.values
        data_crop[i,2,:] = precipitation_land.values
        data_crop[i,3,:] = temperature_land.values
        
    # print(data_crop[1,1,:][~np.isnan(data_crop[1,1,:])])
    # data_crop = data_crop.to_dataset()
    # data_crop[1,0,:] = data_crop
    # print(data_crop)
    # exit(0)
    # df['veg']           =  pd.Series(veg)
    # df['scenario']      =  pd.Series(scenariox)
    # df['media']          =  pd.Series(media)
    # df['media_up']     =  pd.Series(media_up)
    # df['media_down']      =  pd.Series(media_down)
    # df['data_max']     =  pd.Series(data_max)
    # df['data_min']      =  pd.Series(data_min)
    # df.to_csv('box.csv')
    box(data_crop,Figout)
