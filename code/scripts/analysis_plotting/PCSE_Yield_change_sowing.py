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

    path = "/tera04/zhwei/PCSE/data/output/adaptation/sowing/"
    #path = '/tera01/xuqch3/PCSE/sensitivity/harvest_date/sowing'
    # Figout = '/tera04/zhwei/PCSE/Fig'


    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop


    names = ['rice', 'maize', 'soybean']
    idxs=[0,1,2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    # scenarios=['default','co2','precipitation','temperature']
    # scenarios=['default','final','strategy']
    # scenarios=['default','final']
    # ssps=['ssp126','ssp245','ssp370','ssp585']
    ssps = ['ssp585']

    df = pd.DataFrame()
    veg=[]
    sspx=[]
    scenariox=[]
    Yieldmean=[]
    Yieldstd=[]

    for scenario in scenarios:
        for name,idx in zip(names,idxs):
            default_ssp585=xr.open_dataset(f'{path}/{name}_output_ssp585_sowing_Max_Yield_default.nc')["TAGP"] 
            default_ssp585 = default_ssp585.where(crop == idx, drop=True)
            default_ssp585 = default_ssp585.groupby('year').mean(...)[0]

            VarFile = f'{path}/{name}_output_ssp585_sowing_Max_Yield_{scenario}.nc'
            print(VarFile)
            with xr.open_dataset(VarFile) as ds1:
                # ds1 = ds1.where((ds1.year > 2015), drop=True)
                ds1 = ds1["TAGP"] 
                ds_a4 = ds1.where(crop == idx, drop=True)
                ssp585 = ds_a4.groupby('year').mean(...)
                ssp585_land = (ssp585 - default_ssp585) / default_ssp585 * 100
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP585 mean: '+str(ssp585_land.mean(...).values))
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP585 std: '+str(ssp585_land.std(...).values))
                veg.append(str(name))
                scenariox.append(str(scenario))
                sspx.append('ssp585')
                Yieldmean.append(ssp585_land.mean(...).values)
                Yieldstd.append(ssp585_land.std(...).values)
                print(veg,scenario,Yieldmean,Yieldstd)


    df['veg']           =  pd.Series(veg)
    df['scenario']      =  pd.Series(scenariox)
    df['sspx']          =  pd.Series(sspx)
    df['Yieldmean']     =  pd.Series(Yieldmean)
    df['Yieldstd']      =  pd.Series(Yieldstd)
    df.to_csv('Yield_sowing.csv')





   



