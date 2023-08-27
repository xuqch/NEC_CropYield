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

    path = "/tera04/zhwei/PCSE/data/output/sensitivity/"
    Figout = '/tera01/xuqch3/PCSE/sensitivity/Fig/yield_change'
    maskfile_Crop = "/tera04/zhwei/PCSE/data/crop_distribution/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''

    names = ['rice', 'maize', 'soybean']
    idxs=[0,1,2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    scenarios=['default','co2','precipitation','temperature']
    ssps=['ssp126','ssp245','ssp370','ssp585']
    df = pd.DataFrame()
    veg = []
    sspx = []
    scenariox = []
    Yieldmean = []
    Yieldstd = []

    #scenarios=['tas']
    for scenario in scenarios:
        for name,idx,colr in zip(names,idxs,colors):
            t0 = time.strftime('%H:%M:%S')
            VarFile = f'{path}/{scenario}/{name}_output_ssp126_{scenario}.nc'
            print(VarFile)
            with xr.open_dataset(VarFile) as ds1:
                #ds1=ds1.sel(time=slice("2016-01-01", "2099-12-31"))
               # ds1 = ds1.where((ds1.time.dt.month==8)&(ds1.time.dt.day==23)&(ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
                ds1 = ds1.where((ds1.time.dt.month>4)&(ds1.time.dt.month<12)&(ds1.time.dt.year > 2015)&(ds1.time.dt.year < 2100), drop=True)
                ds1 = ds1["TAGP"]
                #ds1 = ds1["TWSO"]
                ds_a1 = ds1.where(crop == idx, drop=True)
                ssp126 = ds_a1.groupby("time.year").max("time").groupby('year').mean(...)
                ssp126_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP126 mean: ' + str(ssp126_land.mean(...).values))
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP126 std: '  +  str(ssp126_land.std(...).values))
                veg.append(str(name))
                scenariox.append(str(scenario))
                sspx.append('ssp126')
                Yieldmean.append(ssp126_land.mean(...).values)
                Yieldstd.append(ssp126_land.std(...).values)
                print(Yieldmean,Yieldstd)


            VarFile = f'{path}/{scenario}/{name}_output_ssp245_{scenario}.nc'
            print(VarFile)
            with xr.open_dataset(VarFile) as ds2:
                #ds2=ds2.sel(time=slice("2016-01-01", "2099-12-31"))
                #ds2 = ds2.where((ds2.time.dt.month==8)&(ds2.time.dt.day==23)&(ds2.time.dt.year > 2015)&(ds2.time.dt.year < 2100), drop=True)
                ds2 = ds2.where((ds2.time.dt.month>4)&(ds2.time.dt.month<12)&(ds2.time.dt.year > 2015)&(ds2.time.dt.year < 2100), drop=True)
                ds2 = ds2["TAGP"]
                #ds2 = ds2["TWSO"]
                ds_a2 = ds2.where(crop == idx, drop=True)
                kk=ds_a2.groupby("time.year").max("time")
                kk.to_netcdf(f'{name}_output_ssp245_{scenario}.nc')
                ssp245 = ds_a2.groupby("time.year").max("time").groupby('year').mean(...)

                ssp245_land = (ssp245 - ssp245[0]) / ssp245[0] * 100
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP245 mean: '+str(ssp245_land.mean(...).values))
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP245 std: '+str(ssp245_land.std(...).values))
                veg.append(str(name))
                scenariox.append(str(scenario))
                sspx.append('ssp245')
                Yieldmean.append(ssp245_land.mean(...).values)
                Yieldstd.append(ssp245_land.std(...).values)

            VarFile = f'{path}/{scenario}/{name}_output_ssp370_{scenario}.nc'
            print(VarFile)
            with xr.open_dataset(VarFile) as ds3:
                #ds3=ds3.sel(time=slice("2016-01-01", "2099-12-31"))
                #ds3 = ds3.where((ds3.time.dt.month==8)&(ds3.time.dt.day==23)&(ds3.time.dt.year > 2015)&(ds3.time.dt.year < 2100), drop=True)
                ds3 = ds3.where((ds3.time.dt.month>4)&(ds3.time.dt.month<12)&(ds3.time.dt.year > 2015)&(ds3.time.dt.year < 2100), drop=True)
                ds3 = ds3["TAGP"]
                #ds3 = ds3["TWSO"]
                ds_a3 = ds3.where(crop == idx, drop=True)

                ssp370 = ds_a3.groupby("time.year").max("time").groupby('year').mean(...)
                ssp370_land = (ssp370 - ssp370[0]) / ssp370[0] * 100
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP370 mean: '+str(ssp370_land.mean(...).values))
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP370 std: '+str(ssp370_land.std(...).values))
                veg.append(str(name))
                scenariox.append(str(scenario))
                sspx.append('ssp370')
                Yieldmean.append(ssp370_land.mean(...).values)
                Yieldstd.append(ssp370_land.std(...).values)

            VarFile = f'{path}/{scenario}/{name}_output_ssp585_{scenario}.nc'
            print(VarFile)
            with xr.open_dataset(VarFile) as ds4:
                #ds4 = ds4.where((ds4.time.dt.month==8)&(ds4.time.dt.day==23)&(ds4.time.dt.year > 2015)&(ds4.time.dt.year < 2100), drop=True)
                #ds4 = ds4["TAGP"]
                ds4 = ds4.where((ds4.time.dt.month>4)&(ds4.time.dt.month<12)&(ds4.time.dt.year > 2015)&(ds4.time.dt.year < 2100), drop=True)
                ds4 = ds4["TAGP"]
                #ds4 = ds4["TWSO"]
                ds_a4 = ds4.where(crop == idx, drop=True)
                ssp585 = ds_a4.groupby("time.year").max("time").groupby('year').mean(...)
                ssp585_land = (ssp585 - ssp585[0]) / ssp585[0] * 100
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP585 mean: '+str(ssp585_land.mean(...).values))
                print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP585 std: '+str(ssp585_land.std(...).values))
                veg.append(str(name))
                scenariox.append(str(scenario))
                sspx.append('ssp585')
                Yieldmean.append(ssp585_land.mean(...).values)
                Yieldstd.append(ssp585_land.std(...).values)
                # print(veg,scenario,Yieldmean,Yieldstd)
            print('plotting now')

            markers = ['*', 'x', '+']
            lines = [1.5, 1.5, 1.5, 1.5]
            alphas = [1., 1., 1., 1.]
            linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ssp126_land.plot.line(x='year', label='ssp126', linewidth=lines[1], linestyle=linestyles[1],
                                  alpha=alphas[1], color=colors[0])  # ,color = 'blue'
            ssp245_land.plot.line(x='year', label='ssp245', linewidth=lines[2], linestyle=linestyles[2],
                                  alpha=alphas[2], color=colors[1])  # ,color = 'green
            ssp370_land.plot.line(x='year', label='ssp370', linewidth=lines[0], linestyle=linestyles[0],
                                  alpha=alphas[0], color=colors[2])  # ,color = 'orangered'
            ssp585_land.plot.line(x='year', label='ssp585', linewidth=lines[3], linestyle=linestyles[3],
                                  alpha=alphas[3], color=colors[3])  # ,color = 'red'

            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_ylabel('Yield change (%)', fontsize=18)
            ax.set_xlabel('Year', fontsize=20)
            # plt.yticks(np.arange(-20, 80, 20), np.arange(-20, 80, 20))
            # plt.yticks(np.arange(-10, 50, 10), np.arange(-10, 50, 10))
            ax.tick_params(axis='both', top='off', labelsize=16)
            #ax.legend(loc='best', shadow=False, fontsize=12)
            ax.set_title('%s_%s' % (name,fnames[i]))
            plt.tight_layout()
            # plt.savefig(f'{Figout}/{name}_output_{scenario}.eps', format='eps', dpi=800)  # timeseries_lines
            plt.savefig(f'{Figout}/{name}_output_{scenario}.png', format='png', dpi=800)  # timeseries_lines

            # plt.show()

            t1 = time.strftime('%H:%M:%S')
            start_date = t0
            end_date = t1
            start_time = datetime.datetime.strptime(start_date, '%H:%M:%S')
            end_time = datetime.datetime.strptime(end_date, '%H:%M:%S')
            during_time = end_time - start_time
            print(during_time)
            # df['time'] = pd.Series(range(2015, 2101))
            # df['ssp_126'] = pd.Series(ssp126_land)
            # df['ssp_245'] = pd.Series(ssp245_land)
            # df['ssp_370'] = pd.Series(ssp370_land)
            # df['ssp_585'] = pd.Series(ssp585_land)
            # df.to_csv('./%s_%s.csv' % ( scenario,name))
        print(name)
    print('plot end')
    # df['veg']           =  pd.Series(veg)
    # df['scenario']      =  pd.Series(scenariox)
    # df['sspx']          =  pd.Series(sspx)
    # df['Yieldmean']     =  pd.Series(Yieldmean)
    # df['Yieldstd']      =  pd.Series(Yieldstd)
    # df.to_csv('Yield.csv')
    # exit()

