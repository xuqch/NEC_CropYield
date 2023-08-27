import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
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

    path = "/stu01/xuqch3/finished/data/"
    pathin = f'{path}/DSSAT_Crop_Yield_SSP/'
    Figout = f'{path}/code/Fig-S10/'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''

    names = ['rice', 'maize', 'soybean']
    idxs = [0, 1, 2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    veg, sspx, Yieldmean, Yieldstd = [], [], [], []

    for name, idx, colr in zip(names, idxs, colors):

        VarFile = f'{pathin}/Crop_Yield_ssp126.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds1:
            ds1 = ds1["Yield"]
            ds_a1 = ds1.where(crop == idx, drop=True)
            print(ds_a1)
            ssp126 = ds_a1.groupby("year").mean(...)

            ssp126_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP126 mean: ' + str(ssp126_land.mean(...).values))
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP126 std: ' + str(ssp126_land.std(...).values))
            veg.append(str(name))
            sspx.append('ssp126')
            Yieldmean.append(ssp126_land.mean(...).values)
            Yieldstd.append(ssp126_land.std(...).values)

        VarFile = f'{pathin}/Crop_Yield_ssp245.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds2:
            ds2 = ds2["Yield"]
            ds_a2 = ds2.where(crop == idx, drop=True)
            ssp245 = ds_a2.groupby("year").mean(...)

            ssp245_land = (ssp245 - ssp245[0]) / ssp245[0] * 100
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP245 mean: ' + str(ssp245_land.mean(...).values))
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP245 std: ' + str(ssp245_land.std(...).values))
            veg.append(str(name))
            sspx.append('ssp245')
            Yieldmean.append(ssp245_land.mean(...).values)
            Yieldstd.append(ssp245_land.std(...).values)

        VarFile = f'{pathin}/Crop_Yield_ssp370.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds3:
            ds3 = ds3["Yield"]
            ds_a3 = ds3.where(crop == idx, drop=True)
            ssp370 = ds_a3.groupby("year").mean(...)
            ssp370_land = (ssp370 - ssp370[0]) / ssp370[0] * 100
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP370 mean: ' + str(ssp370_land.mean(...).values))
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP370 std: ' + str(ssp370_land.std(...).values))
            veg.append(str(name))
            sspx.append('ssp370')
            Yieldmean.append(ssp370_land.mean(...).values)
            Yieldstd.append(ssp370_land.std(...).values)

        VarFile = f'{pathin}/Crop_Yield_ssp585.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds4:
            ds4 = ds4["Yield"]
            ds_a4 = ds4.where(crop == idx, drop=True)
            ssp585 = ds_a4.groupby("year").mean(...)
            ssp585_land = (ssp585 - ssp585[0]) / ssp585[0] * 100
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP585 mean: ' + str(ssp585_land.mean(...).values))
            print(f' ' + f'{name}' + f' {idx} ' + 'SSP585 std: ' + str(ssp585_land.std(...).values))
            veg.append(str(name))

            sspx.append('ssp585')
            Yieldmean.append(ssp585_land.mean(...).values)
            Yieldstd.append(ssp585_land.std(...).values)

        df1['time'] = pd.Series(range(2015, 2101))
        df1['ssp126'] = pd.Series(ssp126_land)
        df1['ssp245'] = pd.Series(ssp245_land)
        df1['ssp370'] = pd.Series(ssp370_land)
        df1['ssp585'] = pd.Series(ssp585_land)
        df1.to_excel('./%s_TAGP.xlsx' % (name), sheet_name=f'{name}', index=True)
        print(name)
    print('plot end')
    df['veg'] = pd.Series(veg)
    df['sspx'] = pd.Series(sspx)
    df['Yieldmean'] = pd.Series(Yieldmean)
    df['Yieldstd'] = pd.Series(Yieldstd)
    df.to_excel(f"{Figout}/Crop_Yield_TAGP.xlsx", sheet_name='Yield', index=True)

