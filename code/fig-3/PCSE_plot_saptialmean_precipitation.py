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


def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    b = np.mean(ys) - m * np.mean(xs)
    return m, b


if __name__ == '__main__':
    import glob, os, shutil

    Inpath = "/stu01/xuqch3/finished/data/"
    Figout = f'{Inpath}/code/fig-3/'
    maskfile_Crop = f"{Inpath}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''

    names = ['rice', 'maize', 'soybean']
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    t0 = time.strftime('%H:%M:%S')
    df = pd.DataFrame()
    for i, ssp in enumerate(ssps):
        VarFile = f'{Inpath}/met/{ssp}/pr_day_ensmean_{ssp}_2015-2100.nc'
        print(VarFile)
        with xr.open_dataset(VarFile) as ds1:
            ds1 = ds1.where((ds1.time.dt.year >= 2015) & (ds1.time.dt.year <= 2100), drop=True)
            ds1 = ds1['pr'].resample(time='1Y').mean()  # .squeeze().values
            ds_a1 = ds1.where(crop >= 0.0, drop=True).squeeze()
            ssp_data = ds_a1.groupby('time').mean(...) * 86400 * 365


        m, b = best_fit_slope_and_intercept(np.arange(2015, 2101), np.array(ssp_data.values))
        regression_line = []
        for a in range(2015, 2101):
            regression_line.append((m * a) + b)

        ssp_data.plot.line(x='time', label=ssp, linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                           color=colors[i], marker='D')
        regression = xr.DataArray(np.array(regression_line), coords={"time": ssp_data.time}, dims=["time"])
        regression.plot.line(x='time', color=colors[i], lw=1.5)
        # plt.plot(ssp_data.time, regression_line, color=colors[i], lw=1.5)  # 预测与实测数据之间的回归线

        ax.legend(loc='best', shadow=False, fontsize=14)
        # ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_ylabel('Precipitation (mm/year)', fontsize=18)
        ax.set_xlabel('Year', fontsize=20)
        ax.set_title(None)

        # ax.tick_params(axis='both', top='off', labelsize=16)
        plt.tight_layout()
        plt.savefig(f'{Figout}/precipitation_year.png', dpi=600)  # timeseries_lines
        plt.savefig(f'{Figout}/precipitation_year.eps', dpi=600)  # timeseries_lines
        plt.show()

        df[f'{ssp}'] = pd.Series(ssp_data.values)
    df['time'] = pd.Series(ssp_data.time)
    df.to_excel('./pr.xlsx', sheet_name=f'pr', index=True)

    t1 = time.strftime('%H:%M:%S')
    start_date = t0
    end_date = t1
    start_time = datetime.datetime.strptime(start_date, '%H:%M:%S')
    end_time = datetime.datetime.strptime(end_date, '%H:%M:%S')
    during_time = end_time - start_time
    print(during_time)
    print('end')
