from email.policy import default
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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


def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    b = np.mean(ys) - m * np.mean(xs)
    return m, b


def line_plot(scenarios, path, Figout, crop,pathvar,runname):
    names = ['rice', 'maize', 'soybean']  #
    # names = ['rice']  #
    idxs = [0, 1, 2]  # 1,
    # idxs = [0]  # 1,
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    veg = []
    sspx = []
    Yieldmean = []
    scenariox = []
    Yieldstd = []

    if os.path.exists(Figout + f'xlsx/{scenarios}_soybean_{runname}.xlsx'):
        for idx, name in enumerate(names):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            stnlist = f"{Figout}{scenarios}_{name}.xlsx"
            station_list = pd.read_excel(stnlist, header=0, sheet_name=f'{name}')  # ,header=0
            for i, ssp in enumerate(ssps):
                ssp_land = xr.DataArray(np.array(station_list[f'{ssp}']), coords={"time": station_list.time.values}, dims=["time"])
                ssp_land.plot.line(x='time', label=ssp, linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                                   color=colors[i], marker='D')

                m, b = best_fit_slope_and_intercept(np.arange(2016, 2100), np.array(ssp_land.values))
                regression_line = []
                for a in range(2016, 2100):
                    regression_line.append((m * a) + b)
                regression = xr.DataArray(np.array(regression_line), coords={"time": ssp_land.time.values}, dims=["time"])
                regression.plot.line(x='time', color=colors[i], lw=1.5)
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_ylabel('Yield change (%)', fontsize=18)
            ax.set_xlabel('Year', fontsize=20)
            plt.yticks(np.arange(-30, 80, 20), np.arange(-30, 80, 20), fontsize=20)  # soybean
            ax.tick_params(axis='both', top='off', labelsize=16)
            ax.legend(loc='best', shadow=False, fontsize=12)
            ax.set_title('%s' % (name))
            plt.tight_layout()
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.eps', format='eps', dpi=800)  # timeseries_lines
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.png', format='png', dpi=800)  # timeseries_lines

    else:
        for name, idx in zip(names, idxs):
            data = []
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for i, ssp in enumerate(ssps):
                VarFile = f'{path}/{name}_output_{ssp}_{scenarios}.nc'
                print(VarFile)
                with xr.open_dataset(VarFile) as ds1:
                    ds1 = ds1.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100),
                                    drop=True)
                    ds1 = ds1[f"{runname}"]
                    ds_a1 = ds1.where(crop == idx, drop=True)
                    ssp126 = ds_a1.groupby("time.year").max("time").groupby('year').mean(...)
                    ssp126_land = (ssp126 - ssp126[0]) / ssp126[0] * 100
                    print(f'{name}' + f' {idx} ' + 'SSP126 mean: ' + str(ssp126_land.mean(...).values))
                    print(f'{name}' + f' {idx} ' + 'SSP126 std: ' + str(ssp126_land.std(...).values))
                    veg.append(str(name))
                    data.append(ssp126_land.values)
                    sspx.append(ssp)
                    Yieldmean.append(ssp126_land.mean(...).values)
                    Yieldstd.append(ssp126_land.std(...).values)

                ssp126_land.plot.line(x='year', label=ssp, linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                                      color=colors[i], marker='D')

                m, b = best_fit_slope_and_intercept(np.arange(2015, 2100), np.array(ssp126_land.values))
                regression_line = []
                for a in range(2015, 2100):
                    regression_line.append((m * a) + b)
                regression = xr.DataArray(np.array(regression_line), coords={"time": ssp126_land.year.values}, dims=["time"])
                regression.plot.line(x='time', color=colors[i], lw=1.5)

            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_ylabel('Yield change (%)', fontsize=18)
            ax.set_xlabel('Year', fontsize=20)
            # plt.yticks(np.arange(-50, 125, 25), np.arange(-50, 125, 25), fontsize=20)  # soybean
            ax.tick_params(axis='both', top='off', labelsize=16)
            ax.legend(loc='best', shadow=False, fontsize=12)
            ax.set_title('%s' % (name))
            plt.tight_layout()
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.eps', format='eps', dpi=800)  # timeseries_lines
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.png', format='png', dpi=800)  # timeseries_lines

            df = pd.DataFrame({'time': ssp126_land.year.values,
                               'ssp126': data[0], 'ssp245': data[1],
                               'ssp370': data[2], 'ssp585': data[3]})  # 'R': R,'KGE': KGE,
            df.to_excel(f'./xlsx/{scenarios}_{name}_{runname}.xlsx', sheet_name=f'{name}', index=True)
        print(name)
        print('plot end')
        df1 = pd.DataFrame({'veg': pd.Series(veg),
                            'sspx': pd.Series(sspx), 'Yieldmean': pd.Series(Yieldmean),
                            'Yieldstd': pd.Series(Yieldstd)})  # 'R': R,'KGE': KGE,
        df1.to_excel(f'./xlsx/Yield_{scenarios}_{runname}.xlsx', sheet_name=f'Yield', index=True)

def irrigate_line_plot(scenarios, path, Figout, crop,pathvar,runname):
    names = ['rice', 'maize', 'soybean']  #
    idxs = [0, 1, 2]  # 1,
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']

    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    veg = []
    sspx = []
    Yieldmean = []
    scenariox = []
    Yieldstd = []

    if os.path.exists(Figout + f'xlsx/{scenarios}_soybean_{runname}.xlsx'):
        for idx, name in enumerate(names):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            stnlist = f"{Figout}{scenarios}_{name}.xlsx"
            station_list = pd.read_excel(stnlist, header=0, sheet_name=f'{name}')  # ,header=0
            for i, ssp in enumerate(ssps):
                ssp_land = xr.DataArray(np.array(station_list[f'{ssp}']), coords={"time": station_list.time.values}, dims=["time"])
                ssp_land.plot.line(x='time', label=ssp, linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                                   color=colors[i], marker='D')

                m, b = best_fit_slope_and_intercept(np.arange(2016, 2100), np.array(ssp_land.values))
                regression_line = []
                for a in range(2016, 2100):
                    regression_line.append((m * a) + b)
                regression = xr.DataArray(np.array(regression_line), coords={"time": ssp_land.time.values}, dims=["time"])
                regression.plot.line(x='time', color=colors[i], lw=1.5)
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_ylabel('Yield change (%)', fontsize=18)
            ax.set_xlabel('Year', fontsize=20)
            plt.yticks(np.arange(-30, 80, 20), np.arange(-30, 80, 20), fontsize=20)  # soybean
            ax.tick_params(axis='both', top='off', labelsize=16)
            ax.legend(loc='best', shadow=False, fontsize=12)
            ax.set_title('%s' % (name))
            plt.tight_layout()
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.eps', format='eps', dpi=800)  # timeseries_lines
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.png', format='png', dpi=800)  # timeseries_lines

    else:
        for name, idx in zip(names, idxs):
            data = []
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for i, ssp in enumerate(ssps):
                VarFile = f'{path}/{name}_output_{ssp}_{scenarios}.nc'
                print(VarFile)
                with xr.open_dataset(VarFile) as ds1:
                    ds1 = ds1.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100),
                                    drop=True)
                    ds1 = ds1[f"{runname}"]
                    ds_a1 = ds1.where(crop == idx, drop=True)
                    ssp126 = ds_a1.groupby("time.year").max("time").groupby('year').mean(...)

                    ds =xr.open_dataset(f'/stu01/xuqch3/PCSE/NEX-GDDP-CMIP6/output/sensitivity/default/{name}_output_{ssp}_default.nc')
                    default = ds.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100),
                                    drop=True)
                    default = default[f"{runname}"]
                    default = default.groupby("time.year").max("time")
                    default = default.where(crop == idx, drop=True)
                    default = default.groupby('year').mean(...)[0]

                    ssp126_land = (ssp126 - default) / default * 100
                    print(f'{name}' + f' {idx} ' + 'SSP126 mean: ' + str(ssp126_land.mean(...).values))
                    print(f'{name}' + f' {idx} ' + 'SSP126 std: ' + str(ssp126_land.std(...).values))
                    veg.append(str(name))
                    data.append(ssp126_land.values)
                    sspx.append(ssp)
                    Yieldmean.append(ssp126_land.mean(...).values)
                    Yieldstd.append(ssp126_land.std(...).values)

                ssp126_land.plot.line(x='year', label=ssp, linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                                      color=colors[i], marker='D')

                m, b = best_fit_slope_and_intercept(np.arange(2015, 2100), np.array(ssp126_land.values))
                regression_line = []
                for a in range(2015, 2100):
                    regression_line.append((m * a) + b)
                regression = xr.DataArray(np.array(regression_line), coords={"time": ssp126_land.year.values}, dims=["time"])
                regression.plot.line(x='time', color=colors[i], lw=1.5)

            # ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_ylabel('Yield change (%)', fontsize=18)
            ax.set_xlabel('Year', fontsize=20)
            # plt.yticks(np.arange(-50, 125, 25), np.arange(-50, 125, 25), fontsize=20)  # soybean
            ax.tick_params(axis='both', top='off', labelsize=16)
            ax.legend(loc='best', shadow=False, fontsize=12)
            ax.set_title('%s' % (name))
            plt.tight_layout()
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.eps', format='eps', dpi=800)  # timeseries_lines
            plt.savefig(f'{Figout}/{pathvar}/{name}_output_{scenarios}_{runname}.png', format='png', dpi=800)  # timeseries_lines

            df = pd.DataFrame({'time': ssp126_land.year.values,
                               'ssp126': data[0], 'ssp245': data[1],
                               'ssp370': data[2], 'ssp585': data[3]})  # 'R': R,'KGE': KGE,
            df.to_excel(f'./xlsx/{scenarios}_{name}_{runname}.xlsx', sheet_name=f'{name}', index=True)
        print(name)
        print('plot end')
        df1 = pd.DataFrame({'veg': pd.Series(veg),
                            'sspx': pd.Series(sspx), 'Yieldmean': pd.Series(Yieldmean),
                            'Yieldstd': pd.Series(Yieldstd)})  # 'R': R,'KGE': KGE,
        df1.to_excel(f'./xlsx/Yield_{scenarios}_{runname}.xlsx', sheet_name=f'Yield', index=True)


def ssp585(scenarios, path, Figout,pathvar,runname):
    names = ['rice', 'maize', 'soybean']

    run_name = 'TAGP'
    idxs = [0, 1, 2]
    colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
    lines = [1.5, 1.5, 1.5, 1.5]
    alphas = [1., 1., 1., 1.]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, name in enumerate(names):

        df = pd.read_excel(f"{path}/xlsx/{scenarios}_{name}_{runname}.xlsx", header=0, sheet_name=f'{name}')  # ,header=0
        year = df.time
        ssp585 = df.ssp585

        lines = [1.5, 1.5, 1.5, 1.5]
        alphas = [1., 1., 1., 1.]
        linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
        ssp585_land = xr.DataArray(np.array(ssp585), coords={"time": year}, dims=["time"])
        ssp585_land.plot.line(x='time', label=f'{name}', linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                              color=colors[i], marker='D')

        m, b = best_fit_slope_and_intercept(np.arange(2015, 2100), np.array(ssp585_land.values))
        regression_line = []
        for a in range(2015, 2100):
            regression_line.append((m * a) + b)
        regression = xr.DataArray(np.array(regression_line), coords={"time": year}, dims=["time"])
        regression.plot.line(x='time', color=colors[i], lw=1.5)

    # ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('Yield Change (%)', fontsize=18)
    ax.set_xlabel('Year', fontsize=20)
    ax.tick_params(axis='both', top='off', labelsize=16)
    ax.legend(loc='best', shadow=False, fontsize=16)

    plt.tight_layout()
    plt.savefig(f'{Figout}/{pathvar}/irrigate_ssp585_output_{runname}_Yield_change.eps', format='eps', dpi=300)  # timeseries_lines
    plt.savefig(f'{Figout}/{pathvar}/irrigate_ssp585_output_{runname}_Yield_change.png', format='png', dpi=300)  # timeseries_lines



if __name__ == '__main__':
    import glob, os, shutil
    path = '/stu01/xuqch3/finished/data/'
    Figout = f'{path}/code/fig-4/'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop
    '''
    rice:0, maize:1, soybean:2
    -'''
    runname = 'TAGP'

    scenarios = 'default'  # default_pr/precipitation
    pathin = f"{path}/PCSE/output/sensitivity/{scenarios}/"  # precipitation
    line_plot(scenarios, pathin, Figout, crop, scenarios, runname)

    scenarios = 'co2'  # default_pr/precipitation
    pathin = f"{path}/PCSE/output/sensitivity/{scenarios}/"  # precipitation
    line_plot(scenarios, pathin, Figout, crop, scenarios, runname)

    scenarios = 'temperature'  # default_pr/precipitation
    pathin = f"{path}/PCSE/output/sensitivity/{scenarios}/"  # precipitation
    line_plot(scenarios, pathin, Figout, crop, scenarios, runname)

    scenarios = 'precipitation'  # default_pr/precipitation
    pathin = f"{path}/PCSE/output/sensitivity/{scenarios}/"  # precipitation
    line_plot(scenarios, pathin, Figout, crop, scenarios, runname)


    scenarios = 'default_add'  # default_pr/precipitation
    pathvar = 'default'
    pathin = f"{path}/PCSE/output/sensitivity/{scenarios}/"  # precipitation
    irrigate_line_plot(scenarios, pathin, Figout, crop, pathinvar, runname)
    pathin = f"{path}/code/fig-4/"  # precipitation
    ssp585(scenarios, pathin, Figout, pathvar, runname)



