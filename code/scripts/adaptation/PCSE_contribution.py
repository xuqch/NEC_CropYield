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
import pymannkendall as mk
import random
import pandas as pd
import statsmodels.formula.api as smf

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


def filter_nan(s=np.array([]), o=np.array([])):
    """
    this functions removed the data from simulated and observed data
    whereever the observed data contains nan

    this is used by all other functions, otherwise they will produce nan as
    output
    """
    data = np.array([s.flatten(), o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]

    return data[:, 0], data[:, 1]


def mk_test(data=np.array([])):
    trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(data)
    return slope  # trend, h, p, z, Tau, s, var_s, slope, intercept


def contribution_slope(xx, yy):  # data_x=np.array([]),data_y=np.array([])
    data_x = pd.Series(xx)
    data_x.name = "a"
    data_y = pd.Series(yy)
    data_y.name = "y"
    data_x

    data = pd.concat([data_x, data_y], axis=1)
    # print(data)
    mod = smf.ols(formula='y~a', data=data)
    res = mod.fit()
    # print(res.summary())
    return res.params['a']


def contribution_analysis(con_1, con_2, con_3, con_main):
    con_final = np.abs(con_main) / (con_1 + con_2 + con_3)
    return con_final


if __name__ == '__main__':
    import glob, os, shutil

    path = "/stu01/xuqch3/finished/data/"
    pathin = f'{path}/PCSE/output/sensitivity/'
    path1 = f"{path}/PCSE/atm/"
    path2 = f'{path}/input/'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop

    '''
    rice:0, maize:1, soybean:2
    '''
    # Vars = ['DVS', 'LAI', 'RD', 'SM', 'TAGP', 'TRA', 'TWLV', 'TWRT', 'TWSO', 'TWST', 'WWLOW']
    # Co = ['', '', '(kg/ha)', '(kg/ha)', '(kg/ha)', '(kg/ha)', '(kg/ha)', '(cm/day)', '(cm)', '', '(cm)']
    # Vars = ['TAGP']
    # Co = ['(kg/ha)']
    # colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    names = ['rice', 'maize', 'soybean']

    idxs = [0, 1, 2]
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    ssps = ['ssp585']
    df = pd.DataFrame()
    # k=len(np.arrange(np.arange(2015,2100,10)))
    veg = []
    sspx = []
    contribution_co2 = []
    contribution_precipitation = []
    contribution_tas = []

    # Apr 15-->105

    for name, idx, colr in zip(names, idxs, colors):
        for ssp in ssps:
            ####
            amt0 = xr.open_dataset(f'{path2}/PCSE_input_{ssp}.nc')
            amt1 = (amt0.where((amt0.time.dt.month > 4) & (amt0.time.dt.month < 12) & (amt0.time.dt.year > 2014) & (amt0.time.dt.year < 2100),
                               drop=True)).where(crop == idx, drop=True)
            amt2 = amt1.groupby("time.year").mean("time").groupby('year').mean(...)

            header_list = ["year", "co2"]
            co2 = pd.read_csv(f'{path1}/../co2/co2_{ssp}_annual_2015_2100.txt', delimiter=r"\s+", names=header_list)
            dco2 = co2["co2"]
            # dco2 = co2.set_index(['year'])

            ds1 = xr.open_dataset(f'{pathin}/default/{name}_output_{ssp}_default.nc')["TAGP"]
            ds1 = (ds1.where((ds1.time.dt.month > 4) & (ds1.time.dt.month < 12) & (ds1.time.dt.year > 2014) & (ds1.time.dt.year < 2100),
                             drop=True)).where(crop == idx, drop=True)
            ds1Yield = ds1.groupby("time.year").max("time").groupby('year').mean(...)

            ds2 = xr.open_dataset(f'{pathin}/co2/{name}_output_{ssp}_co2.nc')["TAGP"]
            ds2 = (ds2.where((ds2.time.dt.month > 4) & (ds2.time.dt.month < 12) & (ds2.time.dt.year > 2014) & (ds2.time.dt.year < 2100),
                             drop=True)).where(crop == idx, drop=True)
            ds2Yield = ds2.groupby("time.year").max("time").groupby('year').mean(...)

            ds3 = xr.open_dataset(f'{pathin}/precipitation/{name}_output_{ssp}_precipitation.nc')["TAGP"]
            ds3 = (ds3.where((ds3.time.dt.month > 4) & (ds3.time.dt.month < 12) & (ds3.time.dt.year > 2014) & (ds3.time.dt.year < 2100),
                             drop=True)).where(crop == idx, drop=True)
            ds3Yield = ds3.groupby("time.year").max("time").groupby('year').mean(...)

            ds4 = xr.open_dataset(f'{pathin}/temperature/{name}_output_{ssp}_temperature.nc')["TAGP"]
            ds4 = (ds4.where((ds4.time.dt.month > 4) & (ds4.time.dt.month < 12) & (ds4.time.dt.year > 2014) & (ds4.time.dt.year < 2100),
                             drop=True)).where(crop == idx, drop=True)
            ds4Yield = ds4.groupby("time.year").max("time").groupby('year').mean(...)

            # for yr in np.arange(2015,2100,10):
            # CO2x=dco2.loc[yr:(yr+9)].values
            # print(CO2x)
            slope_co2 = mk_test(dco2)

            # precip=amt2.where((amt2.year > yr)&(amt2.year < (yr+11)), drop=True).pr
            precip = amt2.pr
            # print(precip)
            slope_precip = mk_test(precip)

            # tas=amt2.where((amt2.year > yr)&(amt2.year < (yr+10)), drop=True)
            tas = (amt2.tasmax + amt2.tasmin) / 2.0
            # print(tas)
            slope_tas = mk_test(tas)

            # Yileld=ds1Yield.where((ds1Yield.year > yr)&(ds1Yield.year < (yr+10)), drop=True)
            # print(Yileld)
            slope_Yileld = mk_test(ds1Yield)

            DeltaYc_co2 = ds1Yield - ds2Yield
            DeltaYc_preciptation = ds1Yield - ds3Yield
            DeltaYc_temperature = ds1Yield - ds4Yield

            Dco2 = dco2[:] - dco2[0]
            Dpreciptation = precip - precip[0]
            Dtas = tas - tas[0]

            sen_co2 = contribution_slope(Dco2, DeltaYc_co2.values)
            print(sen_co2)
            sen_preciptation = contribution_slope(Dpreciptation.values, DeltaYc_preciptation.values)
            print(sen_preciptation)

            sen_tas = contribution_slope(Dtas.values, DeltaYc_temperature.values)
            print(sen_tas)

            con_co2 = abs(sen_co2 * slope_co2) / (
                        abs(sen_tas * slope_tas) + abs(sen_co2 * slope_co2) + abs(sen_preciptation * slope_precip)) * 100.0  # *100  #./slope_Yileld
            con_preciptation = abs(sen_preciptation * slope_precip) / (
                        abs(sen_tas * slope_tas) + abs(sen_co2 * slope_co2) + abs(sen_preciptation * slope_precip)) * 100.0
            con_tas = abs(sen_tas * slope_tas) / (abs(sen_tas * slope_tas) + abs(sen_co2 * slope_co2) + abs(sen_preciptation * slope_precip)) * 100.0

            # con_preciptation=sen_preciptation*slope_precip*100./slope_Yileld
            # con_tas=sen_tas*slope_tas*100./slope_Yileld
            print(name, con_co2, con_preciptation, con_tas)
            # print(con_co2,con_preciptation,con_tas)
            veg.append(str(name))
            sspx.append(ssp)
            contribution_co2.append(con_co2)
            contribution_precipitation.append(con_preciptation)
            contribution_tas.append(con_tas)

    df['veg'] = pd.Series(veg)
    df['sspx'] = pd.Series(sspx)
    df['contribution_co2'] = pd.Series(contribution_co2)
    df['contribution_precipitation'] = pd.Series(contribution_precipitation)
    df['contribution_tas'] = pd.Series(contribution_tas)
    df.to_excel(f"{path}/code/fig-6/contribution.xlsx", sheet_name='contribution', index=True)

    # exit()
