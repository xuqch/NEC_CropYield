import sys, os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import pcse
from pcse.fileinput import YAMLCropDataProvider
from itertools import product
from pcse.models import Wofost72_WLP_FD, Wofost80_WLP_FD_beta
from pcse.fileinput import CABOFileReader
from Xarrayweatherdataprovider import XarrayWeatherDataProvider
from pcse.util import WOFOST80SiteDataProvider
from pcse.base import ParameterProvider
from pcse.exceptions import WeatherDataProviderError
# from numba import jit, prange

from joblib import Parallel, delayed
import multiprocessing

data_dir = os.path.join(os.getcwd(), "/stu01/xuqch3/finished/data/PCSE/")
print(data_dir)
co2data_dir = os.path.join(data_dir, "co2/")
print(co2data_dir)

Syear = 2015
Eyear = 2100
Sdate = str(str(Syear) + "0101")
Edate = str(str(Eyear) + "0101")
years = range(Syear, Eyear)
print(" ")
print("**********************")
print("This code was built with:")
print("python version: %s " % sys.version)
print("PCSE version: %s" % pcse.__version__)
print("Syear: %s" % Syear)
print("Eyear: %s" % Eyear)
data_dir1 = os.path.join(os.getcwd(), "/stu01/xuqch3/finished/data/PCSE/")
fname = os.path.join(data_dir1, "output/adaptation/sowing/")
print("output file: %s" % fname)
print("**********************")
print(" ")


def FilePrapare(metrodata_dir, ssp):
    # get data ready!
    VarFile = ('%s/PCSE_input_%s.nc' % (metrodata_dir, ssp))
    print(VarFile)
    with xr.open_dataset(VarFile, decode_times=True, drop_variables=['height_2', 'height'],
                         chunks={'time': 3650}) as ds:
        try:
            print(" ")
            print("**********************")
            print(" ")
            print("forcing data prepared")
            print(" ")
            print("**********************")
            print(" ")

        except:
            EOFError
    return ds


def agro(year, start):
    agro_maize = """
- {year}-01-01:
    CropCalendar:
        crop_name: '{crop}'
        variety_name: 'Maize_VanHeemst_1988'
        crop_start_date: {startdate}
        crop_start_type: sowing
        crop_end_date:  
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {Nyear}-03-01: null
  """
    agromanagement = yaml.safe_load(agro_maize.format(year=year, Nyear=year + 1, crop='maize', startdate=start))
    return agromanagement


def Initial(data_dir, Start_time, dco2, ds):
    # print(ds)
    crop_data = YAMLCropDataProvider(os.path.join(data_dir, 'crop'))
    crop_data.set_active_crop('maize', 'Maize_VanHeemst_1988')
    soilfile = os.path.join(data_dir, 'soil', 'ec3.soil')
    soild = CABOFileReader(soilfile)
    ## get basic information of array
    all_runs = product(years)
    bk = False
    target_lon = 121.375  #
    target_lat = 38.875
    ds = ds.sel(lon=[target_lon], lat=[target_lat], method="nearest")
    for ilon in enumerate(ds.lon.values):
        for ilat in enumerate(ds.lat.values):
            da = ds.sel(lon=ilon[1], lat=ilat[1])
            S_time = Start_time.sel(lon=ilon[1], lat=ilat[1])
            if da.tasmax.values[0] > -9999.0:
                print("--Continue location: lon-->%s, lat-->%s" % (ilon[1], ilat[1]))
                wdp = XarrayWeatherDataProvider(ELEVE=200.0, xrdata=da)
                appended_data = []
                for i, inputs in enumerate(all_runs):
                    year = inputs[0]
                    m = int(S_time[i])
                    Stime = pd.date_range(f"{year}-01-01", freq='1D', periods=m)
                    start = str(Stime[-1])[0:10]
                    print("       simulation year: %s" % start)
                    agromanagement = agro(year, start)
                    sited = WOFOST80SiteDataProvider(WAV=10, CO2=dco2.loc[year].values, SMLIM=0.35)
                    parameters = ParameterProvider(cropdata=crop_data, soildata=soild, sitedata=sited)

                    wofost = Wofost80_WLP_FD_beta(parameters, wdp, agromanagement)
                    wofost.run_till_terminate()
                    # except:
                    #    print("failed because of missing weather data.")
                    df = pd.DataFrame(wofost.get_output()).set_index(['day'])
                    appended_data.append(df)
                appended_data = pd.concat(appended_data)
                kk = xr.Dataset.from_dataframe(appended_data)

                outputshape = ((len(kk['LAI']), 1, 1))
                temp = {}
                temp['DVS'] = np.full(outputshape, np.nan)
                temp['LAI'] = np.full(outputshape, np.nan)
                temp['TAGP'] = np.full(outputshape, np.nan)
                temp['TWSO'] = np.full(outputshape, np.nan)
                temp['TWLV'] = np.full(outputshape, np.nan)
                temp['TWST'] = np.full(outputshape, np.nan)
                temp['TWRT'] = np.full(outputshape, np.nan)
                temp['TRA'] = np.full(outputshape, np.nan)
                temp['RD'] = np.full(outputshape, np.nan)
                temp['SM'] = np.full(outputshape, np.nan)
                temp['WWLOW'] = np.full(outputshape, np.nan)
                temp['time1'] = (kk['day']).astype(np.datetime64)
                temp['lat'] = np.full((1), np.nan)
                temp['lon'] = np.full((1), np.nan)
                bk = True
                break
        if (bk):
            break
    print(" ")
    print("---------------------")
    print(" ")
    print("initial setting  done")
    print(" ")
    print("---------------------")
    print(" ")

    return temp


def core_cal(ds, ilonn, ilatn, temp, Start_time, dco2):
    crop_data = YAMLCropDataProvider(os.path.join(data_dir, 'crop'))
    crop_data.set_active_crop('maize', 'Maize_VanHeemst_1988')
    soilfile = os.path.join(data_dir, 'soil', 'ec3.soil')
    soild = CABOFileReader(soilfile)
    ilon = ds.lon.values[ilonn]
    ilat = ds.lat.values[ilatn]
    if ds['tasmax'].values[5, ilatn, ilonn] > -9999.0:
        da = ds.sel(lon=ilon, lat=ilat)
        S_time = Start_time.sel(lon=ilon, lat=ilat)
        wdp = XarrayWeatherDataProvider(ELEVE=200.0, xrdata=da)
        all_runs = product(years)
        appended_data = []
        for i, inputs in enumerate(all_runs):
            year = inputs[0]
            # Set the agromanagement with correct year and crop
            m = int(S_time[i])
            Stime = pd.date_range(f"{year}-01-01", freq='1D', periods=m)
            start = str(Stime[-1])[0:10]
            agromanagement = agro(year, start)
            sited = WOFOST80SiteDataProvider(WAV=10, CO2=dco2.loc[year].values, SMLIM=0.35)
            parameters = ParameterProvider(cropdata=crop_data, soildata=soild, sitedata=sited)
            # Encapsulate parameters

            # try:
            # wofost = Wofost72_WLP_FD(parameters, wdp, agromanagement)
            wofost = Wofost80_WLP_FD_beta(parameters, wdp, agromanagement)
            wofost.run_till_terminate()
            # except:
            # except WeatherDataProviderError as e:
            #    msg = "failed because of missing weather data."
            # print(msg)
            df = pd.DataFrame(wofost.get_output()).set_index(['day'])
            appended_data.append(df)
        appended_data = pd.concat(appended_data)
        kk = xr.Dataset.from_dataframe(appended_data)
        # print(kk)
        #
        temp['DVS'][:, 0, 0] = kk['DVS'].values
        temp['LAI'][:, 0, 0] = kk['LAI'].values
        temp['TAGP'][:, 0, 0] = kk['TAGP'].values
        temp['TWSO'][:, 0, 0] = kk['TWSO'].values
        temp['TWLV'][:, 0, 0] = kk['TWLV'].values
        temp['TWST'][:, 0, 0] = kk['TWST'].values
        temp['TWRT'][:, 0, 0] = kk['TWRT'].values
        temp['TRA'][:, 0, 0] = kk['TRA'].values
        temp['RD'][:, 0, 0] = kk['RD'].values
        temp['SM'][:, 0, 0] = kk['SM'].values
        temp['WWLOW'][:, 0, 0] = kk['WWLOW'].values
        temp['time1'] = (kk['day']).astype(np.datetime64)
        temp['lat'] = np.atleast_1d(ilat)
        temp['lon'] = np.atleast_1d(ilon)
        # print(ilatn,temp['lat'])

    return temp


def cal_run(ds, temp, Start_time, dk, dco2):
    max_cpu = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count()
    # n=0
    num_cores = multiprocessing.cpu_count()

    for ilonn in range(len(ds.lon.values)):
        # for ilatn in range(len(ds.lat.values)):
        # n=n+1
        print("now at location: lon-->%s" % (ds.lon.values[ilonn]))
        temp0 = Parallel(n_jobs=num_cores)(
            delayed(core_cal)(ds, ilonn, i, temp, Start_time, dco2) for i in range(len(ds.lat.values)))
        for ilatn in range(len(ds.lat.values)):
            temp1 = temp0[ilatn]
            # print(test['lat'])
            if temp1['lat'] > 0:
                dk['DVS'][:, ilatn, ilonn] = temp1['DVS'][:, 0, 0]
                dk['LAI'][:, ilatn, ilonn] = temp1['LAI'][:, 0, 0]
                dk['TAGP'][:, ilatn, ilonn] = temp1['TAGP'][:, 0, 0]
                dk['TWSO'][:, ilatn, ilonn] = temp1['TWSO'][:, 0, 0]
                dk['TWLV'][:, ilatn, ilonn] = temp1['TWLV'][:, 0, 0]
                dk['TWST'][:, ilatn, ilonn] = temp1['TWST'][:, 0, 0]
                dk['TWRT'][:, ilatn, ilonn] = temp1['TWRT'][:, 0, 0]
                dk['TRA'][:, ilatn, ilonn] = temp1['TRA'][:, 0, 0]
                dk['RD'][:, ilatn, ilonn] = temp1['RD'][:, 0, 0]
                dk['SM'][:, ilatn, ilonn] = temp1['SM'][:, 0, 0]
                dk['WWLOW'][:, ilatn, ilonn] = temp1['WWLOW'][:, 0, 0]
            # core_cal(ds,ilonn,ilatn,parameters,temp) #.compute()
    return dk


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import time

    runname = 'sowing'
    path = '/stu01/xuqch3/finished/data/'
    ssps = ['ssp585']
    for ssp in ssps:
        time_begin = time.time()
        header_list = ["year", "co2"]
        dco2 = pd.read_csv(f'{co2data_dir}co2_{ssp}_annual_2015_2100.txt', delimiter=r"\s+", names=header_list)
        dco2 = dco2.set_index(['year'])
        Sowing_time = xr.open_dataset(f'{path}/PCSE/pr_t/pr_t_Growthday_%s.nc' % (ssp)).start
        print(">>>>time prepared  done")

        ds = FilePrapare(f'{path}/input/{ssp}', ssp)

        for i in np.arange(0, 100, 7):
            print(f'Now the sowing date add {i}')
            Start_time = Sowing_time + float(i)
            temp = Initial(data_dir, Start_time, dco2, ds)
            outputshape = (len(temp['time1']), len(ds.lat), len(ds.lon))
            dk = xr.Dataset({
                'DVS': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'LAI': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'TAGP': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'TWSO': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'TWLV': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'TWST': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'TWRT': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'TRA': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'RD': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'SM': (('time', 'lat', 'lon'), np.full(outputshape, np.nan)),
                'WWLOW': (('time', 'lat', 'lon'), np.full(outputshape, np.nan))
            },
                coords={'time': (('time'), temp['time1'].values),
                        'lat': (('lat'), ds.lat.values),
                        'lon': (('lon'), ds.lon.values),
                        })
            cal_run(ds, temp, Start_time, dk, dco2)
            print(">>>> file calculated done")
            dk.to_netcdf('%s/maize_output_%s_%s_%s.nc' % (fname, ssp, runname, i), engine='netcdf4')
            time_end = time.time()  # 结束时间
            timea = time_end - time_begin
            print('runtime:', timea / 60)  # 结束时间-开始时间
    print('------------end--------------')
