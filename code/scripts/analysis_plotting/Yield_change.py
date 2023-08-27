import numpy as np
import xarray as xr
from pylab import rcParams
import matplotlib
import pymannkendall as mk

font = {'family': 'DejaVu Sans'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def calc_trend(x, y):
    co = np.count_nonzero(~np.isnan(x))
    if co < 4:
        return -9999.0
    slope = mk.original_test(x, alpha=0.01).slope
    return slope  # *100. #,p_value


def trend(x, y, dim='time'):
    return xr.apply_ufunc(
        calc_trend, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )


def save_xarray(outpath, arr, rid):
    """
    Save output
    """
    arr["lat"].attrs["units"] = "degrees_north"
    arr["lon"].attrs["units"] = "degrees_east"
    arr.to_netcdf(os.path.join(outpath, '%s.nc' % (rid)),
                  encoding={'TAGP': {'dtype': 'float32',
                                     'zlib': True,
                                     'complevel': 6,
                                     '_FillValue': -9999.0}})


if __name__ == '__main__':
    import glob, os, shutil

    path = "/tera04/zhwei/PCSE/data/output/adaptation/sowing/"
    pathout = '/tera01/xuqch3/PCSE/sensitivity/Yield_change'

    # scenarios = ['default', 'co2', 'precipitation', 'temperature','sowing', 'strategy']
    scenarios=['sowing']
    names = ['rice', 'maize', 'soybean']
    ssps = [ 'ssp585']
    idxs=[0,1,2]


    maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop



    # for name,idx in zip(names,idxs):
    #     default_ssp585=xr.open_dataset(f'{path}/{name}_output_ssp585_sowing_Max_Yield_default.nc')["TAGP"] 
    #     default_ssp585 = default_ssp585.where(crop == idx, drop=True)

    #     VarFile = f'{path}/{name}_output_ssp585_sowing_Max_Yield_final.nc'
    #     print(VarFile)
    #     with xr.open_dataset(VarFile) as ds1:
    #         ds1 = ds1["TAGP"] 
    #         ssp585 = ds1.where(crop == idx, drop=True)

    #         ssp585_land = (ssp585[:,:,:] - default_ssp585[0,:,:]) / default_ssp585[0,:,:] * 100
    #         ssp585_land.to_netcdf(f'{pathout}/sowing/{name}_output_ssp585_sowing_Yield_change.nc')
    #         print(f'sowing '+f'{name}'+f' {idx} '+'SSP585 mean: '+str(ssp585_land.mean(...).values))
    #         print(f'sowing '+f'{name}'+f' {idx} '+'SSP585 std: '+str(ssp585_land.std(...).values))


    # # optimized_Yield change =====================================
    optimized_distribution = "/tera01/xuqch3/PCSE/sensitivity/harvest_date/strategy/optimized_distribution.nc"
    distribution = xr.open_dataset(optimized_distribution).TAGP

    for name,idx in zip(names,idxs):
        default_ssp585=xr.open_dataset(f'{path}/{name}_output_ssp585_sowing_Max_Yield_default.nc')["TAGP"] 
        default_ssp585 = default_ssp585.where(crop == idx, drop=True)
        # default_ssp585 = default_ssp585.groupby('year').mean(...)
        scenario='optimized'

        VarFile = f"/tera01/xuqch3/PCSE/sensitivity/harvest_date/strategy/optimized_Yield.nc"
        print(VarFile)
        with xr.open_dataset(VarFile) as ds2:
            ds2 = ds2["TAGP"] 
            ssp585 = ds2.where(distribution == idx, drop=True)
            # ssp585 = ds2_a.groupby('year').mean(...)
            ssp585_land = (ssp585[:,:,:] - default_ssp585[0,:,:]) / default_ssp585[0,:,:] * 100
            ssp585_land.to_netcdf(f'{pathout}/strategy/{name}_output_ssp585_strategy_Yield_change.nc')
            print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP585 mean: '+str(ssp585_land.mean(...).values))
            print(f'{scenario} '+f'{name}'+f' {idx} '+'SSP585 std: '+str(ssp585_land.std(...).values))


    # for scenario in scenarios:
    #     for name in (names):
    #         VarFile = f'{pathin}/{scenario}/{name}_output_ssp585_{scenario}_max.nc'
    #         print(VarFile)
    #         with xr.open_dataset(VarFile) as ds4:
    #             # ds4 = ds4.where((ds4.year > 2015), drop=True)
    #             ds4 = ds4["TAGP"] 
    #             ds4_strategy = (ds4[:,:,:] - ds4[0,:,:]) / ds4[0,:,:] *100
    #             ds4_strategy.to_netcdf(f'{pathout}/{scenario}/{name}_output_ssp585_{scenario}_Yield_change.nc')