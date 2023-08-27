import numpy as np
import xarray as xr
from pylab import rcParams
import matplotlib
import pymannkendall as mk
import pandas as pd

### Plot settings
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
                  encoding={'tas': {'dtype': 'float32',
                                    'zlib': True,
                                    'complevel': 6,
                                    '_FillValue': -9999.0}})


if __name__ == '__main__':
    import glob, os, shutil
    path = '/stu01/xuqch3/finished/data/'
    maskfile_Crop = f"{path}/crop/crop.nc"
    crop = xr.open_dataset(maskfile_Crop).crop

    pathout = './'
    maskfile = xr.open_dataset(f'{path}/mask/mask_Dongbei.nc')
    df = pd.DataFrame()
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    tas = []
    for ssp in ssps:
        pathin = f"{path}/met/{ssp}/tas_day_ensmean_{ssp}_2015-2100.nc"
        with xr.open_dataset(pathin) as ds:
            print(ds)
            ds = ds['tas']
            ds2 = ds.pipe(lambda x: x).where(ds > 0.0, drop=True)
            ds2 = ds2.resample(time='1Y').mean()
            x = xr.DataArray(np.arange(len(ds2['time'])) + 1, dims='time', coords={'time': ds2['time']})
            r2 = trend(ds2, x, 'time').compute()
            r2 = r2.to_dataset(name="tas")  # Convert to dataset
            r2 = r2.where(maskfile.mask_array > 0.0, drop=True)
            r2 = r2.where(crop >= 0, drop=True)
            print('save')
            save_xarray(pathout, r2, f'tas_trend_%s' % (ssp))
            tas.append(np.nanmean(r2.tas * 10))  # .pr * 356 * 24 * 60 * 60 * 10  # tas * 10  # pr * 356 * 24 * 60 * 60 * 10
    df['ssps'] = pd.Series(ssps)
    df['tas'] = pd.Series(tas)
    # df.to_csv('%s/tas_trend_year.csv' % (pathout))
    df.to_excel('./tas_trend_year.xlsx', sheet_name=f'tas', index=True)


    #
    # ssps = ['ssp585']
    # for i, ssp in enumerate(ssps):
    #     VarFile = f'{path}/met/{ssp}/tas_day_ensmean_{ssp}_2015-2100.nc'
    #     print(VarFile)
    #     with xr.open_dataset(VarFile) as ds1:
    #         ds1 = ds1.where((ds1.time.dt.year >= 2015) & (ds1.time.dt.year <= 2100), drop=True).tas
    #         ds_a1 = ds1.where(crop >= 0.0, drop=True).groupby('time').mean(...)
    #         ds_a1 = ds_a1 - 273.15
    #         temp = xr.where((ds_a1 <= 30.0) & (ds_a1 >= 10.0), 1, 0)
    #         print(temp)
    #         temp = temp.groupby('time.year').sum(...)
    #         print(temp)
    #         x = xr.DataArray(np.arange(len(temp['year'])) + 1, dims='year', coords={'year': temp['year']})
    #         r2 = trend(temp, x, 'year').compute()
    #         r2 = r2.to_dataset(name="tas")  # Convert to dataset
    #         print(r2)