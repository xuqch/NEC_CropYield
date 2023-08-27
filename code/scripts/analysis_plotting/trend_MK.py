import numpy as np
import matplotlib.pyplot as plt
import os, sys
import xarray as xr
import dask.array as da
from scipy.stats import linregress
from numba import jit  # Speedup for python functions
from matplotlib import colors
import cartopy.crs as ccrs
from pylab import rcParams
import matplotlib
import pymannkendall as mk
from joblib import Parallel, delayed
import multiprocessing

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
                  encoding={'pr': {'dtype': 'float32',
                                   'zlib': True,
                                   'complevel': 6,
                                   '_FillValue': -9999.0}})


if __name__ == '__main__':
    import glob, os, shutil

    #
    # maskfile_Crop = "F:/PCSE/crop/crop.nc"
    # crop = xr.open_dataset(maskfile_Crop)

    pathout = './'
    maskfile = xr.open_dataset('F:/PCSE/mask_Dongbei.nc')


    pathin = "F:/Dongbei/ssp585/pr_ensmean.nc"
    with xr.open_dataset(pathin) as ds:
        print(ds)
        ds = ds['pr']
        ds2 = ds.pipe(lambda x: x).where(ds > 0.0, drop=True)
        ds2 = ds2.resample(time='1Y').mean()
        x = xr.DataArray(np.arange(len(ds2['time'])) + 1, dims='time', coords={'time': ds2['time']})
        r2 = trend(ds2, x, 'time').compute()
        r2 = r2.to_dataset(name="pr")  # Convert to dataset
        r2 = r2.where(maskfile.mask_array > 0.0, drop=True)
        print('save')
        save_xarray(pathout, r2, f'pr_trend')