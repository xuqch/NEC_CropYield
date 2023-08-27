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

    #
    # maskfile_Crop = "F:/PCSE/crop/crop.nc"
    maskfile =xr.open_dataset('/tera01/xuqch3/PCSE/mask_Dongbei.nc')
    '''
    rice:0, maize:1, soybean:2
    -'''
    pathin = "/tera01/xuqch3/PCSE/sensitivity/Yield_change"
    pathout = '/tera01/xuqch3/PCSE/sensitivity/trend'
    # scenarios = ['default', 'co2', 'precipitation', 'temperature','sowing', 'strategy']
    scenarios=['sowing']
    names = ['rice', 'maize', 'soybean']
    # ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']


    for scenario in scenarios:
        print(scenario, ">>>>")
        for name in names:
            print(name, '>>')
            # for ssp in ssps:
            VarFile = f'{pathin}/{scenario}/{name}_output_ssp585_{scenario}_Yield_change.nc'
            print(VarFile)
            with xr.open_dataset(VarFile) as ds:
                ds = ds['TAGP']
                ds2 = ds.pipe(lambda x: x).where(ds > -9999, drop=True)
                x = xr.DataArray(np.arange(len(ds2['year'])) + 1, dims='year', coords={'year': ds2['year']})
                r2 = trend(ds2, x, 'year').compute()
                r2 = r2.to_dataset(name="TAGP")  # Convert to dataset
                r2 = r2.where(maskfile.mask_array > 0.0, drop=True)
                save_xarray(f'{pathout}/{scenario}/', r2, f'{name}_output_ssp585_{scenario}_Yield_change_trend')






        # if os.path.isdir(pathout):
        #     shutil.rmtree(pathout)
        # # re-create the directory
        # os.mkdir(f'{pathout}/{scenario}/')