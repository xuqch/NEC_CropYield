import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import xarray as xr
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pylab import rcParams
import matplotlib
from scipy.stats import linregress
from joblib import Parallel, delayed
import multiprocessing
import pymannkendall as mk

font = {'family': 'Times New Roman'}
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


# @jit(nogil=True)  # Enable JIT compiler
def calc_trend(x, y):
    co = np.count_nonzero(~np.isnan(x))
    if co < 4:
        return -9999.0
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(x)), x)
    return slope  # *100. #,p_value


def trend(x, y, dim='time'):
    return xr.apply_ufunc(
        calc_trend, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )


def plot_trend(path,pathout):
    fname = f'{path}/NE_basemap/NE.shp'
    fig = plt.figure(figsize=(5, 5))  # figsize=(5, 5)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                   ccrs.PlateCarree(), facecolor='none')
    maskfile_Crop = f'{path}/crop/crop.nc'
    crop = xr.open_dataset(maskfile_Crop).crop
    ds1 = xr.open_dataset('./ERA5_trend.nc')

    lats = ds1.variables['lat'][:]
    lons = ds1.variables['lon'][:]
    ds_a1 = ds1.where(crop == 0.0).squeeze()
    ds_a2 = ds1.where(crop == 1.0).squeeze()
    ds_a3 = ds1.where(crop == 2.0).squeeze()

    lons2d, lats2d = np.meshgrid(lons, lats)
    print(ds_a1['trend'].shape)
    print(lons2d.shape)
    p1 = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='^', color='k', label='rice', s=15)
    p2 = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='k', label='maize', s=15)
    p3 = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='D', color='k', label='soybean', s=15)
    plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', s=40)
    ax.legend(loc='upper right', shadow=False, fontsize=15)
    im1 = ax.scatter(lons2d, lats2d, c=ds_a1['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='^',
                     transform=ccrs.PlateCarree(), cmap='YlOrRd')  # Paired\magma\plasma\YlOrRd
    im2 = ax.scatter(lons2d, lats2d, c=ds_a2['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='o',
                     transform=ccrs.PlateCarree(), cmap='YlOrRd')
    im3 = ax.scatter(lons2d, lats2d, c=ds_a3['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='D',
                     transform=ccrs.PlateCarree(), cmap='YlOrRd')

    ax.set_extent([118, 136, 38, 55])
    ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
    ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    fig.colorbar(im2, ax=ax)
    ax.set_title('Annual mean air temperature at 2m ($^\circ$C)', fontsize=15)
    ax.add_feature(shape_feature)
    plt.savefig('2m_temperature_trend_crop_bar.eps', format='eps', dpi=800)
    plt.savefig('2m_temperature_trend_crop.png', format='png', dpi=800)
    plt.show()


def save_xarray(outpath, arr, rid):
    """
    Save output
    """
    arr.to_netcdf(os.path.join(outpath, '%s_trend.nc' % (rid)),
                  encoding={'trend': {'dtype': 'float32',
                                      'zlib': True,
                                      'complevel': 6,
                                      '_FillValue': -9999.0}})


if __name__ == '__main__':
    import glob, os, shutil


    path = '/stu01/xuqch3/finished/data/'
    pathout = './'
    plot_trend(path,pathout)
    # Var = 'ERA5'
    # VarFile = ('F:/PCSE/out/file_mask.nc')
    # print(VarFile)
    # with xr.open_dataset(VarFile) as ds:
    #     ds = ds['t2m']
    #     ds_annual = ds.resample(time='1Y').mean()
    #     x = xr.DataArray(np.arange(len(ds_annual['time'])) + 1, dims='time', coords={'time': ds_annual['time']})
    #     r2 = trend(ds_annual, x, 'time').compute()
    #     r2 = r2.to_dataset(name="trend")  # Convert to dataset
    #     save_xarray(pathout, r2, "ERA5")
    #     plot_trend(pathout)
