import os
import math
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cmcrameri import cm
from pylab import rcParams
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap
from matplotlib.pyplot import MultipleLocator
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


### Plot settings
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

maskfile_Crop = "/tera04/zhwei/PCSE/data/crop_distribution/crop.nc"
crop = xr.open_dataset(maskfile_Crop).crop
fname = '/stu01/xuqch3/PCSE/scripts/NEshp/NE.shp'



def plot_trend(varpath, Varout, tittle, i, name):
    fig = plt.figure()  # figsize=(5, 5)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                   ccrs.PlateCarree(), facecolor='none')
    ds1 = xr.open_dataset('%s' % (varpath))
    ds1['trend'] = ds1.Esd
    lats = ds1.variables['lat'][:]
    lons = ds1.variables['lon'][:]
    ds_a1 = ds1.where(crop == i).squeeze()

    m = ds_a1.trend.values.reshape(-1)[~np.isnan(ds_a1.trend.values.reshape(-1))]
    print(math.floor(min(m)), math.ceil(max(m)))

    lons2d, lats2d = np.meshgrid(lons, lats)
    p1 = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', label=f'{name}', s=10)
    plt.legend(loc='upper right', fontsize=20)


    bins = np.arange(-0.3, -0.1+0.005, 0.001)
    nbin = len(bins) - 1
    mmap=cm.davos
    cmap4 = mpl.cm.get_cmap(mmap, nbin)  # coolwarm/twilight_shifted，cm.batlow,davos,bamko,broc
    norm4 = mpl.colors.BoundaryNorm(bins, nbin)
    locator = mpl.ticker.MultipleLocator(0.1)
    im2 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap4)

    im1 = ax.scatter(lons2d, lats2d, c=ds_a1['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='o',
                     transform=ccrs.PlateCarree(), cmap=cmap4, norm=norm4)  # Paired/magma/plasma/GnBu/YlGn

    ax.set_extent([118, 136, 38, 55])
    ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
    ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=18)
    cbar = fig.colorbar(im1, ax=ax,ticks=locator)
    # cbar.ax.tick_params(labelsize=20)
    # ax.set_title('%s' % (tittle), fontsize=12)
    ax.add_feature(shape_feature)
    plt.tight_layout()
    plt.savefig('%s.eps' % (Varout), format='eps', dpi=800)
    plt.savefig('%s.png' % (Varout), format='png', dpi=300)
    plt.show()
    plt.close()



if __name__ == '__main__':
    import glob, os, shutil
    path = '/stu01/xuqch3/finished/data/'
    pathin = f"{path}/code/Fig-S2/"
    pathout = f'{path}/code/fig-7/'

    names = ['Rice', 'Maize', 'Soybean']

    for i, name in enumerate(names):
        print('>>', name)
        VarFile = f'{pathin}/Esd_trend.nc'
        Varout = f'{pathout}/Esd_Trend_{name}'
        tittle = f'{name}'
        plot_trend(VarFile, Varout, tittle, i, name)
