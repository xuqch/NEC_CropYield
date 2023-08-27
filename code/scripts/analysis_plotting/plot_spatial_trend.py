import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import xarray as xr
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pylab import rcParams
import matplotlib
import matplotlib as mpl   
# import cmaps 
import math

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
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)
fname = '/tera01/xuqch3/PCSE/scripts/NEshp/NE.shp'
maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"  # 'F:/PCSE/crop/crop.nc'
crop = xr.open_dataset(maskfile_Crop).crop


def plot_trend(varpath, Varout, tittle,i,name,scenario):
    fig = plt.figure()  # figsize=(5, 5)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                   ccrs.PlateCarree(), facecolor='none')

    ds1 = xr.open_dataset('%s' % (varpath))
    ds1['trend'] = ds1.TAGP
    lats = ds1.variables['lat'][:]
    lons = ds1.variables['lon'][:]
    ds_a1 = ds1.where(crop == i).squeeze()
    m = ds_a1.trend.values.reshape(-1)[~np.isnan(ds_a1.trend.values.reshape(-1))]
    print(math.floor(min(m)),math.ceil(max(m)))

    lons2d, lats2d = np.meshgrid(lons, lats)
    p1 = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', label=f'{scenario}', s=10)
    # plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', s=22)

    plt.legend(loc='upper right')

    bins = np.arange(-3,3.1,0.2)
    # bins = np.arange(math.ceil(max(m))*-1,math.ceil(max(m))+1,1)
    nbin = len(bins)-1
    cmap4 = mpl.cm.get_cmap('coolwarm', nbin)#coolwarm/bwr
    norm4 = mpl.colors.BoundaryNorm(bins, nbin)
    im2 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap4)
    im1 = ax.scatter(lons2d, lats2d, c=ds_a1['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='o',
                     transform=ccrs.PlateCarree(), cmap=cmap4 ,norm = norm4)  # Paired\magma\plasma\GnBu\YlGn

    ax.set_extent([118, 135, 38, 55])
    ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
    ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    fig.colorbar(im1, ax=ax)
    ax.set_title('%s' % (tittle), fontsize=12)
    ax.add_feature(shape_feature)
    plt.tight_layout()
    # plt.savefig('%s.eps' % (Varout), format='eps', dpi=800)
    plt.savefig('%s.png' % (Varout), format='png', dpi=800)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    import glob, os, shutil

    pathin = '/tera01/xuqch3/PCSE/sensitivity/trend'
    pathout = '/tera01/xuqch3/PCSE/sensitivity/Fig/trend'
    # scenarios = ['default', 'co2', 'precipitation','sowing', 'strategy']#temperature
    # scenarios = ['temperature']#
    scenarios=['sowing']
    # ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    # names = ['rice','maize', 'soybean']
    names = ['rice']
    ssps = ['ssp585']


    for scenario in scenarios:
        print( ">>>>",scenario)
        for i,name in enumerate(names):
            print( '>>',name)
            for ssp in ssps:
                VarFile = f'{pathin}/{scenario}/{name}_output_{ssp}_{scenario}_Yield_change_trend.nc'
                Varout = f'{pathout}/{name}_output_{ssp}_{scenario}_Yield_change_trend'
                tittle = f'{name}'
                plot_trend(VarFile, Varout, tittle,i,name,scenario)
