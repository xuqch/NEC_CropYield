# from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import xarray as xr
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pylab import rcParams
import matplotlib

### Plot settings
font = {'family': 'Times New Roman'}
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
path = '/stu01/xuqch3/finished/data/'

fname = f'{path}/NE_basemap/NE.shp'

def plot_trend(VarFile, figout, ssp, scenario,run):
    ds = xr.open_dataset(VarFile)

    lons2d, lats2d = np.meshgrid(ds.lon, ds.lat)
    colors = ['#62BEA6', '#FDBA6B', '#EB6046']
    for i, year in enumerate(ds.year):
        print(year.values)
        ds0 = ds.sel(year=year)
        ds1 = ds0
        # ds1['trend'] = ds0
        ds1['trend'] = xr.where(ds0[f'{run}'] >= 0, 10.0, np.nan)  # xr.where(crop.crop >= 0, 10, 0)
        ds_a1 = ds1.where(ds0[f'{run}'] == 0.0).squeeze()
        ds_a2 = ds1.where(ds0[f'{run}'] == 1.0).squeeze()
        ds_a3 = ds1.where(ds0[f'{run}'] == 2.0).squeeze()
        # exit(0)
        lons2d_1, lats2d_1 = np.meshgrid(ds_a1.lon, ds_a1.lat)
        lons2d_2, lats2d_2 = np.meshgrid(ds_a2.lon, ds_a2.lat)
        lons2d_3, lats2d_3 = np.meshgrid(ds_a3.lon, ds_a3.lat)
        fig = plt.figure(figsize=(5, 5))  # figsize=(5, 5)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        shape_feature = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none')

        im1 = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', label="Year: %s" % (year.values), s=0)  # label=year.values,
        first_legend = plt.legend(handles=[im1], loc='lower right', fontsize=16)
        rice = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='^', color=colors[0], label='rice', s=16)
        maize = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color=colors[1], label='maize', s=16)
        soybean = plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='D', color=colors[2], label='soybean', s=16)
        plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', s=40)  # label=year.values,
        plt.legend((rice, maize, soybean), ('rice', 'maize', 'soybean'), loc='upper right', shadow=False, fontsize=16)
        plt.gca().add_artist(first_legend)

        plt.scatter(lons2d_1, lats2d_1, s=ds_a1["trend"].values, marker='^', color=colors[0])
        plt.scatter(lons2d_2, lats2d_2, s=ds_a2["trend"].values, marker='o', color=colors[1])
        plt.scatter(lons2d_3, lats2d_3, s=ds_a3["trend"].values, marker='D', color=colors[2])

        ax.set_extent([118, 136, 38, 55])
        ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
        ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(axis='both', top='off', labelsize=14)
        plt.tight_layout()
        ax.add_feature(shape_feature)
        # ax.autoscale(tight=True)
        plt.savefig(f'%s/crop_distribution_{scenario}/{run}_crop_distribution_%s.eps' % (figout, year.values), format='eps', dpi=800)
        plt.savefig(f'%s/crop_distribution_{scenario}/{run}_crop_distribution_%s.png' % (figout, year.values), format='png', dpi=800)
        plt.show()
        plt.close()


if __name__ == '__main__':
    import glob, os, shutil
    run='TAGP'#TWSO
    path = '/stu01/xuqch3/finished/data/'
    figout = f"{path}/code/fig-10/"
    print('plotting now')
    scenario = 'optimized' # optimized CR
    VarFile = f"{path}/PCSE/output/adaptation/{scenario}/{scenario}_distribution_{run}.nc"
    ssp = 'ssp585'
    plot_trend(VarFile, figout, ssp, scenario,run)
