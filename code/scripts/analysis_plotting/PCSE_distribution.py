from joblib import Parallel, delayed
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
fname = '/tera01/xuqch3/PCSE/scripts/NEshp/NE.shp'


def determine(rice_line, maize_line, soybean_line,crop):
    outputshape = (len(rice_line.lat), len(rice_line.lon))
    time = {}
    time['crop'] = np.full(outputshape, np.nan)
    for i, lat in enumerate(rice_line.lat):
        for j, lon in enumerate(rice_line.lon):
            if crop[i,j]>-10:
                m = max(rice_line[i, j], maize_line[i, j], soybean_line[i, j])
                if m.values == rice_line[i, j].values:
                    time['crop'][i, j] = 0
                if m.values == maize_line[i, j].values:
                    time['crop'][i, j] = 1
                if m.values == soybean_line[i, j].values:
                    time['crop'][i, j] = 2
    return time


def cal_run(ds1, ds2, ds3, time,crop):
    max_cpu = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count()
    n = 0
    num_cores = multiprocessing.cpu_count()
    temp0 = Parallel(n_jobs=num_cores)(
        delayed(determine)(ds1[i, :, :], ds2[i, :, :], ds3[i, :, :],crop) for i, itime in enumerate(time))
    # temp0 = Parallel(n_jobs=num_cores)(
    #     delayed(determine)((ds1[i, :, :] / ds1[0, :, :]) * 100, (ds2[i, :, :] / ds2[0, :, :]) * 100,
    #                        (ds3[i, :, :] / ds3[0, :, :]) * 100) for i, itime in enumerate(time))
    return temp0


def plot_trend(VarFile, figout, ssp, scenario):
    ds = xr.open_dataset(VarFile)

    lons2d, lats2d = np.meshgrid(ds.lon, ds.lat)
    colors = ['#62BEA6', '#FDBA6B', '#EB6046']
    for i, year in enumerate(ds.year):
        print(year.values)
        ds0 = ds.sel(year = year)
        ds1 = ds0
        # ds1['trend'] = ds0
        ds1['trend'] = xr.where(ds0.TAGP >= 0, 10.0, np.nan)  # xr.where(crop.crop >= 0, 10, 0)
        ds_a1 = ds1.where(ds0.TAGP == 0.0).squeeze()
        ds_a2 = ds1.where(ds0.TAGP == 1.0).squeeze()
        ds_a3 = ds1.where(ds0.TAGP == 2.0).squeeze()
        # exit(0)
        lons2d_1, lats2d_1 = np.meshgrid(ds_a1.lon, ds_a1.lat)
        lons2d_2, lats2d_2 = np.meshgrid(ds_a2.lon, ds_a2.lat)
        lons2d_3, lats2d_3 = np.meshgrid(ds_a3.lon, ds_a3.lat)
        fig = plt.figure(figsize=(5, 5))  # figsize=(5, 5)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='none')

        plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='^', color=colors[0], label='rice', s=13)
        plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color=colors[1], label='maize', s=13)
        plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='D', color=colors[2], label='soybean', s=13)
        plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w',label=year.values, s=27)
        ax.legend(loc='upper right', shadow=False, fontsize=18)#frameon=False

        plt.scatter(lons2d_1, lats2d_1, s=ds_a1["trend"].values, marker='^', color=colors[0])
        plt.scatter(lons2d_2, lats2d_2, s=ds_a2["trend"].values, marker='o', color=colors[1])
        plt.scatter(lons2d_3, lats2d_3, s=ds_a3["trend"].values, marker='D', color=colors[2])
        print('e')
        # ax.set_extent([118, 136, 38, 55])
        # ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
        # ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
        ax.set_extent([118, 136, 38, 55])
        ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
        ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
        print('f')
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(axis='both', top='off', labelsize=18)
        plt.tight_layout()
        ax.add_feature(shape_feature)
        print('f')
        # ax.autoscale(tight=True)
        plt.savefig('%s/%s/%s/crop_distribution_%s.eps' % (figout, scenario, ssp, year.values), format='eps', dpi=800)
        plt.savefig('%s/%s/%s/crop_distribution_%s.png' % (figout, scenario, ssp, year.values), format='png', dpi=800)
        plt.close()
        print('g')


if __name__ == '__main__':
    import glob, os, shutil

    # maskfile_Crop = "/tera01/xuqch3/PCSE/crop/crop.nc"  # 'F:/PCSE/crop/crop.nc'
    # crop = xr.open_dataset(maskfile_Crop).crop
    # pathin = '/tera01/xuqch3/PCSE/sensitivity/harvest_date'
    # ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    # scenarios = ['default', 'co2', 'precipitation', 'temperature']
    # ssps = ['ssp126']
    # scenarios = ['strategy']#strategy
    # optimized_distribution = "/tera01/xuqch3/PCSE/sensitivity/harvest_date/strategy/optimized_distribution.nc"
    # distribution = xr.open_dataset(optimized_distribution).TAGP

    pathout = '/tera01/xuqch3/PCSE/sensitivity/Crop_distribution'
    figout = '/tera01/xuqch3/PCSE/sensitivity/Fig/Crop_distribution'
    VarFile = "/tera01/xuqch3/PCSE/sensitivity/harvest_date/strategy/optimized_distribution.nc"
    print('plotting now')
    scenario = 'strategy'
    ssp = 'ssp585'
    plot_trend(VarFile, figout, ssp, scenario)



    # for scenario in scenarios:
        # print('>>>>>', scenario)
        # for i, ssp in enumerate(ssps):
        #     print('>>>', ssp)
            # os.mkdir(f'{figout}/{scenario}/{ssp}/')
            # rice = xr.open_dataset('%s/%s/rice_output_%s_%s.nc' % (pathin, scenario, ssp, scenario))  # .TAGP
            # maize = xr.open_dataset('%s/%s/maize_output_%s_%s.nc' % (pathin, scenario, ssp, scenario))
            # soybean = xr.open_dataset('%s/%s/soybean_output_%s_%s.nc' % (pathin, scenario, ssp, scenario))
            # rice_land = rice.where(crop >= 0, np.nan)
            # maize_land = maize.where(crop >= 0, np.nan)
            # soybean_land = soybean.where(crop >= 0, np.nan)
            # # time1 = np.arange(2015, 2100)
            # time1 = np.arange(2016, 2100)
            # outputshape = (len(time1), len(crop.lat), len(crop.lon))
            # dk = xr.Dataset({'TAGP': (('time', 'lat', 'lon'), np.full(outputshape, np.nan))},
            #                 coords={'time': (('time'), time1),
            #                         'lat': (('lat'), crop.lat.values),
            #                         'lon': (('lon'), crop.lon.values),
            #                         })

            # ds1 = rice_land["TAGP"]
            # ds2 = maize_land["TAGP"]
            # ds3 = soybean_land["TAGP"]
            # temp0 = cal_run(ds1, ds2, ds3, time1,crop)
            # for i, year in enumerate(time1):
            #     m = temp0[i]
            #     dk["TAGP"][i, :, :] = m["crop"][:, :]
            # print('%s Done---------------------------------------' % (ssp))
            # dk.to_netcdf(os.path.join(pathout, '%s/crop_distribution_%s_%s.nc' % (scenario, ssp, scenario)))
            # VarFile = f'{pathout}/{scenario}/crop_distribution_{ssp}_{scenario}.nc'
            # print('plotting now')
            # plot_trend(VarFile, figout, ssp, scenario)
