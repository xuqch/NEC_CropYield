import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import xarray as xr
import matplotlib
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pylab import rcParams

font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
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
maskfile_Crop = f"{path}/crop/crop.nc"
crop = xr.open_dataset(maskfile_Crop)
fname = f'{path}/NE_basemap/NE.shp'
fig = plt.figure()  # figsize=(5, 5)figsize=(6, 5)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                               ccrs.PlateCarree(), facecolor='none')
ds1 = xr.open_dataset("./pr_trend_ssp585.nc")
print(ds1)
ds1['trend'] = ds1.pr * 356 * 24 * 60 * 60 * 10  # tas * 10  # pr * 356 * 24 * 60 * 60 * 10

lats = ds1.variables['lat'][:]
lons = ds1.variables['lon'][:]
ds_a1 = ds1.where(crop.crop == 0.0).squeeze()
ds_a2 = ds1.where(crop.crop == 1.0).squeeze()
ds_a3 = ds1.where(crop.crop == 2.0).squeeze()

lons2d, lats2d = np.meshgrid(lons, lats)


ax.legend(loc='upper right', shadow=False, frameon=False)
im1 = ax.scatter(lons2d, lats2d, c=ds_a1['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='^',
                 transform=ccrs.PlateCarree(), cmap='PuBu')  # Paired/magma/plasma/GnBu/YlGn/YlOrRd/PuBu
im2 = ax.scatter(lons2d, lats2d, c=ds_a2['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='o',
                 transform=ccrs.PlateCarree(), cmap='PuBu')  # Paired/magma/plasma/GnBu/YlGn/YlOrRd/PuBu
im3 = ax.scatter(lons2d, lats2d, c=ds_a3['trend'].values, s=10.0, alpha=1.0, zorder=2, marker='D',
                 transform=ccrs.PlateCarree(), cmap='PuBu')  # Paired/magma/plasma/GnBu/YlGn/YlOrRd/PuBu

ax.set_extent([118, 135, 38, 55])
ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
fig.colorbar(im2, ax=ax, format="%.1f")
ax.set_title('Precipitation Trend', fontsize=12)
ax.add_feature(shape_feature)
plt.tight_layout()
plt.savefig('pr_trend.eps', format='eps', dpi=800)
plt.savefig('pr_trend.png', format='png', dpi=800)
plt.show()
