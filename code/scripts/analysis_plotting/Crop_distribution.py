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
fname = 'D:/NCL/data/NEshp/NE.shp'
fig = plt.figure(figsize=(5, 5))  # figsize=(5, 5)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                               ccrs.PlateCarree(), facecolor='none')

maskfile_Crop = 'F:/PCSE/crop/crop.nc'
crop = xr.open_dataset(maskfile_Crop)
ds1 = crop
# ds1['trend'] = crop.crop
ds1['trend'] = xr.where(crop.crop >= 0, 10.0, np.nan)  # xr.where(crop.crop >= 0, 10, 0)
ds_a1 = ds1.where(crop.crop == 0.0).squeeze()
ds_a2 = ds1.where(crop.crop == 1.0).squeeze()
ds_a3 = ds1.where(crop.crop == 2.0).squeeze()
lons2d, lats2d = np.meshgrid(ds1.lon, ds1.lat)
colors = ['#62BEA6', '#FDBA6B', '#EB6046']

plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='^', color=colors[0], label='rice', s=14)
plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color=colors[1], label='maize', s=14)
plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='D', color=colors[2], label='soybean', s=14)
plt.scatter(lons2d[0, 0], lats2d[0, 0], marker='o', color='w', s=20)
plt.scatter(lons2d, lats2d, s=ds_a1["trend"].values, marker='^', color=colors[0])
plt.scatter(lons2d, lats2d, s=ds_a2["trend"].values, marker='o', color=colors[1])
plt.scatter(lons2d, lats2d, s=ds_a3["trend"].values, marker='D', color=colors[2])
# plt.legend(loc='upper right')
ax.legend(loc='upper right', shadow=False, fontsize=12)
ax.set_title('Crop Distribution', fontsize=14)
ax.set_extent([118, 136, 38, 55])
ax.set_xticks([120, 125, 130, 135], crs=ccrs.PlateCarree())
ax.set_yticks([40, 45, 50, 55], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.add_feature(shape_feature)
# ax.autoscale(tight=True)
# plt.savefig('crop_distribution.eps', format='eps', dpi=800)

plt.show()
