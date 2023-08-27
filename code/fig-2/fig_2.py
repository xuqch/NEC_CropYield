import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import dask.array as da
from matplotlib import colors
from pylab import rcParams
import matplotlib
import pymannkendall as mk

### Plot settings
font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)-

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



def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    b = np.mean(ys) - m * np.mean(xs)
    return m, b


if __name__ == '__main__':
    import glob, os, shutil
    path = '/stu01/xuqch3/finished/data/'
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    time = pd.date_range('2015-01-01', '2100-12-31', freq='1Y')

    maskfile_Crop = f"{path}/crop/crop.nc"
    Crop = xr.open_dataset(maskfile_Crop)
    Crop_all = Crop.crop

    '''
    rice:0, maize:1, soybean:2
    '''

    pathin = f'{path}/PCSE/pr_t/'
    pathout = f'{path}/code/fig-2/'
    for i, ssp in enumerate(ssps):
        GrowthFile = ('%s/pr_t_Growthday_%s.nc' % (pathin, ssp))
        date = xr.open_dataset(GrowthFile)
        start_date = date.sowing
        crop = start_date.where(Crop_all >= 0, drop=True)

        crop_area = crop.groupby('time').mean(...)
        crop_area.plot.line(x='time', label=ssp, linewidth=1.2, linestyle='solid', alpha=0.7, markersize=3,
                            color=colors[i], marker='D')


        m, b = best_fit_slope_and_intercept(np.arange(2015, 2101), np.array(crop_area.values))
        regression_line = []
        for a in range(2015, 2101):
            regression_line.append((m * a) + b)

        plt.plot(np.arange(2015, 2101), regression_line, color=colors[i], lw=1.5)  # 预测与实测数据之间的回归线
    # plt.title('Crop', fontsize=18)  # , fontsize=18
    ax.legend(fontsize=18, loc=1)
    ax.legend(loc='best', shadow=False, fontsize=14)
    ax.set_ylabel('Sowing date (DOY) ', fontsize=18)
    ax.set_xlabel('Year', fontsize=18)
    plt.tight_layout()
    plt.savefig('%s/Esd.png' % (pathout), dpi=600)
    plt.show()
