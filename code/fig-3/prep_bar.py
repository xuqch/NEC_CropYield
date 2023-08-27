import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pylab import rcParams
import pandas as pd

# plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码

# st = np.std(ds,axis = (0))

## Plot settings
font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 10,
          'xtick.direction': 'out',
          'ytick.labelsize': 10,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def ensemble(File, crop, ts):
    avg = []
    std = []
    Vars = ['rice', 'maize', 'soybean']

    ds = xr.open_dataset(File).pr * 365 * 86400
    ta = ds.where(ds.time.dt.year >= 2070, drop=True)
    # ta = ta.resample(time='1M').mean()
    for i, var in enumerate(Vars):
        t_bot = np.array(ts.where(crop == i, drop=True).values)  # ts crop past temp area data monmean
        t_top = np.array(ta.where(crop == i, drop=True).values)  # ta crop temperature avg
        data = np.mean(t_top, axis=0) - np.mean(t_bot, axis=0)
        avg.append(np.nanmean(data))
        std.append(np.nanstd(data))

    t_bot = np.array(ts.where(crop >= 0, drop=True))  # ta temperature avg
    t_top = np.array(ta.where(crop >= 0, drop=True))  # ts past temp area data monmean
    data = np.mean(t_top, axis=0) - np.mean(t_bot, axis=0)
    avg.append(np.nanmean(data))
    std.append(np.nanstd(data))
    return avg, std


def bar_plot(path):
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    Figout = './'
    crop = xr.open_dataset(f'{path}/crop/crop.nc').crop

    file = xr.open_dataset(f"{path}/prep/PRECP_remap_day.nc")
    pr = file.PRECT * 356 * 24 * 60 * 60
    print('baseline end')

    Avg = []
    Std = []

    for i, ssp in enumerate(ssps):
        VarFile = f"{path}/met/{ssp}/pr_day_ensmean_{ssp}_2015-2100.nc"
        avg, std = ensemble(VarFile, crop, pr)
        Avg.append(avg)
        Std.append(std)
        print(ssp)

    Avg = np.array(list(Avg))
    error = np.array(list(Std))
    print(error)
    labels = ['Rice', 'Maize', 'Soybean', 'Regional']

    x = np.arange(len(labels)) * 1.5  # 标签位置
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']  ##蓝，绿，黄，红
    width = 0.32
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    rects1 = ax.bar(x - width * 1.5, Avg[0, :], width, label='ssp126', edgecolor='k',yerr=error[0, :],
                    color=colors[0])
    rects2 = ax.bar(x - width * 0.5, Avg[1, :], width, label='ssp245', edgecolor='k',yerr=error[1, :],
                    color=colors[1])
    rects3 = ax.bar(x + width * 0.5, Avg[2, :], width, label='ssp370', edgecolor='k',yerr=error[2, :],
                    color=colors[2])
    rects4 = ax.bar(x + width * 1.5, Avg[3, :], width, label='ssp585', edgecolor='k',yerr=error[3, :],
                    color=colors[3])


    ax.set_ylabel('Future precipitation change (mm/year)', fontsize=11)
    ax.set_xlabel('Area', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best', shadow=False, fontsize=10)
    ax.set_yticks(np.arange(0, 200, 20))
    ax.set_yticklabels(np.arange(0, 200, 20))
    fig.tight_layout()
    plt.savefig('%spr_bar_plot.eps' % (Figout))
    plt.savefig('%spr_bar_plot.png' % (Figout))

    df = pd.DataFrame({'crop': labels + labels,
                       'ssp_126': np.concatenate([Avg[0, :], error[0, :]], axis=0),
                       'ssp_245': np.concatenate([Avg[1, :], error[1, :]], axis=0),
                       'ssp_370': np.concatenate([Avg[2, :], error[2, :]], axis=0),
                       'ssp_585': np.concatenate([Avg[3, :], error[3, :]], axis=0)})  # 'R': R,'KGE': KGE,
    df.to_excel("./prep_bar.xlsx", sheet_name='prep', index=True)



if __name__ == '__main__':
    import glob, os, shutil

    path = '/stu01/xuqch3/finished/data/'
    bar_plot(path)
    # box_plot()
