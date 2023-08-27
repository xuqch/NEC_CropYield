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

    ds = xr.open_dataset(File).tas
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
    Vars = ['tas']

    Figout = './'
    # baseline
    crop = xr.open_dataset(f'{path}/crop/crop.nc').crop
    file = xr.open_dataset(f"{path}/temp/TBOT_remap_day.nc")
    tas = file.TBOT
    print('baseline end')

    Avg = []
    Std = []

    for i, ssp in enumerate(ssps):
        VarFile = f"{path}{ssp}/tas_day_ensmean_{ssp}_2015-2100.nc"
        avg, std = ensemble(VarFile, crop, tas)
        Avg.append(avg)
        Std.append(std)
        print(ssp)

    # print(Avg[:,0])
    Avg = np.array(list(Avg))
    error = np.array(list(Std))
    labels = ['Rice', 'Maize', 'Soybean', 'Regional']
    x = np.arange(len(labels)) * 1.5  # 标签位置
    width = 0.32
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']  ##蓝，绿，黄，红
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    rects1 = ax.bar(x - width * 1.5, Avg[0, :], width, label='ssp126', yerr=error[0, :], edgecolor='k',
                    color=colors[0], alpha=0.7)
    rects2 = ax.bar(x - width * 0.5, Avg[1, :], width, label='ssp245', yerr=error[1, :], edgecolor='k',
                    color=colors[1], alpha=0.7)
    rects3 = ax.bar(x + width * 0.5, Avg[2, :], width, label='ssp370', yerr=error[2, :], edgecolor='k',
                    color=colors[2], alpha=0.7)
    rects4 = ax.bar(x + width * 1.5, Avg[3, :], width, label='ssp585', yerr=error[3, :], edgecolor='k',
                    color=colors[3], alpha=0.7)

    ax.set_ylabel('Future temperature change ($^\circ$C)', fontsize=11)
    ax.set_xlabel('Area', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', shadow=False, fontsize=10)
    ax.set_yticks(np.arange(0, 12))
    ax.set_yticklabels(np.arange(0, 12))
    fig.tight_layout()
    plt.savefig('%stas_bar_plot.eps' % (Figout), dpi=600)
    plt.savefig('%stas_bar_plot.png' % (Figout), dpi=600)
    df = pd.DataFrame({'crop': labels + labels,
                       'ssp_126': np.concatenate([Avg[0, :], error[0, :]], axis=0),
                       'ssp_245': np.concatenate([Avg[1, :], error[1, :]], axis=0),
                       'ssp_370': np.concatenate([Avg[2, :], error[2, :]], axis=0),
                       'ssp_585': np.concatenate([Avg[3, :], error[3, :]], axis=0)})  # 'R': R,'KGE': KGE,
    df.to_excel("./tas_bar.xlsx", sheet_name='tas', index=True)



if __name__ == '__main__':
    import glob, os, shutil

    path = '/stu01/xuqch3/finished/data/'
    bar_plot(path)
    # train_data = get_data()
    # box_plot()
