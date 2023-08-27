import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pylab import rcParams

# plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码

# st = np.std(ds,axis = (0))

## Plot settings
font = {'family': 'DejaVu Sans'}
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
    Vars = ['r', 'm', 's']
    for i, var in enumerate(Vars):
        a1 = []
        for VarFile in File:
            with xr.open_dataset(VarFile) as ds:
                ds = ds.where(ds.time.dt.year >= 2070, drop=True)
                ta = (ds.tasmax + ds.tasmin) / 2
                t_bot = np.mean(ts.where(crop == i, drop=True))  # ts crop past temp area data monmean
                t_top = np.mean(ta.where(crop == i, drop=True))  # ta crop temperature avg
                # t_bot = np.mean(ts.where(var.isnull()==False ,drop=True))
                # t_top = np.mean(ta.where(var.isnull()==False ,drop=True))
                a1.append((t_top - t_bot).data)
        avg.append(np.mean(a1))
        std.append(np.std(a1))
        # del a1[:]
        a1 = []
    a2 = []
    for VarFile in File:
        with xr.open_dataset(VarFile) as ds:
            ds = ds.where(ds.time.dt.year >= 2070, drop=True)
            ta = (ds.tasmax + ds.tasmin) / 2
            ta = np.mean(ta.where(crop >= 0, np.nan))  # ta temperature avg
            ts = np.mean(ts.where(crop >= 0, np.nan))  # ts past temp area data monmean
            a2.append(np.mean(ta - ts))  # regional
    avg.append(np.mean(a2))
    std.append(np.std(a2))
    a2 = []
    return (avg, std)
    print('end')


if __name__ == '__main__':
    import glob, os, shutil

    # define the directory
    # set variables
    ssp126 = []
    ssp245 = []
    ssp370 = []
    ssp585 = []

    # Vars = ['hfls', 'hurs', 'hfss', 'mrro', 'pr', 'prsn','sfcWind','tas','tasmax','tasmin','vap']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    Vars = ['tasmax', 'tasmin']
    path = '/Volumes/xuqch_min/PCSE/ensemble/'
    ssp126 = '/Volumes/xuqch_min/PCSE/ensemble/ssp126/'
    ssp245 = '/Volumes/xuqch_min/PCSE/ensemble/ssp245/'
    ssp370 = '/Volumes/xuqch_min/PCSE/ensemble/ssp370/'
    ssp585 = '/Volumes/xuqch_min/PCSE/ensemble/ssp585/'
    Figout = '/Volumes/xuqch_min/PCSE/temp/'
    path_temp = '/Volumes/xuqch_min/PCSE/temp/'
    # baseline

    VarFile = ('%sTBOT_remap_monmean.nc' % (path_temp))
    with xr.open_dataset(VarFile) as ts:
        ts = ts.TBOT

    # maize rice soybean wheat
    VarFile = ('/Volumes/xuqch_min/PCSE/crop/crop.nc')
    # print(VarFile)
    with xr.open_dataset(VarFile) as ds:
        crop = ds["crop"]
        # r = ds.where(ds == 0, drop=True)
        # m = ds.where(ds == 1, drop=True)
        # s = ds.where(ds == 2, drop=True)
        # print(np.mean(w))
        print('Prepared end')

    File = []
    Avg = []
    Std = []
    avg = []
    std = []
    i = 0
    for ssp in ssps:
        os.chdir("%s%s" % (path, ssp))
        for SubVar in glob.glob("*PCSE_input_monmean.nc"):
            # print(SubVar)
            VarFile = ('%s%s/%s' % (path, ssp, SubVar))
            File.append(VarFile)
        avg, std = ensemble(File, crop, ts)
        print(std)
        # print(type(i))
        Avg.append(avg)
        Std.append(std)
        print(i)
        i = i + 1
    # print(Avg[:,0])
    Avg = np.array(list(Avg))
    error = np.array(list(Std))
    labels = ['Rice', 'Maize', 'Soybean', 'Regional']
    ssp126 = Avg[0, :]
    ssp245 = Avg[1, :]
    ssp370 = Avg[2, :]
    ssp585 = Avg[3, :]

    x = np.arange(len(labels)) * 1.5  # 标签位置
    width = 0.32
    colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']  ##蓝，绿，黄，红
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rects1 = ax.bar(x - width * 1.5, ssp126, width, label='ssp126', yerr=error[0, :], edgecolor='k',
                    color=colors[0])
    rects2 = ax.bar(x - width * 0.5, ssp245, width, label='ssp245', yerr=error[1, :], edgecolor='k',
                    color=colors[1])
    rects3 = ax.bar(x + width * 0.5, ssp370, width, label='ssp370', yerr=error[2, :], edgecolor='k',
                    color=colors[2])
    rects4 = ax.bar(x + width * 1.5, ssp585, width, label='ssp585', yerr=error[3, :], edgecolor='k',
                    color=colors[3])

    # 为y轴、标题和x轴等添加一些文本。
    # plt.figure(figsize=(10, 5))
    ax.set_ylabel('Future temperature change ($^\circ$C)', fontsize=11)
    ax.set_xlabel('Area', fontsize=12)
    # ax.set_title('这里是标题')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', shadow=False, fontsize=8.5)
    ax.set_yticks(np.arange(0, 8))
    ax.set_yticklabels(np.arange(0, 8))
    fig.tight_layout()
    plt.savefig('%stas_bar_plot.eps' % (Figout))
    plt.savefig('%stas_bar_plot.png' % (Figout))
    plt.show()
