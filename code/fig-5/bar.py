import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pylab import rcParams
import pandas as pd
import seaborn as sns

## Plot settings
font = {'family': 'Times New Roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 18,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 17,
          'legend.frameon': False,
          'xtick.labelsize': 18,
          'xtick.direction': 'out',
          'ytick.labelsize': 18,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def markers(data):
    negetive_percentiles = [f'{p:.3f}' if (p < -15)&(p > -30) else '' for p in data]
    positive_percentiles = [f'{p:.3f}' if (p > 20)|((p < -30)) else '' for p in data]
    small_percentiles = [f'{p:.3f}' if (p <= 20) & (p >= -15) else '' for p in data]
    return small_percentiles, negetive_percentiles,positive_percentiles


def plot(run):
    Figout = "./"
    df = pd.read_excel(f"./box_{run}.xlsx", sheet_name=f'{run}')
    mean = np.array(df['mean'])
    crop = np.array(df['veg'])
    default_mean = np.array(df[df['scenario'].isin(['default'])]['mean'])
    data = []

    labels = ['Rice', 'Maize', 'Soybean']
    x = np.arange(len(labels)) * 2  # 标签位置
    width = 0.4
    colors = sns.color_palette("Set3", n_colors=6, desat=.9).as_hex()  #
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    rects1 = ax.barh(x + width, np.array(df[df['scenario'].isin(['co2'])]['mean']) - default_mean, width, label='Co2 - Default', edgecolor='k',
                     error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[0])
    small_percentiles, negetive_p,positive_p = markers(np.array(df[df['scenario'].isin(['co2'])]['mean'])-default_mean)
    ax.bar_label(rects1, negetive_p, padding=-120, color='black', fmt='.3f')
    ax.bar_label(rects1, positive_p, padding=1, color='white',  fmt='.3f',label_type='center')
    ax.bar_label(rects1, small_percentiles, padding=3, color='black',  fmt='.3f')

    rects2 = ax.barh(x, np.array(df[df['scenario'].isin(['precipitation'])]['mean']) - default_mean, width, label='Precipitation - Default',
                     edgecolor='k',
                     error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[5])
    small_percentiles, negetive_p,positive_p = markers(np.array(df[df['scenario'].isin(['precipitation'])]['mean'])-default_mean)
    ax.bar_label(rects2, negetive_p, padding=-120, color='black', fmt='.3f')
    ax.bar_label(rects2, positive_p, padding=1, color='white',  fmt='.3f',label_type='center')
    ax.bar_label(rects2, small_percentiles, padding=3, color='black',  fmt='.3f')

    rects3 = ax.barh(x - width, np.array(df[df['scenario'].isin(['temperature'])]['mean']) - default_mean, width, label='Temperature - Default',
                     edgecolor='k',
                     error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[3])
    small_percentiles, negetive_p,positive_p = markers(np.array(df[df['scenario'].isin(['temperature'])]['mean'])-default_mean)
    ax.bar_label(rects3, negetive_p, padding=-120, color='black', fmt='.3f')
    ax.bar_label(rects3, positive_p, padding=1, color='white',  fmt='.3f',label_type='center')
    ax.bar_label(rects3, small_percentiles, padding=3, color='black',  fmt='.3f',fontsize=12)


    ax.set_xlabel('Mean', fontsize=20)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=18)
    ax.legend(loc='best', shadow=False, fontsize=12)
    fig.tight_layout()
    plt.savefig(f'%s{run}_Yield_change_barplot.eps' % (Figout), dpi=800)
    plt.savefig(f'%s{run}_Yield_change_barplot.png' % (Figout), dpi=800)



if __name__ == "__main__":
    run = 'TAGP'
    plot(run)
