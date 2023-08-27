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
          'font.size': 18,
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


def plot(run):
    Figout = "./"
    df = pd.read_excel("./Yield_sowing.xlsx", sheet_name=f'{run}')
    mean = np.array(df['Yieldmean'])
    error = np.array(df['Yieldstd'])
    x_label = np.array(df['sspx'])
    crop = np.array(df['veg'])
    scenario = np.array(df['scenario'])

    labels = ['Rice', 'Maize', 'Soybean']
    x = np.arange(len(labels)) * 2  # 标签位置
    width = 0.4
    # colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
    colors = sns.color_palette("Set3", n_colors=6, desat=.9).as_hex() #
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    print(df[df['scenario'].isin(['CR'])])

    rects1 = ax.bar(x - width * 1.5,   df[df['scenario'].isin(['Default'])].Yieldmean, width, label='Default',
                    yerr=df[df['scenario'].isin(['Default'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[4])
    rects2 = ax.bar(x - width * 0.5,   df[df['scenario'].isin(['CR'])].Yieldmean, width, label='CR',
                    yerr=df[df['scenario'].isin(['CR'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[0])
    rects3 = ax.bar(x + width * 0.5,   df[df['scenario'].isin(['Sowing'])].Yieldmean, width, label='PDO',
                    yerr=df[df['scenario'].isin(['Sowing'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[5])
    rects4 = ax.bar(x + width * 1.5,   df[df['scenario'].isin(['Optimized'])].Yieldmean, width, label='PDO_CR',
                    yerr=df[df['scenario'].isin(['Optimized'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[3])

    ax.set_ylabel('Yield Change (%)', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(loc='best', shadow=False, fontsize=17)
    fig.tight_layout()
    plt.savefig(f'%s{run}_Yield_change_barplot.eps' % (Figout), dpi=800)
    plt.savefig(f'%s{run}_Yield_change_barplot.png' % (Figout), dpi=800)
    plt.show()


def add_irrigate_plot(run):
    Figout = "./"
    df = pd.read_excel("./Yield_sowing.xlsx", sheet_name=f'{run}')
    mean = np.array(df['Yieldmean'])
    error = np.array(df['Yieldstd'])
    x_label = np.array(df['sspx'])
    crop = np.array(df['veg'])
    scenario = np.array(df['scenario'])

    labels = ['Rice', 'Maize', 'Soybean']
    x = np.arange(len(labels)) * 2.5  # 标签位置
    width = 0.4

    colors = sns.color_palette("Set3", n_colors=6, desat=.9).as_hex() #
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))


    ax.bar(x - width * 2,   df[df['scenario'].isin(['Default'])].Yieldmean, width, label='Default',
                    yerr=df[df['scenario'].isin(['Default'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[4])
    ax.bar(x- width,        df[df['scenario'].isin(['CR'])].Yieldmean, width, label='CR',
                    yerr=df[df['scenario'].isin(['CR'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[0])
    ax.bar(x ,              df[df['scenario'].isin(['Sowing'])].Yieldmean, width, label='PDO',
                    yerr=df[df['scenario'].isin(['Sowing'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[5])
    ax.bar(x + width,       df[df['scenario'].isin(['Optimized'])].Yieldmean, width, label='PDO_CR',
                    yerr=df[df['scenario'].isin(['Optimized'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[3])
    ax.bar(x + width*2,     df[df['scenario'].isin(['Irrigate'])].Yieldmean, width, label='Irrigation',
                    yerr=df[df['scenario'].isin(['Irrigate'])].Yieldstd, edgecolor='k',
                    error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[2])


    ax.set_ylabel('Yield Change (%)', fontsize=20)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(loc='best', shadow=False, fontsize=17)
    fig.tight_layout()
    plt.savefig(f'%s{run}_Yield_change_add_Irrigate_barplot.eps' % (Figout), dpi=800)
    plt.savefig(f'%s{run}_Yield_change_add_Irrigate_barplot.png' % (Figout), dpi=800)
    plt.show()

if __name__ == "__main__":
    run = 'TAGP'
    plot(run)

    add_irrigate_plot(run)
