from email.policy import default
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pylab import rcParams

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
          'xtick.labelsize': 8,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

if __name__ == '__main__':
    import glob, os, shutil
    run='TAGP'
    scenario = 'CR'# optimized CR
    path = '/stu01/xuqch3/finished/data'

    df = pd.DataFrame()
    optimized_distribution = f"{path}/PCSE/output/adaptation/{scenario}/{scenario}_distribution_{run}.nc"
    Figout = "./"
    distribution = xr.open_dataset(optimized_distribution)[f'{run}']
    crop = xr.where(distribution > -1, 1, np.nan).groupby('year').sum(...)
    rice = xr.where(distribution == 0, 1, np.nan).groupby('year').sum(...)
    maize = xr.where(distribution == 1, 1, np.nan).groupby('year').sum(...)
    print(maize)
    soybean = xr.where(distribution == 2, 1, np.nan).groupby('year').sum(...)


    rice_land = rice / crop * 100
    maize_land = maize / crop * 100
    soybean_land = soybean / crop * 100
    print(soybean_land)

    colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
    markers = ['*', 'x', '+']
    lines = [1.5, 1.5, 1.5, 1.5]
    alphas = [1., 1., 1., 1.]
    linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dashed', 'dashdot', 'solid', 'solid']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rice_land.plot.line(x='year', label='Rice', linewidth=lines[1], linestyle=linestyles[1],
                        alpha=alphas[1], color=colors[0])  # ,color = 'blue'
    maize_land.plot.line(x='year', label='Maize', linewidth=lines[2], linestyle=linestyles[2],
                         alpha=alphas[2], color=colors[1])  # ,color = 'green
    soybean_land.plot.line(x='year', label='Soybean', linewidth=lines[0], linestyle=linestyles[0],
                           alpha=alphas[0], color=colors[2])  # ,color = 'orangered'

    # ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('Proportion (%)', fontsize=20)
    ax.set_xlabel('Year', fontsize=17)
    ax.tick_params(axis='both', top='off', labelsize=16)
    ax.legend(loc='best', shadow=False, fontsize=12)
    plt.yticks(np.arange(0, 120, 20), np.arange(0, 120, 20))
    plt.tight_layout()
    plt.savefig(f'{Figout}/{scenario}_crop_change_{run}.eps', format='eps', dpi=300)  # timeseries_lines
    plt.savefig(f'{Figout}/{scenario}_crop_change_{run}.png', format='png', dpi=300)  # timeseries_lines
    plt.show()

    print('plot end')
    df['time'] = pd.Series(soybean_land.year)
    df['rice'] = pd.Series(rice_land)
    df['maize'] = pd.Series(maize_land)
    df['soybean'] = pd.Series(soybean_land)
    df.to_csv(f'./Yield_{scenario}_proportion.csv')