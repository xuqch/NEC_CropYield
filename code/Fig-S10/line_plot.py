import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pylab import rcParams
import time
import datetime
import numpy as np
import seaborn as sns

### Plot settings
font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'grid.linewidth': 0.2,
          'font.size': 20,
          'legend.fontsize': 16,
          'legend.frameon': False,
          'xtick.labelsize': 22,
          'xtick.direction': 'out',
          'ytick.labelsize': 22,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

if __name__ == '__main__':
    import glob, os, shutil

    '''
    rice:0, maize:1, soybean:2
    -'''
    path = '/stu01/xuqch3/finished/data/'
    Figout = f'{path}/code/Fig-S10/'
    names = ['rice', 'maize', 'soybean']

    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    colors_list = sns.color_palette("Set3", n_colors=6, desat=.9).as_hex()  #
    colors = [colors_list[4], colors_list[0], colors_list[5], colors_list[3]]  # colors_list[2],

    for i, name in enumerate(names):
        print(name)
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        for j, ssp in enumerate(ssps):
            stnlist = f'{path}/code/Fig-S10/{name}_TAGP.xlsx'
            DSSAT = pd.read_excel(stnlist, header=0, sheet_name=f'{name}')  # flux Sheet2,header=0
            DSSAT_ssp = xr.DataArray(np.array(DSSAT[f'{ssp}'].values), coords={"time": range(2015, 2101)}, dims=["time"])

            dtnlist = f'{path}/code/fig-4/xlsx/default_add_{name}_TAGP.xlsx'
            default = pd.read_excel(dtnlist, header=0, sheet_name=f'{name}')  # flux Sheet2,header=0
            default_ssp = xr.DataArray(np.array(default[f'{ssp}'].values), coords={"time": range(2015, 2100)}, dims=["time"])

            DSSAT_ssp.plot.line(x='time', label=f'DSSAT {ssp}', linewidth=1.5, linestyle='solid',
                                alpha=1, color=colors[j])  # ,color = 'blue'

            default_ssp.plot.line(x='time', label=f'PCSE {ssp}', linewidth=1.5, linestyle='--',
                                  alpha=1, color=colors[j])  # ,color = 'blue'

        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_ylabel('Yield change (%)', fontsize=25)
        ax.set_xlabel('Year', fontsize=20)
        ax.tick_params(axis='both', top='off', labelsize=22)
        ax.legend(loc='best', shadow=False, fontsize=16, ncol=4)
        ax.set_title('%s' % (name))
        plt.tight_layout()
        plt.savefig(f'{Figout}/{name}_output_irrigate.png', format='png', dpi=800)  # timeseries_lines
        plt.show()

    # names = ['rice']
    # for i, name in enumerate(names):
    #     print(name)
    #     fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    #     for j, ssp in enumerate(ssps):
    #         stnlist = f'{path}/code/Fig-S10/{name}_TAGP.xlsx'
    #         DSSAT = pd.read_excel(stnlist, header=0, sheet_name=f'{name}')  # flux Sheet2,header=0
    #         DSSAT_ssp = xr.DataArray(np.array(DSSAT[f'{ssp}'].values), coords={"time": range(2015, 2101)}, dims=["time"])
    #
    #         dtnlist = f'{path}/code/fig-4/xlsx/default_{name}_TAGP.xlsx'
    #         default = pd.read_excel(dtnlist, header=0, sheet_name=f'{name}')  # flux Sheet2,header=0
    #         default_ssp = xr.DataArray(np.array(default[f'{ssp}'].values), coords={"time": range(2015, 2100)}, dims=["time"])
    #
    #         DSSAT_ssp.plot.line(x='time', ax=axes[0], label=f'DSSAT {ssp}', linewidth=1.5, linestyle='solid',
    #                             alpha=1, color=colors[j])  # ,color = 'blue'
    #
    #         default_ssp.plot.line(x='time', ax=axes[1], label=f'PCSE {ssp}', linewidth=1.5, linestyle='--',
    #                               alpha=1, color=colors[j])  # ,color = 'blue'
    #
    #     axes[0].axhline(y=0, color='gray', linestyle='--')
    #     axes[0].set_ylabel('Yield change (%)', fontsize=25)
    #     axes[0].legend(loc='best', shadow=False, fontsize=16, ncol=4)
    #     axes[0].set_xlabel('', fontsize=0)
    #     axes[0].set_title('%s' % (name), fontsize=25)
    #
    #     axes[1].axhline(y=0, color='gray', linestyle='--')
    #     axes[1].set_ylabel('Yield change (%)', fontsize=25)
    #     axes[1].set_xlabel('Year', fontsize=20)
    #     axes[1].legend(loc='best', shadow=False, fontsize=16, ncol=4)
    #     plt.tight_layout()
    #     plt.savefig(f'{Figout}/{name}_output.png', format='png', dpi=800)  # timeseries_lines
    #     plt.show()
