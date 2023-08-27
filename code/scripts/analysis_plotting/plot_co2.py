# -*- coding: utf-8 -*-
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "0.1"
__release__ = "0.1"
__date__ = "Oct 2020"

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
### Plot settings
font = {'family' : 'Times New Roman'}
#font = {'family' : 'Myriad Pro'}
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
pparam  = dict(xlabel='Year', ylabel='CO2 ($ppm$)')
pathin='/tera04/zhwei/PCSE/data/co2/'
pathout = '/tera04/zhwei/PCSE/Fig/'
if __name__=='__main__':
    import glob, os, shutil
    ssps     = ['ssp126','ssp245','ssp370','ssp585']
    header_list = ["year", "co2"]
    
    with plt.style.context(['science','no-latex']):
        fig, ax = plt.subplots()
        for ssp in ssps:
            dco2 = pd.read_csv(f'{pathin}/co2_{ssp}_annual_2015_2100.txt', delimiter=r"\s+", names=header_list)
            ax.plot(dco2['year'], dco2['co2'], label=ssp)
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    #fig.savefig(f'{pathout}/plot_co2.pdf')
    fig.savefig(f'{pathout}/plot_co2.png', dpi=300)
    fig.savefig(f'{pathout}/plot_co2.eps', dpi=300)



