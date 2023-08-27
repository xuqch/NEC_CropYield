#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:14:27 2019

@author: zhwei
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams

### Plot settings
font = {'family': 'Times New Roman'}
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
    
if __name__=='__main__':
    dir_Fig = './'
    data_dir = './'
    data = pd.read_excel('/tera04/zhwei/PCSE/data/output/sensitivity/mean_std.xlsx' ,sheet_name="Sheet4",index_col=("Scenario"))
    data1=data#.iloc[:10,:]
    #data1=data
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

    #sns.heatmap(data1,vmin=0.3,vmax=0.8,cmap='Blues')
    #sns.heatmap(data1,vmin=10,vmax=40,cmap=cmap)
    sns.heatmap(data1,vmin=-25,vmax=5,cmap='Blues')
   # ax.legend()
    plt.tight_layout()
    plt.savefig('heatmap.eps', format='eps', dpi=600)
    plt.savefig('heatmap.png', format='png', dpi=600)

    plt.show()
