import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pylab import rcParams
import pandas as pd

## Plot settings
font = {'family': 'Times New Roman'}
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

Figout = "F:/Dongbei/plot/bar/"
contribution = pd.read_excel("F:/PCSE/code/scripts/analysis_plotting/contribution.xlsx",
                             sheet_name='contribution')
contribution_co2 = np.array(contribution['contribution_co2'])
contribution_precipitation = np.array(contribution['contribution_precipitation'])
contribution_tas = np.array(contribution['contribution_tas'])
x_label = np.array(contribution['sspx'])
crop = np.array(contribution['veg'])
print(crop)

labels = ['Rice', 'Maize', 'Soybean']
x = np.arange(len(labels)) * 1.5  # 标签位置
width = 0.35
colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

rects1 = ax.bar(x - width * 1.1, contribution_co2, width, label='CO2 constraint', edgecolor='k', color=colors[0])
rects2 = ax.bar(x, contribution_precipitation, width, label='Precipitation constraint', edgecolor='k', color=colors[1])
rects3 = ax.bar(x + width * 1.1, contribution_tas, width, label='Temperature constraint', edgecolor='k', color=colors[2])
ax.set_ylabel('Contribution (%)', fontsize=18)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=18)
ax.set_yticks(np.arange(0, 100, 10))
ax.set_yticklabels(np.arange(0, 100, 10), fontsize=18)
ax.legend(loc='best', shadow=False, fontsize=18)
# fig.tight_layout()
# plt.savefig('%scontribution_barplot.eps' % (Figout))
# plt.savefig('%scontribution_barplot.png' % (Figout))
plt.savefig(f'{Figout}/contribution_barplot_1.png', dpi=800)
plt.show()

# df_default = pd.read_excel("F:/PCSE/code/scripts/analysis_plotting/Yield_sowing.xlsx", sheet_name='Sheet1')
# default_mean = np.array(df_default['Yieldmean'])
# default_error = np.array(df_default['Yieldstd'])
# x_label = np.array(df_default['sspx'])
# crop = np.array(df_default['veg'])
# scenario_d = np.array(df_default['scenario'])[0]
# print(scenario_d)
#
# df_sowing = pd.read_excel("F:/PCSE/code/scripts/analysis_plotting/Yield_sowing.xlsx", sheet_name='Sheet2')
# sowing_mean = np.array(df_sowing['Yieldmean'])
# sowing_error = np.array(df_sowing['Yieldstd'])
# scenario_s = np.array(df_sowing['scenario'])[0]
# print(scenario_s)
#
# df_strategy = pd.read_excel("F:/PCSE/code/scripts/analysis_plotting/Yield_sowing.xlsx", sheet_name='Sheet3')
# strategy_mean = np.array(df_strategy['Yieldmean'])
# strategy_error = np.array(df_strategy['Yieldstd'])
# scenario_o = np.array(df_strategy['scenario'])[0]
# print(scenario_o)
#
# labels = ['Rice', 'Maize', 'Soybean']
# x = np.arange(len(labels)) * 1.5  # 标签位置
# width = 0.35
# # colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']  ##蓝，绿，黄，红
# colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#
# rects1 = ax.bar(x - width * 1.1, default_mean, width, label=scenario_d, yerr=default_error, edgecolor='k',
#                 error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[0])
# rects2 = ax.bar(x, sowing_mean, width, label=scenario_s, yerr=sowing_error, edgecolor='k',
#                 error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[1])
# rects3 = ax.bar(x + width * 1.1, strategy_mean, width, label=scenario_o, yerr=strategy_error, edgecolor='k',
#                 error_kw={'ecolor': 'k', 'capsize': 8}, color=colors[2])
#
# # plt.figure(figsize=(10, 5))
# ax.set_ylabel('Yield Change (%)', fontsize=16)
# # ax.set_xlabel('Area', fontsize=12)
# # ax.set_title('这里是标题')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=14)
# ax.legend(loc='upper right', shadow=False, fontsize=15)
# fig.tight_layout()
# # plt.savefig('%sYield_change_barplot.eps' % (Figout), dpi=800)
# plt.savefig('%sYield_change_barplot.png' % (Figout), dpi=800)
# plt.show()
