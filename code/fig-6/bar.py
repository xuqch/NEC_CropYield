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

Figout = "./"
contribution = pd.read_excel("./contribution.xlsx",
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
ax.set_yticks(np.arange(0, 130, 20))
ax.set_yticklabels(np.arange(0, 130, 20), fontsize=18)
ax.legend(loc='best', shadow=False, fontsize=18)
# fig.tight_layout()

plt.savefig(f'{Figout}/contribution.png', dpi=600)
plt.savefig(f'{Figout}/contribution.eps', dpi=600)
plt.show()
