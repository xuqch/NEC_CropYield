import matplotlib.pylab as plt
import matplotlib
import numpy as np
from pylab import rcParams
import pandas as pd

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
# 绘制误差棒图


df = pd.read_excel("F:/PCSE/code/scripts/analysis_plotting/Yield.xlsx", sheet_name='Sheet1')
mean = np.array(df['Yieldmean'])
error = np.array(df['Yieldstd'])
x_label = np.array(df['sspx'])
crop = np.array(df['veg'])
scenario = np.array(df['scenario'])
print(crop[0], scenario[0])

x_position = [2, 4, 6, 8]
colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
fig = plt.figure(figsize=(6, 4))
plt.errorbar(x_position[0], mean[0], yerr=error[0], ecolor="k", elinewidth=1, marker="s", mfc=colors[0],
             mec="k", mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")  # , label="Observation"
plt.errorbar(x_position[1], mean[1], yerr=error[1], ecolor="k", elinewidth=1, marker="s", mfc=colors[1],
             mec="k", mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")  # , label="Observation"
plt.errorbar(x_position[2], mean[2], yerr=error[2], ecolor="k", elinewidth=1, marker="s", mfc=colors[2],
             mec="k", mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")  # , label="Observation"
plt.errorbar(x_position[3], mean[3], yerr=error[3], ecolor="k", elinewidth=1, marker="s", mfc=colors[3],
             mec="k", mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")  # , label="Observation"

plt.yticks(np.arange(0, 50, 10), np.arange(0, 50, 10), fontsize=14)
plt.xticks([i for i in x_position], x_label, fontsize=14)
plt.ylabel('%s' % (crop[0]))
# plt.grid(linestyle="--")  # 绘制图中虚线 透明度0.3
# fig = plt.gcf()
# plt.savefig('%s_%s.eps' % (scenario[0], crop[0]), format='eps', dpi=800)
# plt.savefig('%s_%s.png'%(scenario[0],crop[0]), format='png', dpi=800)
plt.show()
# plt.close()
