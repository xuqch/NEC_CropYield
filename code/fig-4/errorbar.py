import matplotlib.pylab as plt
import matplotlib
import numpy as np
from pylab import rcParams
import pandas as pd

font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 8,
          'xtick.direction': 'out',
          'ytick.labelsize': 20,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


# 绘制误差棒图
def errorbar(scenarios,pathvar):
    names = ['rice', 'maize', 'soybean']
    dfs = pd.read_excel(f"./xlsx/Yield_{scenarios}_TAGP.xlsx", sheet_name='Yield')
    for name in names:
        df = dfs[dfs['veg'].isin([name])]
        mean = np.array(df['Yieldmean'])
        error = np.array(df['Yieldstd'])
        x_label = np.array(df['sspx'])
        crop = np.array(df['veg'])
        # scenario = np.array(df['scenario'])
        print(crop[0])

        x_label = ['', 'ssp126', '', 'ssp245', '', 'ssp370', '', 'ssp585', '']
        x_position = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        colors = ['#4B66AD', '#62BEA6', '#FDBA6B', '#EB6046']
        fig = plt.figure(figsize=(6, 4))
        plt.errorbar(x_position[1], mean[0], yerr=error[0], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[0],
                     mec=colors[0], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=5, linestyle="none")  # , label="Observation"
        plt.errorbar(x_position[3], mean[1], yerr=error[1], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[1],
                     mec=colors[1], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=3, linestyle="none")  # , label="Observation"
        plt.errorbar(x_position[5], mean[2], yerr=error[2], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[2],
                     mec=colors[2], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=3, linestyle="none")  # , label="Observation"
        plt.errorbar(x_position[7], mean[3], yerr=error[3], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[3],
                     mec=colors[3], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=3, linestyle="none")  # , label="Observation"

        # plt.yticks(np.arange(-10, 50, 10), np.arange(-10, 50, 10), fontsize=20)  # soybean

        plt.xticks([i for i in x_position], x_label, fontsize=25)

        plt.savefig('./%s/errorbar_%s_%s_TAGP.eps' % (pathvar, scenarios, crop[0]), format='eps', dpi=300)
        plt.savefig('./%s/errorbar_%s_%s_TAGP.png' % (pathvar, scenarios, crop[0]), format='png', dpi=300)
        plt.close()

def errorbar_ssp585(scenarios,pathvar):
    names = ['rice', 'maize', 'soybean']
    dfs = pd.read_excel(f"./xlsx/Yield_{scenarios}_TAGP.xlsx", sheet_name='Yield')
    df = dfs[dfs['sspx'].isin(['ssp585'])]
    print(df)
    mean = np.array(df['Yieldmean'])
    error = np.array(df['Yieldstd'])
    x_label = np.array(df['veg'])

    x_label = ['', 'rice', '', 'maize', '', 'soybean', '']
    x_position = [1, 2, 3, 4, 5, 6, 7]
    colors = ['#82B0D2', '#FFBE7A', '#FA7F6F']
    fig = plt.figure(figsize=(6, 4))
    plt.errorbar(x_position[1], mean[0], yerr=error[0], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[0],
                 mec=colors[0], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=5, linestyle="none")  # , label="Observation"
    plt.errorbar(x_position[3], mean[1], yerr=error[1], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[1],
                 mec=colors[1], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=3, linestyle="none")  # , label="Observation"
    plt.errorbar(x_position[5], mean[2], yerr=error[2], ecolor="k", elinewidth=2.5, marker="s", mfc=colors[2],
                 mec=colors[2], mew=2.5, ms=20, alpha=0.75, capsize=5, capthick=3, linestyle="none")  # , label="Observation"


    # plt.yticks(np.arange(-10, 50, 10), np.arange(-10, 50, 10), fontsize=20)  # soybean

    plt.xticks([i for i in x_position], x_label, fontsize=25)

    plt.savefig('./%s/errorbar_%s_%s_TAGP.eps' % (pathvar, scenarios, 'ssp585'), format='eps', dpi=300)
    plt.savefig('./%s/errorbar_%s_%s_TAGP.png' % (pathvar, scenarios, 'ssp585'), format='png', dpi=300)
    plt.close()

if __name__ == "__main__":
    scenarios = 'precipitation'# default_pr/precipitation
    errorbar(scenarios,scenarios)

    scenarios = 'co2'  # default_pr/precipitation
    errorbar(scenarios,scenarios)

    scenarios = 'temperature'  # default_pr/precipitation
    errorbar(scenarios,scenarios)
    #
    scenarios = 'default'# default_pr/precipitation
    errorbar(scenarios,scenarios)

    scenarios = 'default_add'# default_pr/precipitation
    pathvar = 'default'
    errorbar(scenarios,pathvar)
    errorbar_ssp585(scenarios,pathvar)