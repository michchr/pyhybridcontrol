from examples.residential_mg_with_pv_and_dewhs.plotting.plotting_helper import *
import os
import glob
from structdict import OrderedStructDict

plot_with_tex(True)
##LOAD DATA FRAMES ##

paths = glob.glob(r'../data/dewh_dhw_demand_stochastic_scenarios_15Min_200Lpd_mean.pickle')
dfs = OrderedStructDict()
for path in paths:
    if path.endswith('pickle'):
        dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)

df = dfs['dewh_dhw_demand_stochastic_scenarios_15Min_200Lpd_mean']

num_scenarios = 20
df_s = df.loc[:, :num_scenarios]
df_act = df.loc[:, num_scenarios:num_scenarios]

## Plot
fig, axes = get_fig_axes_A4(2, 1, v_scale=1 / 2.5, h_scale=1, sharex='all')

ax0 = axes[0]
ax1 = axes[1]

df_s.plot(drawstyle="steps-post", ax=ax0, lw=lineWidthNorm, alpha=0.25)
df_act.plot(drawstyle="steps-post", ax=ax0, lw=lineWidthThick, color='k')

widths = 60 * 60 * 1e9 / 2  # half and hour
xmin, xmax = ax0.get_xlim()
positions = np.linspace(0, 23 * 3600 * 1e9, 24)

capprops = dict(linestyle='-', linewidth=lineWidthThick, color='b')
meanprops = dict(marker='x', markeredgecolor='k')
(df.resample('1H').sum()).T.boxplot(ax=ax1, whis='range', showmeans=True, positions=positions,
                                    widths=widths, capprops=capprops, meanprops=meanprops)



ax1.xaxis.set_major_formatter(TDelta_formater())
ax1.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=60 * 60 * 1e9 * 2, offset=15 * 60 * 1e9))
ax1.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(base=60 * 60 * 1e9 / 2, offset=15 * 60 * 1e9))
plt.subplots_adjust(hspace=0.1, top=0.95)

## Legends


legend_elements0 = [
    Line2D([0], [0], color='k', lw=lineWidthThick, label=r'Example actual demand'),
    Line2D([0], [0], color='grey', lw=lineWidthNorm,
           label=fr'{num_scenarios} example demand scenarios', alpha=0.3)]

legend_elements1 = [
    Line2D([0], [0], linestyle='-', color='b', lw=lineWidthThick, label=r'Upper and lower bounds'),
    Line2D([], [], linestyle='', color='k', marker='x', lw=lineWidthNorm, label=r'Mean demand')]

ax0.legend(handles=legend_elements0, loc='upper right')
ax1.legend(handles=legend_elements1, loc='upper right', handlelength=0.5)

##PLOT LABELLING#
ax0.set_ylabel(tex_s(r'$\omega_{h}^{\text{nom}}\;[\si{\liter\per15\minute}]$'))
ax1.set_ylabel(tex_s(r'$\omega_{h}^{\text{nom}}\;[\si{\liter\per\hour}]$'))
ax1.set_xlabel('Time of day')

fig.savefig(FIG_SAVE_PATH + "plot_dwh_scenarios.pdf", bbox_inches='tight')
