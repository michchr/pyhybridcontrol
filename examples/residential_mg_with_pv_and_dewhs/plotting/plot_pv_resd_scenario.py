from examples.residential_mg_with_pv_and_dewhs.plotting.plotting_helper import *
import os
import glob
from structdict import OrderedStructDict

plot_with_tex(True)
##LOAD DATA FRAMES ##

paths = glob.glob(r'../data/pv_supply_norm_1w_max_15min_from_091218_150219.pickle')
paths.extend(glob.glob(r'../data/res_demand_norm_1w_mean_15min_from_091218_150219.pickle'))
dfs = OrderedStructDict()
for path in paths:
    if path.endswith('pickle'):
        dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)

pv_df = dfs['pv_supply_norm_1w_max_15min_from_091218_150219']
resd_df = dfs['res_demand_norm_1w_mean_15min_from_091218_150219']

dates = IDX['2018-12-10':'2018-12-19']
pv_df = pv_df.loc[dates]
resd_df = resd_df[dates]

# Plot
fig, axes = get_fig_axes_A4(2, 1, v_scale=1 / 3, h_scale=1, sharex='all')

ax0 = axes[0]
ax1 = axes[1]

pv_df.plot(drawstyle="steps-post", ax=ax0, lw=lineWidthNorm)
resd_df.plot(drawstyle="steps-post", ax=ax1, lw=lineWidthNorm, color='r')


## Legends

ax0.legend().remove()
ax1.legend().remove()

##PLOT LABELLING#
ax0.set_ylabel(tex_s(r'$P_{\text{pv}}\;[\si{\watt}]$'), wrap=True)
ax1.set_ylabel(tex_s(r'$P_{r}\;[\si{\watt}]$'), wrap=True)
ax1.set_xlabel("Date", labelpad=-10)

fig.savefig(FIG_SAVE_PATH + "plot_pv_resd_scenario.pdf", bbox_inches='tight')
