from examples.residential_mg_with_pv_and_dewhs.plotting.plotting_helper import *
import glob

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from datetime import datetime
from structdict import OrderedStructDict

plot_with_tex(True)
##LOAD DATA FRAMES ##

paths = glob.glob(r'../sim_out/soft_constraint/*')
dfs = OrderedStructDict()
for path in paths:
    if path.endswith('sim_out'):
        dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)

## Plot

fig, axes = get_fig_axes_A4(2, 1, v_scale=1 / 3, h_scale=1, sharex='all')

ax0 = axes[0]
ax1 = axes[1]
axins_top = inset_axes(ax0, width="20%", height="40%", loc=2)
axins_bot = inset_axes(ax0, width="20%", height="40%", loc=3)

q_values = []

for ind, (name, df) in enumerate(dfs.copy().items()):
    s_name = name.split('_')
    st = int(s_name[s_name.index('st') + 1])
    sb = int(s_name[s_name.index('sb') + 1])
    q_values.append((st, sb))
    dfs[name] = df['2018-12-14':'2018-12-15']

axes = [ax0, axins_top, axins_bot]
for _ax in axes:
    for df in dfs.values():
        df.dewh[1].loc[:, IDX[('mpc_pb',), 'x']].plot.line(ax=_ax, drawstyle='steps-post', lw=lineWidthNorm)
    _ax.axhline(y=65, linestyle='--', color='r', lw=lineWidthNorm, label='T_max')
    _ax.axhline(y=50, linestyle='--', color='b', lw=lineWidthNorm, label='T_min')

omega_data = dfs['sim_Np_48_st_1_sb_1_Ns_20_Nsr_8_N_h_1'].dewh[1].loc[:, IDX[('mpc_pb',), 'omega']] * 900
omega_data.plot.line(ax=ax1, drawstyle='steps-post', lw=lineWidthNorm)

##INSET PLOT LOCATIONS
x1, x2, y1, y2 = datetime(2018, 12, 15, 3, 30), datetime(2018, 12, 15, 6), 60, 75
axins_top.set_xlim(x1, x2)
axins_top.set_ylim(y1, y2)

x1, x2, y1, y2 = datetime(2018, 12, 15, 5), datetime(2018, 12, 15, 7), 43, 58
axins_bot.set_xlim(x1, x2)
axins_bot.set_ylim(y1, y2)

axins_top.tick_params(which='both', labelleft=False, labelbottom=False, bottom=False, top=False)
axins_top.legend().remove()
axins_bot.tick_params(which='both', labelleft=False, labelbottom=False, bottom=False, top=False)
axins_bot.legend().remove()

mark_inset(ax0, axins_top, loc1=1, loc2=4, fc="none", ec="0.5")
mark_inset(ax0, axins_bot, loc1=1, loc2=4, fc="none", ec="0.5")

##PLOT LABELLING
ax0.legend([fr"$q_{{h}}^{{\mu}}={list(q_values[0])}^{{\top}}$",
            fr"$q_{{h}}^{{\mu}}={list(q_values[1])}^{{\top}}$",
            fr"$q_{{h}}^{{\mu}}={list(q_values[2])}^{{\top}}$",
            tex_s(r'$T_\text{max}$'),
            tex_s(r'$T_\text{min}$')],
           loc='center right', bbox_to_anchor=(1.235, 0.5))
ax0.set_ylabel(tex_s(r'$x_{h}\;[\si{\celsius}]$'))

ax1.set_ylabel(tex_s(r'$\omega_{h}^{\text{nom}}\;[\si{\liter}]$'))
# ax1.set_xlabel('Date', labelpad=-5)
ax1.get_legend().remove()

fig.savefig(FIG_SAVE_PATH + "plot_soft_penalty_tradeoff.pdf", bbox_inches='tight')
