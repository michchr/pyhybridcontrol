from examples.residential_mg_with_pv_and_dewhs.plotting.plotting_helper import *
import os
import glob
from structdict import OrderedStructDict
from collections import namedtuple
import pickle
import time
import itertools

plot_with_tex(True)
##LOAD DATA FRAMES ##

st = time.time()
paths = glob.glob(r'../sim_out/single_*/*')
dfs = OrderedStructDict()
for path in paths:
    if path.endswith('sim_out'):
        dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)
print(f"Time to load data: {time.time() - st}")

st = time.time()
## Get meta data
Meta = namedtuple('Meta', ['N_h', 'T_max', 'T_min', 'N_p', 'N_s', 'N_sr', 'scen'])
dfs_meta = OrderedStructDict()
for name, df in dfs.items():
    s_name = name.split('_')
    N_h = int(s_name[s_name.index('Nh') + 1])
    N_p = int(s_name[s_name.index('Np') + 1])
    N_s = int(s_name[s_name.index('Ns') + 1])
    N_sr = int(s_name[s_name.index('Nsr') + 1])
    scen = int(s_name[s_name.index('scen') + 1])
    T_max = int(s_name[s_name.index('Tmax') + 1])
    T_min = int(s_name[s_name.index('Tmin') + 1])
    dfs_meta[Meta(N_h=N_h, T_max=T_max, T_min=T_min, N_p=N_p, N_s=N_s, N_sr=N_sr, scen=scen)] = df

print(f"Time to meta: {time.time() - st}")


class _IDXmeta(type(IDX)):
    def __call__(self, *args, N_h=None, T_max=None, T_min=None, N_p=None, N_s=None, N_sr=None, scen=None):
        def _sl(item):
            if item is None:
                return IDX[:]
            else:
                return IDX[item]

        ind = [_sl(N_h), _sl(T_max), _sl(T_min), _sl(N_p), _sl(N_s), _sl(N_sr), _sl(scen)]
        for arg in args:
            ind.append(_sl(arg))
        return self[tuple(ind)]


IDXmeta = _IDXmeta()
df_select = pd.concat(list(dfs_meta.values()), keys=dfs_meta.keys(), names=Meta._fields, axis=1)

controllers = ['mpc_pb', 'mpc_ce', 'mpc_minmax', 'mpc_sb_full', 'mpc_sb_reduced', 'thermo']

c_exc_reduced = controllers.copy()
c_exc_reduced.remove('mpc_sb_reduced')
drop_reduced = [(controller, N_sr) for controller, N_sr in itertools.product(c_exc_reduced, [6, 8])]

# PV self consumption

pv_gen = df_select.loc[:, IDXmeta('pv', 1, None, 'y')].sum(axis=0)
p_exp = df_select.loc[:, IDXmeta('grid', 1, None, 'p_exp')].sum(axis=0)
pv_self = (pv_gen - p_exp.values) / (pv_gen.values)
pv_self = pd.DataFrame(
    pv_self.mean(level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0]))

pv_self = pv_self.drop(labels=drop_reduced, axis=0)
pv_self = pv_self.reindex(axis=0, level=0, labels=controllers)

pv_self_increase = (pv_self - pv_self.loc['thermo'].values) * 100
pv_self_increase = pv_self_increase.drop(axis=0, level=0, labels=['thermo'])

pv_self_increase_65 = pv_self_increase[65]
pv_self_increase_80 = pv_self_increase[80]

# Cost data
cost_df = df_select.loc[:, IDXmeta('grid', 1, None, 'cost')].sum(axis=0).mean(
    level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])

cost_df = cost_df.drop(labels=drop_reduced, axis=0)
cost_df = cost_df.reindex(axis=0, level=0, labels=controllers)

# Savings
savings_df = (1 - cost_df / cost_df.loc['thermo'].values) * 100
savings_df = savings_df.drop(axis=0, level=0, labels=['thermo'])

savings_df_65 = savings_df[65]
savings_df_80 = savings_df[80]

# Constraint data

cons_df_over = df_select.loc[:, IDXmeta('dewh', 1, None, 'mu', 0)].sum(axis=0).mean(
    level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])
cons_df_over = cons_df_over / 4  # to hours

cons_df_over = cons_df_over.drop(labels=drop_reduced, axis=0)
cons_df_over = cons_df_over.reindex(axis=0, level=0, labels=controllers)

cons_df_over_65 = cons_df_over[65]
cons_df_over_80 = cons_df_over[80]

cons_df_under = df_select.loc[:, IDXmeta('dewh', 1, None, 'mu', 1)].sum(axis=0).mean(
    level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])
cons_df_under = cons_df_under / 4  # to hours

cons_df_under = cons_df_under.drop(labels=drop_reduced, axis=0)
cons_df_under = cons_df_under.reindex(axis=0, level=0, labels=controllers)

cons_df_under_65 = cons_df_under[65]
cons_df_under_80 = cons_df_under[80]

## Computation time

# 'time_in_solver', 'time_solve_overall'
time_solve_df = df_select.loc[:, IDXmeta('grid', 1, None, 'time_in_solver')]
time_solve_df = time_solve_df.mean(axis=0)
time_solve_df = time_solve_df.mean(level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])
#
time_solve_df = time_solve_df.drop(labels=drop_reduced, axis=0)
time_solve_df = time_solve_df.reindex(axis=0, level=0, labels=controllers)
time_solve_df = time_solve_df.drop(axis=0, level=0, labels=['thermo'])

######################################
#######  PLOT SAVINGS ##############
######################################
fig, axes = get_fig_axes_A4(2, 2, v_scale=0.7, h_scale=1, sharex='all', sharey='row')
ax0 = axes[0,0]
ax1 = axes[0,1]
ax2 = axes[1,0]
ax3 = axes[1,1]

savings_df_65.plot.bar(ax=ax0)
ax0.grid(linestyle='-.', alpha=0.5)
savings_df_80.plot.bar(ax=ax1)
ax1.grid(linestyle='-.', alpha=0.5)

pv_self_increase_65.plot.bar(ax=ax2)
ax2.grid(linestyle='-.', alpha=0.5)
pv_self_increase_80.plot.bar(ax=ax3)
ax3.grid(linestyle='-.', alpha=0.5)

fig.subplots_adjust(wspace=0.1, top=0.96, bottom=0.2, hspace=0.1)
## Labelling

ax0.legend().remove()
ax1.legend().set_title(tex_s('$N_{p}$'))

ax2.legend().remove()
ax3.legend().set_title(tex_s('$N_{p}$'))

ax0.set_title(tex_s('$T_{h}^{\max} = 65\si{\celsius}$'))
ax1.set_title(tex_s('$T_{h}^{\max} = 80\si{\celsius}$'))

xtick_labels = [
    r'PB-EMPC',
    r'CE-EMPC',
    r'MM-EMPC',
    r'SB-EMPC',
    tex_s(r'$\text{SBR-EMPC}\\(N_{\text{SBR}} = 4)$'),
    tex_s(r'$\text{SBR-EMPC}\\(N_{\text{SBR}} = 6)$'),
    tex_s(r'$\text{SBR-EMPC}\\(N_{\text{SBR}} = 8)$'),
]

ax2.set_xticklabels(xtick_labels)
ax3.set_xticklabels(xtick_labels)
ax2.set_xlabel('Controller', labelpad=5)
ax3.set_xlabel('Controller', labelpad=5)

ax0.set_ylabel(tex_s(r'Cost saving $[\si{\percent}]$'), wrap=True)
ax2.set_ylabel(tex_s(r'Self-consumption increase $[\si{\percent}]$'), wrap=True)

temp_alpha_patch=0.075
for ax in [ax0,ax2]:
    ax.patch.set_facecolor('blue')
    ax.patch.set_alpha(temp_alpha_patch)

for ax in [ax1,ax3]:
    ax.patch.set_facecolor('red')
    ax.patch.set_alpha(temp_alpha_patch)

fig.savefig(FIG_SAVE_PATH + "plot_savings_single_bar.pdf", bbox_inches='tight')


######################################
### PLOT CONS_VIOLATION            ###
######################################

fig2, axes2 = get_fig_axes_A4(2, 2, v_scale=0.7, h_scale=1, sharey='row', sharex='all')

ax2_0 = axes2[0,0]
ax2_1 = axes2[0,1]

ax2_2 = axes2[1,0]
ax2_3 = axes2[1,1]

cons_df_over_65.plot.bar(ax=ax2_0)
ax2_0.grid(linestyle='-.', alpha=0.5)
cons_df_over_80.plot.bar(ax=ax2_1)
ax2_1.grid(linestyle='-.', alpha=0.5)

cons_df_under_65.plot.bar(ax=ax2_2)
ax2_2.grid(linestyle='-.', alpha=0.5)
cons_df_under_80.plot.bar(ax=ax2_3)
ax2_3.grid(linestyle='-.', alpha=0.5)

fig2.subplots_adjust(wspace=0.1, top=0.96, bottom=0.2, hspace=0.1)
## Labelling

ax2_0.legend().remove()
ax2_1.legend().set_title(tex_s('$N_{p}$'))
ax2_2.legend().remove()
ax2_3.legend().set_title(tex_s('$N_{p}$'))

ax2_0.set_title(tex_s('$T_{h}^{\max} = 65\si{\celsius}$'))
ax2_1.set_title(tex_s('$T_{h}^{\max} = 80\si{\celsius}$'))

xtick_labels_cons = xtick_labels + ['TSRB']


ax2_2.set_xticklabels(xtick_labels_cons)
ax2_3.set_xticklabels(xtick_labels_cons)
ax2_2.set_xlabel('Controller', labelpad=5)
ax2_3.set_xlabel('Controller', labelpad=5)

ax2_0.set_ylabel(tex_s(r'Sum Violation over $T_{h}^{\max}\;[\si{\celsius\hour}]$'))
ax2_2.set_ylabel(tex_s(r'Sum Violation below $T_{h}^{\min}\;[\si{\celsius\hour}]$'))

ax2_0.set_yscale('log')
ax2_2.set_yscale('log')

for ax in [ax2_0,ax2_2]:
    ax.patch.set_facecolor('blue')
    ax.patch.set_alpha(temp_alpha_patch)

for ax in [ax2_1,ax2_3]:
    ax.patch.set_facecolor('red')
    ax.patch.set_alpha(temp_alpha_patch)

fig2.savefig(FIG_SAVE_PATH + "plot_single_cons_vio_bar.pdf", bbox_inches='tight')

