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
paths = glob.glob(r'../sim_out/multi_agent/*')
dfs = OrderedStructDict()
for path in paths:
    if path.endswith('sim_out'):
        dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)
print(f"Time to load data: {time.time() - st}")

st = time.time()
## Get meta data
Meta = namedtuple('Meta', ['N_h', 'T_max', 'T_min', 'N_p', 'N_s', 'N_sr'])
dfs_meta = OrderedStructDict()
for name, df in dfs.items():
    s_name = name.split('_')
    N_h = int(s_name[s_name.index('Nh') + 1])
    N_p = int(s_name[s_name.index('Np') + 1])
    N_s = int(s_name[s_name.index('Ns') + 1])
    N_sr = int(s_name[s_name.index('Nsr') + 1])
    T_max = int(s_name[s_name.index('Tmax') + 1])
    T_min = int(s_name[s_name.index('Tmin') + 1])
    dfs_meta[Meta(N_h=N_h, T_max=T_max, T_min=T_min, N_p=N_p, N_s=N_s, N_sr=N_sr)] = df

print(f"Time to meta: {time.time() - st}")


class _IDXmeta(type(IDX)):
    def __call__(self, *args, N_h=None, T_max=None, T_min=None, N_p=None, N_s=None, N_sr=None):
        def _sl(item):
            if item is None:
                return IDX[:]
            else:
                return IDX[item]

        ind = [_sl(N_h), _sl(T_max), _sl(T_min), _sl(N_p), _sl(N_s), _sl(N_sr)]
        for arg in args:
            ind.append(_sl(arg))
        return self[tuple(ind)]


IDXmeta = _IDXmeta()

df_select = pd.concat(list(dfs_meta.values()), keys=dfs_meta.keys(), names=Meta._fields, axis=1)
controllers = ['mpc_pb', 'mpc_ce', 'mpc_minmax', 'mpc_sb_full', 'mpc_sb_reduced', 'thermo']

controller_names = {
    'mpc_pb'        : 'PB-EMPC',
    'mpc_ce'        : 'CE-EMPC',
    'mpc_minmax'    : 'MM-EMPC',
    'mpc_sb_full'   : 'SB-EMPC',
    'mpc_sb_reduced': 'SBR-EMPC',
    'thermo'        : 'TSRB'
}

date_range = IDX['2018-12-10':'2018-12-19']

## Cost data ##
cost_df = df_select.loc[:, IDXmeta('grid', 1, None, 'cost')].sum(axis=0).mean(
    level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])
cost_df = cost_df.reindex(axis=0, level=0, labels=controllers)

# Savings
savings_df = (1 - cost_df / cost_df.loc['thermo'].values) * 100
savings_df = savings_df.drop(axis=0, level=0, labels=['thermo'])

savings_df_65 = savings_df[65]
savings_df_80 = savings_df[80]

# PV self consumption

pv_gen = df_select.loc[:, IDXmeta('pv', 1, None, 'y')].sum(axis=0)
p_exp = df_select.loc[:, IDXmeta('grid', 1, None, 'p_exp')].sum(axis=0)
pv_self = (pv_gen - p_exp.values) / (pv_gen.values)
pv_self = pd.DataFrame(
    pv_self.mean(level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0]))

pv_self = pv_self.reindex(axis=0, level=0, labels=controllers)

pv_self_increase = (pv_self - pv_self.loc['thermo'].values) * 100
pv_self_increase = pv_self_increase.drop(axis=0, level=0, labels=['thermo'])

pv_self_increase_65 = pv_self_increase[65]
pv_self_increase_80 = pv_self_increase[80]

## Power data ##

pv_df = df_select.loc[:, IDXmeta('pv', 1, None, 'y')]
res_df = df_select.loc[:, IDXmeta('resd', 1, None, 'y')]
p_g_df = df_select.loc[:, IDXmeta('grid', 1, None, 'y')]
p_exp_df = df_select.loc[:, IDXmeta('grid', 1, None, 'p_exp')]
p_h_df = p_g_df - (pv_df.values + res_df.values)

## Temperature data ##

temp_df_65 = pd.DataFrame(df_select.loc[date_range, IDXmeta('dewh', None, None, 'x', T_max=65)])
temp_df_65 = temp_df_65.droplevel(list(range(0, 6)), axis=1)
temp_df_65 = temp_df_65.reindex(axis=1, level=2,
                                labels=controllers)

temp_df_80 = pd.DataFrame(df_select.loc[date_range, IDXmeta('dewh', None, None, 'x', T_max=80)])
temp_df_80 = temp_df_80.droplevel(list(range(0, 6)), axis=1)
temp_df_80 = temp_df_80.reindex(axis=1, level=2,
                                labels=controllers)

## Constraint Vio ##

cons_df_over = df_select.loc[:, IDXmeta('dewh', None, None, 'mu', 0)].sum(axis=0).sum(
    level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])
cons_df_over = cons_df_over / 4  # to hours

cons_df_over = cons_df_over.reindex(axis=0, level=0, labels=controllers)

cons_df_over_65 = cons_df_over[65]
cons_df_over_80 = cons_df_over[80]

cons_df_under = df_select.loc[:, IDXmeta('dewh', None, None, 'mu', 1)].sum(axis=0).sum(
    level=['N_sr', 'N_p', 'T_max', 'controller']).unstack([2, 1]).reorder_levels([1, 0])
cons_df_under = cons_df_under / 4  # to hours

cons_df_under = cons_df_under.reindex(axis=0, level=0, labels=controllers)

cons_df_under_65 = cons_df_under[65]
cons_df_under_80 = cons_df_under[80]

## Average solve time per step ##

time_solve_df = df_select.loc[:, IDXmeta('grid', 1, None, 'time_in_solver')].mean()
time_solve_df = time_solve_df.droplevel(
    list(set(time_solve_df.index.names).difference(['N_sr', 'N_p', 'T_max', 'controller']))).unstack(
    [0, 1]).reorder_levels([1, 0])

## All data ##

all_df = pd.concat(
    [cost_df.T, savings_df.T, pv_self.T, pv_self_increase.T, cons_df_over.T, cons_df_under.T, time_solve_df.T],
    keys=['cost', 'savings', 'self', 'self_increase', 'cons_over', 'cons_under', 'time_solve'], axis=0, sort=False)

all_df = all_df.reindex(axis=1, level=0, labels=controllers)
all_df = all_df.sort_index(axis=0, level=1, sort_remaining=False)

####### PLOT 65 ##########

fig, axes = get_fig_axes_A4(6, 1, v_scale=0.77, h_scale=1, sharex='all', sharey='all')
fig.subplots_adjust(top=0.98, bottom=0.02, hspace=0.4)

for ind, controller in enumerate(controllers):
    temp_df_65.loc[:, IDX[:, :, (controller,)]].plot(drawstyle="steps-post", ax=axes[ind], lw=lineWidthNorm)
    axes[ind].legend().remove()
    axes[ind].axhline(y=65, linestyle='--', color='r', lw=lineWidthNorm, label='T_max')
    axes[ind].axhline(y=50, linestyle='--', color='b', lw=lineWidthNorm, label='T_min')

    axes[ind].set_ylabel(tex_s(fr'$x_{{h,i}}\;[\si{{\celsius}}]$'))

    cost = f'{float(cost_df[65].values[ind]):.2f}'
    if ind < 5:
        saving = f'{float(savings_df[65].values[ind]):.2f}'
    else:
        saving = f'-'

    self = f'{float(pv_self[65].values[ind]):.2f}'

    title = tex_s(f'{controller_names[controller]} ('
                  f'$J_{{\\text{{bill}}}}^{{\\text{{cl}}}}$ : R{cost};    '
                  f'Cost saving : {saving}\%;    '
                  f'$\\nu_{{\\text{{self}}}}^{{\\text{{cl}}}}$ : {self})')

    axes[ind].set_title(title, fontsize=12)

legend_elements = [
    Line2D([0], [0], color='r', linestyle='--', lw=lineWidthNorm, label=tex_s(r'${T}_{h,i}^{\text{max}}$')),
    Line2D([0], [0], color='b', lw=lineWidthNorm, linestyle='--', label=tex_s(r'${T}_{h,i}^{\text{min}}$'))]
axes[0].legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.16, 0.5))

axes[-1].set_xlabel('Date', labelpad=-14)

temp_alpha_patch=0.05
for ax in axes:
    ax.patch.set_facecolor('blue')
    ax.patch.set_alpha(temp_alpha_patch)

fig.savefig(FIG_SAVE_PATH + "plot_multi_temp_65.pdf", bbox_inches='tight')



####### PLOT 80 ##########

fig1, axes1 = get_fig_axes_A4(6, 1, v_scale=0.77, h_scale=1, sharex='all', sharey='all')
fig1.subplots_adjust(top=0.98, bottom=0.02, hspace=0.4)

for ind, controller in enumerate(controllers):
    temp_df_80.loc[:, IDX[:, :, (controller,)]].plot(drawstyle="steps-post", ax=axes1[ind], lw=lineWidthNorm)
    axes1[ind].legend().remove()
    axes1[ind].axhline(y=80, linestyle='--', color='r', lw=lineWidthNorm, label='T_max')
    axes1[ind].axhline(y=50, linestyle='--', color='b', lw=lineWidthNorm, label='T_min')

    axes1[ind].set_ylabel(tex_s(fr'$x_{{h,i}}\;[\si{{\celsius}}]$'))

    cost = f'{float(cost_df[80].values[ind]):.2f}'
    if ind < 5:
        saving = f'{float(savings_df[80].values[ind]):.2f}'
    else:
        saving = f'-'

    self = f'{float(pv_self[80].values[ind]):.2f}'

    title = tex_s(f'{controller_names[controller]} ('
                  f'$J_{{\\text{{bill}}}}^{{\\text{{cl}}}}$ : R{cost};    '
                  f'Cost saving : {saving}\%;    '
                  f'$\\nu_{{\\text{{self}}}}^{{\\text{{cl}}}}$ : {self})')

    axes1[ind].set_title(title, fontsize=12)

legend_elements = [
    Line2D([0], [0], color='r', linestyle='--', lw=lineWidthNorm, label=tex_s(r'${T}_{h,i}^{\text{max}}$')),
    Line2D([0], [0], color='b', lw=lineWidthNorm, linestyle='--', label=tex_s(r'${T}_{h,i}^{\text{min}}$'))]
axes1[0].legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.16, 0.5))

axes1[-1].set_xlabel('Date', labelpad=-14)

for ax in axes1:
    ax.patch.set_facecolor('red')
    ax.patch.set_alpha(temp_alpha_patch)

fig1.savefig(FIG_SAVE_PATH + "plot_multi_temp_80.pdf", bbox_inches='tight')
