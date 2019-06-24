from examples.residential_mg_with_pv_and_dewhs.plotting.plotting_helper import *
import os
import glob
from structdict import OrderedStructDict
from collections import namedtuple
import pickle
import time
import itertools

plot_with_tex(False)
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
    'thermo'        : 'thermo'
}

date_range = IDX['2018-12-10':'2018-12-20']
## Power data ##

power_df = pd.DataFrame(df_select.loc[date_range, IDXmeta(('grid','pv','resd'), None, None, None, T_max=65)])
power_df = power_df.droplevel(list(range(0, 6)), axis=1)



## Temperature data ##



temp_df_65 = pd.DataFrame(df_select.loc[date_range, IDXmeta('dewh', None, None, 'x', T_max=65)])
temp_df_65 = temp_df_65.droplevel(list(range(0, 6)), axis=1)
temp_df_65 = temp_df_65.reindex(axis=1, level=2,
                                labels=controllers)


temp_df_80 = pd.DataFrame(df_select.loc[date_range, IDXmeta('dewh', None, None, 'x', T_max=80)])
temp_df_80 = temp_df_80.droplevel(list(range(0, 6)), axis=1)
temp_df_80 = temp_df_80.reindex(axis=1, level=2,
                                labels=controllers)

####### PLOT 65 ##########

fig, axes = get_fig_axes_A4(6, 1, v_scale=0.8, h_scale=1, sharex='all', sharey='all')


for ind, controller in enumerate(controllers):
    temp_df_65.loc[:, IDX[:, :, (controller,)]].plot(drawstyle="steps-post", ax=axes[ind], lw=lineWidthNorm)
    axes[ind].legend().remove()
    axes[ind].axhline(y=65, linestyle='--', color='r', lw=lineWidthNorm, label='T_max')
    axes[ind].axhline(y=50, linestyle='--', color='b', lw=lineWidthNorm, label='T_min')

    axes[ind].set_ylabel(tex_s(fr'$x_{{h}}\;[\si{{\celsius}}]$'))
    axes[ind].set_title(controller_names[controller])

fig.subplots_adjust(top=0.96, bottom=0.04, hspace=0.25)

fig.savefig(FIG_SAVE_PATH + "plot_multi_temp_65.pdf", bbox_inches='tight')



####### PLOT 80 ##########

fig1, axes1 = get_fig_axes_A4(6, 1, v_scale=0.8, h_scale=1, sharex='all', sharey='all')


for ind, controller in enumerate(controllers):
    temp_df_80.loc[:, IDX[:, :, (controller,)]].plot(drawstyle="steps-post", ax=axes1[ind], lw=lineWidthNorm)
    axes1[ind].legend().remove()
    axes1[ind].axhline(y=80, linestyle='--', color='r', lw=lineWidthNorm, label='T_max')
    axes1[ind].axhline(y=50, linestyle='--', color='b', lw=lineWidthNorm, label='T_min')

    axes1[ind].set_ylabel(tex_s(fr'$x_{{h}}\;[\si{{\celsius}}]$'))
    axes1[ind].set_title(controller_names[controller])

fig1.subplots_adjust(top=0.96, bottom=0.04, hspace=0.25)

fig1.savefig(FIG_SAVE_PATH + "plot_multi_temp_80.pdf", bbox_inches='tight')
