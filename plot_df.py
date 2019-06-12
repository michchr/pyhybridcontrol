import pandas as pd
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker
import os
import sys
import glob

from models.micro_grid_agents import DewhAgentMpc

from structdict import StructDict, OrderedStructDict
from models.parameters import dewh_param_struct

IDX = pd.IndexSlice

paths = glob.glob(r'./experiments/sims/*')
dfs = OrderedStructDict()
for path in paths:
    dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)

for df in dfs.values():
    df.index = pd.date_range(start=pd.datetime(2018, 12, 1), periods=len(df), freq='15Min')


def extract_df_data(df):
    mu_over = df.dewh.loc[:, IDX[:, :, 'mu', 0]].sum(axis=1, level=1).sum()
    mu_under = df.dewh.loc[:, IDX[:, :, 'mu', 1]].sum(axis=1, level=1).sum()
    cost = df.grid[1].loc[:, IDX[:, 'cost']].sum()
    saving = cost / cost.max()
    export = df.grid[1].loc[:, IDX[:, 'p_exp']].sum() / df.grid[1].loc[:, IDX[:, 'p_exp']].sum().max()
    return OrderedStructDict(cost=cost, saving=saving, mu_over=mu_over, mu_under=mu_under,
                                      export=export)

df_data = OrderedStructDict()
for name, df in dfs.items():
    df_data[name] =  extract_df_data(df)
print(df_data)

dfs_list = list(dfs.values())
dfs_data_list = list(df_data.values())

df = dfs_list[2]
df_data = dfs_data_list[2]

def use_tex(string, use=True):
    if use:
        ret = string.replace('$','')
        ret = ret.replace("\\", '')
    else:
        ret = string

    return ret


def plot_with_tex(enable=True):
    if enable:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        matplotlib.verbose.level = 'debug-annoying'
    else:
        plt.rc('text', usetex=False)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        matplotlib.verbose.level = 'helpful'


def plot_temps(df, controllers = ('mpc_pb', 'mpc_ce', 'mpc_scenario', 'thermo'), usetex=False):
    T_max = dewh_param_struct.T_h_max
    T_min = dewh_param_struct.T_h_min
    df_data = extract_df_data(df)

    df: pd.DataFrame = df.copy()
    df.columns = df.columns.reorder_levels([2, 0, 3, 1, 4])

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    titleFontSize = 19
    normFontSize = 12
    normLineWidth = 2

    plt.rc('axes', titlesize=titleFontSize)
    plt.rc('font', size=normFontSize)
    plt.rc('lines', lw=normLineWidth)

    axes = []
    for ind, cname in enumerate(controllers):
        if axes:
            ax: plt.Axes = fig.add_subplot(len(controllers), 1, ind + 1, sharex=axes[0])
        else:
            ax: plt.Axes = fig.add_subplot(len(controllers), 1, ind + 1)
        axes.append(ax)

        df_p = df[cname].dewh.x.copy()
        df_p.columns = df_p.columns.droplevel(level=1)
        df_p.plot(drawstyle="steps-post", ax=ax, lw=1)
        if cname == 'mpc_pb':
            title_name = r'Performance Bound EMPC (Assumes Perfect Predictions)'
            legend_elements = [
                Line2D([0], [0], color='r', linestyle='--', lw=normLineWidth, label=r'${T}_{h}^{\text{max}}$'),
                Line2D([0], [0], color='b', lw=normLineWidth, linestyle='--', label=r'${T}_{h}^{\text{min}}$')]
            ax.legend(handles=legend_elements, loc=4)
        elif cname == 'mpc_ce':
            title_name = r'Certainty Equivalent EMPC'
            ax.legend().remove()
        elif cname == 'mpc_scenario':
            title_name = r'Scenario Based EMPC'
            ax.legend().remove()
        elif cname == 'thermo':
            title_name = r'Thermostatic Rule Based'
            # ax.set_xlabel('Time (days)')
            ax.legend().remove()
        else:
            title_name = cname.replace('_','-')
            ax.legend().remove()

        title = (f'\\textbf{{{title_name}}}\n '
                 f'Total Energy Bill: \${float(df_data.cost[cname].values)/100:.2f},   '
                 f'Bill saving: {100 - float(df_data.saving[cname].values * 100):.1f}\%,   '
                 f'Constraint violation: (over={int(df_data.mu_over[cname])}, '
                 f'under={int(df_data.mu_under[cname])}) '
                 f'$\\Delta t\,{{}}^{{\\circ}}\\text{{c}}$')

        print(title)
        ax.set_title(title, fontsize=titleFontSize)

        ax.axhline(y=T_max, linestyle='--', color='r', label='T_max')
        ax.axhline(y=T_min, linestyle='--', color='b', label='T_min')
        ax.set_ylabel('Temp\n(deg c)', wrap=True)
        plt.subplots_adjust(hspace=0.6, top=0.93)

    return fig


def plot_omega(df):
    df: pd.DataFrame = df.copy()
    df.columns = df.columns.reorder_levels([2, 0, 3, 1, 4])

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    axes = []
    for ind, dev in enumerate(['grid', 'grid', 'pv', 'resd', 'dewh']):
        if axes:
            ax: plt.Axes = fig.add_subplot(5, 1, ind + 1, sharex=axes[0])
        else:
            ax: plt.Axes = fig.add_subplot(5, 1, ind + 1)
        axes.append(ax)

        if dev == 'dewh':
            df_p = df.loc[:, IDX[('thermo',), ('dewh',), 'x', :]].mean(axis=1, level=0).copy()
        elif ind == 1:
            df_p = df.loc[:, IDX[('thermo',), (dev,), 'p_exp', :]].copy()
            df_p.columns = df_p.columns.droplevel(level=2)
        else:
            df_p = df.loc[:, IDX[('thermo',), (dev,), 'y', :]].copy()
            df_p.columns = df_p.columns.droplevel(level=2)

        df_p.plot(drawstyle="steps-post", ax=ax)

        if ind == 1:
            ax.set_title(f'Power profile - {dev} grid power export.')
        else:
            ax.set_title(f'Power profile - {dev}')

        ax.set_ylabel('Power\n(W)', wrap=True)
        plt.legend(loc=1)

def plot_omega2(df):
    df: pd.DataFrame = df.copy()
    df.columns = df.columns.reorder_levels([2, 0, 3, 1, 4])

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    axes = []
    for ind, dev in enumerate(['grid', 'grid', 'pv', 'resd', 'dewh']):
        if axes:
            ax: plt.Axes = fig.add_subplot(5, 1, ind + 1, sharex=axes[0])
        else:
            ax: plt.Axes = fig.add_subplot(5, 1, ind + 1)
        axes.append(ax)

        if dev == 'dewh':
            df_p = df.loc[:, IDX[:, ('dewh',), 'u', :]].sum(axis=1, level=0).copy()*dewh_param_struct.P_h_Nom
        elif ind == 1:
            df_p = df.loc[:, IDX[('thermo',), (dev,), 'p_exp', :]].copy()
            df_p.columns = df_p.columns.droplevel(level=2)
        else:
            df_p = df.loc[:, IDX[('thermo',), (dev,), 'y', :]].copy()
            df_p.columns = df_p.columns.droplevel(level=2)

        ax.legend().remove()
        df_p.plot(drawstyle="steps-post", ax=ax)


        if ind == 1:
            ax.set_title(f'Power profile - {dev} grid power export.')
        else:
            ax.set_title(f'Power profile - {dev}')

        ax.set_ylabel('Power\n(W)', wrap=True)
        plt.legend(loc=1)


def plot_pv_resd(df):
    df: pd.DataFrame = df.copy()
    df.columns = df.columns.reorder_levels([2, 0, 3, 1, 4])

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    titleFontSize = 25
    normFontSize = 16
    normLineWidth = 2

    plt.rc('axes', titlesize=titleFontSize)
    plt.rc('font', size=normFontSize)
    plt.rc('lines', lw=normLineWidth)


    axes = []
    df_p = 0
    for ind, dev in enumerate(['pv', 'resd']):
        if axes:
            ax: plt.Axes = fig.add_subplot(2, 1, ind + 1, sharex=axes[0])
        else:
            ax: plt.Axes = fig.add_subplot(2, 1, ind + 1)
        axes.append(ax)

        if dev == 'pv':
            df_p = -1 * (df.loc[:, IDX['thermo', (dev,), 'y', :]].copy()).droplevel(level=[0, 2], axis=1)
            df_p.columns = df_p.columns.droplevel(level=2)
            df_p.plot(color='g', drawstyle="steps-post", ax=ax)
            ax.set_title(r'PV Generation Power Profile: $P_{\text{pv}}$')
            ax.set_ylabel('Power Generation\n(W)', wrap=True)
        elif dev == 'resd':
            df_p = (df.loc[:, IDX['thermo', (dev,), 'y', :]].copy()).droplevel(level=[0, 2], axis=1)
            df_p.columns = df_p.columns.droplevel(level=2)
            df_p.plot(color='r', drawstyle="steps-post", ax=ax)
            ax.set_ylabel('Power Demand\n(W)', wrap=True)
            ax.set_title(r'Residual Power Demand Profile: $P_{r}$')
        ax.set_xlabel('Time', labelpad=-20)
        plt.legend(loc=1)
    return df_p


# omega_profile = pd.read_pickle(
#     os.path.realpath(r'./experiments/data/dewh_omega_profile_df.pickle')) / dewh_param_struct.ts
#
# dewh = DewhAgentMpc()
# dewh.set_omega_scenarios(omega_scenarios_profile=omega_profile)


def plot_eg_water_demand_scenarios(num_scenarios=10):
    df = dewh.omega_scenarios.loc[:, :num_scenarios].droplevel(level=1) * 900
    df_act = dewh.omega_scenarios.loc[:, num_scenarios:num_scenarios].droplevel(level=1) * 900

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    titleFontSize = 25
    normFontSize = 16
    normLineWidth = 2

    plt.rc('axes', titlesize=titleFontSize)
    plt.rc('font', size=normFontSize)
    plt.rc('lines', lw=normLineWidth)

    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    df_p = df
    df_p.plot(drawstyle="steps-post", ax=ax, alpha=0.3)
    df_act.plot(drawstyle="steps-post", ax=ax, linewidth=3, color='b')
    ax.set_title(f'Hot Water Demand Scenario Realizations ($N_{{s}}={num_scenarios}$)')
    ax.set_ylabel('Hot Water Draw\n(Litres)', wrap=True)
    ax.set_xlabel('Time of day', wrap=True)
    ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=60 * 60 * 1e9 * 2, offset=0))

    legend_elements = [Line2D([0], [0], color='b', lw=3, label=r'Example actual demand ($\tilde{\omega}_{h}^{k}$)'),
                       Line2D([0], [0], color='grey', lw=normLineWidth,
                              label=r'Demand scenarios ($\tilde{\omega}_{h}^{k,(s)}$)', alpha=0.3)]

    ax.legend(handles=legend_elements, loc=0)

    return ax

plot_with_tex(True)
# plot_temps(df)
# plot_omega(df)
# plot_omega2(df)
# df_p = plot_pv_resd(df)
# ax = plot_eg_water_demand_scenarios(20)
