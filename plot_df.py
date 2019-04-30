import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import os
import sys
import glob

from structdict import StructDict, OrderedStructDict
from models.parameters import dewh_param_struct

IDX = pd.IndexSlice

paths = glob.glob(r'./experiments/sims/*')
dfs = OrderedStructDict()
for path in paths:
    dfs[os.path.basename(path).split('.')[0]] = pd.read_pickle(path)

for df in dfs.values():
    df.index = pd.date_range(start=pd.datetime(2018, 12, 1), periods=len(df), freq='15Min')

df_data = OrderedStructDict()
for name, df in dfs.items():
    max_mu = df.dewh.loc[:, IDX[:, :, 'mu']].sum(axis=1, level=1).sum()
    cost = df.grid[1].loc[:, IDX[:, 'cost']].sum()
    saving = cost / cost.max()
    export = df.grid[1].loc[:, IDX[:, 'p_exp']].sum()/df.grid[1].loc[:, IDX[:, 'p_exp']].sum().max()
    df_data[name] = OrderedStructDict(cost=cost, saving=saving, max_mu=max_mu, export=export)
print(df_data)

dfs_list = list(dfs.values())
dfs_data_list = list(df_data.values())

df = dfs_list[0]
df_data = dfs_data_list[0]

def plot_temps(df):
    T_max = dewh_param_struct.T_h_max
    T_min = dewh_param_struct.T_h_min

    df: pd.DataFrame = df.copy()
    df.columns = df.columns.reorder_levels([2, 0, 3, 1, 4])

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    axes = []
    for ind, cname in enumerate(['mpc_pb', 'mpc_ce', 'mpc_scenario', 'thermo']):
        if axes:
            ax: plt.Axes = fig.add_subplot(4, 1, ind + 1, sharex=axes[0])
        else:
            ax: plt.Axes = fig.add_subplot(4, 1, ind + 1)
        axes.append(ax)
        df_p = df[cname].dewh.x.copy()
        df_p.columns = df_p.columns.droplevel(level=1)
        df_p.plot(drawstyle="steps-post", ax=ax)
        ax.set_title(f'Temperature Profile - {cname} (Total Cost = {int(df_data.cost[cname].values)}, '
                     f'%saving = {float(df_data.saving[cname].values):.2f}, '
                     f'sum_cons_violation = {float(df_data.max_mu[cname]):.2f})')
        ax.axhline(y=T_max, linestyle='--', color='r', label='T_max')
        ax.axhline(y=T_min, linestyle='--', color='b', label='T_min')
        ax.set_ylabel('Temp\n(deg c)', wrap=True)
        ax.legend().remove()

    return fig

def plot_omega(df):
    df: pd.DataFrame = df.copy()
    df.columns = df.columns.reorder_levels([2, 0, 3, 1, 4])

    fig: plt.Figure = plt.figure(figsize=(19.2, 9.28))

    axes = []
    for ind, dev in enumerate(['grid','grid', 'pv', 'resd', 'dewh']):
        if axes:
            ax: plt.Axes = fig.add_subplot(5, 1, ind + 1, sharex=axes[0])
        else:
            ax: plt.Axes = fig.add_subplot(5, 1, ind + 1)
        axes.append(ax)

        if dev == 'dewh':
            df_p = df.loc[:, IDX[('thermo',), ('dewh',), 'x', :]].mean(axis=1, level=0).copy()
        elif ind==1:
            df_p = df.loc[:, IDX[('thermo',), (dev,), 'p_exp', :]].copy()
            df_p.columns = df_p.columns.droplevel(level=2)
        else:
            df_p = df.loc[:,IDX[('thermo',),(dev,),'y',:]].copy()
            df_p.columns = df_p.columns.droplevel(level=2)

        df_p.plot(drawstyle="steps-post", ax=ax)

        if ind==1:
            ax.set_title(f'Power profile - {dev} grid power export.')
        else:
            ax.set_title(f'Power profile - {dev}')

        ax.set_ylabel('Power\n(W)', wrap=True)
        plt.legend(loc=1)

plot_temps(df)
plot_omega(df)