import pandas as pd
from pandas import DataFrame
from pandas.tseries.frequencies import to_offset
import numpy as np
from datetime import datetime as DateTime
import inspect
import re
import functools

from utils.mongo_tools import MongoInterface

IDX = pd.IndexSlice

DEWH_RESAMPLE_FUNC_MAP = {'Temp': 'last',
                          'Status': np.mean,
                          'Error': np.max,
                          'is_Nan_Raw': np.mean}

PM_RESAMPLE_FUNC_MAP = {'A_L1': np.mean, 'A_L2': np.mean, 'A_L3': np.mean,
                        'ActivePower_L1': np.mean, 'ActivePower_L2': np.mean, 'ActivePower_L3': np.mean,
                        'ActivePower_SF': np.mean, 'ActivePower_Tot': np.mean,
                        'ApparentPower_L1': np.mean, 'ApparentPower_L2': np.mean, 'ApparentPower_L3': np.mean,
                        'ApparentPower_Tot': np.mean,
                        'ForActiveEnergy_L1': 'last', 'ForActiveEnergy_L2': 'last', 'ForActiveEnergy_L3': 'last',
                        'ForActiveEnergy_Tot': 'last',
                        'GridFreq': np.mean,
                        'ID': 'last',
                        'PF': np.mean,
                        'PF_L1': np.mean, 'PF_L2': np.mean, 'PF_L3': np.mean,
                        'ReactivePower_L1': np.mean, 'ReactivePower_L2': np.mean, 'ReactivePower_L3': np.mean,
                        'ReactivePower_Tot': np.mean,
                        'RevActiveEnergy_L1': 'last', 'RevActiveEnergy_L2': 'last', 'RevActiveEnergy_L3': 'last',
                        'RevActiveEnergy_Tot': 'last',
                        'SN': 'last',
                        'V_L1': np.mean, 'V_L2': np.mean, 'V_L3': np.mean,
                        'is_Nan_Raw': np.mean}


class MicroGridDataFrame(DataFrame):

    def __init__(self, *args, **kwargs):
        super(MicroGridDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return MicroGridDataFrame

    @staticmethod
    def _con(*args, **kwargs):
        return MicroGridDataFrame(*args, **kwargs)

    def stair_plot(self, *args, **kwargs):
        self.plot(*args, **kwargs, drawstyle="steps-post")

    stair_plot.__signature__ = inspect.signature(DataFrame.plot.__call__)

    def align_samples(self, sampling_time='1Min'):
        df = self.groupby(pd.Grouper(freq=sampling_time, closed='right', label='left')).last()

        for dev in df.columns.levels[0]:
            df.loc[:, (dev, 'is_Nan_Raw')] = np.any(df.loc[:, IDX[dev, :]].isna().values, axis=1).astype(int)

        df.sort_index(axis=1, inplace=True)

        return self._con(df)

    def dewh_resample(self, sample_time='15Min', func_map=None):
        df = self.copy()

        df.loc[:, IDX[:, 'Temp']] = df.loc[:, IDX[:, 'Temp']].interpolate()
        df.ffill(axis=0, inplace=True)
        df.bfill(axis=0, inplace=True)

        df:DataFrame.__getitem__()

        func_map = func_map or DEWH_RESAMPLE_FUNC_MAP
        funcs = {key: func_map[key[1]] for key in df.columns.get_values()}

        # df = df.groupby(pd.Grouper(freq=sample_time)).agg(funcs)
        df = df.resample(sample_time, closed='right', label='right').agg(funcs)

        return self._con(df)

    def pm_resample(self, sample_time='15Min', func_map=None):
        df = self.copy()

        df = df._rename_join(
            (df.loc[:, IDX[:, 'is_Nan_Raw']].rolling(to_offset(sample_time).n).mean() >= 1).astype(int),
            name_map = {'is_Nan_Raw':'is_Nan_Rol'}, axis=1
        )

        fields = df.columns.levels[1].tolist()
        energy_fields = list(filter(re.compile("(energy)", re.IGNORECASE).match, fields))

        df = df.interpolate()
        df.ffill(axis=0, inplace=True)
        df.bfill(axis=0, inplace=True)

        # create mask to remove non-increasing energy values
        mask = np.isclose(
            df.loc[:, IDX[:, energy_fields]].apply(np.maximum.accumulate).astype(np.int).diff(), 0.0)

        df.loc[:, IDX[:, energy_fields]] = np.where(mask, np.nan, df.loc[:, IDX[:, energy_fields]])
        df.loc[:, IDX[:, energy_fields]] = df.loc[:, IDX[:, energy_fields]].interpolate()

        func_map = func_map or PM_RESAMPLE_FUNC_MAP
        df = self._groupby_resample(df, sample_time, func_map)

        return self._con(df)

    def _groupby_resample(self, df, sample_time, func_map, *arg, **kwargs):
        funcs = {key: func_map.get(key[1], 'last') for key in df.columns.get_values()}

        col_names = df.columns.namess
        df = df.groupby(pd.Grouper(*arg, freq=sample_time, closed='right', label='right', **kwargs)).agg(funcs)
        df.columns.names = col_names

        return self._con(df)

    def _rename_join(self, other, axis=0, name_map=None, level=None, how='left', sort=True):

        if name_map is not None:
            index = name_map if axis == 0  else None
            columns = name_map if axis == 1 else None
            other = other.rename(index=index, columns=columns, level=level)

        joined = self.join(other, how=how, lsuffix='_self', rsuffix='_other')

        if sort:
            joined.sort_index(inplace=True, axis=axis)

        return self._con(joined)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mi = MongoInterface("site_data", "Kikuyu")

    start_date = DateTime(2018, 6, 10, 13, 0)
    end_time = DateTime(2018, 7, 10, 14, 10)

    raw_data = mi.get_many_dev_raw_dataframe('pm', [10], fields=None,
                                             start_time=start_date, end_time=end_time)

    mg_df = MicroGridDataFrame(raw_data)

    al_df = mg_df.align_samples()

    df = al_df.pm_resample(sample_time='15Min')

    # df[17].ForActiveEnergy_Tot.interpolate().diff().plot()

    df_r = al_df.copy()
    df_r.iloc[:, :] = np.random.randint(100, size=al_df.shape)

    df_r_i: pd.MultiIndex = df_r.columns








    # from bokeh.models import ColumnDataSource
    # from bokeh.plotting import figure, show, output_file
    # from bokeh.models.glyphs import Step
    #
    # df.columns = df.columns.droplevel(0)
    #
    # source = ColumnDataSource(df)
    #
    # p = figure(x_axis_type="datetime", sizing_mode='stretch_both')
    #
    # glyph1 = Step(x='TimeStamp', y='Temp', line_color="#1d91d0", mode="center")
    # p.add_glyph(source, glyph1)
    #
    # output_file("ts.html")
    # show(p)
