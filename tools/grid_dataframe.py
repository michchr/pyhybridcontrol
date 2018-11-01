__all__ = ['MicroGridDataFrame', 'MicroGridSeries', 'IDX']

import pandas as pd
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from pandas.tseries.frequencies import to_offset
import numpy as np
from datetime import datetime as DateTime
import inspect
import re

IDX = pd.IndexSlice
pd.set_option('mode.chained_assignment', 'raise')

_VALID_DEVICE_TYPES = ['dewh', 'pm']

_DEWH_RESAMPLE_FUNC_MAP = {'Temp': 'last',
                           'Status': np.mean,
                           'Error': np.max,
                           'is_Nan_Raw': np.mean,
                           'is_Nan_Rol': 'last',
                           'Demand': np.mean}

_DEWH_FILLNA_FUNC_MAP = {'Status': 'ffill',
                         'Error': 'ffill',
                         'is_Nan_Raw': 'ffill'}

_PM_FILLNA_FUNC_MAP = {'ID': 'ffill',
                       'SN': 'ffill',
                       'is_Nan_Raw': 'ffill'}

_PM_RESAMPLE_FUNC_MAP = {'A_L1': np.mean, 'A_L2': np.mean, 'A_L3': np.mean,
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
    # normal properties
    _metadata = DataFrame._metadata + ['_device_type', 'is_aligned']

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, device_type=None):

        super(MicroGridDataFrame, self).__init__(data=data, index=index, columns=columns, dtype=dtype,
                                                 copy=copy)

        if not hasattr(self, 'is_aligned'):
            self.is_aligned = False
        self._device_type = device_type

    @property
    def device_type(self):
        return self._device_type

    @device_type.setter
    def device_type(self, device_type):
        device_type = device_type or self._device_type
        if device_type not in _VALID_DEVICE_TYPES:
            raise ValueError("Invalid device type: '{}'".format(device_type))
        else:
            self._device_type = device_type

    @property
    def values_2d(self):
        return np.atleast_2d(self.values)

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return MicroGridDataFrame(*args, **kwargs).__finalize__(self)

        return _c

    @property
    def _construct_sliced(self):
        def _c(*args, **kwargs):
            return MicroGridSeries(*args, **kwargs).__finalize__(self)

        return _c

    @property
    def _constructor_override(self):
        def _c(*args, **kwargs):
            data = args[0] if len(args) else kwargs.get('data')
            if isinstance(data, NDFrame):
                return self._constructor(*args, **kwargs).__finalize__(data)
            else:
                return self._constructor

        return _c

    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, NDFrame):
            for name in self._metadata:
                if hasattr(other, name):
                    object.__setattr__(self, name, getattr(other, name, None))
        return self

    def __getattr__(self, item):
        attribute = super(MicroGridDataFrame, self).__getattr__(item)
        if isinstance(attribute, Series):
            return self._construct_sliced(attribute)
        else:
            return attribute

    def __getitem__(self, key):
        item = super(MicroGridDataFrame, self).__getitem__(key)
        if isinstance(item, Series):
            return self._construct_sliced(item)
        else:
            return item

    def _dir_additions(self):
        additions = set(self._metadata)
        return super(MicroGridDataFrame, self)._dir_additions().union(additions)

    def stair_plot(self, *args, **kwargs):
        self.plot(*args, **kwargs, drawstyle="steps-post")

    stair_plot.__signature__ = inspect.signature(DataFrame.plot.__call__)

    def align_samples(self, sampling_time='1Min'):
        grouped = self.groupby(pd.Grouper(freq=sampling_time, closed='right', label='left')).last()

        for dev in grouped.columns.levels[0]:
            grouped.loc[:, (dev, 'is_Nan_Raw')] = np.any(grouped.loc[:, IDX[dev, :]].isna().values, axis=1).astype(int)

        grouped.sort_index(axis=1, inplace=True)
        grouped.is_aligned = True

        return self._constructor_override(grouped)

    def resample_device(self, sample_time, func_map=None, device_type=None):
        self.device_type = device_type

        if not self.is_aligned:
            new_df = self.align_samples()
        else:
            new_df = self.copy()

        try:
            new_df = new_df.drop(labels='is_Nan_Rol', axis=1, level=1)
        except KeyError:
            pass

        try:
            new_df = new_df._rename_join(
                (new_df.loc[:, IDX[:, 'is_Nan_Raw']].rolling(to_offset(sample_time).n).mean() >= 1).astype(int),
                name_map={'is_Nan_Raw': 'is_Nan_Rol'}, axis=1
            )
        except KeyError:
            pass

        new_df = new_df._dev_fillna_default()

        if self.device_type == 'dewh':
            func_map = func_map or _DEWH_RESAMPLE_FUNC_MAP
        elif self.device_type == 'pm':
            energy_fields = new_df.get_device_fields(regex_filter="Energy")
            new_df = new_df._non_increasing_to_nan(fields=energy_fields)
            new_df.loc[:, IDX[:, energy_fields]] = new_df.loc[:, IDX[:, energy_fields]].interpolate(
                limit_direction='both')
            func_map = func_map or _PM_RESAMPLE_FUNC_MAP

        new_df = new_df._groupby_resample(sample_time, func_map)

        return self._constructor_override(new_df)

    def compute_power_from_energy(self, fields=None, find_filter=None, replace_text=None):

        find_filter = find_filter or "Energy"
        replace_text = replace_text or "Power"
        fields = fields or self.get_device_fields(regex_filter=find_filter)

        rename_map = {field: re.sub(find_filter, replace_text, field) + "_Calc" for field in fields}

        new_df = self.copy()

        to_w_factor = 1
        to_hour_factor = 3600 / (new_df.index.freq.nanos / 10 ** 9)

        power = new_df.loc[:, IDX[:, fields]].diff() * to_hour_factor * to_w_factor
        new_df = new_df._rename_join(power, name_map=rename_map, axis=1)
        return self._constructor_override(new_df)

    def get_device_fields(self, regex_filter=None, regex_flags=re.IGNORECASE):
        fields = self.columns.levels[1].tolist()
        if regex_filter is not None:
            reg = re.compile(regex_filter, regex_flags)
            return list(filter(reg.search, fields))
        else:
            return fields

    def _dev_fillna_default(self, func_map=None, device_type=None):
        self.device_type = device_type

        if self.device_type == 'dewh':
            func_map = func_map or _DEWH_FILLNA_FUNC_MAP
        elif self.device_type == 'pm':
            func_map = func_map or _PM_FILLNA_FUNC_MAP

        new_df = self.copy()
        fields = new_df.get_device_fields()
        for field in fields:
            func = func_map.get(field)
            if func is not None:
                new_df.loc[:, IDX[:, field]] = self.loc[:, IDX[:, field]].fillna(method=func)

        new_df.interpolate(limit_direction='both', inplace=True)

        return self._constructor_override(new_df)

    def _groupby_resample(self, sample_time, func_map, *arg, **kwargs):
        funcs = {}
        for key in self.columns.get_values():
            try:
                funcs[key] = func_map[key[1]]
            except KeyError:
                print("Warning no function map for key:'{}', using default: 'last'".format(key))
                funcs[key] = func_map.get(key[1], 'last')

        col_names = self.columns.names
        grouped = self.groupby(pd.Grouper(*arg, freq=sample_time, closed='right', label='right', **kwargs)).agg(funcs)
        grouped.columns.names = col_names

        return self._constructor_override(grouped)

    def _rename_join(self, other, axis=0, name_map=None, level=None, how='left', on=None, sort=True):

        if name_map is not None:
            index = name_map if axis == 0 else None
            columns = name_map if axis == 1 else None
            other = other.rename(index=index, columns=columns, level=level)

        joined = self.join(other, how=how, on=None, lsuffix='_self', rsuffix='_other')

        if sort:
            joined.sort_index(inplace=True, axis=axis)

        return self._constructor_override(joined)

    def _non_increasing_to_nan(self, fields=None):
        # create mask to remove non-increasing values
        if fields is not None:
            mask = np.isclose(
                self.loc[:, IDX[:, fields]].apply(np.maximum.accumulate).astype(np.int).diff(), 0.0)
            new_df = self.copy()
            new_df.loc[:, IDX[:, fields]] = np.where(mask, np.nan, self.loc[:, IDX[:, fields]])
        else:
            new_df = self

        return self._constructor_override(new_df)


class MicroGridSeries(Series):
    _metadata = Series._metadata + MicroGridDataFrame._metadata

    def __init__(self, data=None, index=None, dtype=None, name=None,
                 copy=False, fastpath=False, device_type=None):

        super(MicroGridSeries, self).__init__(data=data, index=index, dtype=dtype, name=name, copy=copy,
                                              fastpath=fastpath)
        self._device_type = device_type

    def stair_plot(self, *args, **kwargs):
        self.plot(*args, **kwargs, drawstyle="steps-post")

    stair_plot.__signature__ = inspect.signature(Series.plot.__call__)

    @property
    def device_type(self):
        return self._device_type

    @device_type.setter
    def device_type(self, device_type):
        device_type = device_type or self._device_type
        if device_type not in _VALID_DEVICE_TYPES:
            raise ValueError("Invalid device type: '{}'".format(device_type))
        else:
            self._device_type = device_type

    @property
    def values_2d(self):
        return np.atleast_2d(self.values).transpose()

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return MicroGridSeries(*args, **kwargs).__finalize__(self)

        return _c

    @property
    def _constructor_expanddim(self):
        def _c(*args, **kwargs):
            return MicroGridDataFrame(*args, **kwargs).__finalize__(self)

        return _c

    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, NDFrame):
            for name in self._metadata:
                if hasattr(other, name):
                    object.__setattr__(self, name, getattr(other, name, None))
        return self


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tools.mongo_interface import MongoInterface

    mi = MongoInterface("site_data", "Kikuyu")

    start_datetime = DateTime(2018, 6, 27, 13, 0)
    end_datetime = DateTime(2018, 10, 30, 14, 10)

    raw_data = mi.get_many_dev_raw_dataframe('pm', [0], fields=None, start_datetime=start_datetime,
                                             end_datetime=end_datetime)

    al_df = raw_data.align_samples()
    #
    df = al_df.resample_device(sample_time='15Min')

    df = df.compute_power_from_energy()

    # for i in range(1,2):
    #     df.loc[:, IDX[i, ['Temp', 'Status', 'is_Nan_Rol']]].stair_plot(subplots=True)
    #     figManager = plt.get_current_fig_manager()
    #     figManager.window.showMaximized()
    #     plt.show()

    for i in range(0, 1):
        df.loc[:, IDX[i, ['ForActivePower_Tot_Calc']]].stair_plot(subplots=True)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    #
    # # Verify Power
    # df.loc[:, (0, 'A_L2')] = (df.loc[:, (0, 'A_L1')] + df.loc[:, (0, 'A_L3')]) / 2.0
    # df.loc[:, (0, 'Ave_Amps')] = (df.loc[:, (0, 'A_L1')] + df.loc[:, (0, 'A_L2')] + df.loc[:, (0, 'A_L3')]) / 3.0
    # df.loc[:, (0, 'Power_Calc')] = np.sqrt(3) * df.loc[:, (0, 'Ave_Amps')] * 11000
    # df.loc[:, IDX[:, ['ActivePower_Tot']]] *= -1
    # ax1 = plt.subplot(211)
    # plt.title('Kikuyu Power Analysis using Wattnode: PM0')
    # plt.ylabel('Power (W)')
    # df.loc[:, IDX[:, ['ForActivePower_Tot_Calc', 'Power_Calc', 'ActivePower_Tot']]].stair_plot(ax=ax1)
    # labels=['Measured power taken from Kikuyu: Wattnode PM0',
    #         'Power computed using ForActiveEnergyMeasurement and computing rate of change',
    #         'Power computed from average of current measurements and assumption that voltage is fixed at 11kV'
    # ]
    # ax1.legend(labels)
    # ax2 = plt.subplot(212, sharex=ax1)
    # plt.ylabel('Voltage (V)')
    # plt.title('Voltage Measurements: PM0')
    # df.loc[:, IDX[:, ['V_L1', 'V_L2', 'V_L3']]].stair_plot(ax=ax2)
    #
    # plt.show()
