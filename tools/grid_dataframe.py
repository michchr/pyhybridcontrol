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

_DEWH_RESAMPLE_FUNC_MAP = {'Temp'      : 'last',
                           'Status'    : np.mean,
                           'Error'     : np.max,
                           'is_Nan_Raw': np.mean,
                           'is_Nan_Rol': 'last',
                           'Demand'    : np.mean}

_DEWH_FILLNA_FUNC_MAP = {'Status'    : 'ffill',
                         'Error'     : 'ffill',
                         'is_Nan_Raw': 'ffill'}

_PM_FILLNA_FUNC_MAP = {'ID'                 : 'ffill',
                       'SN'                 : 'ffill',
                       'is_Nan_Raw'         : 'ffill',
                       'ForActiveEnergy_L1' : 'ffill', 'ForActiveEnergy_L2': 'ffill', 'ForActiveEnergy_L3': 'ffill',
                       'ForActiveEnergy_Tot': 'ffill',
                       'RevActiveEnergy_L1' : 'ffill', 'RevActiveEnergy_L2': 'ffill', 'RevActiveEnergy_L3': 'ffill',
                       'RevActiveEnergy_Tot': 'ffill',
                       }

_PM_RESAMPLE_FUNC_MAP = {'A_L1'               : np.mean, 'A_L2': np.mean, 'A_L3': np.mean,
                         'ActivePower_L1'     : np.mean, 'ActivePower_L2': np.mean, 'ActivePower_L3': np.mean,
                         'ActivePower_SF'     : np.mean, 'ActivePower_Tot': np.mean,
                         'ApparentPower_L1'   : np.mean, 'ApparentPower_L2': np.mean, 'ApparentPower_L3': np.mean,
                         'ApparentPower_Tot'  : np.mean,
                         'ForActiveEnergy_L1' : 'last', 'ForActiveEnergy_L2': 'last', 'ForActiveEnergy_L3': 'last',
                         'ForActiveEnergy_Tot': 'last',
                         'GridFreq'           : np.mean,
                         'ID'                 : 'last',
                         'PF'                 : np.mean,
                         'PF_L1'              : np.mean, 'PF_L2': np.mean, 'PF_L3': np.mean,
                         'ReactivePower_L1'   : np.mean, 'ReactivePower_L2': np.mean, 'ReactivePower_L3': np.mean,
                         'ReactivePower_Tot'  : np.mean,
                         'RevActiveEnergy_L1' : 'last', 'RevActiveEnergy_L2': 'last', 'RevActiveEnergy_L3': 'last',
                         'RevActiveEnergy_Tot': 'last',
                         'SN'                 : 'last',
                         'V_L1'               : np.mean, 'V_L2': np.mean, 'V_L3': np.mean,
                         'is_Nan_Raw'         : np.mean}


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

        for dev in grouped.columns.get_level_values(0):
            grouped.loc[:, (dev, 'is_Nan_Raw')] = np.any(grouped.loc[:, IDX[dev, :]].isna().values, axis=1).astype(int)

        grouped.sort_index(axis=1, inplace=True)
        grouped.is_aligned = True

        return self._constructor_override(grouped)

    def resample_device(self, sample_time, func_map=None, device_type=None):
        if device_type is not None:
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
            energy_df = new_df.loc[:, IDX[:, energy_fields]].copy()
            energy_df[np.isclose(energy_df, 0.0)] = np.NaN
            new_df.loc[:, IDX[:, energy_fields]] = energy_df
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

    def _unwrap(self, discont, axis=0, fields=None):
        if fields is not None:
            diff = self.loc[:, IDX[:, fields]].ffill().diff(axis=axis).values
            correct = diff * -1
            correct[np.abs(diff) < discont] = 0
            new_df = self.copy()
            new_df.loc[:, IDX[:, fields]] = new_df.loc[:, IDX[:, fields]] + np.nancumsum(correct, axis=axis)
        else:
            new_df = self

        return self._constructor_override(new_df)

    def _non_increasing_to_nan(self, fields=None):
        # create mask to remove non-increasing values
        if fields is not None:
            unwrapped = self._unwrap(discont=1e7, fields=fields)
            mask = np.isclose((unwrapped.loc[:, IDX[:, fields]].apply(np.fmax.accumulate)).diff(), 0.0)
            new_df = unwrapped
            new_df.loc[:, IDX[:, fields]] = np.where(mask, np.nan, new_df.loc[:, IDX[:, fields]].values)
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
    import os

    from tools.mongo_interface import MongoInterface

    mi = MongoInterface("site_data", "Kikuyu")

    start_datetime = DateTime(2018, 12, 1, 0, 0)
    end_datetime = DateTime(2019, 2, 1, 0, 0)

    raw_data = mi.get_many_dev_raw_dataframe('pm', mi.get_one_dev_ids('pm'), fields=None,
                                             start_datetime=start_datetime,
                                             end_datetime=end_datetime)

    # raw_data = pd.read_pickle(
    #     os.path.abspath(r"C:\Users\chris\Documents\University_local\Thesis_datasets\raw_power_meter_kikuyu"))

    al_df = raw_data.align_samples()
    df = al_df.resample_device(sample_time='15Min')
    df_pow = df.compute_power_from_energy()
