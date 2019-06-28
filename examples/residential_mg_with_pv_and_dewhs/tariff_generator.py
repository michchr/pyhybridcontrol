from datetime import datetime as DateTime, timedelta as TimeDelta
from enum import Enum
from collections import namedtuple
from structdict import named_fixed_struct_dict
import numpy as np
from utils.matrix_utils import atleast_2d_col

_TARIFF_TYPES = ['low_off_peak',
                 'low_stnd',
                 'low_peak',
                 'high_off_peak',
                 'high_stnd',
                 'high_peak']

TariffTypes = namedtuple('TariffTypes', _TARIFF_TYPES)(*_TARIFF_TYPES)
TariffRatesStruct = named_fixed_struct_dict('TariffRatesStruct', _TARIFF_TYPES)

class Day_e(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


START_HIGH = (6, 1)  # 1st of June High Season Starts
END_HIGH = (8, 31)  # 31st of August High Season Ends


class TariffGenerator():

    def __init__(self, low_off_peak=0.0, low_stnd=0.0, low_peak=0.0, high_off_peak=0.0, high_stnd=0.0, high_peak=0.0):
        self.rates = TariffRatesStruct(low_off_peak=low_off_peak, low_stnd=low_stnd, low_peak=low_peak,
                                       high_off_peak=high_off_peak, high_stnd=high_stnd, high_peak=high_peak)

    def get_price_vector(self, date_time_0, N_tilde, control_ts):
        if isinstance(control_ts, TimeDelta):
            delta_time = control_ts
        else:
            delta_time = TimeDelta(seconds=control_ts)

        if not isinstance(date_time_0, DateTime):
            raise TypeError(f"date_time_0 must be of type {DateTime.__name__!r}")
        time_vector = [date_time_0 + i * delta_time for i in range(N_tilde)]
        price_vector = atleast_2d_col([self.get_import_price(time_vector[i]) for i in range(N_tilde)])
        return price_vector

    def get_import_price(self, date_time: DateTime):
        tariff_type = self.get_tariff_type(date_time)
        return self.rates[tariff_type]

    def is_high_demand(self, date_time: DateTime):
        month_day = (date_time.month, date_time.day)
        return START_HIGH <= month_day <= END_HIGH

    def get_tariff_type(self, date_time: DateTime):
        tariff_type = self.TariffTypeSwitcher[Day_e(date_time.weekday())](self, date_time=date_time)
        return tariff_type

    def _get_tariff_rate_type_weekday(self, date_time: DateTime):
        hour = date_time.hour
        high_demand = self.is_high_demand(date_time)

        if high_demand:
            if 22 <= hour or 0 <= hour < 6:
                return TariffTypes.high_off_peak
            elif 9 <= hour < 17:
                return TariffTypes.high_stnd
            elif 19 <= hour < 22:
                return TariffTypes.high_stnd
            else:
                return TariffTypes.high_peak
        else:  # not high demand season
            if 22 <= hour or 0 <= hour < 6:
                return TariffTypes.low_off_peak
            elif 6 <= hour < 7:
                return TariffTypes.low_stnd
            elif 10 <= hour < 18:
                return TariffTypes.low_stnd
            elif 20 <= hour < 22:
                return TariffTypes.low_stnd
            else:
                return TariffTypes.low_peak

    def _get_tariff_rate_type_saturday(self, date_time: DateTime):
        hour = date_time.hour
        high_demand = self.is_high_demand(date_time)

        if high_demand:
            if 20 <= hour or 0 <= hour < 7:
                return TariffTypes.high_off_peak
            elif 12 <= hour < 18:
                return TariffTypes.high_off_peak
            else:
                return TariffTypes.high_stnd
        else:  # not high demand season
            if 20 <= hour or 0 <= hour < 7:
                return TariffTypes.low_off_peak
            elif 12 <= hour < 18:
                return TariffTypes.low_off_peak
            else:
                return TariffTypes.low_stnd

    def _get_tarrif_rate_type_sunday(self, date_time: DateTime):
        high_demand = self.is_high_demand(date_time)
        if high_demand:
            return TariffTypes.high_off_peak
        else:
            return TariffTypes.low_off_peak

    TariffTypeSwitcher = {
        Day_e.MONDAY   : _get_tariff_rate_type_weekday,
        Day_e.TUESDAY  : _get_tariff_rate_type_weekday,
        Day_e.WEDNESDAY: _get_tariff_rate_type_weekday,
        Day_e.THURSDAY : _get_tariff_rate_type_weekday,
        Day_e.FRIDAY   : _get_tariff_rate_type_weekday,
        Day_e.SATURDAY : _get_tariff_rate_type_saturday,
        Day_e.SUNDAY   : _get_tarrif_rate_type_sunday
    }


if __name__ == '__main__':
    import pprint

    tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                                 high_stnd=102.95, high_peak=339.77)

    test_date = DateTime(2018, 11, 20, 18, 30)
    #
    print(test_date)
    print(tariff_gen.get_import_price(test_date))

    pprint.pprint(tariff_gen.get_price_vector(24, test_date, 15 * 60))
