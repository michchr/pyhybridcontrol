from datetime import datetime as DateTime, timedelta as TimeDelta
from enum import Enum
from structdict import StructDict
import numpy as np


class Tariff_e(Enum):
    OFF_PEAK = 0
    STANDARD = 1
    PEAK = 2


class Day_e(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class TariffGenerator():
    def __init__(self, low_off_peak=0.0, low_stnd=0.0, low_peak=0.0, high_off_peak=0.0, high_stnd=0.0, high_peak=0.0):
        self.rates = StructDict(low_off_peak=low_off_peak, low_stnd=low_stnd, low_peak=low_peak,
                                high_off_peak=high_off_peak, high_stnd=high_stnd, high_peak=high_peak)

    def get_price_vector(self, N_p, date_time_0, control_ts):
        delta_time = TimeDelta(hours=control_ts)
        time_vector = [date_time_0 + i * delta_time for i in range(N_p)]
        price_vector = np.atleast_2d([self.get_import_price(time_vector[i]) for i in range(N_p)]).T
        return price_vector

    def get_import_price(self, date_time: DateTime):
        tarrif_type, high_demand = self.get_tariff_type(date_time)
        if high_demand:
            if tarrif_type == Tariff_e.OFF_PEAK:
                return self.rates.high_off_peak
            elif tarrif_type == Tariff_e.STANDARD:
                return self.rates.high_stnd
            elif tarrif_type == Tariff_e.PEAK:
                return self.rates.high_peak
            else:
                raise ValueError("Invalid tariff type: {}".format(tarrif_type))
        else:
            if tarrif_type == Tariff_e.OFF_PEAK:
                return self.rates.low_off_peak
            elif tarrif_type == Tariff_e.STANDARD:
                return self.rates.low_stnd
            elif tarrif_type == Tariff_e.PEAK:
                return self.rates.low_peak
            else:
                raise ValueError("Invalid tariff type: {}".format(tarrif_type))

    def is_high_demand(self, date_time: DateTime):
        start_high = (6, 1)  # 1st of June High Season Starts
        end_high = (8, 31)  # 31st of August High Season Ends

        if start_high <= (date_time.month, date_time.day) <= end_high:
            return True
        else:
            return False

    def get_tariff_type(self, date_time: DateTime):
        high_demand = self.is_high_demand(date_time)
        switcher = {
            Day_e.MONDAY: self._get_tariff_type_weekday,
            Day_e.TUESDAY: self._get_tariff_type_weekday,
            Day_e.WEDNESDAY: self._get_tariff_type_weekday,
            Day_e.THURSDAY: self._get_tariff_type_weekday,
            Day_e.FRIDAY: self._get_tariff_type_weekday,
            Day_e.SATURDAY: self._get_tariff_type_saturday,
            Day_e.SUNDAY: self._get_tarrif_type_sunday
        }
        return switcher.get(Day_e(date_time.weekday()))(date_time), high_demand

    def _get_tariff_type_weekday(self, date_time: DateTime):
        hour = date_time.hour

        if self.is_high_demand(date_time):
            if 22 <= hour or 0 <= hour < 6:
                return Tariff_e.OFF_PEAK
            elif 9 <= hour < 17:
                return Tariff_e.STANDARD
            elif 19 <= hour < 22:
                return Tariff_e.STANDARD
            else:
                return Tariff_e.PEAK
        else:  # not high demand season
            if 22 <= hour or 0 <= hour < 6:
                return Tariff_e.OFF_PEAK
            elif 6 <= hour < 7:
                return Tariff_e.STANDARD
            elif 10 <= hour < 18:
                return Tariff_e.STANDARD
            elif 20 <= hour < 22:
                return Tariff_e.STANDARD
            else:
                return Tariff_e.PEAK

    def _get_tariff_type_saturday(self, date_time: DateTime):
        hour = date_time.hour

        if self.is_high_demand(date_time):
            if 20 <= hour or 0 <= hour < 7:
                return Tariff_e.OFF_PEAK
            elif 12 <= hour < 18:
                return Tariff_e.OFF_PEAK
            else:
                return Tariff_e.STANDARD
        else:  # not high demand season
            if 20 <= hour or 0 <= hour < 7:
                return Tariff_e.OFF_PEAK
            elif 12 <= hour < 18:
                return Tariff_e.OFF_PEAK
            else:
                return Tariff_e.STANDARD

    def _get_tarrif_type_sunday(self, date_time: DateTime):
        return Tariff_e.OFF_PEAK


if __name__ == '__main__':
    import pprint

    tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                                 high_stnd=102.95, high_peak=339.77)

    test_date = DateTime(2018, 11, 20, 18, 30)
    #
    print(test_date)
    print(tariff_gen.get_import_price(test_date))

    pprint.pprint(tariff_gen.get_price_vector(24, test_date, 0.25))
