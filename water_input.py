import os
import sys
import numpy as np

import pandas as pd
from datetime import datetime as DateTime, timedelta as TimeDelta

data = []
for i in range(1, 12):
    file_path = os.path.normpath(
        fr"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW_Act"
        fr"\DHW_Act_{i}\DHW_Act_{i}_DHW.txt")

    print(i)
    with open(file_path, 'r') as file_object:
        line_reader = file_object.readlines()
        for line in line_reader:
            data.append(float(line.strip()) / 60.0)

raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018, 12, 9), None, len(data), '1Min'), columns=['actual'])

dewh_omega_profile_df: pd.DataFrame = raw_df.resample('15Min').sum()

dewh_omega_scenarios_df = pd.DataFrame(
    dewh_omega_profile_df.values.reshape(pd.Timedelta('1D') // pd.Timedelta('15Min'), -1, order='F'),
    index=pd.timedelta_range(0, '1D', freq='15Min', closed='left'))




# path = os.path.realpath(r"./experiments/data/dewh_dhw_demand_scenario_profile_15Min_200Lpd_mean.pickle")
# df =pd.read_pickle(path)
