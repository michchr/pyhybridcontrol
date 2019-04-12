import os
import sys
import numpy as np

import pandas as pd
from datetime import datetime as DateTime, timedelta as TimeDelta

#
# file_path = os.path.normpath(
#     r"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW0001\DHW0001_DHW.txt")
# #
# with open(file_path,'r') as file_object:
#     line_reader = file_object.readlines()
#     data = []
#     for line in line_reader:
#         data.append(float(line.strip())/60.0)
#
# data = np.array(data)
#
# raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018,1,1),None, len(data), '1Min'), columns=['actual'])
#
# df:pd.DataFrame = raw_df.resample('24H').sum()
#
# df['forecast'] = df.shift(1, freq='24H', fill_value=0).rename(columns={'actual': 'forecast'})
# df['err'] = df.actual - df.forecast
# df = df.fillna(0)
#
# df.iloc[:,:1].plot()

data = []
for i in range(1, 12):
    file_path = os.path.normpath(
        fr"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW10"
        fr"00\DHW1000_{i}\DHW1000_{i}_DHW.txt")

    print(i)
    with open(file_path, 'r') as file_object:
        line_reader = file_object.readlines()
        for line in line_reader:
            data.append(float(line.strip()) / 60.0)

raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018, 12, 3), None, len(data), '1Min'), columns=['actual'])

dewh_omega_profile_df: pd.DataFrame = raw_df.resample('15Min').sum()

dewh_omega_scenarios_df = pd.DataFrame(
    dewh_omega_profile_df.values.reshape(pd.Timedelta('1D') // pd.Timedelta('15Min'), -1, order='F'),
    index=pd.timedelta_range(0, '1D', freq='15Min', closed='left'))



