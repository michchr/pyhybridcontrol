import os
import sys
import numpy as np

file_path = os.path.normpath(
    r"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW0005\DHW0005_DHW.txt")
#
with open(file_path,'r') as file_object:
    line_reader = file_object.readlines()
    data = []
    for line in line_reader:
        data.append(float(line.strip())/60.0)

data = np.array(data)

import pandas as pd
from datetime import datetime as DateTime
raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018,1,1),None, len(data), '1Min'), columns=['actual'])

df:pd.DataFrame = raw_df.resample('15Min').sum()
shift = df.diff(-4*24*7).rename(columns={'actual':'shifted'}).dropna()

from matplotlib import pyplot as plt
ax = plt.axes()

df.plot(ax=ax, drawstyle="steps-post")
shift.plot(ax=ax, drawstyle="steps-post")

from matplotlib import pyplot as plt

from scipy.stats.kde import gaussian_kde

kde = gaussian_kde(shift.values.flatten())