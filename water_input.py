import os
import sys
import numpy as np

import pandas as pd
from datetime import datetime as DateTime, timedelta as TimeDelta

file_path = os.path.normpath(
    r"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW0001\DHW0001_DHW.txt")
#
with open(file_path,'r') as file_object:
    line_reader = file_object.readlines()
    data = []
    for line in line_reader:
        data.append(float(line.strip())/60.0)

data = np.array(data)

raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018,1,1),None, len(data), '1Min'), columns=['actual'])

df:pd.DataFrame = raw_df.resample('24H').sum()

df['forecast'] = df.shift(1, freq='24H', fill_value=0).rename(columns={'actual': 'forecast'})
df['err'] = df.actual - df.forecast
df = df.fillna(0)

df.iloc[:,:1].plot()


# ind = pd.timedelta_range('0H', '1D', freq='1H', closed='left')
#
# df2 = pd.DataFrame(df.err.values.reshape(24, -1), index=ind)
# df2[df2==0.0] = np.NaN
#
# import seaborn as sns
# for i in range(24):
#     ax = sns.distplot(df2.iloc[i,:], norm_hist=True, kde=True, bins=200)
#

# from matplotlib.patches import Ellipse
# from matplotlib import pyplot as plt
# import numpy as np
# import pymc3 as pm
# import seaborn as sns
#
# x = df2.values.T
#
# if __name__ == '__main__':
#     with pm.Model() as model:
#         packed_L = pm.LKJCholeskyCov('packed_L', n=24,
#                                      eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
#
#     with model:
#         L = pm.expand_packed_triangular(24, packed_L)
#         sig = pm.Deterministic('sig', L.dot(L.T))
#
#
#     with model:
#         mu = pm.Normal('mu', 0., 10., shape=24,
#                       testval=x.mean(axis=0))
#         obs = pm.MvNormal('obs', mu, chol=L, observed=x)
#
#     with model:
#         step = pm.Metropolis([L])
#         trace = pm.sample(step=step)
#

