import os
import sys
import numpy as np

file_path = os.path.normpath(
    r"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW0001\DHW0001_DHW.txt")
#
with open(file_path,'r') as file_object:
    line_reader = file_object.readlines()
    data = []
    for line in line_reader:
        data.append(float(line.strip())/60.0)

data = np.array(data)
data = data.reshape(60*24,-1).sum(axis=0)
from matplotlib import pyplot as plt
plt.plot(np.array(data), drawstyle="steps-post")