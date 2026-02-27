import numpy as np
import os
from pathlib import Path
import datetime as dt
import matplotlib.pyplot as plt
import mrr
import spc
from netCDF4 import Dataset



data_path_mrr = Path('/home/engbers/Documents/PhD/Data/MRR_from_Jerome/20190101/')

df = Dataset(data_path_mrr / '20190101_060000.nc')
print(df)

# Get time and range variables
df_time = df.variables['time'][:]  # shape: (450,)
df_range = df.variables['range'][:]  # shape: (32,)

# Get velocity data
df_vel = df.variables['VEL'][:, :]  # shape: (450, 32)

plt.contourf(df_time, df_range, df_vel.T)  # transpose so range is y-axis
plt.xlabel('Time ')
plt.ylabel('Range')
plt.show()