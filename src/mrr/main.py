from pathlib import Path
import os
import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

import mrr
import spc

# %% PARAMETERS
#data_path_mrr = Path('C:/Users\dujardin\AWS_Data\MRR')
data_path_mrr = Path('/home/engbers/Documents/PhD/Data/MRR_from_Jerome/')
# data_path_spc2 = Path('C:/Users/dujardin/AWS_Data/SPC2')ex
# data_path_spc = Path('C:/Users/dujardin/AWS_Data/SPC')
#data_path_spc = Path('D:/Users/dujardin/Desktop/Data_Arrival/SPC/Retrieved_20190102_1500(Functionnal)/Converted')

arg_mrr = dict(
                file_path=Path('/home/engbers/Documents/PhD/Data/MRR_from_Jerome/'),
                var_name='spectrum_raw',
                vel_min=1,  # [m/s] Min horizontal wind speed
                vel_max=30,  # [m/s] Max horizontal wind speed
                x_min=20,  # [m] Min horizontal distance to the dish center
                x_max=300,  # [m] Max horizontal distance to the dish center
                z_min=0,  # [m] Min vertical height above ground
                z_max=1e12,  # [m] Max vertical height above ground
                dish_height=1,  # [m] Height of the dish center from the ground
                beam_angle=7,  # [deg] From the horizontal oward the sky
                dist_first_gate=5,  # [m] Distance of the center of the 1ts range gate from the dish
                time_offset=0,  # [s] Number of seconds to add to the MRR timestamps to be exactly at UTC
                visu=False,  # [bool] To plot the data as an animation
                save_movie=False # [bool] To save the plots in a .mp4 file (need to install ffmpeg and specify its path in mrr.py)
                )


# %%
mrr_vel_all = np.empty((0,), dtype=np.float32)
mrr_x_vel_all = np.empty((0,), dtype=np.float32)
mrr_reflect_all = np.empty((0,), dtype=np.float32)
mrr_timestamp_all = []

month_list = sorted(os.listdir(data_path_mrr))
for month in month_list:
    day_list = sorted(os.listdir(data_path_mrr / month))
    for day in day_list:
        file_list = sorted(os.listdir(data_path_mrr / month / day))
        for file in file_list:
            file_path = data_path_mrr / month / day / file
            if (file[15:] == '.nc') & (os.lstat(file_path).st_size > 0):
                print(file_path)
                arg_mrr['file_path'] = file_path
                if int(file[:8]+file[9:-3]) <= 20181228180000:
                    arg_mrr['time_offset'] = 52
                else:
                    arg_mrr['time_offset'] = 0
                print(arg_mrr['time_offset'])

                (mrr_vel, mrr_x_vel, mrr_reflect, mrr_timestamp) = mrr.import_mrr(**arg_mrr)
                mrr_vel_all = np.append(mrr_vel_all, mrr_vel)
                mrr_x_vel_all = np.append(mrr_x_vel_all, mrr_x_vel)
                mrr_reflect_all = np.append(mrr_reflect_all, mrr_reflect)
                mrr_timestamp_all = mrr_timestamp_all + mrr_timestamp

if mrr_reflect_all[0] is not None:
    ind_zero = np.argwhere((mrr_reflect_all == 0) | (mrr_reflect_all >= 300000))
    mrr_vel_all[ind_zero] = 0
    mrr_reflect_all[ind_zero] = 0


# %%
spc_temperature_all = np.empty((0,), dtype=np.float32)
spc_timestamp_all = []

file_list = sorted(os.listdir(data_path_spc))
i = 0
for file in file_list:
    if file[-3:] == 'csv':
        file_path = data_path_spc / file
        print(file_path)
        (spc_counts, spc_temperature, spc_diameter, spc_timestamp) = spc.import_spc(file_path)
        if i == 0:
            spc_counts_all = spc_counts
        else:
            spc_counts_all = np.append(spc_counts_all, spc_counts, axis=0)
        spc_temperature_all = np.append(spc_temperature_all, spc_temperature)
        spc_timestamp_all = spc_timestamp_all + spc_timestamp
        i += 1

# %%
spc_temperature_all2 = np.empty((0,), dtype=np.float32)
spc_timestamp_all2 = []

file_list = sorted(os.listdir(data_path_spc2))
i = 0
for file in file_list:
    if file[-3:] == 'csv':
        file_path = data_path_spc2 / file
        print(file_path)
        (spc_counts, spc_temperature, spc_diameter, spc_timestamp) = spc.import_spc(file_path)
        if i == 0:
            spc_counts_all2 = spc_counts
        else:
            spc_counts_all2 = np.append(spc_counts_all2, spc_counts, axis=0)
        spc_temperature_all2 = np.append(spc_temperature_all2, spc_temperature)
        spc_timestamp_all2 = spc_timestamp_all2 + spc_timestamp
        i += 1
spc2_ind=np.argwhere(np.sum(spc_counts_all2[:], axis=1)>0)[:,0]
spc_timestamp_all2_sub = [spc_timestamp_all2[i] for i in spc2_ind]
spc2_ind=np.asarray(spc2_ind,dtype=np.int32)
# %%
#mrr_timestamp_all_corrected = mrr_timestamp_all.copy()
#for i in range(len(mrr_timestamp_all_corrected)):
#    if mrr_vel_all[i] > 0:
#        mrr_timestamp_all_corrected[i] +=dt.timedelta(seconds=mrr_x_vel_all[i]/mrr_vel_all[i])
plt.figure()
plt.plot(mrr_timestamp_all, mrr_vel_all)
plt.plot(mrr_timestamp_all, mrr_vel_all, '.')
if mrr_reflect_all[0] is not None:
    plt.plot(mrr_timestamp_all, mrr_reflect_all/100)
    plt.plot(mrr_timestamp_all, mrr_reflect_all/100, '.')
#plt.plot(spc_timestamp_all, np.convolve(np.sum(spc_counts_all*(spc_diameter**3), axis=1)/(10e8), np.ones((8,))/8,'same'))
#plt.plot(spc_timestamp_all, np.convolve(spc_counts_all[:,5]/5, np.ones((8,))/8,'same'))

plt.plot(spc_timestamp_all2_sub, np.sum(spc_counts_all2[spc2_ind, :], axis=1)/100)
plt.plot(spc_timestamp_all2_sub, np.sum(spc_counts_all2[spc2_ind, :], axis=1)/100, '.')

plt.plot(spc_timestamp_all, np.sum(spc_counts_all, axis=1)/100)
plt.plot(spc_timestamp_all, np.sum(spc_counts_all, axis=1)/100, '.')


# %%
if True:
#    arg_mrr['file_path'] = data_path_mrr / Path('201812/20181227/20181227_110000.nc')
    arg_mrr['file_path'] = data_path_mrr / Path('201901/20190124/20190124_030000.nc')
    var_list = ['Za', 'Z', 'Zea', 'Ze', 'RR', 'LWC', 'PIA', 'VEL', 'WIDTH', 'ML', 'SNR', 'index_spectra', 'spectrum_raw', 'N']
    arg_mrr['var_name'] = 'spectrum_raw'
    arg_mrr['visu'] = True
    arg_mrr['save_movie'] = True
    arg_mrr['time_offset'] = 0
    (mrr_vel, mrr_x_vel, mrr_reflect, mrr_timestamp) = mrr.import_mrr(**arg_mrr)
