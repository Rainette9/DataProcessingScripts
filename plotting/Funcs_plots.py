import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.dates as mdates
sys.path.append(os.path.join(os.getcwd(), 'EC'))
import Func_read_data
from Func_read_data import convert_RH_liquid_to_ice

def resample_with_threshold(data, resample_time, min_valid_percent=80):
    """
    Returns NaN if the percentage of valid values within the resample time is less than min_valid_percent.

    Parameters:
        data (pd.Series): The input data to be resampled.
        resample_time (str): The resampling frequency (e.g., '10min', '1H').
        min_valid_percent (float): Minimum percentage of valid values required to keep the resampled value.

    Returns:
        pd.Series: The resampled data with insufficient valid data set to NaN.
    """
    # Resample the data
    resampled_data = data.resample(resample_time).mean()
    # Count the number of valid (non-NaN) values in each resample period
    valid_counts = data.resample(resample_time).count()
    # Calculate the total number of values in each resample period
    total_counts = data.resample(resample_time).size()
    # Calculate the percentage of valid values
    valid_percent = (valid_counts / total_counts) * 100
    # Apply the threshold and valid percentage filter
    filtered_data = resampled_data.where((valid_percent >= min_valid_percent))
    return filtered_data

def plot_SFC_slowdata_and_fluxes(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, sensor, start, end, resample_time='10min'):

    fig, ax = plt.subplots(9, 1, figsize=(13, 18), sharex=True)

    ax[0].plot(resample_with_threshold(slowdata['SFTempK'][start:end] - 273.15, resample_time),
               label='TS', color='darkblue', alpha=0.8, linestyle='dashed')
    ax[0].plot(resample_with_threshold(slowdata['TA'][start:end], resample_time),
               label='TA', color='deepskyblue')
    ax[0].plot(resample_with_threshold(fluxes_SFC['sonic_temperature'][start:end] - 273.15, resample_time),
               label='T_sonic_SFC', color='deepskyblue', alpha=0.8, linestyle='-.')
    ax[0].set_ylabel('Temperature [oC]')
    ax[0].plot(resample_with_threshold(fluxes_16m['sonic_temperature'][start:end] - 273.15, resample_time),
               label='T_16m', color='limegreen')
    ax[0].plot(resample_with_threshold(fluxes_26m['sonic_temperature'][start:end] - 273.15, resample_time),
               label='T_26m', color='gold')
    ax[0].legend(frameon=False)

    ax[1].plot(resample_with_threshold(convert_RH_liquid_to_ice(slowdata['RH'], slowdata['TA'])[start:end],
                                       resample_time),
               label='RH', color='deepskyblue')
    ax[1].set_ylabel('RH wrt ice [%]')
    ax[1].legend(frameon=False)
    ax[1].set_ylim(0, 100)

    ax[2].scatter(resample_with_threshold(slowdata['WD1'][start:end], resample_time).index,
                  resample_with_threshold(slowdata['WD1'][start:end], resample_time),
                  label='WD1', s=5, color='deepskyblue')
    ax[2].scatter(resample_with_threshold(slowdata['WD2'][start:end], resample_time).index,
                  resample_with_threshold(slowdata['WD2'][start:end], resample_time),
                  label='WD2', s=5, color='darkblue')
    ax[2].scatter(resample_with_threshold(fluxes_16m['wind_dir'][start:end], resample_time).index,
                  resample_with_threshold(fluxes_16m['wind_dir'][start:end], resample_time),
                  label='WD_16m', s=5, color='limegreen')
    ax[2].scatter(resample_with_threshold(fluxes_26m['wind_dir'][start:end], resample_time).index,
                  resample_with_threshold(fluxes_26m['wind_dir'][start:end], resample_time),
                  label='WD_26m', s=5, color='gold')
    ax[2].set_ylabel('Wind Direction')
    ax[2].legend(frameon=False)
    ax[2].set_ylim(0, 360)

    ax[3].plot(resample_with_threshold(slowdata['WS1_Avg'][start:end], resample_time),
               label='WS1_Avg', color='deepskyblue')
    ax[3].plot(resample_with_threshold(slowdata['WS2_Avg'][start:end], resample_time),
               label='WS2_Avg', color='darkblue')
    ax[3].plot(resample_with_threshold(fluxes_16m['wind_speed'][start:end], resample_time),
               label='WS_16m', color='limegreen')
    ax[3].plot(resample_with_threshold(fluxes_26m['wind_speed'][start:end], resample_time),
               label='WS_26m', color='gold')
    ax[3].set_ylabel('Wind Speed [ms-1]')
    ax[3].legend(frameon=False)

    ax[4].plot(resample_with_threshold(-(slowdata['SWdown1'] - slowdata['SWup1'])[start:end], resample_time),
               label='SW_net1', color='gold')
    ax[4].plot(resample_with_threshold(-(slowdata['LWdown1'] - slowdata['LWup1'])[start:end], resample_time),
               label='LW_net1', color='limegreen')
    ax[4].plot(resample_with_threshold(-(slowdata['SWdown2'] - slowdata['SWup2'])[start:end], resample_time),
               label='SW_net2', color='gold', linestyle='dashed', alpha=0.8)
    ax[4].plot(resample_with_threshold(-(slowdata['LWdown2'] - slowdata['LWup2'])[start:end], resample_time),
               label='LW_net2', color='limegreen', linestyle='dashed', alpha=0.8)
    ax[4].set_ylabel('Net Radiation [Wm-2]')
    ax[4].legend(frameon=False)

    ax[5].plot(resample_with_threshold(slowdata['HS_Cor'][start:end], resample_time),
               label='HS_Cor', color='deepskyblue')
    ax[5].set_ylabel('HS_Cor [m]')
    ax[5].legend(frameon=False)

    ax[6].plot(resample_with_threshold(slowdata['PF_FC4'][start:end], resample_time),
               label='PF_FC4', color='deepskyblue')
    ax[6].set_ylabel('Flowcapt [g/m2/s]')

    ax[7].plot(resample_with_threshold(fluxes_SFC['H'][start:end], resample_time),
               label='H SFC', color='deepskyblue')
    ax[7].plot(resample_with_threshold(fluxes_16m['H'][start:end], resample_time),
               label='H 16m', color='limegreen')
    ax[7].plot(resample_with_threshold(fluxes_26m['H'][start:end], resample_time),
               label='H 26m', color='gold')
    ax[7].set_ylabel('H [Wm-2]')
    ax[7].set_ylim(-180, 80)
    ax[7].legend(frameon=False)

    ax[8].plot(resample_with_threshold(fluxes_SFC['LE'][start:end], resample_time),
               label='LE SFC', color='deepskyblue')
    ax[8].set_ylabel('LE [Wm-2]')
    ax[8].legend(frameon=False)

    fig.suptitle(f'{sensor} slowdata {start} - {end}', y=0.92, fontsize=16)
    plt.savefig(f'./plots/{sensor}_{start}_slowdata_and_fluxes.png', bbox_inches='tight')
    return fig, ax

