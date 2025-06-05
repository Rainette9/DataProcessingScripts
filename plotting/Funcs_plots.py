import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.dates as mdates
sys.path.append(os.path.join(os.getcwd(), 'EC'))
import Func_read_data
from Func_read_data import convert_RH_liquid_to_ice





def find_consecutive_periods(slowdata, SPC, threshold=1, duration='3h'):
    """
    Finds periods where both slowdata['PF_FC4'] and SPC['Corrected Mass Flux(kg/m^2/s)']
    are greater than a threshold for consecutive hours, while removing occurrences where
    slowdata['HS_Cor'] decreases by more than 1 per hour.

    Parameters:
        slowdata (pd.DataFrame): DataFrame containing 'PF_FC4' and 'HS_Cor' columns.
        SPC (pd.DataFrame): DataFrame containing 'Corrected Mass Flux(kg/m^2/s)' column.
        threshold (float): The threshold value to check against.
        duration (str): Minimum duration of consecutive periods (e.g., '3H').

    Returns:
        list: A list of tuples containing the start and end times of consecutive periods.
    """
    # Remove occurrences where 'HS_Cor' decreases by more than 1 per hour
    hs_cor_diff = slowdata['HS_Cor'].resample('1h').mean().diff()
    slowdata = slowdata[hs_cor_diff.reindex(slowdata.index, method='ffill') >= -0.02/60] ### 2cm per hour
    slowdata =slowdata[slowdata['WS1_Avg']>3]
    # Create masks for values greater than the threshold
    mask_slowdata = slowdata['PF_FC4'] > threshold
    # mask_SPC = SPC['Corrected Mass Flux(kg/m^2/s)']  >= threshold /1000
    mask_SPC = SPC['Corrected Mass Flux(kg/m^2/s)']  >= 0
    # Combine masks to find periods where both conditions are met
    combined_mask = mask_slowdata & mask_SPC
    # Resample to hourly frequency and check for consecutive periods
    resampled_mask = combined_mask.resample('3h').mean() > 0

    # Identify consecutive periods
    consecutive_periods = resampled_mask.astype(int).diff().fillna(0)
    start_times = resampled_mask[consecutive_periods == 1].index
    end_times = resampled_mask[consecutive_periods == -1].index

    # Ensure start and end times align correctly
    if len(end_times) < len(start_times):
        end_times = end_times.append(pd.Index([resampled_mask.index[-1]]))

    # Filter periods based on duration
    valid_periods = []
    for starts, ends in zip(start_times, end_times):
        if (ends - starts) >= pd.Timedelta(duration):
            valid_periods.append((starts, ends))

    return valid_periods




def resample_with_threshold(data, resample_time, interpolate=False, max_gap='1h', min_valid_percent=80):
    """
    Returns NaN if the percentage of valid values within the resample time is less than min_valid_percent.
    Linearly interpolates gaps in the data only if the gaps are smaller than 1H.

    Parameters:
        data (pd.Series): The input data to be resampled.
        resample_time (str): The resampling frequency (e.g., '10min', '1h').
        min_valid_percent (float): Minimum percentage of valid values required to keep the resampled value.

    Returns:
        pd.Series: The resampled data with insufficient valid data set to NaN.
    """
    if interpolate == True:
        # Calculate the data's frequency in seconds
        freq = (data.index[1] - data.index[0]).total_seconds()
        # Convert the max_gap to seconds
        max_gap_seconds = pd.to_timedelta(max_gap).total_seconds()
        # Calculate the limit as the number of consecutive NaNs within the max_gap
        limit = int(max_gap_seconds / freq)
        data = data.interpolate(limit=limit, limit_direction='both', limit_area='inside')
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
    # Interpolate gaps smaller than 1H

    return filtered_data

def plot_SFC_slowdata_and_fluxes(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, sensor, start, end, SPC=None, resample_time='10min', interpolate=False, interp_time='1h'):

    consecutive_periods = find_consecutive_periods(slowdata, SPC) ### find 3h hour consecutive BS periods
    filtered_periods = [period for period in consecutive_periods if period[0] >= pd.Timestamp(start) and period[1] <= pd.Timestamp(end)]

    fig, ax = plt.subplots(9, 1, figsize=(13, 18), sharex=True)

    for a in ax:
        for starts, ends in filtered_periods:
            a.axvspan(starts, ends, color='grey', alpha=0.2)

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
    # ax[2].scatter(resample_with_threshold(fluxes_16m['wind_dir'][start:end], resample_time).index,
    #               resample_with_threshold(fluxes_16m['wind_dir'][start:end], resample_time),
    #               label='WD_16m', s=5, color='limegreen')
    # ax[2].scatter(resample_with_threshold(fluxes_26m['wind_dir'][start:end], resample_time).index,
    #               resample_with_threshold(fluxes_26m['wind_dir'][start:end], resample_time),
    #               label='WD_26m', s=5, color='gold')
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

    ax[6].plot(resample_with_threshold(SPC['Corrected Mass Flux(kg/m^2/s)'][start:end], resample_time)*1000, 
               label='SPC Mass Flux', color='darkblue')
    ax[6].plot(resample_with_threshold(slowdata['PF_FC4'][start:end], resample_time),
               label='PF_FC4', color='deepskyblue')
    ax[6].legend(frameon=False)
    ax[6].set_ylabel('Mass flux [g/m2/s]')

    ax[7].plot(resample_with_threshold(fluxes_SFC['H'][start:end], resample_time, interpolate, interp_time),
               label='H SFC', color='deepskyblue')
    ax[7].plot(resample_with_threshold(fluxes_16m['H'][start:end], resample_time, interpolate, interp_time),
               label='H 16m', color='limegreen')
    ax[7].plot(resample_with_threshold(fluxes_26m['H'][start:end], resample_time, interpolate, interp_time),
               label='H 26m', color='gold')
    ax[7].set_ylabel('H [Wm-2]')
    # ax[7].set_ylim(-180, 80)
    ax[7].legend(frameon=False)

    ax[8].plot(resample_with_threshold(fluxes_SFC['LE'][start:end], resample_time, interpolate, interp_time),
               label='LE SFC', color='deepskyblue')
    ax[8].set_ylabel('LE [Wm-2]')
    ax[8].legend(frameon=False)


    fig.suptitle(f'{resample_time} resampled {start} - {end}', y=0.92, fontsize=16)
    plt.savefig(f'./plots/{sensor}_{start}_slowdata_and_fluxes.png', bbox_inches='tight')
    return fig, ax

def check_log_profile(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, start, end, heights=[0,2,3,16,26], log=False):
    """
    Check the log profile for the slow data and fluxes.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)

    # Wind Speed Profile
    wind_speeds = [0, slowdata['WS2_Avg'][start:end].mean(), slowdata['WS1_Avg'][start:end].mean(), fluxes_16m['wind_speed'][start:end].mean(), fluxes_26m['wind_speed'][start:end].mean()]
    axes[0].scatter(wind_speeds, heights, label='Wind Speed Data Points')
    if log:
        log_wind_speeds = np.log(wind_speeds[1:])  # Exclude the first zero value
        log_heights = np.log(heights[1:])  # Exclude the first zero value
        slope, intercept = np.polyfit(log_wind_speeds, log_heights, 1)
        fitted_heights = np.exp(intercept) * np.array(wind_speeds[1:])**slope
        axes[0].plot(wind_speeds[1:], fitted_heights, label=f'Fit: slope={slope:.2f}', color='red')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel('Wind Speed (m/s)')
    axes[0].set_ylabel('Height (m)')
    # axes[0].legend()
    axes[0].set_title('Wind Speed Profile')

    # Temperature Profile
    temperatures = [slowdata['SFTempK'][start:end].mean() - 273.15, slowdata['TA'][start:end].mean(), fluxes_16m['sonic_temperature'][start:end].mean() - 273.15, fluxes_26m['sonic_temperature'][start:end].mean() - 273.15]
    axes[1].scatter(temperatures, heights[:2] + heights[3:], label='Temperature Data Points')
    axes[1].set_xlabel('Temperature (°C)')
    # axes[1].legend()
    axes[1].set_title('Temperature Profile')

    # Sensible Heat Flux Profile
    sensible_heat_fluxes = [fluxes_SFC['H'][start:end].mean(), fluxes_16m['H'][start:end].mean(), fluxes_26m['H'][start:end].mean()]
    axes[2].scatter(sensible_heat_fluxes, [heights[1]] + heights[3:], label='Sensible Heat Flux Data Points')
    axes[2].set_xlabel('Sensible Heat Flux (W/m²)')
    # axes[2].legend()
    axes[2].set_title('Sensible Heat Flux Profile')

    # Sensible Heat Flux Profile
    heat_fluxes = [fluxes_SFC['TKE'][start:end].mean(), fluxes_16m['TKE'][start:end].mean(), fluxes_26m['TKE'][start:end].mean()]
    axes[3].scatter(heat_fluxes, [heights[1]] + heights[3:], label='Sensible Heat Flux Data Points')
    axes[3].set_xlabel('TKE')
    # axes[3].legend()
    axes[3].set_title('TKE') 

    plt.suptitle(f'Wind, Temperature, and Sensible Heat Flux Profiles from {start} to {end}', fontsize=16, y=0.97)
    plt.tight_layout()
    plt.savefig(f'./plots/log_profile_{start}_to_{end}.png', bbox_inches='tight')
    plt.show()
