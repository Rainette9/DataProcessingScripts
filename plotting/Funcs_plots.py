import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.dates as mdates
sys.path.append(os.path.join(os.getcwd(), 'EC'))
import Func_read_data
from Func_read_data import convert_RH_liquid_to_ice





def find_consecutive_periods(slowdata, SPC,  threshold=1, duration='4h'):
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
    ax[1].set_ylim(0, 115)

    wd1 = resample_with_threshold(slowdata['WD1'][start:end], resample_time)
    wd2 = resample_with_threshold(slowdata['WD2'][start:end], resample_time)

    # WD1 markers
    ax[2].scatter(wd1[wd1.between(0, 90)].index, wd1[wd1.between(0, 90)],
                  label='WD1 (0-90)', s=10, color='deepskyblue', marker='s')
    ax[2].scatter(wd1[wd1.between(90, 180)].index, wd1[wd1.between(90, 180)],
                  label='WD1 (90-180)', s=10, color='deepskyblue',  marker='o', facecolors='none')
    ax[2].scatter(wd1[~wd1.between(0, 180)].index, wd1[~wd1.between(0, 180)],
                  label='WD1 (180-360)', s=10, color='deepskyblue', marker='x')

    # WD2 markers
    ax[2].scatter(wd2[wd2.between(0, 90)].index, wd2[wd2.between(0, 90)],
                  label='WD2 (0-90)', s=10, color='darkblue', marker='s')
    ax[2].scatter(wd2[wd2.between(90, 180)].index, wd2[wd2.between(90, 180)],
                  label='WD2 (90-180)', s=10, color='darkblue', marker='o', facecolors='none')
    ax[2].scatter(wd2[~wd2.between(0, 180)].index, wd2[~wd2.between(0, 180)],
                  label='WD2 (180-360)', s=10, color='darkblue', marker='x')

    # Add a secondary y-axis (twinx) on the right
    ax2_right = ax[2].twinx()
    ax2_right.scatter(resample_with_threshold(fluxes_SFC['(z-d)/L'][start:end], resample_time).index, resample_with_threshold(fluxes_SFC['(z-d)/L'][start:end], resample_time, interpolate, interp_time),color='darkorange', label='z/L', s=10, marker='^')
    ax2_right.set_ylabel('z/L', color='darkorange')
    ax2_right.set_ylim(-0.4, 2)
    ax2_right.legend(frameon=False, loc='upper right')

    ax[2].set_ylabel('Wind Direction')
    ax[2].legend(frameon=False)
    ax[2].set_ylim(0, 360)

    ax[3].plot(resample_with_threshold(slowdata['WS2_Avg'][start:end], resample_time),
               label='WS2_Avg', color='darkblue')
    ax[3].plot(resample_with_threshold(fluxes_SFC['wind_speed'][start:end], resample_time),
               label='WS_SFC', color='royalblue', alpha=0.8, linestyle='dashed')
    ax[3].plot(resample_with_threshold(slowdata['WS1_Avg'][start:end], resample_time),
               label='WS1_Avg', color='deepskyblue')
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
    plt.savefig(f'./plots_specific_events/{sensor}_{start}_slowdata_and_fluxes.png', bbox_inches='tight')
    return fig, ax

def check_log_profile(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, start, end, heights=[0,1.5,1.9,3.5,16,26], log=False):
    """
    Check the log profile for the slow data and fluxes.
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 6), sharey=True)
    time_diff=pd.Timestamp(end) - pd.Timestamp(start)
    # Wind Speed Profile
    wind_speeds = [0, resample_with_threshold(slowdata['WS2_Avg'][start:end], time_diff, True).mean(), 
                   resample_with_threshold(fluxes_SFC['wind_speed'][start:end], time_diff, True).mean(),
                   resample_with_threshold(slowdata['WS1_Avg'][start:end], time_diff, True).mean(), 
                   resample_with_threshold(fluxes_16m['wind_speed'][start:end], time_diff, True).mean(), 
                   resample_with_threshold(fluxes_26m['wind_speed'][start:end], time_diff, True).mean()]
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
    temperatures = [
        resample_with_threshold(slowdata['SFTempK'][start:end] - 273.15, time_diff, True).mean(),
        resample_with_threshold(slowdata['TA'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_SFC['sonic_temperature'][start:end]- 273.15, time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['sonic_temperature'][start:end] - 273.15, time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['sonic_temperature'][start:end] - 273.15, time_diff, True).mean()
    ]
    axes[1].scatter(temperatures, heights[:3] + heights[4:], label='Temperature Data Points')
    axes[1].set_xlabel('Temperature (°C)')
    # axes[1].legend()
    axes[1].set_title('Temperature Profile')

    # Sensible Heat Flux Profile
    sensible_heat_fluxes = [
        resample_with_threshold(fluxes_SFC['H'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['H'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['H'][start:end], time_diff, True).mean()
    ]
    axes[2].scatter(sensible_heat_fluxes, [heights[2]] + heights[4:], label='Sensible Heat Flux Data Points')
    axes[2].set_xlabel('Sensible Heat Flux (W/m²)')
    # axes[2].legend()
    axes[2].set_title('Sensible Heat Flux Profile')

    # TKE Profile
    tke_fluxes = [
        resample_with_threshold(fluxes_SFC['TKE'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['TKE'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['TKE'][start:end], time_diff, True).mean()
    ]
    axes[3].scatter(tke_fluxes, [heights[2]] + heights[4:], label='TKE Data Points')
    axes[3].set_xlabel('TKE')
    # axes[3].legend()
    axes[3].set_title('TKE Profile')

    # TKE Profile
    tke_fluxes = [
        resample_with_threshold(fluxes_SFC['(z-d)/L'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['(z-d)/L'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['(z-d)/L'][start:end], time_diff, True).mean()
    ]
    axes[4].scatter(tke_fluxes, [heights[2]] + heights[4:], label='stability Data Points')
    axes[4].set_xlabel('stability')
    # axes[4].legend()
    axes[4].set_title('stability Profile')

    plt.suptitle(f'Wind, Temperature, and Sensible Heat Flux Profiles from {start} to {end}', fontsize=16, y=0.97)
    plt.tight_layout()
    # plt.savefig(f'./plots/log_profile_{start}_to_{end}.png', bbox_inches='tight')
    # plt.show()

def check_log_profiles(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m,consecutive_days, heights=[0,2,3,16,26], log=False):
    """
    Check the log profile for the slow data and fluxes.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
    for start,end in consecutive_days:
        # Wind Speed Profile
        wind_speeds = [0, slowdata['WS2_Avg'][start:end], slowdata['WS1_Avg'][start:end].mean(), fluxes_16m['wind_speed'][start:end].mean(), fluxes_26m['wind_speed'][start:end].mean()]
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
    # plt.savefig(f'./plots/log_profile_{start}_to_{end}.png', bbox_inches='tight')
    # plt.show()


def plot_bi_monthly_mean_H(fluxes_SFC, fluxes_16m, fluxes_26m, heights, variable):
    """
    Plots the bi-monthly mean H with 25th and 75th percentiles for different heights.

    Parameters:
        fluxes_SFC (pd.DataFrame): DataFrame containing SFC flux data.
        fluxes_16m (pd.DataFrame): DataFrame containing 16m flux data.
        fluxes_26m (pd.DataFrame): DataFrame containing 26m flux data.
        heights (list): List of heights corresponding to SFC, 16m, and 26m.
        variable (str): The variable to plot.
    """
    # Add 'Month' and 'Day' columns to group data
    fluxes_SFC['Month'] = fluxes_SFC.index.month
    fluxes_SFC['Day'] = fluxes_SFC.index.day
    fluxes_16m['Month'] = fluxes_16m.index.month
    fluxes_16m['Day'] = fluxes_16m.index.day
    fluxes_26m['Month'] = fluxes_26m.index.month
    fluxes_26m['Day'] = fluxes_26m.index.day

    # Add 'Month_Name' column for labelingfluxes_SFC['TI']= np.sqrt(fluxes_SFC['u_var']**2 + fluxes_SFC['v_var']**2 + fluxes_SFC['w_var']**2) / fluxes_SFC['wind_speed']
    fluxes_SFC['Month_Name'] = fluxes_SFC.index.month_name()
    fluxes_16m['Month_Name'] = fluxes_16m.index.month_name()
    fluxes_26m['Month_Name'] = fluxes_26m.index.month_name()

    # Initialize the figure
    fig, axes = plt.subplots(2, 12, figsize=(30, 10), sharey=True, sharex=True)
    axes = axes.flatten()

    # Loop through each month and create subplots for the first and second halves
    for month in range(1, 13):
        for part in [1, 2]:  # 1 for first half, 2 for second half
            ax = axes[(month - 1) * 2 + (part - 1)]

            # Filter data for the current month and part
            if part == 1:
                sfc_part = fluxes_SFC[(fluxes_SFC['Month'] == month) & (fluxes_SFC['Day'] <= 15)][variable]
                m16_part = fluxes_16m[(fluxes_16m['Month'] == month) & (fluxes_16m['Day'] <= 15)][variable]
                m26_part = fluxes_26m[(fluxes_26m['Month'] == month) & (fluxes_26m['Day'] <= 15)][variable]
                title_suffix = " (1st Half)"
            else:
                sfc_part = fluxes_SFC[(fluxes_SFC['Month'] == month) & (fluxes_SFC['Day'] > 15)][variable]
                m16_part = fluxes_16m[(fluxes_16m['Month'] == month) & (fluxes_16m['Day'] > 15)][variable]
                m26_part = fluxes_26m[(fluxes_26m['Month'] == month) & (fluxes_26m['Day'] > 15)][variable]
                title_suffix = " (2nd Half)"

            # Calculate mean, 25th, and 75th percentiles
            means = [
                resample_with_threshold(sfc_part, '15D', True, '30min', 60).mean() if not sfc_part.empty else np.nan,
                resample_with_threshold(m16_part, '15D', True, '3h', 60).mean() if not m16_part.empty else np.nan,
                resample_with_threshold(m26_part, '15D', True, '3h', 60).mean() if not m26_part.empty else np.nan
            ]
            percentiles_25 = [
                resample_with_threshold(sfc_part, '1h', True, '30min', 60).quantile(0.25) if not sfc_part.empty else np.nan,
                resample_with_threshold(m16_part, '1h', True, '1h', 60).quantile(0.25) if not m16_part.empty else np.nan,
                resample_with_threshold(m26_part, '1h', True, '1h', 60).quantile(0.25) if not m26_part.empty else np.nan
            ]
            percentiles_75 = [
                resample_with_threshold(sfc_part, '1h', True, '30min', 60).quantile(0.75) if not sfc_part.empty else np.nan,
                resample_with_threshold(m16_part, '1h', True, '1h', 60).quantile(0.75) if not m16_part.empty else np.nan,
                resample_with_threshold(m26_part, '1h', True, '1h', 60).quantile(0.75) if not m26_part.empty else np.nan
            ]
            # Plot the means with whiskers
            # Ensure error bars are non-negative
            lower_error = np.maximum(0, np.array(means) - np.array(percentiles_25))
            upper_error = np.maximum(0, np.array(percentiles_75) - np.array(means))

            # Plot the means with whiskers
            ax.errorbar(
                means, heights,
                xerr=[lower_error, upper_error],
                fmt='o-', capsize=5, label='H'
            )

            # Set titles and labels
            if not fluxes_SFC['Month_Name'][fluxes_SFC['Month'] == month].empty:
                ax.set_title(f"{fluxes_SFC['Month_Name'][fluxes_SFC['Month'] == month].iloc[0]}{title_suffix}", fontsize=10)
            else:
                ax.set_title(f"Month {month}{title_suffix}", fontsize=10)
            if (month - 1) * 2 + (part - 1) % 4 == 0:  # First column
                ax.set_ylabel("Height (m)")
            if (month - 1) * 2 + (part - 1) >= 20:  # Last row
                ax.set_xlabel(f"{variable} (Mean ± IQR)")
            ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(f"HalfMonthly Mean {variable} with 25th and 75th Percentiles", fontsize=16, y=1.02)
    plt.savefig(f'./plots/BiMonthly_Mean_{variable}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_wind_speed_binned(fluxes_SFC, fluxes_16m, fluxes_26m, slowdata, heights, variable):
    """
    Plots the mean H with 25th and 75th percentiles for different heights, grouped into bins based on wind speed.

    Parameters:
        fluxes_SFC (pd.DataFrame): DataFrame containing SFC flux data.
        fluxes_16m (pd.DataFrame): DataFrame containing 16m flux data.
        fluxes_26m (pd.DataFrame): DataFrame containing 26m flux data.
        slowdata (pd.DataFrame): DataFrame containing wind speed data.
        heights (list): List of heights corresponding to SFC, 16m, and 26m.
        variable (str): The variable to plot.
    """
    # Define wind speed bins (0-20 m/s in steps of 2.5 m/s)
    bins = np.arange(0, 18, 3)
    bin_labels = [f"{bins[i]}-{bins[i+1]} m/s" for i in range(len(bins) - 1)]
    slowdata_mean = slowdata.resample('30min').mean()  # Resample slowdata to 30-minute intervals

    slowdata_mean['Wind_Speed_Bin'] = pd.cut(slowdata_mean['WS1_Avg'], bins=bins, labels=bin_labels, include_lowest=True)

    # Add wind speed bins to flux data
    fluxes_SFC['Wind_Speed_Bin'] = slowdata_mean['Wind_Speed_Bin']
    fluxes_16m['Wind_Speed_Bin'] = slowdata_mean['Wind_Speed_Bin']
    fluxes_26m['Wind_Speed_Bin'] = slowdata_mean['Wind_Speed_Bin']

    # Initialize the figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 8), sharey=True, sharex=True)
    axes = axes.flatten()

    # Loop through each wind speed bin and create subplots
    for i, wind_bin in enumerate(bin_labels):
        ax = axes[i]

        # Filter data for the current wind speed bin
        sfc_bin = fluxes_SFC[fluxes_SFC['Wind_Speed_Bin'] == wind_bin][variable]
        m16_bin = fluxes_16m[fluxes_16m['Wind_Speed_Bin'] == wind_bin][variable]
        m26_bin = fluxes_26m[fluxes_26m['Wind_Speed_Bin'] == wind_bin][variable]
        # Calculate mean, 25th, and 75th percentiles
        means = [
            fluxes_SFC.median() if not sfc_bin.empty else np.nan,
            fluxes_16m.median() if not m16_bin.empty else np.nan,
            fluxes_26m.median() if not m26_bin.empty else np.nan
        ]
        percentiles_25 = [
            fluxes_SFC.quantile(0.25) if not sfc_bin.empty else np.nan,
            fluxes_16m.quantile(0.25) if not m16_bin.empty else np.nan,
            fluxes_26m.quantile(0.25) if not m26_bin.empty else np.nan
        ]
        percentiles_75 = [
            fluxes_SFC.quantile(0.75) if not sfc_bin.empty else np.nan,
            fluxes_16m.quantile(0.75) if not m16_bin.empty else np.nan,
            fluxes_26m.quantile(0.75) if not m26_bin.empty else np.nan
        ]

        # Ensure error bars are non-negative
        lower_error = np.maximum(0, np.array(means) - np.array(percentiles_25))
        upper_error = np.maximum(0, np.array(percentiles_75) - np.array(means))

        # Plot the means with whiskers
        ax.errorbar(
            means, heights,
            xerr=[lower_error, upper_error],
            fmt='o-', capsize=5, label='H'
        )

        # Set titles and labels
        ax.set_title(f"Wind Speed Bin: {wind_bin}", fontsize=10)
        if i == 0:  # First column
            ax.set_ylabel("Height (m)")
        ax.set_xlabel(f"{variable} (Mean ± IQR)")
        ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(f"{variable} by Wind Speed Bins with 25th and 75th Percentiles", fontsize=16, y=1.02)
    plt.savefig(f'./plots/wind_speed_binned_{variable}.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_filtered_wind_speed(fluxes_SFC, fluxes_16m, fluxes_26m, slowdata, heights, variable):
    """
    Plots two subplots for filtered cases:
    1. When slowdata['PF_FC4'] > 0.1 and slowdata['WS1_Avg'] > 5.
    2. When slowdata['PF_FC4'] == 0 and slowdata['WS1_Avg'] > 5.

    Parameters:
        fluxes_SFC (pd.DataFrame): DataFrame containing SFC flux data.
        fluxes_16m (pd.DataFrame): DataFrame containing 16m flux data.
        fluxes_26m (pd.DataFrame): DataFrame containing 26m flux data.
        slowdata (pd.DataFrame): DataFrame containing slow data.
        heights (list): List of heights corresponding to SFC, 16m, and 26m.
        variable (str): The variable to plot.
    """
    # Filter data for the two cases
    # Define wind speed bins (0-20 m/s in steps of 2.5 m/s)

    # Case 1: Filter for PF_FC4 > 0.1 and WS1_Avg > 5
    slowdata_mean = slowdata.resample('30min').mean()  # Resample slowdata to 30-minute intervals
    case_BS_condition = (slowdata_mean['PF_FC4'] > 0.1) & (slowdata_mean['WS1_Avg'] > 5) & (slowdata_mean['WS1_Avg'] < 10)
    case_noBS_condition = (slowdata_mean['PF_FC4'] <= 0.00001) & (slowdata_mean['WS1_Avg'] > 5) & (slowdata_mean['WS1_Avg'] < 10)

    # Add bins to flux data based on conditions using .loc to avoid SettingWithCopyWarning
    fluxes_SFC.loc[:, 'BS_bin'] = np.where(case_BS_condition, 'BS', np.where(case_noBS_condition, 'no_BS', 'else'))
    fluxes_16m.loc[:, 'BS_bin'] = np.where(case_BS_condition, 'BS', np.where(case_noBS_condition, 'no_BS', 'else'))
    fluxes_26m.loc[:, 'BS_bin'] = np.where(case_BS_condition, 'BS', np.where(case_noBS_condition, 'no_BS', 'else'))


    # Initialize the figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    axes = axes.flatten()
    bin_labels = ['BS', 'no_BS', 'else']
    # Loop through each wind speed bin and create subplots
    for i, BS_bin in enumerate(bin_labels):
        
        ax = axes[i]

        # Filter data for the current wind speed bin
        sfc_bin = fluxes_SFC[fluxes_SFC['BS_bin'] == BS_bin][variable]
        m16_bin = fluxes_16m[fluxes_16m['BS_bin'] == BS_bin][variable]
        m26_bin = fluxes_26m[fluxes_26m['BS_bin'] == BS_bin][variable]
        # Calculate mean, 25th, and 75th percentiles
        means = [
            resample_with_threshold(sfc_bin, '30min', False, '30min', 50).mean() if not sfc_bin.empty else np.nan,
            resample_with_threshold(m16_bin, '30min', False, '30min', 50).mean() if not m16_bin.empty else np.nan,
            resample_with_threshold(m26_bin, '30min', False, '30min', 50).mean() if not m26_bin.empty else np.nan
        ]
        percentiles_25 = [
            resample_with_threshold(sfc_bin, '30min', False, '30min', 50).quantile(0.25) if not sfc_bin.empty else np.nan,
            resample_with_threshold(m16_bin, '30min', False, '30min', 50).quantile(0.25) if not m16_bin.empty else np.nan,
            resample_with_threshold(m26_bin, '30min', False, '30min', 50).quantile(0.25) if not m26_bin.empty else np.nan
        ]
        percentiles_75 = [
            resample_with_threshold(sfc_bin, '30min', False, '30min', 50).quantile(0.75) if not sfc_bin.empty else np.nan,
            resample_with_threshold(m16_bin, '30min', False, '30min', 50).quantile(0.75) if not m16_bin.empty else np.nan,
            resample_with_threshold(m26_bin, '30min', False, '30min', 50).quantile(0.75) if not m26_bin.empty else np.nan
        ]

        # Ensure error bars are non-negative
        lower_error = np.maximum(0, np.array(means) - np.array(percentiles_25))
        # lower_error = np.array(percentiles_25)
        upper_error = np.maximum(0, np.array(percentiles_75) - np.array(means))
        # upper_error = np.array(percentiles_75)

        # Plot the means with whiskers
        ax.errorbar(
            means, heights,
            xerr=[lower_error, upper_error],
            fmt='o-', capsize=5, label='H'
        )

        # Set titles and labels
        ax.set_title(f"{BS_bin}", fontsize=10)
        if i % 4 == 0:  # First column
            ax.set_ylabel("Height (m)")
        if i >= 4:  # Last row
            ax.set_xlabel(f"{variable} (Mean ± IQR)")
        ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(f"{variable} by BS Bins with 25th and 75th Percentiles", fontsize=16, y=1.02)
    plt.savefig(f'./plots/BS_binned_{variable}.png', bbox_inches='tight', dpi=300)
    plt.show()
    # return case_BS, case_noBS