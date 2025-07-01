import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter
from scipy.optimize import root_scalar


def apply_plausibility_limits(fastdata, plim):
    """
    This function applies plausibility limits to the fast data.
    """
    fastdata_plaus = fastdata.copy()
    # Check 'u' values
    in_out = fastdata['Ux'].abs() > plim['abs.u'].iloc[0]
    if in_out.any():
        print(f"Plausibility limits: Discarding {in_out.sum()} 'u' records.")
        fastdata_plaus.loc[in_out, 'Ux'] = np.nan

    # Check 'v' values
    in_out = fastdata_plaus['Uy'].abs() > plim['abs.v'].iloc[0]
    if in_out.any():
        print(f"Plausibility limits: Discarding {in_out.sum()} 'v' records.")
        fastdata_plaus.loc[in_out, 'Uy'] = np.nan

    # Check 'w' values
    in_out = fastdata_plaus['Uz'].abs() > plim['abs.w'].iloc[0]
    if in_out.any():
        print(f"Plausibility limits: Discarding {in_out.sum()} 'w' records.")
        fastdata_plaus.loc[in_out, 'Uz'] = np.nan

    # Check 'Ts' values
    in_out = (fastdata_plaus['Ts'] > plim['Ts.up'].iloc[0]) | (fastdata_plaus['Ts'] < plim['Ts.low'].iloc[0])
    if in_out.any():
        print(f"Plausibility limits: Discarding {in_out.sum()} 'Ts' records.")
        fastdata_plaus.loc[in_out, 'Ts'] = np.nan

    if 'LI_H2Om' in fastdata_plaus.columns:
        # Check 'LI_H2Om' values
        in_out = (fastdata_plaus['LI_H2Om'] < plim['h2o.low'].iloc[0]) | (fastdata_plaus['LI_H2Om'] > plim['h2o.up'].iloc[0])
        if in_out.any():
            print(f"Plausibility limits: Discarding {in_out.sum()} 'H2O' records.")
            fastdata_plaus.loc[in_out, 'LI_H2Om'] = np.nan
    #### Check sensor diagnostics
        in_out = fastdata_plaus['LI_diag'].abs() <= 240
        if in_out.any():
            print(f"LI_diag: Discarding {in_out.sum()} 'LI' records.")
            fastdata_plaus.loc[in_out, 'LI_H2Om'] = np.nan
            fastdata_plaus.loc[in_out, 'LI_Pres'] = np.nan
    in_out = fastdata_plaus['diag_csat'].abs() > 4096 #Sergi uses 4960
    if in_out.any():
        print(f"diag_csat: Discarding {in_out.sum()} 'CSAT' records.")
        fastdata_plaus.loc[in_out, 'Ux'] = np.nan
        fastdata_plaus.loc[in_out, 'Uy'] = np.nan
        fastdata_plaus.loc[in_out, 'Uz'] = np.nan
        fastdata_plaus.loc[in_out, 'Ts'] = np.nan
    


    # # Check 'LI_Pres' values if 'LI_Pres' column exists
    # if 'LI_Pres' in fastdata_plaus.columns:
    #     # in_out = (fastdata_plaus['LI_Pres'] < plim['pres.low'].iloc[0]) | (fastdata_plaus['LI_Pres'] > plim['pres.up'].iloc[0])
    #     in_out = (fastdata_plaus['LI_Pres'] < 0)
    #     if in_out.any():
    #         print(f"Plausibility limits: Discarding {in_out.sum()} 'pressure' records.")
    #         fastdata_plaus.loc[in_out, 'LI_Pres'] = np.nan
    return fastdata_plaus



def compute_h2o_concentration(RH, TA):
    """
    Compute H2O concentration from relative humidity (RH) and temperature (T).
    Parameters:
    RH (pd.Series): Relative humidity in percentage.
    TA (pd.Series): Temperature in degrees Celsius.
    """
    es = 611.2 * np.exp(17.67 * TA / (TA + 243.5)) *(RH/100) #Pa
    h2o_concentration = (1000 * es/  (8.314 * (TA+273.15)))   #mmol m^-3
    
    return h2o_concentration


def h2o_calibration(calibration_coefficients, fastdata, slowdata):
    """
    This function calibrates the H2O mixing ratio measurements using the calibration coefficients and slow data.
    """
    freq_LF=slowdata.index[1]-slowdata.index[0]
    freq=fastdata.index[1]-fastdata.index[0]
    fastdata_plaus=fastdata.copy()
    #Calibration coefficients and polynomial
    df_LF=pd.DataFrame()
    if 'TA' not in slowdata.columns:
        height=str(fastdata_plaus.attrs['heights']['Ts'][0])

        df_LF['TA']=slowdata[f'Temp_{height}m_Avg']
        df_LF['RH']=slowdata[f'RH_{height}m_Avg']*100 #Convert RH to percentage
        df_LF[['LI_H2Om_Avg','LI_Pres_Avg']]=slowdata[[f'LI_H2Om_{height}m_Avg', f'LI_Pres_{height}m_Avg']] #.groupby(pd.Grouper(freq='1min')).mean()
    else:
        df_LF[['LI_H2Om_Avg','LI_Pres_Avg']]=fastdata_plaus[['LI_H2Om', 'LI_Pres']].groupby(pd.Grouper(freq=freq_LF)).mean()
        df_LF[['TA','RH']] = slowdata[['TA', 'RH']].resample(freq_LF).mean()

    ###Bias correction of Water Vapour
    #Create a dataframe with the LF variables and calculate the molar density difference between RH and LI
    df_vap=pd.DataFrame()
    df_vap= df_LF.copy()
    df_vap['LI_Pres_Avg'] = (df_vap['LI_Pres_Avg']) * 1000 #Convert Pres units to Pa
    df_vap['RH_H2Om_Avg'] = compute_h2o_concentration(df_vap['RH'], df_vap['TA'])
    df_vap['H2Om_Diff'] = df_vap['LI_H2Om_Avg'] - df_vap['RH_H2Om_Avg']
    df_vap['LI_y'] = df_vap['LI_H2Om_Avg'] / df_vap['LI_Pres_Avg'] * 1000 #mmmol m^-3 kPa^-1
    df_vap['RH_y'] = df_vap['RH_H2Om_Avg'] / df_vap['LI_Pres_Avg'] * 1000 #mmmol m^-3 kPa^-1
    print('Mean H2O concentration difference: ' + str(df_vap['H2Om_Diff'].mean()))
    #Calculate minutal absorptance using calibration polynomial
    def polyapp(y, calibration_coefficients):
        if np.isnan(y):
            return np.nan
        # global counter
        # counter = counter+1; print(np.round(counter/total_items*100,3), end="\r")
        p = np.poly1d([calibration_coefficients['C'],calibration_coefficients['B'],calibration_coefficients['A'], y])
        # print(p.roots[0].real)
        return p.roots[0].real
    # def polyapp(y, calibration_coefficients, bounds=(0, 0.002)):
    #     if np.isnan(y):
    #         print("NaN value encountered in polyapp")
    #         return np.nan
        
    #     # Define the cubic polynomial function
    #     def f(a):
    #         return calibration_coefficients['A'] * a + \
    #             calibration_coefficients['B'] * a**2 + \
    #             calibration_coefficients['C'] * a**3 + y  # Note: y is already negated

    #     try:
    #         sol = root_scalar(f, bracket=bounds, method='brentq')
    #         if sol.converged:
    #             print(f"Root found for y={y} with bounds {bounds}: {sol.root}")
    #             return sol.root
    #         else:
    #             print(f"Root finding failed for y={y} with bounds {bounds}")
    #             return np.nan  # fallback if solver fails
    #     except ValueError:
    #         return np.nan  # happens when no root in bracket

    total_items = df_vap.shape[0]
    counter = 0
    df_vap['LI_a'] = df_vap['LI_y'].apply(lambda y: polyapp(-y, calibration_coefficients)) #LI absorptance
    df_vap['LI_a'] = df_vap['LI_y'].apply(lambda y: polyapp(-y, calibration_coefficients)) #LI absorptance
    #print(df_vap['LI_a'])
    counter = 0
    df_vap['RH_a'] = df_vap['RH_y'].apply(lambda y: polyapp(-y, calibration_coefficients)) #RH absorptance
    df_vap['RH_a'] = df_vap['RH_y'].apply(lambda y: polyapp(-y, calibration_coefficients)) #RH absorptance
    df_vap['LI_a_raw'] = df_vap['LI_a'] * df_vap['LI_Pres_Avg']/1000 / calibration_coefficients['H20_Span'] #LI raw absorptance
    df_vap['RH_a_raw'] = df_vap['RH_a'] * df_vap['LI_Pres_Avg']/1000 / calibration_coefficients['H20_Span'] #RH raw absorptance
    df_vapHF = df_vap[['LI_a_raw', 'RH_a_raw']].resample(freq).ffill() #High-resolution absorptances


    #Calculate 10Hz absorptance using calibration polynomial and correct the H2O mol
    df_p = fastdata_plaus.copy()

    df_p = pd.concat([df_p,df_vapHF], axis=1) #Add 30 minutely absorptances to fast data
    # df_p = df_p.dropna()
    df_p['LI_Pres'] = (df_p['LI_Pres'])  * 1000 #so this is in Pa
    df_p['LI_y_fast'] = df_p['LI_H2Om'] / df_p['LI_Pres'] *1000 #mmmol m^-3 kPa^-1
    total_items = df_p.shape[0]
    counter = 0

    df_p['LI_a_fast'] = df_p['LI_y_fast'].apply(lambda y: polyapp(-y, calibration_coefficients)) #LI absorptance
    df_p['LI_a_fast'] = df_p['LI_y_fast'].apply(lambda y: polyapp(-y, calibration_coefficients)) #LI absorptance
    df_p['LI_a_raw_fast'] = df_p['LI_a_fast'] * df_p['LI_Pres']/1000 / calibration_coefficients['H20_Span'] #LI raw absorptance
    df_p['LI_a_corr_fast'] = ((1 - df_p['RH_a_raw']) * df_p['LI_a_raw_fast'] - df_p['LI_a_raw'] + df_p['RH_a_raw']) / (1 - df_p['LI_a_raw']) #correction of raw absorptance
    df_p['LI_a_norm_fast'] =  df_p['LI_a_corr_fast'] / df_p['LI_Pres']*1000 * calibration_coefficients['H20_Span']
    df_p['LI_y_norm_fast'] = calibration_coefficients['A']*df_p['LI_a_norm_fast']+ calibration_coefficients['B']*df_p['LI_a_norm_fast']**2 + calibration_coefficients['C']*df_p['LI_a_norm_fast']**3
    df_p['LI_H2Om_corr'] = df_p['LI_y_norm_fast'] * df_p['LI_Pres']/1000 #mmol/m^-3
    df_p['LI_H2Om_corr'] = df_p['LI_H2Om_corr'].round(1)

    df_p.drop(columns=['LI_a_raw','RH_a_raw', 'LI_y_fast', 'LI_a_fast', 'LI_a_raw_fast', 'LI_a_corr_fast', 'LI_a_norm_fast', 'LI_y_norm_fast'], inplace=True)
    return df_p, df_LF

def plot_despiking_results(fastdata, fastdata_plaus, df_p, sensor, slowdata=None):
    """
    This function plots the results of the despiking algorithm.
    """
    if 'LI_H2Om' in fastdata.columns:
        fig, ax=plt.subplots(5,1, figsize=(15,10))
        ax[4].plot(fastdata['LI_H2Om'], label='LI_H2Om', color='grey')
        ax[4].plot(fastdata_plaus['LI_H2Om'], label='LI_H2Om_plaus', color='blue')
        if 'LI_H2Om_corr' in df_p.columns:
            ax[4].plot(df_p['LI_H2Om_corr'], label='LI_H2Om_despiked', color='red')
        if slowdata is not None:
            ax[4].plot(compute_h2o_concentration(slowdata['RH'], slowdata['TA']), label='RH_H2Om', color='green')
        ax[4].legend()
        ax[4].set_ylabel('LI_H2Om (mmol/m^-3)')
        ax[4].set_xlim(fastdata.index[0], fastdata.index[-1])
        ax[4].set_ylim(0, 700)

    else:
        fig, ax=plt.subplots(4,1, figsize=(15,10))
    ax[0].plot(fastdata['Ux'], label='Ux', color='grey')
    ax[0].plot(fastdata_plaus['Ux'], label='Ux_plaus', color='blue')
    ax[0].plot(df_p['Ux'], label='Ux_despiked', color='red')
    ax[0].set_ylabel('Ux (m/s)')
    ax[0].legend(loc='upper left')
    ax[0].set_ylim(-30, 30)
    ax[1].plot(fastdata['Uy'], label='Uy', color='grey')
    ax[1].plot(fastdata_plaus['Uy'], label='Uy_plaus', color='blue')
    ax[1].plot(df_p['Uy'], label='Uy_despiked', color='red')
    ax[1].legend(loc='upper left')
    ax[1].set_ylim(-30, 30)
    ax[1].set_ylabel('Uy (m/s)')
    ax[2].plot(fastdata['Uz'], label='Uz', color='grey')
    ax[2].plot(fastdata_plaus['Uz'], label='Uz_plaus', color='blue')
    ax[2].plot(df_p['Uz'], label='Uz_despiked', color='red')
    ax[2].legend(loc='upper left')
    ax[2].set_ylim(-10, 10)
    ax[2].set_ylabel('Uz (m/s)')
    ax[3].plot(fastdata['Ts'], label='Ts', color='grey')
    ax[3].plot(fastdata_plaus['Ts'], label='Ts_plaus', color='blue')
    ax[3].plot(df_p['Ts'], label='Ts_despiked', color='red')
    ax[3].legend(loc='upper left')
    ax[3].set_ylabel('Ts (C)')
    ax[3].set_ylim(-30, 10)
    fig.suptitle('Fast data despiked for sensor ' + sensor + fastdata.index[0].strftime('%Y%m%d'), y=0.93)
    plt.savefig(f'/home/engbers/Documents/PhD/EC_data_convert/SFC/plots_despiking/EC_despiked_{sensor}_{fastdata.index[0].strftime('%Y%m%d')}.png', bbox_inches='tight')
    plt.close()






def despike_fast_MAD(fastdata, slowdata, plim, sensor, calibration_coefficients=None, plot_despike=False):
    """
    This function  based on "modified_mad_filter" (Sigmund et al., 2022)
    """
    ### Applying plausibility limits 
    freq=fastdata.index[1]-fastdata.index[0]
    fastdata_plaus=apply_plausibility_limits(fastdata, plim)
    print('Plausibility limits applied')

    ### H2O corection
    if calibration_coefficients is not None:
        print('Applying H2O calibration')
        fastdata_plaus_calib, df_LF= h2o_calibration(calibration_coefficients, fastdata_plaus, slowdata)
        print('H2O calibration applied')
        # fastdata_plaus = fastdata_plaus.rename(columns={'LI_H2Om_corr': 'LI_H2Om'})
        df_p=fastdata_plaus_calib.copy()
    else:
        df_p=fastdata_plaus.copy()

    ###Spike correction
    print('Processing large dataset (%)')
    window = int(5 * 60 / freq.total_seconds()) # 5 minutes
    df_di = np.abs(df_p.rolling(window=window, center=True).median()-df_p)
    #df_MAD = df_p.rolling(window=6000, center=True).apply(median_abs_deviation)
    df_MAD = (np.abs(df_p-df_p.rolling(window=window, center=True).median())).rolling(window=window, center=True).median()
    #df_di = 7 * df_MAD / 0.6745
    df_hat = np.abs(df_di) - 0.5 * (np.abs(df_di.shift(-1)) + np.abs(df_di.shift(1)))
    df_hat_MAD = df_hat / df_MAD
    spike_condition = np.abs(df_hat_MAD['Ux'] + df_hat_MAD['Uy'] + df_hat_MAD['Uz'] + df_hat_MAD['Ts']) >= 6 / 0.6745
    if calibration_coefficients is not None:
        spike_condition |= df_hat_MAD['LI_H2Om_corr'] >= 6 / 0.6745
        # Set spikes to NaN
        df_p.loc[spike_condition, ['Ux', 'Uy', 'Uz', 'Ts', 'LI_H2Om_corr']] = np.nan
        print('Spikes removed from Ux,Uy,Uz,Ts:' + str(df_p['Ux'].isna().sum()-fastdata_plaus['Ux'].isna().sum()), 'Spikes removed from LI_H2Om_corr:' + str(df_p['LI_H2Om_corr'].isna().sum()-fastdata_plaus['LI_H2Om'].isna().sum()))
    elif 'LI_H2Om' not in df_p.columns:
        df_p.loc[spike_condition, ['Ux', 'Uy', 'Uz', 'Ts']] = np.nan
        print('Spikes removed from Ux,Uy,Uz,Ts:' + str(df_p['Ux'].isna().sum()-fastdata_plaus['Ux'].isna().sum()))
    elif calibration_coefficients is None:
        spike_condition |= df_hat_MAD['LI_H2Om'] >= 6 / 0.6745
        # Set spikes to NaN
        df_p.loc[spike_condition, ['Ux', 'Uy', 'Uz', 'Ts', 'LI_H2Om']] = np.nan
        print('Spikes removed from Ux,Uy,Uz,Ts:' + str(df_p['Ux'].isna().sum()-fastdata_plaus['Ux'].isna().sum()), 'Spikes removed from LI_H2Om:' + str(df_p['LI_H2Om'].isna().sum()-fastdata_plaus['LI_H2Om'].isna().sum()))        
    else:
        print('Despiking LI_H2Om_corr')
        spike_condition |= df_hat_MAD['LI_H2Om_corr'] >= 6 / 0.6745
        # Set spikes to NaN
        df_p.loc[spike_condition, ['Ux', 'Uy', 'Uz', 'Ts', 'LI_H2Om_corr']] = np.nan
        print('Spikes removed from Ux,Uy,Uz,Ts:' + str(df_p['Ux'].isna().sum()-fastdata_plaus['Ux'].isna().sum()), 'Spikes removed from LI_H2Om:' + str(df_p['LI_H2Om_corr'].isna().sum()-fastdata_plaus['LI_H2Om'].isna().sum()))
    
    ### Interpolate if gaps < 1s
    nan_mask = df_p['Ux'].isna()
    # Group consecutive NaNs and calculate their lengths
    nan_groups = nan_mask.astype(int).groupby((~nan_mask).cumsum(), group_keys=False).cumsum()
    last_values = nan_groups.groupby((~nan_mask).cumsum()).transform('last')
    nan_groups = np.where((last_values > 10) & (last_values != 0), last_values, nan_groups)
    # Identify gaps with fewer than 10 consecutive NaNs
    small_nan_gaps = nan_groups <= 10
    # Interpolate only for small NaN gaps
    df_p['Ux'] = df_p['Ux'].where(~nan_mask | ~small_nan_gaps, df_p['Ux'].interpolate(method='linear', limit_direction='both'))
    df_p['Uy'] = df_p['Uy'].where(~nan_mask | ~small_nan_gaps, df_p['Uy'].interpolate(method='linear', limit_direction='both'))
    df_p['Uz'] = df_p['Uz'].where(~nan_mask | ~small_nan_gaps, df_p['Uz'].interpolate(method='linear', limit_direction='both'))
    df_p['Ts'] = df_p['Ts'].where(~nan_mask | ~small_nan_gaps, df_p['Ts'].interpolate(method='linear', limit_direction='both'))
    # print('Interpolated Ux,Uy,Uz,Ts:' + str(df_p['Ux'].isna().sum()-fastdata_plaus['Ux'].isna().sum()))
    if 'LI_H2Om_corr' in df_p.columns:
        # Identify where the data is NaN
        nan_mask = df_p['LI_H2Om_corr'].isna()
        # Group consecutive NaNs and calculate their lengths
        nan_groups = nan_mask.astype(int).groupby((~nan_mask).cumsum(), group_keys=False).cumsum()
        last_values = nan_groups.groupby((~nan_mask).cumsum()).transform('last')
        nan_groups = np.where((last_values > 10) & (last_values != 0), last_values, nan_groups)
        # Identify gaps with fewer than 10 consecutive NaNs
        small_nan_gaps = nan_groups <= 10
        df_p['LI_H2Om_corr'] = df_p['LI_H2Om_corr'].where(~nan_mask | ~small_nan_gaps, df_p['LI_H2Om_corr'].interpolate(method='linear', limit_direction='both'))
        # print('Interpolated LI_H2Om_corr:' + str(df_p['LI_H2Om_corr'].isna().sum()-fastdata_plaus['LI_H2Om'].isna().sum()))

    ### Plotting
    if plot_despike==True and calibration_coefficients is not None:
        print('Plotting despiking results')
        plot_despiking_results(fastdata, fastdata_plaus, df_p, sensor, df_LF)
    if plot_despike==True and calibration_coefficients is None:
        print('Plotting despiking results')
        plot_despiking_results(fastdata, fastdata_plaus, df_p, sensor)

    return df_p





# def despiking(datain, windowwidth=3000, maxsteps=10, breakcrit=1.05):
#     """
#     Despiking algorithm after Sigmund et al. (2022) based on the despiking algorith from Michis Julia scripts
#     """
#     print("Despiking...")

#     # Make windowwidth odd if it's even
#     if windowwidth % 2 == 0:
#         windowwidth += 1

#     # Criterion to compare to
#     criterion = 6 / 0.6745

#     # Replace missing values with NaN
#     datain = datain.replace({pd.NA: np.nan})

#     # Initialize spike detection arrays
#     spike = np.zeros(len(datain), dtype=bool)
#     spikeirg = np.zeros(len(datain), dtype=bool)
#     tolook = np.ones(len(datain), dtype=bool)

#     # Columns to despike
#     if 'h2o' in datain.columns:
#         colstodespike = ['u', 'v', 'w', 'T', 'h2o']
#     else:
#         colstodespike = ['u', 'v', 'w', 'T']

#     devtomedian = datain[colstodespike].copy()

#     nrsteps = 0
#     nrspikes = np.zeros(maxsteps, dtype=int)
#     nrspikesirg = np.zeros(maxsteps, dtype=int)
#     nrspikestot = np.zeros(maxsteps, dtype=int)
#     mad_data = np.zeros((len(datain), len(colstodespike)))

#     while nrsteps < maxsteps:
#         nrsteps += 1
#         for idx, col in enumerate(colstodespike):
#             tmp = median_filter(datain[col], size=windowwidth)
#             devtomedian[col] = np.abs(datain[col] - tmp)
#             mad_data[:, idx] = median_filter(devtomedian[col], size=windowwidth)

#         # Set previously detected spikes to equal the median
#         devtomedian.loc[spike, ['u', 'v', 'w', 'T']] = 0
#         if 'h2o' in colstodespike:
#             devtomedian.loc[spikeirg, 'h2o'] = 0

#         for jdx in range(1, len(devtomedian) - 1):
#             if tolook[jdx]:
#                 leftside = 0
#                 leftsideirg = 0
#                 for jcol in range(len(colstodespike)):
#                     a = devtomedian.iloc[jdx, jcol] - np.nanmean([devtomedian.iloc[jdx-1, jcol], devtomedian.iloc[jdx+1, jcol]])
#                     if colstodespike[jcol] != 'h2o':
#                         leftside += a / mad_data[jdx, jcol]
#                     else:
#                         leftsideirg = a / mad_data[jdx, jcol]
#                 if leftside > criterion:
#                     spike[jdx] = True
#                 if leftsideirg > criterion:
#                     spikeirg[jdx] = True

#         nrspikes[nrsteps-1] = np.sum(spike)
#         nrspikesirg[nrsteps-1] = np.sum(spikeirg)
#         nrspikestot[nrsteps-1] = nrspikes[nrsteps-1] + nrspikesirg[nrsteps-1]
#         print(f"nrspikes[{nrsteps-1}] = {nrspikes[nrsteps-1]}")
#         print(f"nrspikesirg[{nrsteps-1}] = {nrspikesirg[nrsteps-1]}")
#         print(f"nrspikestot[{nrsteps-1}] = {nrspikestot[nrsteps-1]}")

#         # Only check neighboring to spikes in next step
#         spiketot = spike | spikeirg
#         spikeloc = np.where(spiketot)[0]
#         tolook[:] = False
#         tolook[spikeloc-1] = True
#         tolook[spikeloc+1] = True

#         devtomedian.loc[spike, ['u', 'v', 'w', 'T']] = 0
#         if np.sum(spikeirg) != 0:
#             devtomedian.loc[spikeirg, 'h2o'] = 0
#         datain.loc[spike, ['u', 'v', 'w', 'T']] = np.nan
#         if np.sum(spikeirg) != 0:
#             datain.loc[spikeirg, 'h2o'] = np.nan
#         if nrsteps > 1 and nrspikestot[nrsteps-1] / nrspikestot[nrsteps-2] < breakcrit:
#             break

#     return datain


# def clean_slowdata(slowdata):
