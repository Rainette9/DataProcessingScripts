import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def double_rotation(fastdata, blockdur='30min'):
    """
    Transform wind to mean streamline coordinate system using double rotation in timesteps of 30 minutes.

    Parameters
    ----------
    fastdata: pd.DataFrame
        DataFrame containing the wind data with columns 'Ux', 'Uy', 'Uz'.
    blockdur: str
        Duration of each block (e.g., '30T' for 30 minutes).

    Returns
    -------
    pd.DataFrame
        DataFrame with rotated wind data.
    """
    fastdata_rot=fastdata.copy()

    # Convert block duration to Timedelta
    blockdur = pd.Timedelta(blockdur)
    freq=(fastdata.index[1]-fastdata.index[0]).total_seconds()
    blockduridx = int(blockdur / pd.Timedelta(f'{freq}s'))
    
    startidx = 0
    endidcs = []
    startidcs = []

    print(f"Double rotation for blocks of {blockdur}")
    while startidx < len(fastdata_rot)- blockduridx:
        startidcs.append(startidx)
        endidcs.append(startidx + blockduridx -1)
        startidx += blockduridx 
    endidcs[-1] = len(fastdata_rot) -1
    angles = pd.DataFrame(columns=['theta', 'phi'])
    for startidx, endidx in zip(startidcs, endidcs):

        datatouse = fastdata_rot.iloc[startidx:endidx+1]
        u_unrot = datatouse['Ux'].values
        v_unrot = datatouse['Uy'].values
        w_unrot = datatouse['Uz'].values
        # Combine winds into matrix
        wind_unrot = np.c_[u_unrot, v_unrot, w_unrot]

        # # Mirror y-axes to get right-handed coordinate system (depends on the sonic)
        # wind_unrot[:, 1] = -wind_unrot[:, 1]

        # First rotation to set mean(v) = 0
        theta = np.arctan2(np.nanmean(wind_unrot[:, 1]), np.nanmean(wind_unrot[:, 0]))

        rot1 = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0, 0, 1]])
        wind1 = np.dot(wind_unrot, rot1)

        # Second rotation to set mean(w) = 0
        phi = np.arctan2(np.nanmean(wind1[:, 2]), np.nanmean(wind1[:, 0]))
        rot2 = np.array([[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0], [np.sin(phi), 0, np.cos(phi)]])
        wind_rot = np.dot(wind1, rot2)

        u_rot = wind_rot[:, 0]
        v_rot = wind_rot[:, 1]
        w_rot = wind_rot[:, 2]

        # Overwrite the input data with rounded values
        fastdata_rot.loc[fastdata_rot.index[startidx:endidx + 1], 'Ux'] = np.round(u_rot, 5)
        fastdata_rot.loc[fastdata_rot.index[startidx:endidx + 1], 'Uy'] = np.round(v_rot, 5)
        fastdata_rot.loc[fastdata_rot.index[startidx:endidx + 1], 'Uz'] = np.round(w_rot, 5)
        angles.loc[fastdata_rot.index[startidx]] = [theta, phi]
    return fastdata_rot, angles


def check_heat_flux(fastdata, blockdur, rho=1.225, plot=False):
    """
    Calculate and plot the heat flux for the averaging time window blockdur.

    Parameters
    ----------
    fastdata: pd.DataFrame
        DataFrame containing the wind data with columns 'Ux', 'Uy', 'Uz'.
    blockdur: str
        Duration of each block (e.g., '30min' for 30 minutes).


    Returns
    -------
    pd.DataFrame
        DataFrame with senisble and latent (if H2O present) heat flux for each block.
    """
    rho=1.29 # kg/m^3
    cp =1005
    # Convert block duration to Timedelta
    blockdur = pd.Timedelta(blockdur)
    freq = (fastdata.index[1] - fastdata.index[0]).total_seconds()
    blockduridx = int(blockdur / pd.Timedelta(f'{freq}s'))

    startidx = 0
    endidcs = []
    startidcs = []

    while startidx < len(fastdata)- blockduridx:
        startidcs.append(startidx)
        endidcs.append(startidx + blockduridx -1)
        startidx += blockduridx 
    endidcs[-1] = len(fastdata) -1
    df_heatflux = pd.DataFrame(columns=['SHF', 'LHF'])
    for startidx, endidx in zip(startidcs, endidcs):
        datatouse = fastdata.iloc[startidx:endidx+1]
        Ts = datatouse['Ts'].values
        w = datatouse['Uz'].values
        if datatouse.Uz.isna().sum()   > blockduridx*0.4:
            print(f"Warning: NaNs in block {startidx} to {endidx} bigger than 40%")
            SHF = np.nan
            LHF = np.nan
            wTsprime = np.nan
            wqprime = np.nan
        else:       
            # Calculate the 
            Tsprime = Ts - np.nanmean(Ts)
            wprime = w - np.nanmean(w)
            wTsprime = np.nanmean(wprime * Tsprime)
            if 'LI_H2Om_corr' in fastdata.columns:
                q = datatouse['LI_H2Om_corr'].values
            elif 'LI_H2Om' in fastdata.columns:
                q = datatouse['LI_H2Om'].values
            else:
                q = np.nan

            if not np.isnan(q).all():
                qprime = (q - np.nanmean(q) )*(0.018/1000) #from mmol/kg to kg/m3
                wqprime = np.nanmean(wprime * qprime)
                LHF = rho * 2.834*10**6 * wqprime 
            else:
                LHF = np.nan

        SHF= rho * cp * wTsprime
        df_heatflux.loc[fastdata.index[startidx]] = SHF, LHF



    if plot==True:
        plt.figure(figsize=(10, 6))
        plt.plot(df_heatflux['SHF'].resample('30min').mean(), label='Sensible Heat Flux')
        plt.plot(df_heatflux['LHF'].resample('30min').mean(), label='Latent Heat Flux')
        plt.xlabel('Time')
        plt.ylabel(r'Heat Flux [$W/m^2$]')
        plt.title('Heat Flux over time')
        plt.ylim(-200,100)
        plt.legend()
        plt.show()
    
    return df_heatflux





