import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def despike_fast_MAD(fastdata, slowdata, window, calibration_data):
    """
    This function .. based on "modified_mad_filter" (Sigmund et al., 2022)
    """
        
    # #Plausibility limits
    fastdata = fastdata[(fastdata['Ux'] < 40) & (fastdata['Ux'] > -40)]
    fastdata = fastdata[(fastdata['Uy'] < 40) & (fastdata['Uy'] > -40)]
    fastdata = fastdata[(fastdata['Uz'] < 10) & (fastdata['Uz'] > -10)]
    fastdata = fastdata[(fastdata['Ts'] < 20) & (fastdata['Ts'] > -25)]
    fastdata = fastdata[(fastdata['LI_H2Om'] < 1000) & (fastdata['LI_H2Om'] > -1000)]
    freq=fastdata.index[1]-fastdata.index[0]
    fastdata = fastdata.resample(freq).mean() 


    df_WindLF=pd.DataFrame()
    df_WindLF[['LI_H2Om_Avg','LI_Pres_Avg']]=fastdata[['H2O', 'P']].groupby(pd.Grouper(freq='30min')).mean()
    slowdata=pd.DataFrame()
    slowdata['TA']=fastdata['Ts'].groupby(pd.Grouper(freq='30min')).mean()


    ###Bias correction of Water Vapour
    #Create a dataframe with the LF variables and calculate the molar density difference between RH and LI
    df_vap=pd.DataFrame()
    df_vap = pd.concat([slowdata[['TA','RH']], df_WindLF[['LI_H2Om_Avg','LI_Pres_Avg']]], axis=1)
    df_vap = df_vap.astype(float)
    df_vap['LI_Pres_Avg'] = (df_vap['LI_Pres_Avg']) * 1000 #Convert Pres units to Pa
    df_vap['es'] = 611.2 * np.exp(17.67 * df_vap['TA'] / (df_vap['TA'] + 243.5)) *(df_vap['RH']/100) #Pa
    df_vap['RH_H2Om_Avg'] = (1000 * df_vap['es'] /  (8.314 * (df_vap['TA']+273.15)))   #mmol m^-3
    df_vap['H2Om_Diff'] = df_vap['LI_H2Om_Avg'] - df_vap['RH_H2Om_Avg']
    df_vap['LI_y'] = df_vap['LI_H2Om_Avg'] / df_vap['LI_Pres_Avg'] * 1000 #mmmol m^-3 kPa^-1
    df_vap['RH_y'] = df_vap['RH_H2Om_Avg'] / df_vap['LI_Pres_Avg'] * 1000 #mmmol m^-3 kPa^-1
    print('Mean H2O concentration difference: ' + str(df_vap['H2Om_Diff']))

    #Calibration coefficients and polynomial
    A = 5.49957E3
    B = 4.00024E6
    C = -1.11280E8
    H2O_Zero = 0.8164
    H20_Span = 1.0103

    #Calculate minutal absorptance using calibration polynomial
    df_vap = df_vap.dropna()
    def polyapp(y):
        #global counter
        #counter = counter+1; print(np.round(counter/total_items*100,3), end="\r")
        p = np.poly1d([C,B,A, y])
        return p.roots[1].real

    print(df_vap.shape[0])
    total_items = df_vap.shape[0]
    counter = 0
    print('Processing large dataset (%)')
    df_vap['LI_a'] = df_vap['LI_y'].apply(lambda y: polyapp(-y)) #LI absorptance
    #print(df_vap['LI_a'])
    counter = 0
    print('Processing large dataset (%)')
    df_vap['RH_a'] = df_vap['RH_y'].apply(lambda y: polyapp(-y)) #RH absorptance
    df_vap['LI_a_raw'] = df_vap['LI_a'] * df_vap['LI_Pres_Avg']/1000 / H20_Span #LI raw absorptance
    df_vap['RH_a_raw'] = df_vap['RH_a'] * df_vap['LI_Pres_Avg']/1000 / H20_Span #RH raw absorptance
    df_vapHF = df_vap[['LI_a_raw', 'RH_a_raw']].resample('0.1S').ffill() #High-resolution absorptances
    print(df_vapHF)    
    df_vap

    #Calculate 10Hz absorptance using calibration polynomial and correct the H2O mol
    df_Wind_p = fastdata
    df_Wind_p = pd.concat([df_Wind_p,df_vapHF], axis=1) #Add 30 minutely absorptances to fast data
    df_Wind_p = df_Wind_p.dropna()
    df_Wind_p['LI_Pres'] = (df_Wind_p['P'])  * 1000 #so this is in Pa
    df_Wind_p['LI_y_fast'] = df_Wind_p['H2O'] / df_Wind_p['LI_Pres'] *1000 #mmmol m^-3 kPa^-1
    print(df_Wind_p.shape[0])
    total_items = df_Wind_p.shape[0]
    counter = 0

    print('Processing large dataset (%)')
    df_Wind_p['LI_a_fast'] = df_Wind_p['LI_y_fast'].apply(lambda y: polyapp(-y)) #LI absorptance
    df_Wind_p['LI_a_raw_fast'] = df_Wind_p['LI_a_fast'] * df_Wind_p['LI_Pres']/1000 / H20_Span #LI raw absorptance
    df_Wind_p['LI_a_corr_fast'] = ((1 - df_Wind_p['RH_a_raw']) * df_Wind_p['LI_a_raw_fast'] - df_Wind_p['LI_a_raw'] + df_Wind_p['RH_a_raw']) / (1 - df_Wind_p['LI_a_raw']) #correction of raw absorptance
    df_Wind_p['LI_a_norm_fast'] =  df_Wind_p['LI_a_corr_fast'] / df_Wind_p['LI_Pres']*1000 * H20_Span
    df_Wind_p['LI_y_norm_fast'] = A*df_Wind_p['LI_a_norm_fast']+ B*df_Wind_p['LI_a_norm_fast']**2 + C*df_Wind_p['LI_a_norm_fast']**3
    df_Wind_p['LI_H2Om_corr'] = df_Wind_p['LI_y_norm_fast'] * df_Wind_p['LI_Pres']/1000 #mmol/m^-3
    df_Wind_p['LI_H2Om_corr'] = df_Wind_p['LI_H2Om_corr'].round(1)

    print(df_Wind_p[['LI_H2Om_corr']])
    df_Wind_p = df_Wind_p.drop(columns=['LI_a_raw','RH_a_raw','LI_Pres', 'LI_y_fast', 'LI_a_fast', 'LI_a_raw_fast', 'LI_a_corr_fast', 'LI_a_norm_fast', 'LI_y_norm_fast'])
    df_Wind_p


    ###Spike correction
    print('Processing large dataset (%)')
    df_Wind_di = np.abs(df_Wind_p.rolling(window=3000, center=True).median()-df_Wind_p)
    #df_Wind_MAD = df_Wind_p.rolling(window=6000, center=True).apply(median_abs_deviation)
    df_Wind_MAD = (np.abs(df_Wind_p-df_Wind_p.rolling(window=3000, center=True).median())).rolling(window=3000, center=True).median()
    #df_Wind_di = 7 * df_Wind_MAD / 0.6745
    df_Wind_hat = np.abs(df_Wind_di) - 0.5 * (np.abs(df_Wind_di.shift(-1)) + np.abs(df_Wind_di.shift(1)))
    df_Wind_hat_MAD = df_Wind_hat / df_Wind_MAD
    df_Wind_sp = df_Wind_p[np.abs(df_Wind_hat_MAD['U'] + df_Wind_hat_MAD['V'] + df_Wind_hat_MAD['W'] + df_Wind_hat_MAD['Ts']) < 6/0.6745]
    df_Wind_sp = df_Wind_sp[df_Wind_hat_MAD['LI_H2Om_corr'] < 6/0.6745]
    df_Wind_sp = df_Wind_sp.resample('0.1S').mean() #resample to have 0.1s values
    print('Data after Spike Correction:' + str(df_Wind_sp.shape[0]))