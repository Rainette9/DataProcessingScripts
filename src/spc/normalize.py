import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bins = np.array([36,46,53,60,67,74,81,88,95,102,109,116,123,130,137,144,151,158,165,172,179,186,193,201,208,215,222,229,236,243,250,257,264,271,278,285,292,300,307,314,321,328,335,342,349,356,364,371,378,385,392,399,406,414,421,428,435,442,449,456,464,471,478,490])
slopes = np.array([[-0.9434,412.77],[-0.9636,371.09],[-0.9885,285.26],[-1.0831,197.83],[-1.4838,93.839]])
bins_str = list(bins.astype(str))

def diam(temperature,slope): #Compute 5 slopes of datasheet
    return slopes[slope,0]*temperature + slopes[slope,1]

def diam2(temperature,slope,bin_size):# Returns bin size at -30 for specific original temperature and slope
    return bin_size - slopes[slope,0]*(temperature + 30)
    
def binCorrection(temperature):# Returns correction table for specific temperature
    newbins = []
    y = diam(temperature,range(5))
    for bin in bins:
        slope = np.argmin(np.abs(y-bin))
        corrected_bin = diam2(temperature,slope,bin)
        corrected_bin = bins[np.argmin(np.abs(corrected_bin-bins))]
        newbins.append(str(corrected_bin))
    return newbins

def getCorrectionTable(): # Returns correction table for all bins and temperatures
    bin_corrected = pd.DataFrame(columns=['Temperature'] + bins_str)
    for temp in np.linspace(-38,0,381): #Discretize every 0.1C
        # if temp < -30 or temp > 0: # Only correct between -30 and 0
        #     bin_corrected.loc[len(bin_corrected)] = [temp] 
        #     temp = round(temp, 1)
        # else:      
        temp = round(temp, 1)
        correction = binCorrection(temp)
        bin_corrected.loc[len(bin_corrected)] = [temp] + correction
    bin_corrected.set_index('Temperature', inplace=True)
    return bin_corrected


def tempCorrection(raw_data):
    conversions_df = getCorrectionTable() #Get conversion table
    raw_data['temp_round'] = raw_data['Temperature(C)'].round(1) #Discretize temperature
    grouped_raw_data = raw_data.groupby('temp_round',sort=False) # Group by discretized temperature
    corrected_data = grouped_raw_data.apply(correctTempBloc,conversions_df) #Apply correction by bloc of temperatures
    return corrected_data.reset_index(level='temp_round',drop=True)
    
def correctTempBloc(group_temp,conversions_df): #Apply correction to all lines that have same temperature
    convert = conversions_df.loc[group_temp.iloc[0]['temp_round'] ] # Select temp specific conversion table
    new_group_temp = group_temp.copy()
    new_group_temp[bins_str] = 0
    for bin_str in bins_str:
        new_group_temp[convert[bin_str]] += group_temp[bin_str]
        
    return new_group_temp
    
def computeMassFlux(SPC):
    A = 0.00005 
    rho = 920
    mass = 4/3*np.pi*1e-18*np.power(bins,3)/8*rho # in kg
    SPC['Corrected Mass Flux(kg/m^2/s)'] = (mass*SPC[bins_str].values).sum(axis=1)/A
    return SPC
    
def getNormalizedData(SPC_filenames, slowdata, OneMin_filenames=None): # correct temperature and particle sizes
    
    #SPC
    SPC_CSVS = []
    for file in sorted(SPC_filenames): # Sort files before reading
        df = pd.read_csv(file, sep='\t', header=0)
        SPC_CSVS.append(df)
    SPC = pd.concat(SPC_CSVS,ignore_index=True) #Merge
    SPC['Time(UTC)'] = pd.to_datetime(SPC['Time(UTC)'], format="%d.%m.%Y %H:%M:%S") # To datetime format
    # SPC.dropna(subset=['Mass Flux(g/cm^2/sec)'],inplace=True) #Drop all empty data lines
    SPC.set_index('Time(UTC)',inplace=True)

    #OneMin
    # ONEMIN_CSVS = []
    # for file in OneMin_filenames:
    #     df=pd.read_csv(file,skiprows=1,usecols=['TIMESTAMP','TA'])
    #     df.drop([0,1],inplace=True)
    #     ONEMIN_CSVS.append(df)
    # OneMin = pd.concat(ONEMIN_CSVS,ignore_index=True)
    # OneMin['TIMESTAMP'] = pd.to_datetime(OneMin['TIMESTAMP']) #Convert format
    # OneMin['TA'] = OneMin['TA'].astype(float)
    # OneMin.set_index('TIMESTAMP',inplace=True)
    OneMin=slowdata[['TA']].copy() # Use slowdata for temperature, as it is more accurate


    #Correct temperature
    SPC['Time_min'] = SPC.index.floor('min')
    SPC = SPC.join(OneMin['TA'],how='left',on='Time_min')
    SPC['Temperature(C)'] = SPC['TA']
    SPC = tempCorrection(SPC)
    computeMassFlux(SPC)
    SPC.drop(columns=['TA','Total Second','Temperature(C)','Time_min','temp_round','Time(Julian)'],inplace=True) #Keep only important data
    SPC.sort_index(inplace=True)
    return SPC

def getRawData(SPC_filenames): # Returns raw data (only bins counts and flux mass)
    SPC_CSVS = []
    for file in sorted(SPC_filenames): # Sort files before reading
        df = pd.read_csv(file, sep='\t', header=0)
        SPC_CSVS.append(df)
    SPC = pd.concat(SPC_CSVS,ignore_index=True) #Merge
    # SPC['Time(UTC)'] = pd.to_datetime(SPC['Time(UTC)'], format="%d.%m.%Y %H:%M:%S") # To datetime format
    SPC['Time(UTC)'] = pd.to_datetime(SPC['Time(UTC)'], format="%Y-%m-%d %H:%M:%S") # To datetime format
    # SPC.drop(columns=['Total Second','Time(Julian)'],inplace=True) #Keep only important data
    SPC.set_index('Time(UTC)',inplace=True)
    SPC.sort_index(inplace=True)
    return SPC
