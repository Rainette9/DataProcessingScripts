import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def read_data(folder_path, fastorslow, shortorlong):
    """
    Function to read .dat data files from a folder and concatenate them into a single DataFrame. 
    Columns should be in order of 'TIMESTAMP', 'RECORD', 'Ux', 'Uy', 'Uz', 'Ts', 'diag_csat', 'LI_H2Om', 'LI_Pres', 'LI_diag' for fast data
    and in order of 'RECORD', 'rmcutcdate', 'rmcutctime', 'rmclatitude', 'rmclongitude',
       'BattV_Min', 'PTemp_Avg', 'PowerSPC', 'PowerLIC', 'PowerHtr', 'WD1',
       'WD2', 'TA', 'RH', 'HS_Cor', 'HS_Qty', 'SBTempK', 'SFTempK', 'SWdown1',
       'SWup1', 'LWdown1', 'LWup1', 'SWdown2', 'SWup2', 'LWdown2', 'LWup2',
       'SWdn', 'SensorT', 'PF_FC4', 'WS_FC4'
    When reading in the data define whether it is fast or slow data and whether you want all columns or only selected amount of columns
    """
    if fastorslow == 'fast':
        name='*FAST*.dat'
    if fastorslow == 'slow':
        name='*SLOW*.dat'

    # Initialize an empty list to store DataFrames
    data_frames = []
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith(name) and file_name.endswith('.dat'):
            file_path = os.path.join(folder_path, file_name)
            # Read the data from the file
            if shortorlong == 'short' and fastorslow == 'slow':
                data = pd.read_csv(file_path, delimiter=',', header=1, low_memory=False, usecols=['TIMESTAMP', 'PTemp_Avg', 'WD1', 'WD2', 'TA', 'RH', 'HS_Cor', 'HS_Qty', 'SBTempK', 'SFTempK', 'SWdown1', 'SWup1', 'LWdown1', 'LWup1', 'SWdown2', 'SWup2', 'LWdown2', 'LWup2', 'SWdn', 'SensorT', 'PF_FC4', 'WS_FC4'])
            else:
                data = pd.read_csv(file_path, delimiter=',', header=1, low_memory=False)
            # Read the units from the second row
            units = pd.read_csv(file_path, delimiter=',', header=1, nrows=1).iloc[0]
            # Drop the second and third rows with units and empty strings
            data = data.drop([0, 1])
            data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='mixed')
            data.set_index('TIMESTAMP', inplace=True)
             # Convert all columns to numeric, coercing errors to NaN
            data = data.apply(pd.to_numeric, errors='coerce')
            # Append the DataFrame to the list
            data_frames.append(data)
                
    
    # Concatenate all DataFrames in the list
    combined_data = pd.concat(data_frames)   
    # Store units in a separate attribute
    data.attrs['units'] = units.to_dict()
    return data

def read_data_OneMin(folder_path):
    data_frames = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('TOA5_STN1OneMin') and file_name.endswith('.dat'):
            file_path = os.path.join(folder_path, file_name)
            # Read the data from the file
            data = pd.read_csv(file_path, delimiter=',', header=1, low_memory=False)
            # Read the units from the second row
            units = pd.read_csv(file_path, delimiter=',', header=1, nrows=1).iloc[0]
            # Drop the second and third rows with units and empty strings
            data = data.drop([0, 1])
            data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='mixed')
            data.set_index('TIMESTAMP', inplace=True)
             # Convert all columns to numeric, coercing errors to NaN
            data = data.apply(pd.to_numeric, errors='coerce')
            # Append the DataFrame to the list
            data_frames.append(data)
    
    # Concatenate all DataFrames in the list
    combined_data = pd.concat(data_frames)   
    # Store units in a separate attribute
    data.attrs['units'] = units.to_dict()
    return data

def read_data_wind(file_path):
    # Read the data from the file
    data = pd.read_csv(file_path)
    return data

