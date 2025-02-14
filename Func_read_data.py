import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re



def read_data(folder_path, fastorslow, sensor, start=None, end=None, plot_data=False):
    """
    Function to read .da data files from a folder and concatenate them into a single DataFrame. 
    Columns should be in order of 'TIMESTAMP', 'RECORD', 'Ux', 'Uy', 'Uz', 'Ts', 'diag_csat', 'LI_H2Om', 'LI_Pres', 'LI_diag' for fast data
    and in order of 'RECORD', 'rmcutcdate', 'rmcutctime', 'rmclatitude', 'rmclongitude',
       'BattV_Min', 'PTemp_Avg', 'PowerSPC', 'PowerLIC', 'PowerHtr', 'WD1',
       'WD2', 'TA', 'RH', 'HS_Cor', 'HS_Qty', 'SBTempK', 'SFTempK', 'SWdown1',
       'SWup1', 'LWdown1', 'LWup1', 'SWdown2', 'SWup2', 'LWdown2', 'LWup2',
       'SWdn', 'SensorT', 'PF_FC4', 'WS_FC4'
    When reading in the data define whether it is fast or slow data and whether you want all columns or only selected amount of columns
    """
    if fastorslow == 'fast':
        name='FAST'
        if sensor=='SFC':
            name2='TOA5_STN1fast'
        if sensor=='L':
            name2='TOA'
        if sensor=='U':
            name2='TOA'
        if sensor=='B':
            name2='Fast'
    if fastorslow == 'slow':
        name='Slow'
        if sensor=='SFC':
            name2='TOA5_STN1OneMin'
        if sensor=='L':
            name2='SLOW'
        if sensor=='U':
            name2='SLOW'
        if sensor=='B':
            name2='TOA5_STN1OneMin'
    # Initialize an empty list to store DataFrames
    data_frames = []
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if (name in file_name or name2 in file_name) and file_name.endswith('.dat'):

            file_path = os.path.join(folder_path, file_name)
            # Read the data from the file
            if fastorslow == 'slow':
                data = pd.read_csv(file_path, delimiter=',', header=1, low_memory=False)
                usecols=['TIMESTAMP', 'PTemp_Avg', 'WD1', 'WD2', 'TA', 'RH', 'HS_Cor', 'HS_Qty', 'SBTempK', 'SFTempK', 'SWdown1', 'SWup1', 'LWdown1', 'LWup1', 'SWdown2', 'SWup2', 'LWdown2', 'LWup2', 'SWdn', 'SensorT', 'PF_FC4', 'WS_FC4']
                usecols = ['']
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
            # print(data)
            # Append the DataFrame to the list
            data_frames.append(data)
                
    
    # Concatenate all DataFrames in the list
    combined_data = pd.concat(data_frames)   
    combined_data= combined_data[~combined_data.index.duplicated(keep='first')]
    combined_data = combined_data.sort_index()

    if start is not None:
        combined_data = combined_data.loc[start:]
        combined_data = combined_data.loc[:end]
    if fastorslow == 'fast':
        data=rename_columns(combined_data)
    # Store units in a separate attribute
    data.attrs['units'] = units.to_dict()

    if plot_data:
        plot_slow_data(combined_data, sensor)
    return combined_data



def extract_height_from_column(column_name):
    """
    Function to extract height from a column name.
    Returns the height if present, otherwise returns None.
    """
    pattern = re.compile(r'_(\d+)m(_\w+)?$')
    match = pattern.search(column_name)
    if match:
        return int(match.group(1))
    return None


def rename_columns(df):
    # Dictionary to store heights
    heights = {}
    # Regular expression to match columns with height suffix
    pattern = re.compile(r'_(\d+)m(_\w+)?$')
    # Iterate over the columns
    new_columns = {}
    for col in df.columns:
        match = pattern.search(col)
        if match:
            # Extract the height
            height = int(match.group(1))
            # Remove the height suffix from the column name
            new_col = pattern.sub(r'\2', col)
            # Store the height in the dictionary
            base_col = pattern.sub('', col)
            if base_col not in heights:
                heights[base_col] = []
            heights[base_col].append(height)
            # Add the new column name to the dictionary
            new_columns[col] = new_col
            # print(f"Renaming column {col} to {new_col} with height {height}")
        else:
            # If no match, keep the column name as is
            new_columns[col] = col

    # Rename the columns
    df.rename(columns=new_columns, inplace=True)
    print("Heights dictionary:", heights)
    # Store the heights as an attribute of the DataFrame
    df.attrs['heights'] = heights
    return df



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



def plot_slow_data(slowdata, sensor): 
    # Plot TA, TS, RH, WD, WS, SWdown, SWup, LWdown, LWup 
    fig, ax= plt.subplots(6,1, figsize=(15,10))
    for column_name in slowdata.columns:
        if 'TA' in column_name or 'Temp' in column_name:
            if 'PTemp' in column_name:
                continue    
            ax[0].plot(slowdata[column_name], label=column_name)
            ax[0].set_ylabel('Temperature')
            ax[0].legend()
        if 'RH' in column_name:
            ax[1].plot(slowdata[column_name], label=column_name)
            ax[1].set_ylabel('Relative Humidity')
            ax[1].legend()
        if 'WD' in column_name:
            ax[2].plot(slowdata[column_name], label=column_name)
            ax[2].set_ylabel('Wind Direction')
            ax[2].legend()
        if 'WS' in column_name:
            if 'Max' in column_name or 'Std' in column_name:
                continue
            ax[3].plot(slowdata[column_name], label=column_name)
            ax[3].set_ylabel('Wind Speed')
            ax[3].legend()
        if 'SWdown' in column_name or 'Incoming_SW' in column_name:
            ax[4].plot(slowdata[column_name], label=column_name)
            ax[4].set_ylabel('Shortwave Radiation')
            ax[4].legend()
        if 'SWup' in column_name or 'Outgoing_SW' in column_name:
            ax[4].plot(slowdata[column_name], label=column_name)
            ax[4].set_ylabel('Shortwave Radiation')
            ax[4].legend()
        if 'LWdown' in column_name or 'Incoming_LW' in column_name or 'UW' in column_name:
            ax[5].plot(slowdata[column_name], label=column_name)
            ax[5].set_ylabel('Longwave Radiation')
            ax[5].legend()
        if 'LWup' in column_name or 'Outgoing_LW' in column_name:
            ax[5].plot(slowdata[column_name], label=column_name)
            ax[5].set_ylabel('Longwave Radiation')
            ax[5].legend()


    fig.suptitle(f'{sensor} slowdata')
    plt.savefig(f'{sensor}_slowdata.png')
        
    return fig, ax