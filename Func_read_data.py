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
        name=['Fast', 'FAST', 'fast']
    if fastorslow == 'slow':
        name=['Slow', 'SLOW', 'OneMin']
    # Initialize an empty list to store DataFrames
    data_frames = []
    file_count=0
    # Iterate over all files in the folder
    for root, dirs, files in os.walk(folder_path):
        if sensor in root:
            for file_name in files:            
                if file_count >= 1000:  # Stop after processing 100 files
                    break
                # Check if the file name contains the sensor name and ends with .dat
                if any(n in file_name for n in name) and file_name.endswith('.dat'):
                    print(file_name)
                    file_path = os.path.join(root, file_name)
                    # Read the data from the file
                    if fastorslow == 'slow':
                        data = pd.read_csv(file_path, delimiter=',', header=1, low_memory=False)
                        data = data.drop([0, 1])
                        #open and append the wind file
                        if sensor=='SFC':
                            file_path = os.path.join(root, file_name)
                            match = re.search(r'_(\d+)\.dat$', file_name)
                            if match:
                                number = match.group(1)  # Extract the number as a string

                                # Search for a file with 'wind' and the same number in the name
                                for wind_file in files:
                                    if f'wind_{number}' in wind_file and wind_file.endswith('.dat'):
                                        wind_file_path = os.path.join(root, wind_file)
                                        # Open and process the wind file
                                        wind_data = pd.read_csv(wind_file_path, delimiter=',', header=1, low_memory=False)
                                        wind_data = wind_data.drop([0, 1])
                                        wind_data['TIMESTAMP'] = pd.to_datetime(wind_data['TIMESTAMP'], format='mixed')
                                        data = data.join(wind_data, how='left', rsuffix='_wind')

                            units_wind = pd.read_csv(wind_file_path, delimiter=',', header=1, nrows=1).iloc[0]
                                        
                    if fastorslow == 'fast':
                        data = pd.read_csv(file_path, delimiter=',', header=1, low_memory=False)
                        data = data.drop([0, 1])
                    file_count += 1  # Increment the counter
                    # Read the units from the second row
                    units = pd.read_csv(file_path, delimiter=',', header=1, nrows=1).iloc[0]
                    # Drop the second and third rows with units and empty strings
                    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='mixed')
                    data.set_index('TIMESTAMP', inplace=True)
                     # Convert all columns to numeric, coercing errors to NaN
                    data = data.apply(pd.to_numeric, errors='coerce')
                    # Append the DataFrame to the list
                    data_frames.append(data)
                
    
    # Concatenate all DataFrames in the list
    combined_data = pd.concat(data_frames)   
    combined_data= combined_data[~combined_data.index.duplicated(keep='first')]
    combined_data = combined_data.sort_index()
    if fastorslow == 'slow':
        combined_data = combined_data.resample('1min').mean()

    if start is not None:
        combined_data = combined_data.loc[start:]
        combined_data = combined_data.loc[:end]
    if fastorslow == 'fast':
        data=rename_columns(combined_data)
    # Store units in a separate attribute
    combined_data.attrs['units'] = units.to_dict()
    if units_wind is not None:
        combined_data.attrs['units_wind'] = units_wind.to_dict()


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





def plot_slow_data(slowdata, sensor): 
    # Plot TA, TS, RH, WD, WS, SWdown, SWup, LWdown, LWup 
    fig, ax= plt.subplots(6,1, figsize=(15,10))
    for column_name in slowdata.columns:
        if 'TA' in column_name or 'Temp' in column_name:
            if 'PTemp' in column_name or 'SBTemp' in column_name:
                continue    
           
            else:
                ax[0].plot(slowdata[column_name], label=column_name)
                ax[0].set_ylabel('Temperature')
                ax[0].legend()
                ax[0].set_ylim(-50, 10)
                if 'SFTemp' in column_name:
                    ax[0].plot(slowdata[column_name]-273.15, label=column_name, linestyle='--')
                # ax[0].plot(slowdata[column_name]-273.15, label=column_name, linestyle='--')           

        if 'RH' in column_name:
            ax[1].plot(slowdata[column_name], label=column_name)
            ax[1].set_ylabel('Relative Humidity')
            ax[1].legend()
        if 'WD' in column_name:
            ax[2].plot(slowdata[column_name], label=column_name)
            ax[2].set_ylabel('Wind Direction')
            ax[2].legend()
            ax[2].set_ylim(0, 360)
        if 'WS' in column_name:
            if 'Max' in column_name or 'Std' in column_name:
                continue
            ax[3].plot(slowdata[column_name], label=column_name)
            ax[3].set_ylabel('Wind Speed')
            ax[3].legend()
            ax[3].set_ylim(-10, 40)
        if 'SWdown' in column_name or 'Incoming_SW' in column_name:
            ax[4].plot(slowdata[column_name], label=column_name)
            # ax[4].set_ylabel('Shortwave Radiation')
            ax[4].legend()
        if 'SWup' in column_name or 'Outgoing_SW' in column_name:
            ax[4].plot(slowdata[column_name], label=column_name)
            ax[4].set_ylabel('Shortwave Radiation')
            ax[4].legend()
            ax[4].set_ylim(0,1200)
        if 'LWdown' in column_name or 'Incoming_LW' in column_name or 'UW' in column_name:
            ax[5].plot(slowdata[column_name], label=column_name)
            # ax[5].set_ylabel('Longwave Radiation')
            ax[5].legend()
        if 'LWup' in column_name or 'Outgoing_LW' in column_name:
            ax[5].plot(slowdata[column_name], label=column_name)
            ax[5].set_ylabel('Longwave Radiation')
            ax[5].legend()
            ax[5].set_ylim(50,400)


    fig.suptitle(f'{sensor} slowdata')
    plt.savefig(f'./plots/{sensor}_slowdata.png')
        
    return fig, ax


def plot_SFC_slowdata(slowdata, sensor, start_time, end_time):

    # Plot TA, TS, RH, WD, WS, SWdown, SWup, LWdown, LWup 
    fig, ax= plt.subplots(6,1, figsize=(15,10), sharex=True)

    ax[0].plot(slowdata['TA'], label='TA')
    ax[0].set_ylabel('Temperature [oC]')
    ax[0].legend()
    ax[0].set_ylim(-50, 10)
    ax[0].plot(slowdata['SFTempK']-273.15, label='TS', linestyle='--')

    ax[1].plot(slowdata['RH'], label='RH')
    ax[1].set_ylabel('Relative Humidity')
    ax[1].legend()
    ax[1].set_ylim(0, 100)

    ax[2].plot(slowdata['WD1'], label='WD1')
    ax[2].plot(slowdata['WD2'], label='WD2')
    ax[2].set_ylabel('Wind Direction')
    ax[2].legend()
    ax[2].set_ylim(0, 360)

    ax[3].plot(slowdata['WS1_Avg'], label='WS1_Avg')
    ax[3].plot(slowdata['WS2_Avg'], label='WS2_Avg')
    ax[3].set_ylabel('Wind Speed[ms-1]')
    ax[3].legend()
    ax[3].set_ylim(-10, 40)

    ax[4].plot(slowdata[column_name], label=column_name)
    # ax[4].set_ylabel('Shortwave Radiation')
    ax[4].legend()
    ax[4].plot(slowdata[column_name], label=column_name)
    ax[4].set_ylabel('Shortwave Radiation')
    ax[4].legend()
    ax[4].set_ylim(0,1200)
    ax[5].plot(slowdata[column_name], label=column_name)
    # ax[5].set_ylabel('Longwave Radiation')
    ax[5].legend()
    ax[5].plot(slowdata[column_name], label=column_name)
    ax[5].set_ylabel('Longwave Radiation')
    ax[5].legend()
    ax[5].set_ylim(50,400)


    fig.suptitle(f'{sensor} slowdata', y=0.95)
    plt.savefig(f'./plots/{sensor}_slowdata.png')
        
    return fig, ax


def plot_fast_data(fastdata, sensor): 
    # fastdata_res = fastdata.resample('0.1s').mean()
    fig, ax= plt.subplots(4,1, figsize=(15,10))
    for column_name in fastdata_res.columns:
        if 'Ux' in column_name:
            ax[0].plot(fastdata_res[column_name], label=column_name)
            ax[0].set_ylabel('Wind Speed Ux')
        if 'Uy' in column_name:
            ax[1].plot(fastdata_res[column_name], label=column_name)
            ax[1].set_ylabel('Wind Speed Uy')
        if 'Uz' in column_name:
            ax[2].plot(fastdata_res[column_name], label=column_name)
            ax[2].set_ylabel('Wind Speed Uz')
        if 'Ts' in column_name:
            ax[3].plot(fastdata_res[column_name], label=column_name)
            ax[3].set_ylabel('Temperature')


    fig.suptitle(f'{sensor} fastdata')
    plt.savefig(f'./plots/{sensor}_fastdata.png')
        
    return fig, ax