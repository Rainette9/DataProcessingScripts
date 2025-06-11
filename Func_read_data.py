import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob



def read_data(folder_path, fastorslow, sensor, start=None, end=None, plot_data=False, file_numbers=None):
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
    # Iterate over all files in the folder, walking through roots in order
    for root, dirs, files in sorted(os.walk(folder_path)):
        if sensor in root:
            print(f"Reading data from {root}")
            # Sort files to ensure they are read in order
            files.sort()
            for file_name in files:
                # Check if file_numbers is defined and filter files accordingly
                if file_numbers is None or any('0'+str(nums) in file_name or 'T'+str(nums) in file_name for nums in file_numbers):
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
                                match = re.search(r'_(\d+)', file_name)
                                if match:
                                    number = match[1]  # Extract the number as a string
                                    # print(f"Extracted number: {number}")
                                    units_wind = None
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
                            # if file_count <=1 or file_count >= 4:
                            # # if file_count >= 1:
                            #     file_count += 1  # Increment the counter
                            #     continue
                            print(f'reading data {file_name}')
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
        combined_data=rename_columns(combined_data)  
        unique_dates = pd.Series(combined_data.index.date).drop_duplicates().astype(str).tolist()
        print("Unique dates in the DataFrame:", unique_dates)

    # Store units in a separate attribute
    combined_data.attrs['units'] = units.to_dict()
    if sensor=='SFC' and fastorslow == 'slow':
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

def vapor_pressure_ice_MK2005(T):
    T_k = T + 273.15
    ln_esi = (-9.09718 * ((273.16 / T_k) - 1)
              - 3.56654 * np.log10(273.16 / T_k)
              + 0.876793 * (1 - (T_k / 273.16))
              + np.log10(6.1071))
    return 10**ln_esi * 100  # Pa

def vapor_pressure_liquid_MK2005(T):
    T_k = T + 273.15
    return np.exp(54.842763 - 6763.22 / T_k - 4.210 * np.log(T_k)
                  + 0.000367 * T_k
                  + np.tanh(0.0415 * (T_k - 218.8)) *
                  (53.878 - 1331.22 / T_k - 9.44523 * np.log(T_k)
                   + 0.014025 * T_k))  # Pa

def convert_RH_liquid_to_ice(RH_liquid, T):
    e_s_liquid = vapor_pressure_liquid_MK2005(T)
    e_s_ice = vapor_pressure_ice_MK2005(T)
    RH_ice = RH_liquid * (e_s_liquid / e_s_ice)
    return RH_ice

def plot_SFC_slowdata(slowdata, sensor, start, end):

    fig, ax= plt.subplots(7,1, figsize=(13,14), sharex=True)

    ax[0].plot(slowdata['TA'][start:end], label='TA', color='deepskyblue')
    ax[0].set_ylabel('Temperature [oC]')
    # ax[0].set_ylim(-45, 5)
    ax[0].plot(slowdata['SFTempK'][start:end]-273.15, label='TS', color='gold', alpha=0.8)
    ax[0].legend()


    ax[1].plot(convert_RH_liquid_to_ice(slowdata['RH'], slowdata['TA'])[start:end], label='RH', color='deepskyblue')
    ax[1].set_ylabel('RH wrt ice [%]')
    ax[1].legend()
    ax[1].set_ylim(0, 100)

    ax[2].scatter(slowdata.loc[start:end].index, slowdata['WD1'][start:end], label='WD1', s=5, color='deepskyblue')
    ax[2].scatter(slowdata.loc[start:end].index, slowdata['WD2'][start:end], label='WD2', s=5, color='limegreen')
    ax[2].set_ylabel('Wind Direction')
    ax[2].legend()
    ax[2].set_ylim(0, 360)

    ax[3].plot(slowdata['WS1_Avg'][start:end], label='WS1_Avg', color='deepskyblue')
    ax[3].plot(slowdata['WS2_Avg'][start:end], label='WS2_Avg', color='limegreen')
    ax[3].set_ylabel('Wind Speed[ms-1]')
    ax[3].legend()
    # ax[3].set_ylim(-1, 30)

    ax[4].plot(-(slowdata['SWdown1']-slowdata['SWup1'])[start:end], label='SW_net1', color='gold')
    ax[4].plot(-(slowdata['LWdown1']-slowdata['LWup1'])[start:end], label='LW_net1', color='limegreen')
    ax[4].set_ylabel('Net Radiation [Wm-2]')
    ax[4].legend()
    # ax[4].set_ylim(-400, 400)
    ax[5].plot(-(slowdata['SWdown2']-slowdata['SWup2'])[start:end], label='SW_net2', color='gold')
    ax[5].plot(-(slowdata['LWdown2']-slowdata['LWup2'])[start:end], label='LW_net2', color='limegreen')
    ax[5].set_ylabel('Net Radiation [Wm-2]')
    ax[5].legend()

    ax[6].plot(slowdata['PF_FC4'][start:end], label='PF_FC4', color='deepskyblue')
    ax[6].set_ylabel('Flowcapt [g/m2/s]')
    fig.suptitle(f'{sensor} slowdata {start} - {end}', y=0.92, fontsize=16)
    # plt.tight_layout()
    plt.savefig(f'./plots/{sensor}_{start}_slowdata.png', bbox_inches='tight')
        
    return fig, ax


def plot_fast_data(fastdata, sensor): 
    # fastdata_res = fastdata.resample('0.1s').mean()
    fig, ax= plt.subplots(4,1, figsize=(15,10))
    for column_name in fastdata.columns:
        if 'Ux' in column_name:
            ax[0].plot(fastdata[column_name], label=column_name)
            ax[0].set_ylabel('Wind Speed Ux')
        if 'Uy' in column_name:
            ax[1].plot(fastdata[column_name], label=column_name)
            ax[1].set_ylabel('Wind Speed Uy')
        if 'Uz' in column_name:
            ax[2].plot(fastdata[column_name], label=column_name)
            ax[2].set_ylabel('Wind Speed Uz')
        if 'Ts' in column_name:
            ax[3].plot(fastdata[column_name], label=column_name)
            ax[3].set_ylabel('Temperature')


    fig.suptitle(f'{sensor} fastdata')
    plt.savefig(f'./plots/{sensor}_fastdata.png')
    # plt.close()
    return fig, ax

def despike_snow_height(data, column_name='HS_Cor'):
    """
    Function to despike snow height data using a one-day moving median filter.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame containing snow height data.
    - column_name (str): The name of the column containing snow height data.
    
    Returns:
    - pd.Series: The despiked snow height data.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    # Apply a one-day moving median filter
    window = '1D'  # 1-day window
    median_filtered = data[column_name].rolling(window=window, center=True, min_periods=1).median()
    # Remove values above 2m
    median_filtered[median_filtered > 2] = np.nan
    # Interpolate NaN values
    median_filtered = median_filtered.interpolate(method='linear', limit_direction='both')
    return median_filtered


def clean_slowdata(slowdata):
    """
    Function to clean slowdata by removing outliers and renaming columns.
    """
    slowdata_cleaned = slowdata[['WD1', 'WD2', 'TA', 'RH', 'HS_Cor', 'HS_Qty', 'SFTempK', 'SWdown1', 'SWdown2', 'SWup1', 'SWup2', 'LWdown1', 'LWdown2', 'LWup1', 'LWup2', 'SWdn', 'PF_FC4', 'WS_FC4', 'WS1_Avg', 'WS2_Avg', 'WS1_Max', 'WS2_Max', 'WS1_Std', 'WS2_Std']].copy()

    # Add corresponding units from slowdata to attrs
    for column in slowdata_cleaned.columns:
        if 'units' in slowdata.attrs and column in slowdata.attrs['units']:
            slowdata_cleaned[column].attrs['units'] = slowdata.attrs['units'][column]

    slowdata_cleaned.loc[:, 'WS_FC4'] = slowdata_cleaned['WS_FC4'] / 3.6
    slowdata_cleaned['WS_FC4'].attrs['units'] = 'm/s'
    for var in ['SWdown1', 'SWdown2', 'SWup1', 'SWup2']:
        slowdata_cleaned.loc[:, var] = slowdata_cleaned[var].where(slowdata_cleaned[var] <= 1300, np.nan)
        slowdata_cleaned.loc[:, var] = slowdata_cleaned[var].where(slowdata_cleaned[var] >= -20, np.nan)
    for var in ['LWdown1', 'LWdown2', 'LWup1', 'LWup2']:
        slowdata_cleaned.loc[:, var] = slowdata_cleaned[var].where(slowdata_cleaned[var] <= 400, np.nan)
        slowdata_cleaned.loc[:, var] = slowdata_cleaned[var].where(slowdata_cleaned[var] >= 10, np.nan)
    # slowdata_cleaned = slowdata_cleaned.copy()
    slowdata_cleaned.loc[:,'SWdown1'] = slowdata_cleaned['SWdown1'].where(slowdata_cleaned['SWdown1'] > slowdata_cleaned['SWup1'], np.nan)
    slowdata_cleaned.loc[:, 'SWdown1'] = slowdata_cleaned['SWdown1'].interpolate(method='linear', limit_direction='both')
    slowdata_cleaned.loc[:,'SWdown2'] = slowdata_cleaned['SWdown2'].where(slowdata_cleaned['SWdown2'] > slowdata_cleaned['SWup2'], np.nan)
    slowdata_cleaned.loc[:, 'SWdown2'] = slowdata_cleaned['SWdown2'].interpolate(method='linear', limit_direction='both')
    slowdata_cleaned.loc[:, 'SFTempK'] = slowdata_cleaned['SFTempK'].where(slowdata_cleaned['SFTempK'] <= 283, np.nan)
    slowdata_cleaned.loc[:, 'SFTempK'] = slowdata_cleaned['SFTempK'].where(slowdata_cleaned['SFTempK'] >= 210, np.nan)
    slowdata_cleaned.loc[:, 'TA'] = slowdata_cleaned['TA'].where(slowdata_cleaned['TA'] >= -60, np.nan)
    slowdata_cleaned.loc[:, 'HS_Cor'] = slowdata_cleaned['HS_Cor'].where((slowdata_cleaned['HS_Qty'] >= 152) & (slowdata_cleaned['HS_Qty'] <= 210), np.nan)
    slowdata_cleaned.loc[:, 'HS_Cor'] = despike_snow_height(slowdata_cleaned, column_name='HS_Cor')
    slowdata_cleaned = slowdata_cleaned.drop(columns=['HS_Qty'])
    return slowdata_cleaned





def save_despiked_data(fastdata, despiked_fastdata, output_folder, sensor):
    """
    Save the despiked fast data to .dat file per hour
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Ensure the timestamp is a datetime object
    if not pd.api.types.is_datetime64_any_dtype(despiked_fastdata.index):
        despiked_fastdata.index = pd.to_datetime(despiked_fastdata.index)

    # Rename and select the desired columns
    save_fastdata = pd.DataFrame()
    if 'LI_H2Om' in despiked_fastdata.columns:
        save_fastdata[['Ux', 'Uy', 'Uz', 'Ts', 'LI_H2Om', 'LI_Pres']] = despiked_fastdata[
            ['Ux', 'Uy', 'Uz', 'Ts', 'LI_H2Om_corr', 'LI_Pres']]
    else:
        save_fastdata[['Ux', 'Uy', 'Uz', 'Ts']] = despiked_fastdata[
            ['Ux', 'Uy', 'Uz', 'Ts']]
    save_fastdata = save_fastdata[~save_fastdata.index.duplicated(keep='first')]
    save_fastdata = save_fastdata.resample('100ms').asfreq()

    # Group by hour
    for date, group in save_fastdata.groupby(pd.Grouper(freq='h')):
        print(len(group))
        print(f"Processing date: {date}")
        if len(group) == 36000:  # Check if the data consists of a full hour (100ms frequency, 36000 rows per hour)
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d_%H%M")
            file_path = os.path.join(output_folder, f"{sensor}_Fastdata_proc_{date_str}.dat")

            # Add units as the second row
            units_row = {col: fastdata.attrs['units'].get(col, '') for col in group.columns}
            group_with_units = pd.concat([pd.DataFrame([units_row], index=['units']), group])

            # Save to .dat 
            group_with_units.to_csv(file_path, sep='\t', index=True, header=True)

            print(f"Data saved per hour in {file_path}")



def read_eddypro_data(folder, sensor, qc=False):
    # Read eddypro data from subfolders of the sensor folder
    sensor_folder = os.path.join(folder, sensor)
    files = glob.glob(os.path.join(sensor_folder, '**', 'eddypro_*_full_output*.csv'), recursive=True)
    if qc==True:
        files = glob.glob(os.path.join(sensor_folder, '**', 'eddypro_*_qc_details*.csv'), recursive=True)
    print("Files found:", files)
    # Extract the first row as metadata and remove it from the dataframe
    eddypro_data = pd.concat(
        [pd.read_csv(file, header=1) for file in files],
        ignore_index=True
    )
    eddypro_data.drop(columns=eddypro_data.columns[0], inplace=True)  # Drop the first column
    # Ensure the 'date' and 'time' columns are strings before concatenation
    eddypro_data['date'] = eddypro_data['date'].astype(str)
    eddypro_data['time'] = eddypro_data['time'].astype(str)
    
    # Merge 'date' and 'time' columns and parse them into datetime format
    eddypro_data['datetime'] = pd.to_datetime(eddypro_data['date'] + ' ' + eddypro_data['time'], format='%Y-%m-%d %H:%M', errors='coerce')
    
    # Drop rows with invalid datetime values
    eddypro_data.dropna(subset=['datetime'], inplace=True)
    
    # Set the merged column as the index
    eddypro_data.set_index('datetime', inplace=True)
    dt=eddypro_data.index[1]-eddypro_data.index[0]
    eddypro_data.index=eddypro_data.index-dt
    eddypro_data.sort_index(inplace=True)
    # metadata = eddypro_data.iloc[0]  # Extract the first row as metadata
    # eddypro_data = eddypro_data.iloc[1:]  # Remove the first row from the dataframe
    if qc==False:
        eddypro_data.drop(columns=['date', 'time', 'DOY', 'daytime', 'file_records', 'used_records'], inplace=True)  # Drop unnecessary columns
    # print("Metadata extracted:", metadata.to_dict())
    eddypro_data= eddypro_data.apply(pd.to_numeric, errors='coerce')
    if 'H' in eddypro_data.columns:
        eddypro_data.loc[eddypro_data['qc_H']>=1, 'H'] = np.nan
        eddypro_data.loc[(eddypro_data['H'] > 200) | (eddypro_data['H'] < -400), 'H'] = np.nan
    if 'LE' in eddypro_data.columns:
        eddypro_data.loc[eddypro_data['qc_LE']>=1, 'LE'] = np.nan
        eddypro_data.loc[(eddypro_data['LE'] > 200) | (eddypro_data['LE'] < -200), 'LE'] = np.nan
    if 'Tau' in eddypro_data.columns:
        eddypro_data.loc[eddypro_data['qc_Tau']>=1, 'Tau'] = np.nan

    return eddypro_data