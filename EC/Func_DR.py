import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def double_rotation(fastdata, blockdur='30T', periodwise=True, gapthresh='10T'):
    """
    Perform double rotation on the wind data in blocks.
    
    Parameters:
    fastdata (pd.DataFrame): DataFrame containing the wind data.
    blockdur (str): Duration of each block (e.g., '30T' for 30 minutes).
    periodwise (bool): Whether to consider data gaps.
    gapthresh (str): Threshold for detecting gaps (e.g., '10T' for 10 minutes).
    
    Returns:
    pd.DataFrame: DataFrame with rotated wind data.
    """
    # Convert block duration and gap threshold to Timedelta
    blockdur = pd.Timedelta(blockdur)
    gapthresh = pd.Timedelta(gapthresh)
    
    # Initialize indices
    endidcs = []
    startidcs = []
    currnanidx = 0
    
    print(f"Double rotation for blocks of {blockdur}")
    blockduridx = int(blockdur / pd.Timedelta('50ms'))
    startidx = 0
    
    if periodwise:
        gaps = detect_nan_and_gap(fastdata, gapthresh)
        nanendidcs = gaps['idx_before_gap'].values
        while startidx < len(fastdata):
            startidcs.append(startidx)
            nanendidx = fastdata.index.get_loc(nanendidcs[currnanidx])
            if currnanidx < len(nanendidcs) and nanendidx <= startidx + blockduridx - 1:
                endidcs.append(nanendidx)
                startidx = fastdata.index.get_loc(gaps['idx_after_gap'].iloc[currnanidx])
                currnanidx += 1
            else:
                endidcs.append(startidx + blockduridx - 1)
                startidx += blockduridx - 1
        endidcs[-1] = len(fastdata) - 1
        print(f"Double rotation period-wise. Considering data gaps (e.g., due to reposition).")
        print(f"{len(gaps) + 1} periods")
    else:
        print("Performing in-place Double Rotation without considering data gaps (DR over gaps as well).")
        while startidx < len(fastdata):
            startidcs.append(startidx)
            endidcs.append(startidx + blockduridx - 1)
            startidx += blockduridx - 1
        endidcs[-1] = len(fastdata) - 1
    
    for startidx, endidx in zip(startidcs, endidcs):
        datatouse = fastdata.iloc[startidx:endidx + 1]
        
        # Calculate averages
        mean_ux = datatouse['Ux'].mean()
        mean_uy = datatouse['Uy'].mean()
        mean_uz = datatouse['Uz'].mean()
        
        # Calculate first rotation angle alpha [rad]
        alpha = np.arctan2(mean_uy, mean_ux)
        
        # Rotate the windfield to obtain mean_uy=0 (Uz-component stays the same)
        data1_ux = datatouse['Ux'] * np.cos(alpha) + datatouse['Uy'] * np.sin(alpha)
        data1_uy = -datatouse['Ux'] * np.sin(alpha) + datatouse['Uy'] * np.cos(alpha)
        
        # Calculate second rotation angle beta [rad]
        beta = np.arctan2(mean_uz, data1_ux.mean())
        
        # Rotate the windfield to obtain mean_uz=0 (Uy-component stays the same)
        data2_ux = data1_ux * np.cos(beta) + datatouse['Uz'] * np.sin(beta)
        data2_uy = data1_uy
        data2_uz = -data1_ux * np.sin(beta) + datatouse['Uz'] * np.cos(beta)
        
        # Overwrite the input data
        fastdata.loc[startidx:endidx, 'Ux'] = data2_ux
        fastdata.loc[startidx:endidx, 'Uy'] = data2_uy
        fastdata.loc[startidx:endidx, 'Uz'] = data2_uz
    
    return fastdata

def detect_nan_and_gap(data, gapthresh):
    """
    Detect NaN values and gaps in the data.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    gapthresh (pd.Timedelta): Threshold for detecting gaps.
    
    Returns:
    pd.DataFrame: DataFrame with indices before and after gaps.
    """
    # Detect NaN values
    nan_indices = data.index[data.isna().any(axis=1)]
    
    # Detect gaps
    time_diffs = data.index.to_series().diff().fillna(pd.Timedelta(seconds=0))
    gap_indices = time_diffs[time_diffs > gapthresh].index
    
    # Combine NaN and gap indices
    idx_before_gap = sorted(set(nan_indices).union(set(gap_indices)))
    idx_after_gap = [data.index[data.index.get_loc(idx) + 1] for idx in idx_before_gap]
    
    return pd.DataFrame({'idx_before_gap': idx_before_gap, 'idx_after_gap': idx_after_gap})

