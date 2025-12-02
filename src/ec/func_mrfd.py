import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt


def detect_gaps(data, threshold):
    """Detects gaps in the data where time difference exceeds threshold."""
    time_diffs = data.index.to_series().diff().dt.total_seconds()
    gap_indices = np.where(time_diffs > threshold.total_seconds())[0]
    gaps = pd.DataFrame({'idx_before_gap': gap_indices, 
                         'time_before_gap': data.iloc[gap_indices].index.values, 
                         'gaplength': time_diffs.iloc[gap_indices]})
    return gaps

def moving_average(data, window_size):
    """Computes a simple moving average while handling NaNs."""
    data_series = pd.Series(data)
    return data_series.rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()


def mrd(data_a, data_b, M, Mx):
    """
    NEW Orthogonal Multiresolution Flux Decomposition.
    Adapted from Ivana Stiperski's code.
    See Vickers&Mahrt 2003 and Howell&Mahrt 1997. With uncertainty estimation.
    """
    D = np.zeros(M - Mx)
    Dstd = np.zeros_like(D)
    data_a2 = np.copy(data_a)
    data_b2 = np.copy(data_b)
    
    for ims in range(M - Mx + 1):
        ms = M - ims
        l = 2 ** ms
        nw = int(round((2 ** M) / l))
        wmeans_a = np.zeros(nw)
        wmeans_b = np.zeros(nw)
        
        for i in range(nw):
            k = int(round((i) * l))
            wmeans_a[i] = np.nanmean(data_a2[k:k+l]) #changed mean to nanmean
            wmeans_b[i] = np.nanmean(data_b2[k:k+l]) #changed mean to nanmean
            data_a2[k:k+l] -= wmeans_a[i]
            data_b2[k:k+l] -= wmeans_b[i]
        
        if nw > 1:
            D[ms] = np.nanmean(wmeans_a * wmeans_b) #changed mean to nanmean
            Dstd[ms] = np.nanstd(wmeans_a * wmeans_b, ddof=1)  #changed mean to nanmean
    
    return D, Dstd


def completemrd(data, col1, col2, M, shift, normed=False, plot=False):
    """"""
    print("MRD for DataFrame")
    timestep = data.index[1] - data.index[0]
    checktime = data.index[len(data) // 2 + 1] - data.index[len(data) // 2]
    if timestep != checktime:
        print("Warning: Timestep and check-timestep do not agree! Careful!")
    
    blocklength = 2**M * timestep
    timeshift = shift * timestep
    
    gaps = detect_gaps(data, timedelta(seconds=10))
    # Create additional row with explicit dtypes to avoid FutureWarning
    additional_gap = pd.DataFrame({
        'idx_before_gap': [len(data)], 
        'time_before_gap': [data.index[-1] + timedelta(seconds=1)], 
        'gaplength': [timedelta(seconds=99)]
    })
    # Ensure dtypes match if gaps is not empty
    if not gaps.empty:
        additional_gap = additional_gap.astype(gaps.dtypes)
    gaps = pd.concat([gaps, additional_gap], ignore_index=True)
    print("Number of gaps:", len(gaps))
    
    fx = np.ones(len(data))
    if normed:
        tmp_fx = moving_average(data[col1] * data[col2], 2**11)
        fx = moving_average(tmp_fx, 2**M)
    
    mrd_x = np.array([(2**i) * timestep for i in range(1, M+1)])
    data_cont_mrd = []
    time_middle = []
    
    startidx, endidx, gapidx, nrblocks, normfct = 0, 2**M, 0, 0, 0
    
    with tqdm(total=len(data)) as pbar:
        while endidx + shift <= len(data):
            if nrblocks != 0:
                startidx += shift
                endidx += shift
                # print(nrblocks, '!=0')
            
            if gapidx < len(gaps) and endidx <= gaps.iloc[gapidx, 0]:
                datatouse1 = data[col1].iloc[startidx:endidx].to_numpy()
                datatouse2 = data[col2].iloc[startidx:endidx].to_numpy()
                time_middle.append(data.index[startidx + (endidx - startidx) // 2])
                
                (mrd_data_tmp, mrd_data_std) = mrd(datatouse1, datatouse2, M, 0)
                if normed:
                    normfct = np.sum(mrd_data_tmp[:11]) / fx[startidx + (endidx - startidx) // 2]
                    mrd_data_tmp /= normfct
                
                data_cont_mrd.append(mrd_data_tmp)
                # print(mrd_data_tmp)
                nrblocks += 1
                # print(gapidx, '<', len(gaps), 'and', endidx, '<=', gaps.iloc[gapidx, 0])

            else:
                startidx = gaps.iloc[gapidx]['idx_before_gap'] + 1 - shift
                endidx = startidx + (2**M) - 1

                theonrstepsyet = round(1 + ((data.index[startidx + shift] - data.index[0] - blocklength).total_seconds() / timeshift.total_seconds()))
                nrstepsyet = len(time_middle)

                print("Theoretical steps yet:", theonrstepsyet)
                print("Number of steps yet:", nrstepsyet)

                nrstepstodo = theonrstepsyet - nrstepsyet

                for istep in range(nrstepstodo):
                    if istep == 0 and len(time_middle) == 0:
                        time_middle.append(data['time'].iloc[0] + timeshift)
                    else:
                        time_middle.append(time_middle[-1] + timeshift)
                    print('concatenating')
                    # print(data_cont_mrd, np.full((M, 1), np.nan))
                    data_cont_mrd = np.concatenate((data_cont_mrd, np.full((M, 1), np.nan)), axis=1)
                    nrblocks += 1
                
                gapidx += 1

            pbar.update(shift)

    if plot==True:
        mrd_data=np.array(data_cont_mrd).T
        seconds_array = np.vectorize(lambda td: td.total_seconds())(mrd_x)

        fig, ax = plt.subplots()

        # Set title dynamically using first and last time values from evaldf1
        ax.set_title(f"MRD {data.index[0]} - {data.index[-1]}")
        ax.set_xlabel("avg. time [s]")
        ax.set_ylabel(r"$C_{w\theta} [\cdot 10^{-3} \mathrm{Kms^{-1}}]$")
        ax.grid(True)

        ax.set_xscale("log")

        # Plot the median MRD values
        ax.plot(np.array(seconds_array) ,(np.nanmedian(mrd_data, axis=1))*1000)

        # Fill between the quantiles
        ax.fill_between(np.array(seconds_array), 
                        np.nanquantile(mrd_data, 0.25, axis=1) * 1000, 
                        np.nanquantile(mrd_data, 0.75, axis=1) * 1000, 
                        alpha=0.4)

        plt.show()
    
    return mrd_x, np.array(data_cont_mrd).T, np.array(time_middle)

def plot_mrd(mrd_x, mrd_data, title="MRD", xlabel="avg. time [s]", ylabel=r"$C_{w\theta} [\cdot 10^{-3} \mathrm{Kms^{-1}}]$", ax=None, label=None, color=None, alpha=0.25):
    """
    Plot MRD results with median and interquartile range.
    
    Parameters:
    -----------
    mrd_x : array-like
        Time scale array (timedeltas)
    mrd_data : array-like
        MRD data (2D array)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes
    label : str, optional
        Label for the plot legend
    color : str, optional
        Color for the line and fill
    alpha : float, optional
        Transparency for the fill area (default 0.25)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    seconds_array = np.vectorize(lambda td: td.total_seconds())(mrd_x)

    # Create new figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        fig = ax.get_figure()
        show_plot = False
    
    # Set labels and formatting
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_xscale("log")

    # Plot the median MRD values
    line = ax.plot(np.array(seconds_array), 
                   (np.nanmedian(mrd_data, axis=1))*1000, 
                   label=label, 
                   color=color)
    
    # Get the color from the line if not specified
    if color is None:
        color = line[0].get_color()

    # Fill between the quantiles
    ax.fill_between(np.array(seconds_array), 
                    np.nanquantile(mrd_data, 0.25, axis=1) * 1000, 
                    np.nanquantile(mrd_data, 0.75, axis=1) * 1000, 
                    alpha=alpha,
                    color=color)
    
    # Add legend if labels are present
    if label is not None:
        ax.legend()
    
    # Only show if we created the figure
    if show_plot:
        plt.show()
    
    return fig, ax


def mrdpp(mrd_x, mrd):
    """Post-processing of 'completemrd' to obtain quantiles, median, etc."""
    mrd_D_median = np.nanmedian(mrd, axis=1)
    mrd_D_min = np.nanmin(mrd, axis=1)
    mrd_D_max = np.nanmax(mrd, axis=1)
    mrd_D_quantile1 = np.nanquantile(mrd, 0.25, axis=1)
    mrd_D_quantile3 = np.nanquantile(mrd, 0.75, axis=1)
    
    return mrd_x, mrd_D_median, mrd_D_min, mrd_D_max, mrd_D_quantile1, mrd_D_quantile3
