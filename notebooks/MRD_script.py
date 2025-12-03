import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from scipy.ndimage import median_filter
import sys
import os
import struct
import pywt
from pathlib import Path
import pickle
from multiprocessing import Pool, cpu_count

# Add project 'src' directory to sys.path when running from the notebooks/ folder
# (notebooks/ is expected to be inside the repo; repo_root = parent of cwd)
repo_root = Path.cwd().parent
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Import modules from the installed package. If this fails, the editable install
# (`pip install -e .`) may be missing or kernel needs restart.
try:
    from ec.func_read_data import *
    from mo.func_mo import *
    from spc.normalize import *
    from plotting.funcs_plots import *
    from ec.sensor_info import *
    from ec.func_dr import *
    from ec.func_mrfd import *
except Exception as e:
    print('Package import failed:', e)
    print('Make sure you ran `pip install -e .` (editable install) and restart the kernel, or that src/ exists at:', src_path)

print('All imports successful!')

# ========== CONFIGURATION ==========
# Configure which heights to analyze
main_folder= '/capstor/scratch/cscs/rengbers/ec_data/'
SFC_height_folder = main_folder + 'SFC_DR/'
_16m_height_folder = main_folder + 'CSAT_16m_DR/'
_26m_height_folder = main_folder + 'CSAT_26m_DR/'
print(f"Data folder for SFC height: {SFC_height_folder}")

# ========== FILTER CONFIGURATION ==========
# Choose ONE filter type: 'wind_direction', 'stability', or 'blowing_snow'
filter_type = 'wind_direction'  # Options: 'wind_direction', 'stability', 'blowing_snow'

# Wind Direction Filter (only used if filter_type = 'wind_direction')
# Available files: 'wd1_45_90_periods.pkl', 'wd1_90_180_periods.pkl'
wind_direction_filters = {
    'synoptic': 'wd1_45_100_periods.pkl',
    'katabatic': 'wd1_110_180_periods.pkl'
}

# Stability Filter (only used if filter_type = 'stability')
# Available files: 'stability_0_02_periods.pkl', 'stability_02_10_periods.pkl', 'stability_10_01_periods.pkl'.
stability_filters = {
    'neutral': 'stability_01_01_periods.pkl',
    'stable': 'stability_015_10_periods.pkl',
    'unstable': 'stability_10_01_periods.pkl'
}

# Blowing Snow Filter (only used if filter_type = 'blowing_snow')
# This uses the find_consecutive_periods function from plotting.funcs_plots
blowing_snow_filters = {
    'with_BS': 'BS_periods.pkl',
    'without_BS': 'noBS_periods.pkl'
}

# Color scheme for different categories
color_scheme = {
    # Wind direction colors
    'synoptic': {'main': '#2E86AB', 'fill': '#2E86AB', 'alpha': 0.25},
    'katabatic': {'main': '#A23B72', 'fill': '#A23B72', 'alpha': 0.25},
    # Stability colors
    'neutral': {'main': '#27AE60', 'fill': '#27AE60', 'alpha': 0.25},
    'stable': {'main': '#E67E22', 'fill': '#E67E22', 'alpha': 0.25},
    'unstable': {'main': '#E74C3C', 'fill': '#E74C3C', 'alpha': 0.25},
    # Blowing snow colors
    'with_BS': {'main': '#8E44AD', 'fill': '#8E44AD', 'alpha': 0.25},
    'without_BS': {'main': '#16A085', 'fill': '#16A085', 'alpha': 0.25}
}

# ====================================
for filter_type in ['wind_direction']:
    # Load event data based on filter type
    events_folder = '/capstor/scratch/cscs/rengbers/DataProcessingScripts/events/'
    events_data = {}

    if filter_type == 'wind_direction':
        filters_to_use = wind_direction_filters
        colors = {k: color_scheme[k] for k in wind_direction_filters.keys()}
    elif filter_type == 'stability':
        filters_to_use = stability_filters
        colors = {k: color_scheme[k] for k in stability_filters.keys()}
    elif filter_type == 'blowing_snow':
        filters_to_use = blowing_snow_filters
        colors = {k: color_scheme[k] for k in blowing_snow_filters.keys()}
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}. Choose 'wind_direction', 'stability', or 'blowing_snow'")

    # Load period files for wind_direction and stability
    if filter_type in ['wind_direction', 'stability', 'blowing_snow']:
        for category, filename in filters_to_use.items():
            filepath = events_folder + filename
            try:
                with open(filepath, 'rb') as f:
                    events_data[category] = pickle.load(f)
                print(f"Loaded {category}: {len(events_data[category])} periods from {filename}")
            except FileNotFoundError:
                print(f"Warning: {filepath} not found. Skipping {category}.")
                events_data[category] = []


    # Store all MRD results for each category
    mrd_results = {category: {'mrd_x': [], 'mrd_data': []} 
                    for category in filters_to_use.keys()}

    # Define worker function for parallel processing
    def process_single_period(args):
        """Process a single period and return MRD results."""
        period_start, period_end, category, SFC_height_folder = args
        start_str = period_start.strftime('%Y-%m-%d_%H:%M:%S')
        end_str = period_end.strftime('%Y-%m-%d_%H:%M:%S')
        
        try:
            fastdata = load_fastdata(SFC_height_folder, start_str, end_str)
            M = 17
            (mrd_x_temp, mrd_data_temp, time_middle) = completemrd(
                fastdata, 'Uz', 'Ts', M, 
                shift=round(int(0.1 * 2**M)), 
                plot=False
            )

            return {
                'category': category,
                'mrd_x': mrd_x_temp,
                'mrd_data': mrd_data_temp,
                'success': True,
                'period': (period_start, period_end)
            }
        except Exception as e:
            return {
                'category': category,
                'success': False,
                'error': str(e),
                'period': (period_start, period_end)
            }

    # Prepare all tasks for parallel processing
    all_tasks = []
    for category in filters_to_use.keys():
        periods = events_data.get(category, [])
        for period_start, period_end in periods:
            all_tasks.append((period_start, period_end, category, SFC_height_folder))
    
    print(f"\nProcessing {len(all_tasks)} total periods across all categories...")
    num_cpus=64
    print(f"Using {num_cpus} CPU cores for parallel processing\n")
    
    # Process periods in parallel
    with Pool(processes=num_cpus) as pool:  # Use only 4 cores
        results = pool.map(process_single_period, all_tasks)
        
        # Save results to pickle file
        output_filename = f'mrd_results_{filter_type}.pkl'
        with open(output_filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {output_filename}")

    
    # Organize results by category
    for result in results:
        if result['success']:
            category = result['category']
            mrd_results[category]['mrd_x'].append(result['mrd_x'])
            mrd_results[category]['mrd_data'].append(result['mrd_data'])
            print(f"✓ {category}: {result['period'][0]} to {result['period'][1]}")
        else:
            print(f"✗ {result['category']}: Error - {result['error']}")
    
    # Print summary
    print("\n" + "="*60)
    for category in mrd_results.keys():
        n_success = len(mrd_results[category]['mrd_data'])
        print(f"{category}: Successfully processed {n_success} periods")
    print("="*60 + "\n")

    # Plot combined results with median and quantiles across all events
    fig, ax = plt.subplots(figsize=(10, 6))

    for category in mrd_results.keys():
        if len(mrd_results[category]['mrd_data']) > 0:
            # Stack all MRD data for this category
            # Each mrd_data_temp is shape (M, n_windows), we want to combine across all events
            all_mrd_data = np.concatenate(mrd_results[category]['mrd_data'], axis=1)
            
            # Use the first event's x-axis (should be same for all)
            x_data = mrd_results[category]['mrd_x'][0]
            
            # Convert timedelta to seconds
            seconds_array = np.vectorize(lambda td: td.total_seconds())(x_data)
            
            # Calculate median and quantiles across all windows from all events
            median_data = np.nanmedian(all_mrd_data, axis=1) * 1000
            q25_data = np.nanquantile(all_mrd_data, 0.25, axis=1) * 1000
            q75_data = np.nanquantile(all_mrd_data, 0.75, axis=1) * 1000
            
            # Plot
            color = color_scheme[category]['main']
            alpha = color_scheme[category]['alpha']
            
            ax.plot(seconds_array, median_data, color=color, label=category, linewidth=2)
            ax.fill_between(seconds_array, q25_data, q75_data, color=color, alpha=alpha)
            
            print(f"\n{category}: Combined {len(mrd_results[category]['mrd_data'])} events")
        else:
            print(f"No data available for {category}")
    
    ax.set_title("MRD - Wind Direction Comparison")
    ax.set_xlabel("avg. time [s]")
    ax.set_ylabel(r"$C_{w\theta} [\cdot 10^{-3} \mathrm{Kms^{-1}}]$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'MRD_wind_direction_comparison_SFC.png', dpi=300, bbox_inches='tight')
    plt.show()