import numpy as np
import pandas as pd

from .constants import epsilon, R_dry_air, R_w


def resample_with_threshold(data, resample_time, interpolate=False, max_gap='1h', min_valid_percent=80):
    """
    Returns NaN if the percentage of valid values within the resample time is less than min_valid_percent.
    Linearly interpolates gaps in the data only if the gaps are smaller than max_gap.

    Parameters:
        data (pd.Series): The input data to be resampled.
        resample_time (str): The resampling frequency (e.g., '10min', '1h').
        min_valid_percent (float): Minimum percentage of valid values required to keep the resampled value.

    Returns:
        pd.Series: The resampled data with insufficient valid data set to NaN.
    """
    if interpolate == True:
        # Calculate the data's frequency in seconds
        freq = (data.index[1] - data.index[0]).total_seconds()
        # Convert the max_gap to seconds
        max_gap_seconds = pd.to_timedelta(max_gap).total_seconds()
        # Calculate the limit as the number of consecutive NaNs within the max_gap
        limit = int(max_gap_seconds / freq)
        data = data.interpolate(limit=limit, limit_direction='both', limit_area='inside')
    # Resample the data
    resampled_data = data.resample(resample_time).mean()
    # Count the number of valid (non-NaN) values in each resample period
    valid_counts = data.resample(resample_time).count()
    # Calculate the total number of values in each resample period
    total_counts = data.resample(resample_time).size()
    # Calculate the percentage of valid values
    valid_percent = (valid_counts / total_counts) * 100
    # Apply the threshold and valid percentage filter
    filtered_data = resampled_data.where((valid_percent >= min_valid_percent))
    # Interpolate gaps smaller than 1H

    return filtered_data




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

def RH_to_specific_humidity(RH, T, P):
    """
    Convert relative humidity to specific humidity (water vapor mixing ratio).
    
    Parameters:
        RH (float or array-like): Relative humidity (0-100 for percent, or 0-1 for fraction)
        T (float or array-like): Temperature (°C)
        P (float or array-like): Atmospheric pressure (Pa)
    
    Returns:
        float or array-like: Specific humidity q (kg/kg)
    """
    # Convert to numpy arrays for vectorized operations
    RH = np.asarray(RH)
    T = np.asarray(T)
    P = np.asarray(P)
    
    # If RH is in percent (0-100), convert to fraction
    # Check if maximum value suggests it's in percent
    if np.nanmax(RH) > 1:
        RH = RH / 100
    
    # Get saturation vapor pressure (Pa)
    # Use ice formula if T < 0, liquid if T >= 0
    e_s = np.where(T < 0, 
                   vapor_pressure_ice_MK2005(T),
                   vapor_pressure_liquid_MK2005(T))
    
    # Actual vapor pressure (Pa)
    e = RH * e_s
    
    # Specific humidity (kg/kg)
    q = epsilon * e / (P - (1 - epsilon) * e)
    
    return q

def calc_es(temp, ice=False):
    """
    Calculate saturation vapor pressure [Pa]
    
    Approximation of Clausius-Clapeyron equation, semiempirical formula as used in 
    Lehning et al., 2002: A physical SNOWPACK model for the Swiss avalanche.
    
    Parameters:
    -----------
    temp : float or array-like
        Air temperature [degC]
    ice : bool, optional
        If True, calculate for ice (sublimation), otherwise for water (evaporation)
        Default is False
    
    Returns:
    --------
    float or ndarray
        Saturation vapor pressure [Pa]
    """
    # Triple point pressure [Pa]
    p_t = 610.5
    # Triple point temperature [K]
    T_t = 273.16
    # Specific gas constant for water vapor [J kg-1 K-1]
    R_v = 461.9
    
    # Latent heat of sublimation (or evaporation)
    if ice:
        # Formula as in LES-LSM [J kg-1]
        L = (2834.1 - 0.29 * temp - 0.004 * temp**2) * 1e3
    else:
        # From Foken: Micrometeorology, 2008 [J kg-1]
        L = (2.501 - 0.00237 * temp) * 1e6
    
    # Convert temperature from degC to K
    Tmp = temp + 273.15
    
    # Calculate saturation vapor pressure
    # (equivalent to a common approximation of the Clausius-Clapeyron equation)
    es = p_t * np.exp(L * (Tmp - T_t) / (R_v * T_t * Tmp))  # [Pa]
    
    return es


def Lsubl(temp):
    """
    Calculate latent heat of sublimation [J kg^-1]
    
    Formula as in LES-LSM
    
    Parameters:
    -----------
    temp : float or array-like
        Temperature [degC]
    
    Returns:
    --------
    float or ndarray
        Latent heat of sublimation [J kg^-1]
    """
    return (2834.1 - 0.29 * temp - 0.004 * temp**2) * 1e3


def calc_qv_surface(T_surf, pressure, ice=True):
    """
    Calculate specific humidity at the surface [kg kg^-1]
    
    This calculation assumes saturation at the surface and follows the approach
    from the R code provided, which is equivalent to q = 0.622 * e / (p - 0.378*e)
    
    Parameters:
    -----------
    T_surf : float or array-like
        Surface temperature [degC]
    pressure : float or array-like
        Atmospheric pressure [Pa]
    ice : bool, optional
        If True, calculate for ice surface (default), otherwise for water
    
    Returns:
    --------
    dict with keys:
        'qv_surf' : float or ndarray - Specific humidity at surface [kg kg^-1]
        'rho_air_surf' : float or ndarray - Air density at surface [kg m^-3]
        'Ls' : float or ndarray - Latent heat of sublimation [J kg^-1]
    """
    # Convert to numpy arrays for vectorized operations
    T_surf = np.asarray(T_surf)
    pressure = np.asarray(pressure)
    
    # (Saturation) vapor pressure at surface [Pa]
    e_surf = calc_es(temp=T_surf, ice=ice)
    
    # Dry air density (kg m^-3) at surface
    rho_dry_air_surf = (pressure - e_surf) / (R_dry_air * (T_surf + 273.15))
    
    # Water vapor partial density (kg m^-3) at surface
    rho_h2o_surf = e_surf / (R_w * (T_surf + 273.15))
    
    # Air density (kg m^-3) at surface
    rho_air_surf = rho_dry_air_surf + rho_h2o_surf
    
    # Specific humidity (kg kg^-1) at surface
    # The calculation here is equivalent to the formula q = 0.622 * e / (p - 0.378*e)
    qv_surf = rho_h2o_surf / rho_air_surf
    
    # Latent heat of sublimation (J kg^-1) at surface temperature
    # Formula as in LES-LSM
    Ls = Lsubl(temp=T_surf)
    
    return {
        'qv_surf': qv_surf,
        'rho_air_surf': rho_air_surf,
        'Ls': Ls
    }