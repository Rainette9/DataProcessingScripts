import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_roughness_length(eddypro_data, eddypro_qc, z_wind):
    """This function computes the roughness length from the wind speed profile and ustar."""
    eddypro_data = eddypro_data.copy()
    k = 0.4
    u_mean = eddypro_data['wind_speed']
    u_star = eddypro_data['u*']
    z0 = z_wind * np.exp(-k * u_mean / u_star)
    
    # Set z0 to NaN when (z-d)/L is not neutral
    z0[(eddypro_data['(z-d)/L'] < -0.1) | (eddypro_data['(z-d)/L'] > 0.1)] = np.nan
    # z0[(eddypro_qc['flag(u)'])]
    z0rolling = z0.rolling(window='28D', center=True, min_periods=1).median()

    return z0, z0rolling



# Stable stratification ---------------------------------------------------

def calc_psi_stable_stearns_weidner(zeta):
    """
    Universal functions for stable and neutral conditions according to Stearns and Weidner (1993).
    
    Parameters:
        zeta (array-like): Stability parameter (must be >= 0)
    
    Returns:
        dict: Dictionary with 'm' (momentum) and 's' (scalars/temperature) psi values
    """
    zeta = np.asarray(zeta)
    if np.any(zeta < 0):
        raise ValueError("zeta is not >= 0")
    
    # Momentum
    y = (1 + 5 * zeta)**0.25
    psi_m = np.log((1 + y)**2) + np.log(1 + y**2) - 2 * np.arctan(y) - 4/3 * y**3 + 0.8247
    
    # Temperature, scalars
    y = (1 + 5 * zeta)**0.5
    psi_s = np.log((1 + y)**2) - 2 * y - 2/3 * y**3 + 1.2804
    
    return {'m': psi_m, 's': psi_s}


def calc_psi_stable_holtslag(zeta):
    """
    Universal functions for stable and neutral conditions according to Holtslag and DeBruin (1988).
    
    Parameters:
        zeta (array-like): Stability parameter (must be >= 0)
    
    Returns:
        dict: Dictionary with 'm' (momentum) and 's' (scalars) psi values
    """
    zeta = np.asarray(zeta)
    if np.any(zeta < 0):
        raise ValueError("zeta is not >= 0")
    
    # Same for momentum and scalars
    psi_m = -(0.7 * zeta + 0.75 * (zeta - 14.28) * np.exp(-0.35 * zeta) + 10.71)
    psi_s = psi_m
    
    return {'m': psi_m, 's': psi_s}


# Unstable stratification -------------------------------------------------

def calc_psi_unstable(zeta):
    """
    Universal functions after Businger et al., 1971 using the normalization after Hoegstroem, 1988.
    
    Parameters:
        zeta (array-like): Stability parameter (should be < 0 for unstable conditions)
    
    Returns:
        dict: Dictionary with 'm' (momentum) and 's' (scalars) psi values
    """
    zeta = np.asarray(zeta)
    
    # Momentum
    x_term = (1 - 19.3 * zeta)**0.25
    psi_m = (np.log(((1 + x_term**2) / 2) * ((1 + x_term) / 2)**2) - 
             2 * np.arctan(x_term) + np.pi / 2)
    
    # Scalars
    psi_s = 2 * np.log((1 + (0.95 * (1 - 11.6 * zeta)**0.5)**2) / 2)
    
    return {'m': psi_m, 's': psi_s}


def calc_psi_unstable_paulson_stearns_weidner(zeta):
    """
    Universal functions of Paulson (original) for momentum and of Stearns and Weidner (1993) for scalars.
    [The Stearns and Weidner (1993) formula for momentum is not correct, something (at least a sign 
    of one term) seems to be missing in the paper.]
    
    Parameters:
        zeta (array-like): Stability parameter (must be < 0)
    
    Returns:
        dict: Dictionary with 'm' (momentum) and 's' (scalars/temperature) psi values
    """
    zeta = np.asarray(zeta)
    if np.any(zeta >= 0):
        raise ValueError("zeta is not < 0")
    
    # Momentum
    x = (1 - 15 * zeta)**0.25
    psi_m = (2 * np.log(0.5 * (1 + x)) + 
             np.log(0.5 * (1 + x**2)) - 
             2 * np.arctan(x) + 
             0.5 * np.pi)
    
    # Temperature, scalars
    x = (1 - 22.5 * zeta)**(1/3)
    psi_s = (np.log((1 + x + x**2)**1.5) - 
             np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3)) + 
             0.1659)
    # Note: The paper is not clear about whether to first apply the power of 1.5 and then the 
    # logarithm or first the logarithm and then the power of 1.5; however, the former makes more 
    # sense because otherwise psi_s does not approach zero but -0.496 for zeta approaching zero.
    
    return {'m': psi_m, 's': psi_s}


def calc_bulk_kinemat_flux(u_up, u_low, scalar_up, scalar_low, coeff):
    """
    Calculate bulk kinematic flux.
    
    Parameters:
        u_up (float): Wind speed at upper height (m/s)
        u_low (float): Wind speed at lower height (m/s)
        scalar_up (float): Scalar value at upper height
        scalar_low (float): Scalar value at lower height
        coeff (float): Bulk transfer coefficient
    
    Returns:
        float: Kinematic flux
    """
    return -1 * coeff * (u_up - u_low) * (scalar_up - scalar_low)


def calc_coeff_scalar(psi_m, psi_s, z_u_up, z_u_low, z_scalar_up, z_scalar_low):
    """
    Calculate bulk transfer coefficient for scalars.
    
    Parameters:
        psi_m (float): Integrated universal function for momentum
        psi_s (float): Integrated universal function for scalars
        z_u_up (float): Upper height for wind measurement (m)
        z_u_low (float): Lower height for wind (roughness length for momentum, m)
        z_scalar_up (float): Upper height for scalar measurement (m)
        z_scalar_low (float): Lower height for scalar (roughness length for scalar, m)
    
    Returns:
        float: Bulk transfer coefficient
    """
    return 0.4**2 / ((np.log(z_u_up/z_u_low) - psi_m) * (np.log(z_scalar_up/z_scalar_low) - psi_s))


def calc_fluxes_iter(z_ref_vw=None, z_ref_scalar=None, rough_len_m=None, rough_len_tq_Andreas=False,
                     rough_len_t=None, rough_len_q=None, vw_ref=None, T_ref=None, qv_ref=None,
                     T_surf=None, qv_surf=None, FUN_psi_stable=None, FUN_psi_unstable=None,
                     prescribe_ustar=None):
    """
    Iterative flux calculation using Monin-Obukhov similarity theory.
    
    Parameters:
        z_ref_vw (float): Reference height for wind speed (m), defaults to z_ref_scalar
        z_ref_scalar (float): Reference height for temperature and humidity (m)
        rough_len_m (float): Roughness length for momentum (m)
        rough_len_tq_Andreas (bool): Use Andreas (1987) parametrization for scalar roughness lengths
        rough_len_t (float): Roughness length for temperature (m), defaults to 0.1*rough_len_m
        rough_len_q (float): Roughness length for humidity (m), defaults to 0.1*rough_len_m
        vw_ref (float): Wind speed at reference height (m/s)
        T_ref (float): Temperature at reference height (K)
        qv_ref (float): Specific humidity at reference height (kg/kg)
        T_surf (float): Surface temperature (K)
        qv_surf (float): Surface specific humidity (kg/kg)
        FUN_psi_stable (callable): Function for stable universal functions, defaults to calc_psi_stable_holtslag
        FUN_psi_unstable (callable): Function for unstable universal functions, defaults to calc_psi_unstable_paulson_stearns_weidner
        prescribe_ustar (float or None): Prescribed friction velocity (m/s), or None to calculate
    
    Returns:
        dict: Dictionary with u_star, Tw_flux, qw_flux, zeta, psi_m, psi_s, converged, and optionally Re_star
    """
    # Set defaults
    if z_ref_vw is None:
        z_ref_vw = z_ref_scalar
    if z_ref_scalar is None:
        z_ref_scalar = z_ref_vw
    if rough_len_t is None:
        rough_len_t = 0.1 * rough_len_m
    if rough_len_q is None:
        rough_len_q = 0.1 * rough_len_m
    if FUN_psi_stable is None:
        FUN_psi_stable = calc_psi_stable_holtslag
    if FUN_psi_unstable is None:
        FUN_psi_unstable = calc_psi_unstable_paulson_stearns_weidner
    
    # Constants
    nu_air = 1.24e-05  # kinematic viscosity of air
    kappa = 0.4  # von Karman constant
    g = 9.81  # gravitational acceleration
    
    # Check if iteration is needed
    iterate = True
    if prescribe_ustar is None or not np.isfinite(prescribe_ustar):
        if vw_ref == 0:
            return {
                'u_star': 0.0,
                'Tw_flux': 0.0,
                'qw_flux': 0.0,
                'zeta': np.nan,
                'psi_m': np.nan,
                'psi_s': np.nan,
                'converged': np.nan,
                'Re_star': np.nan
            }
    
    # Iterative calculation
    Re_star = np.nan
    converged = 0
    
    for i in range(100):
        if i == 0:
            # Start with neutral conditions
            zeta = 1e-6
        else:
            zeta = zeta_new
        
        # Stable or neutral conditions
        if zeta >= 0:
            psi = FUN_psi_stable(zeta)
        else:  # Unstable conditions
            psi = FUN_psi_unstable(zeta)
        
        # Friction velocity
        if prescribe_ustar is not None and np.isfinite(prescribe_ustar):
            u_star = prescribe_ustar
        else:
            u_star = kappa * vw_ref / (np.log(z_ref_vw / rough_len_m) - psi['m'])
        
        # Scalar roughness lengths using Andreas parametrization (if requested)
        if rough_len_tq_Andreas:
            Re_star = u_star * rough_len_m / nu_air
            # Note: calc_Andreas_model function would need to be implemented
            # rough_len_t = rough_len_m * calc_Andreas_model(Re_star, "temperature")
            # rough_len_q = rough_len_m * calc_Andreas_model(Re_star, "humidity")
            pass  # Placeholder for Andreas parametrization
        
        # Temperature flux
        if prescribe_ustar is not None and np.isfinite(prescribe_ustar):
            Tw_flux = -kappa * u_star * (T_ref - T_surf) / (np.log(z_ref_scalar/rough_len_t) - psi['s'])
        else:
            c_h = calc_coeff_scalar(psi['m'], psi['s'], z_ref_vw, rough_len_m, z_ref_scalar, rough_len_t)
            Tw_flux = calc_bulk_kinemat_flux(vw_ref, 0, T_ref, T_surf, c_h)
        
        # Stability parameter z/L
        zeta_new = -z_ref_scalar * kappa * (g / T_surf) * Tw_flux / u_star**3
        
        # Check convergence
        if np.isfinite(zeta) and np.isfinite(zeta_new):
            if abs(zeta_new - zeta) <= 0.001 * abs(zeta):
                converged = 1
                break
        else:
            converged = 0
            break
    
    # If not converged, assume neutral conditions
    if converged == 0:
        zeta = 0
        psi = {'m': 0, 's': 0}
        if prescribe_ustar is not None and np.isfinite(prescribe_ustar):
            Tw_flux = -kappa * u_star * (T_ref - T_surf) / np.log(z_ref_scalar/rough_len_t)
        else:
            u_star = kappa * vw_ref / np.log(z_ref_vw / rough_len_m)
            c_h = calc_coeff_scalar(psi['m'], psi['s'], z_ref_vw, rough_len_m, z_ref_scalar, rough_len_t)
            Tw_flux = calc_bulk_kinemat_flux(vw_ref, 0, T_ref, T_surf, c_h)
    
    # Limit zeta to [-10, 10]
    if abs(zeta) > 10:
        if zeta > 10:
            zeta = 10
            psi = FUN_psi_stable(zeta)
        if zeta < -10:
            zeta = -10
            psi = FUN_psi_unstable(zeta)
        
        if prescribe_ustar is None or not np.isfinite(prescribe_ustar):
            u_star = kappa * vw_ref / (np.log(z_ref_vw / rough_len_m) - psi['m'])
        
        if rough_len_tq_Andreas:
            Re_star = u_star * rough_len_m / nu_air
            # rough_len_t = rough_len_m * calc_Andreas_model(Re_star, "temperature")
            # rough_len_q = rough_len_m * calc_Andreas_model(Re_star, "humidity")
        
        if prescribe_ustar is not None and np.isfinite(prescribe_ustar):
            Tw_flux = -kappa * u_star * (T_ref - T_surf) / (np.log(z_ref_scalar/rough_len_t) - psi['s'])
        else:
            c_h = calc_coeff_scalar(psi['m'], psi['s'], z_ref_vw, rough_len_m, z_ref_scalar, rough_len_t)
            Tw_flux = calc_bulk_kinemat_flux(vw_ref, 0, T_ref, T_surf, c_h)
    
    # Latent heat flux
    if prescribe_ustar is not None and np.isfinite(prescribe_ustar):
        qw_flux = -kappa * u_star * (qv_ref - qv_surf) / (np.log(z_ref_scalar/rough_len_q) - psi['s'])
    else:
        c_q = calc_coeff_scalar(psi['m'], psi['s'], z_ref_vw, rough_len_m, z_ref_scalar, rough_len_q)
        qw_flux = calc_bulk_kinemat_flux(vw_ref, 0, qv_ref, qv_surf, c_q)

    # # Sensible heat flux
    # if prescribe_ustar is not None and np.isfinite(prescribe_ustar):
    #     Tw_flux = -kappa * u_star * (T_ref - T_surf) / (np.log(z_ref_scalar/rough_len_t) - psi['s'])
    # else:
    #     c_q = calc_coeff_scalar(psi['m'], psi['s'], z_ref_vw, rough_len_m, z_ref_scalar, rough_len_t)
    #     qw_flux = calc_bulk_kinemat_flux(vw_ref, 0, T_ref, T_surf, c_q)
    
    result = {
        'u_star': u_star,
        'Tw_flux': Tw_flux,
        'qw_flux': qw_flux,
        'zeta': zeta,
        'psi_m': psi['m'],
        'psi_s': psi['s'],
        'converged': converged
    }
    
    if rough_len_tq_Andreas:
        result['Re_star'] = Re_star
    
    return result


def calc_MO_profile(z_ref, x_ref, x_star, psi, z_out):
    """
    Calculate vertical profile using Monin-Obukhov similarity theory.
    
    Parameters:
        z_ref (float): Reference height (m)
        x_ref (float): Quantity of interest at reference height
        x_star (float): Vertical flux divided by friction velocity (same unit as x_ref)
        psi (float or array): Integrated universal function (stability correction)
        z_out (float or array): Output heights (m)
    
    Returns:
        float or array: Profile values at z_out
    """
    return x_ref + x_star / 0.4 * (np.log(z_out / z_ref) - psi)


def compute_MO(slowdata, z0=0.002):
    """This function computes the Monin-Obukhov turbulent fluxes"""
    
    # Placeholder - needs implementation
    SHF = None
    LHF = None

    return SHF, LHF

# import pandas as pd
# import numpy as np
# from scipy.constants import R

# # Constants
# R_dry_air = 287.05  # J/(kg·Kplim = get_sensor_info(sensor, 2024)

# R_w = 461.5  # J/(kg·K)

# # Functions (placeholders for external functions)
# def calc_es(temp, ice=False):
#     """
#     Calculate saturation vapor pressure (Pa).
#     """
#     if ice:
#         # Formula for saturation vapor pressure over ice
#         return 6.112 * np.exp((22.46 * temp) / (temp + 272.62)) * 100
#     else:
#         # Formula for saturation vapor pressure over liquid water
#         return 6.112 * np.exp((17.62 * temp) / (temp + 243.12)) * 100

# def Lsubl(temp):
#     """
#     Calculate latent heat of sublimation (J/kg) at a given temperature.
#     """
#     return 2.834e6 - 2.1e3 * temp

# def calc_fluxes_iter(z_ref_vw, z_ref_scalar, rough_len_m, rough_len_t, rough_len_q, vw_ref, T_ref, qv_ref, T_surf, qv_surf):
#     """
#     Placeholder for iterative flux calculation.
#     """
#     # Replace with actual implementation
#     return {
#         "u_star": np.nan,
#         "Tw_flux": np.nan,
#         "qw_flux": np.nan,
#         "zeta": np.nan,
#         "psi_m": np.nan,
#         "psi_s": np.nan,
#         "converged": False
#     }

# # Read input file and prepare data
# path_input = "csv_for_monin_obukhov.csv"
# dat = pd.read_csv(path_input, na_values=["NA", "NaN", '"NAN"', '"NaN"', "INF", '"INF"'])

# # Vapor pressure (Pa) at height z_TA_RH
# dat["e_z"] = dat["RH"] / 100 * calc_es(dat["TA"], ice=False)

# # Dry air density (kg/m³)
# dat["rho_dry_air"] = (dat["pressure"] * 1000 - dat["e_z"]) / (R_dry_air * (dat["TA"] + 273.15))

# # Water vapor partial density (kg/m³)
# dat["rho_h2o"] = dat["e_z"] / (R_w * (dat["TA"] + 273.15))

# # Air density (kg/m³)
# dat["rho_z"] = dat["rho_dry_air"] + dat["rho_h2o"]

# # Specific humidity (kg/kg)
# dat["qv"] = dat["rho_h2o"] / dat["rho_z"]

# # Saturation vapor pressure at surface (Pa)
# dat["e_surf"] = calc_es(dat["T_surf"], ice=True)

# # Dry air density at surface (kg/m³)
# dat["rho_dry_air_surf"] = (dat["pressure"] * 1000 - dat["e_surf"]) / (R_dry_air * (dat["T_surf"] + 273.15))

# # Water vapor partial density at surface (kg/m³)
# dat["rho_h2o_surf"] = dat["e_surf"] / (R_w * (dat["T_surf"] + 273.15))

# # Air density at surface (kg/m³)
# dat["rho_air_surf"] = dat["rho_dry_air_surf"] + dat["rho_h2o_surf"]

# # Specific humidity at surface (kg/kg)
# dat["qv_surf"] = dat["rho_h2o_surf"] / dat["rho_air_surf"]

# # Latent heat of sublimation (J/kg) at surface temperature
# dat["Ls"] = Lsubl(dat["T_surf"])

# # Iterative flux calculation
# result = pd.DataFrame({
#     "time": dat["time"],
#     "u_star": np.nan,
#     "Tw_flux": np.nan,
#     "qw_flux": np.nan,
#     "zeta": np.nan,
#     "psi_m": np.nan,
#     "psi_s": np.nan,
#     "converged": np.nan
# })

# # Time loop
# for i, d in dat.iterrows():
#     # Skip iteration if any value is NA
#     if d.isna().any():
#         print(f"Skipping row {i} due to missing or infinite values.")
#         continue

#     # Iterative flux calculation
#     fluxes = calc_fluxes_iter(
#         z_ref_vw=d["z_ws"],
#         z_ref_scalar=d["z_TA_RH"],
#         rough_len_m=d["z0"],
#         rough_len_t=d["z0"],
#         rough_len_q=d["z0"],
#         vw_ref=d["ws"],
#         T_ref=d["TA"] + 273.15,
#         qv_ref=d["qv"],
#         T_surf=d["T_surf"] + 273.15,
#         qv_surf=d["qv_surf"]
#     )
#     result.loc[i, 1:] = list(fluxes.values())

# # Convert units: vapor flux (kg/kg·m/s) to latent heat flux (W/m²)
# result["LE"] = dat["Ls"] * dat["rho_air_surf"] * result["qw_flux"]

# # Save results to CSV
# result = result.round({"qw_flux": 10, "u_star": 5, "Tw_flux": 5, "zeta": 5, "psi_m": 5, "psi_s": 5, "LE": 5})
# result.to_csv("Monin_Obukhov_results.csv", na_rep="NaN", index=False)