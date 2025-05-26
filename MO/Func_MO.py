import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_roughness_length(eddypro_data, eddypro_qc, z_wind):
    """This function computes the roughness length from the wind speed profile and ustar."""
    eddypro_data = eddypro_data.copy()
    eddypro_data = eddypro_data
    k = 0.4
    u_mean = eddypro_data['wind_speed']
    u_star = eddypro_data['u*']
    z0 = z_wind * np.exp(-k * u_mean / u_star)
    
    # Set z0 to NaN when (z-d)/L is not neutral
    z0[(eddypro_data['(z-d)/L'] < -0.1) | (eddypro_data['(z-d)/L'] > 0.1)] = np.nan
    # z0[(eddypro_qc['flag(u)'])]
    z0rolling = z0.rolling(window='28D', center=True, min_periods=1).median()

    return z0, z0rolling


# def compute_MO(slowdata, fastdata_or_z0):
#     """This function computes the Monin-Obukhov turbulent fluxes"""
#     if isinstance(fastdata_or_z0, pd.DataFrame):
#         z0=compute_roughness_length(fastdata_or_z0)
#     else:
#         z0=fastdata_or_z0

#     return SHF

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