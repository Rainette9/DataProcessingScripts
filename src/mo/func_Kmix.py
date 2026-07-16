import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .func_mo import calc_phi_stable_holtslag, calc_phi_unstable_paulson_stearns_weidner
from utils.utils import (
    convert_RH_liquid_to_ice,
    RH_to_specific_humidity,
    resample_with_threshold,
)

GAMMA_D = 0.0098  # Dry adiabatic lapse rate [K m^-1]
M_W = 0.018016    # Molar mass of water [kg mol^-1]


def Kmix_ustar(z, ustar, L):
    """
    Calculate the mixing coefficient Kmix based on Monin-Obukhov similarity theory.

    Parameters:
    z : float or array-like
        Height above the ground (m).
    ustar : float or array-like
        Friction velocity (m/s).
    L : float or array-like
        Monin-Obukhov length (m).

    Returns:
    Kmix : float or array-like
        Mixing coefficient (m^2/s).
    """
    # Ensure inputs are numpy arrays for element-wise operations
    z = np.asarray(z)
    ustar = np.asarray(ustar)
    L = np.asarray(L)

    # Calculate the stability correction function phi_m, using the same universal functions
    # as the calc_fluxes_iter defaults: Holtslag when stable, Paulson/Stearns-Weidner when
    # unstable. Both branches are evaluated on clipped zeta so that each stays in its valid
    # range, and np.where then selects per element.
    zeta = z / L
    phi_m = np.where(
        zeta >= 0,
        calc_phi_stable_holtslag(np.clip(zeta, 0, None))['m'],
        calc_phi_unstable_paulson_stearns_weidner(np.clip(zeta, None, -np.finfo(float).tiny))['m'],
    )

    # Calculate Kmix using the formula Kmix = k * ustar * z / phi_m
    k = 0.4  # von Karman constant
    Kmix = k * ustar * z / phi_m

    return Kmix


