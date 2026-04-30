import numpy as np

# Constants
_KARMAN = 0.4
_G = 9.81
_T_FREEZE = 273.15

SUPPORTED_STABILITY = (
    'NEUTRAL', 'RICHARDSON',
    'MO_HOLTSLAG', 'MO_STEARNS', 'MO_MICHLMAYR', 'MO_LOG_LINEAR',
    'MO_SCHLOEGL_UNI', 'MO_SCHLOEGL_MULTI', 'MO_SCHLOEGL_MULTI_OFFSET',
)


# ---------------------------------------------------------------------------
# Saturation vapor pressure
# ---------------------------------------------------------------------------

def _vapor_saturation_pressure(T):
    """
    Tetens formula as in SNOWPACK Atmosphere::vaporSaturationPressure.
    T: Kelvin → returns Pa.
    """
    Tc = T - _T_FREEZE
    if T >= _T_FREEZE:
        return 611.0 * np.exp(17.269 * Tc / (T - 35.86))
    else:
        return 611.0 * np.exp(21.874 * Tc / (T - 7.66))


# ---------------------------------------------------------------------------
# Stability functions (stable)
# ---------------------------------------------------------------------------

def _psi_stable_holtslag(x):
    v = -(0.7 * x + 0.75 * (x - 14.28) * np.exp(-0.35 * x) + 10.71)
    return v, v  # psi_m, psi_s


def _psi_stable_stearns(x):
    d1 = (1. + 5. * x) ** 0.25
    psi_m = (np.log((1. + d1)**2) + np.log(1. + d1**2)
             - 2. * np.arctan(d1) - 4./3. * d1**3 + 0.8247)
    d2 = d1**2
    psi_s = np.log((1. + d2)**2) - 2. * d2 - 2./3. * d2**3 + 1.2804
    return psi_m, psi_s


def _psi_stable_michlmayr(x):
    # Stearns & Weidner (1993) modified by Michlmayr (2008)
    d1 = (1. + 5. * x) ** 0.25
    psi_m = (np.log(1. + d1)**2 + np.log(1. + d1**2)
             - np.arctan(d1) - 0.5 * d1**3 + 0.8247)
    d2 = d1**2
    psi_s = np.log(1. + d2)**2 - d2 - 0.3 * d2**3 + 1.2804
    return psi_m, psi_s


# ---------------------------------------------------------------------------
# Stability functions (unstable) — Paulson (momentum) + Stearns-Weidner (scalars)
# ---------------------------------------------------------------------------

def _psi_unstable_paulson_stearns(x):
    d1 = (1. - 15. * x) ** 0.25
    psi_m = (2. * np.log(0.5 * (1. + d1)) + np.log(0.5 * (1. + d1**2))
             - 2. * np.arctan(d1) + 0.5 * np.pi)
    d2 = (1. - 22.5 * x) ** (1./3.)
    psi_s = (np.log((1. + d2 + d2**2) ** 1.5)
             - np.sqrt(3.) * np.arctan((1. + 2.*d2) / np.sqrt(3.))
             + 0.1659)
    return psi_m, psi_s


# ---------------------------------------------------------------------------
# Core stability routines matching SNOWPACK's MOStability / RichardsonStability
# ---------------------------------------------------------------------------

def _richardson_stability(ta_v, t_surf_v, zref, vw):
    """SNOWPACK RichardsonStability. Returns (psi_m, psi_s, Ri)."""
    Ri = _G / t_surf_v * (ta_v - t_surf_v) * zref / vw**2
    if Ri < 0.:
        d = (1. - 15. * Ri) ** 0.25
        psi_m = (np.log((0.5*(1.+d**2)) * (0.5*(1.+d))**2)
                 - 2.*np.arctan(d) + 0.5*np.pi)
        psi_s = 2. * np.log(0.5 * (1. + d**2))
    elif Ri < 0.1999:
        sr = Ri / (1. - 5. * Ri)
        psi_m = psi_s = -5. * sr
    else:
        sr = Ri / (1. - 5. * 0.1999)
        psi_m = psi_s = -5. * sr
    return psi_m, psi_s, Ri


def _mo_stability(stability, ta_v, t_surf_v, T_surf, zref, vw, z_ratio, psi_s_prev, ustar):
    """
    SNOWPACK MOStability. Computes stab_ratio from Tstar, then dispatches to
    the chosen stability function. Returns (psi_m, psi_s, stab_ratio).
    """
    Tstar = _KARMAN * (t_surf_v - ta_v) / (z_ratio - psi_s_prev)
    stab_ratio = -_KARMAN * zref * Tstar * _G / (T_surf * ustar**2)
    stab_ratio = min(stab_ratio, 1.0)  # SNOWPACK clamps at 1

    if stab_ratio > 0.:
        if stability == 'MO_HOLTSLAG':
            psi_m, psi_s = _psi_stable_holtslag(stab_ratio)
        elif stability == 'MO_STEARNS':
            psi_m, psi_s = _psi_stable_stearns(stab_ratio)
        elif stability == 'MO_MICHLMAYR':
            psi_m, psi_s = _psi_stable_michlmayr(stab_ratio)
        elif stability == 'MO_LOG_LINEAR':
            psi_m = psi_s = -5. * stab_ratio
        elif stability == 'MO_SCHLOEGL_UNI':
            psi_m = -1.62 * stab_ratio
            psi_s = -2.96 * stab_ratio
        elif stability == 'MO_SCHLOEGL_MULTI':
            dT = (ta_v - t_surf_v) / (0.5 * (ta_v + t_surf_v))
            psi_m = -65.35 * dT + 0.0017 * zref * _G / vw**2
            psi_s = -813.21 * dT - 0.0014 * zref * _G / vw**2
        elif stability == 'MO_SCHLOEGL_MULTI_OFFSET':
            dT = (ta_v - t_surf_v) / (0.5 * (ta_v + t_surf_v))
            psi_m = -0.69 - 15.47 * dT + 0.0059 * zref * _G / vw**2
            psi_s =  6.73 - 688.18 * dT - 0.0023 * zref * _G / vw**2
        else:
            raise ValueError(f"Unsupported stability: '{stability}'")
    else:
        psi_m, psi_s = _psi_unstable_paulson_stearns(stab_ratio)

    return psi_m, psi_s, stab_ratio


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def calc_fluxes_snowpack(z_ref, rough_len_m, vw_ref, T_ref, T_surf,
                          qv_ref, qv_surf, pressure,
                          stability='MO_HOLTSLAG',
                          rough_len_scalar=None,
                          max_iter=100, eps=1e-3):
    """
    Iterative turbulent flux calculation following SNOWPACK's MicroMet / MOStability.

    Key differences from the standard MO bulk method in func_mo.py / func_MO_bulk_fluxes.R:

    1. Iteration converges on u* change  (|u*_new - u*_old| < eps)  instead of zeta change.
    2. Stability parameter uses virtual temperatures and Tstar:
           Tstar      = kappa * (T_surf_v - T_air_v) / (z_ratio - psi_s)
           stab_ratio = -kappa * z_ref * Tstar * g / (T_surf * u*^2)
       rather than the direct zeta formula kappa*g*Tw_flux / (T_surf * u*^3).
    3. stab_ratio is clamped to 1.0 on the stable side during each iteration.
    4. Virtual temperatures computed using SNOWPACK's approach:
           sat_vap  = saturation vapor pressure at T_ref  (Tetens formula)
           T_air_v  = T_ref  * (1 + 0.377 * sat_vap / pressure)
           T_surf_v = T_surf * (1 + 0.377 * sat_vap / pressure)
    5. rough_len_scalar defaults to rough_len_m (single z0 as in SNOWPACK MicroMet).
    6. Wind speed is clipped to a minimum of 0.3 m/s (as in SNOWPACK).
    7. All SNOWPACK atmospheric stability models are supported.

    Parameters
    ----------
    z_ref : float
        Reference height for wind speed and scalars (m).
    rough_len_m : float
        Roughness length for momentum (m).
    vw_ref : float
        Wind speed at z_ref (m/s). Clipped to min 0.3 m/s internally.
    T_ref : float
        Air temperature at z_ref (K).
    T_surf : float
        Surface temperature (K).
    qv_ref : float
        Specific humidity at z_ref (kg/kg).
    qv_surf : float
        Surface specific humidity (kg/kg).
    pressure : float
        Atmospheric pressure (Pa).
    stability : str
        Stability model. One of: 'NEUTRAL', 'RICHARDSON', 'MO_HOLTSLAG' (default),
        'MO_STEARNS', 'MO_MICHLMAYR', 'MO_LOG_LINEAR', 'MO_SCHLOEGL_UNI',
        'MO_SCHLOEGL_MULTI', 'MO_SCHLOEGL_MULTI_OFFSET'.
    rough_len_scalar : float or None
        Roughness length for scalar transfer (m). Defaults to rough_len_m.
    max_iter : int
        Maximum iterations (default 100).
    eps : float
        Convergence threshold on u* (m/s). Default 1e-3 as in SNOWPACK.

    Returns
    -------
    dict with:
        u_star    : friction velocity (m/s)
        Tw_flux   : kinematic sensible heat flux (K m/s); positive = upward
        qw_flux   : kinematic moisture flux (kg/kg m/s); positive = upward (sublimation)
        zeta      : stability parameter (stab_ratio, or Richardson number for RICHARDSON)
        psi_m     : integrated stability correction for momentum
        psi_s     : integrated stability correction for scalars
        converged : 1 if converged, 0 if not
    """
    if stability not in SUPPORTED_STABILITY:
        raise ValueError(f"stability must be one of {SUPPORTED_STABILITY}, got '{stability}'")

    if rough_len_scalar is None:
        rough_len_scalar = rough_len_m

    if vw_ref == 0.:
        return {'u_star': 0., 'Tw_flux': 0., 'qw_flux': 0.,
                'zeta': np.nan, 'psi_m': np.nan, 'psi_s': np.nan, 'converged': np.nan}

    vw = max(0.3, vw_ref)

    # Virtual temperatures — SNOWPACK uses sat_vap at T_ref for both (approximation)
    sat_vap  = _vapor_saturation_pressure(T_ref)
    ta_v     = T_ref  * (1. + 0.377 * sat_vap / pressure)
    t_surf_v = T_surf * (1. + 0.377 * sat_vap / pressure)

    z_ratio        = np.log(z_ref / rough_len_m)
    z_ratio_scalar = np.log(z_ref / rough_len_scalar)

    psi_m, psi_s, stab_ratio = 0., 0., 0.
    ustar = _KARMAN * vw / z_ratio
    converged = 0

    for _ in range(max_iter):
        ustar_old = ustar

        if stability == 'NEUTRAL':
            psi_m, psi_s, stab_ratio = 0., 0., 0.
        elif stability == 'RICHARDSON':
            psi_m, psi_s, stab_ratio = _richardson_stability(ta_v, t_surf_v, z_ref, vw)
        else:
            psi_m, psi_s, stab_ratio = _mo_stability(
                stability, ta_v, t_surf_v, T_surf, z_ref, vw, z_ratio, psi_s, ustar
            )

        ustar = _KARMAN * vw / (z_ratio - psi_m)

        if abs(ustar_old - ustar) <= eps:
            converged = 1
            break

    if converged == 0:
        psi_m = psi_s = stab_ratio = 0.
        ustar = _KARMAN * vw / z_ratio

    # Bulk transfer coefficient — same z0 for momentum and scalars (SNOWPACK default)
    c_bulk = _KARMAN**2 / ((z_ratio - psi_m) * (z_ratio_scalar - psi_s))

    # Kinematic fluxes (positive = upward, away from surface)
    Tw_flux = -c_bulk * vw * (T_ref - T_surf)
    qw_flux = -c_bulk * vw * (qv_ref - qv_surf)

    return {
        'u_star':    ustar,
        'Tw_flux':   Tw_flux,
        'qw_flux':   qw_flux,
        'zeta':      stab_ratio,
        'psi_m':     psi_m,
        'psi_s':     psi_s,
        'converged': converged,
    }
