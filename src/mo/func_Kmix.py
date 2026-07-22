import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .func_mo import calc_phi_stable_holtslag, calc_phi_unstable_paulson_stearns_weidner
from utils.utils import (
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


def log_mean_height(z_low, z_up):
    """
    Height to which a finite difference between two levels actually attributes.

    A difference (x_up - x_low) / (z_up - z_low) approximates dx/dz at the log-mean height
    rather than at the arithmetic mean, because x varies as ln(z) in the surface layer.

    Parameters:
    z_low, z_up : float
        Lower and upper measurement heights (m), both > 0.

    Returns:
    float
        Log-mean height (m).
    """
    return (z_up - z_low) / np.log(z_up / z_low)


def Kh_Kq_gradient(eddypro_data, T_low, T_up, RH_low, RH_up, z_low, z_up,
                   resample_time='15min', qc_level=1, min_valid_percent=80, interpolate=True, max_gap='5min',
                   wts_cov_col='w/ts_cov', wh2o_cov_col='w/h2o_cov',
                   qc_wts_col='qc_H', qc_wh2o_col='qc_LE',
                   pressure_col='air_pressure', density_col='air_density',
                   min_dthetav_dz=1e-2, min_dq_dz=1e-6):
    """
    Eddy diffusivities for heat and moisture from EddyPro covariances over a slow-data gradient.

        Kh = -<w'Ts'>  / (d(theta_v)/dz)
        Kq = -<w'q'>   / (d(q)/dz)

    The flux comes from the sonic/IRGA at one height; the gradient is a finite difference
    between two slow-data levels that should straddle it. The difference attributes to
    log_mean_height(z_low, z_up), NOT to z_up or the arithmetic mean -- check that it lands
    on the sonic before trusting the result (e.g. 10 & 26 m -> 16.7 m, i.e. the 16 m sonic).

    Pairings follow "same variable on both sides": <w'Ts'> is the sonic temperature
    covariance, i.e. essentially the buoyancy flux, so it is divided by the virtual potential
    temperature gradient. Temperature is converted to potential temperature (+GAMMA_D * z)
    before differencing.

    Note that w/ts_cov and w/h2o_cov are the raw covariances: unlike H and LE they carry no
    WPL or spectral corrections (here w/h2o_cov runs ~4% below the corrected h2o_flux). Pass
    corrected fluxes via the *_col arguments if that matters for the application.

    Parameters:
    eddypro_data : pd.DataFrame
        EddyPro full output, from read_eddypro_data. Supplies the covariances, air_pressure
        and air_density.
    resample_time : str
        Averaging window that fluxes and slow data are both resampled onto with
        resample_with_threshold, e.g. '30min'. This sets the output time grid, so it does not
        need to match either input's native frequency (typically 3 or 30 min for the fluxes
        and 1 min for the slow data).
    T_low, T_up : pd.Series
        Slow-data air temperature (degC) at z_low and z_up.
    RH_low, RH_up : pd.Series
        Slow-data relative humidity (%) at z_low and z_up, with respect to liquid water.
    z_low, z_up : float
        Heights of the two slow-data levels (m).
    qc_level : int or None
        Discard covariances whose quality flag is >= qc_level, matching what
        read_eddypro_data does for H and LE (it leaves the raw covariances unflagged, so the
        gating has to happen here). EddyPro flags 0/1/2 = good/intermediate/poor, so the
        default of 1 keeps only flag 0. None disables the filter.
    wts_cov_col : str
        Column holding <w'Ts'> (m K s^-1).
    wh2o_cov_col : str
        Column holding <w'rho_h2o'> (mmol m^-2 s^-1), converted to kinematic <w'q'> here.
    qc_wts_col, qc_wh2o_col : str
        Flag columns applied to the two covariances. <w'Ts'> and <w'q'> have no flags of their
        own, so the flags of the fluxes derived from them (H and LE) stand in.
    pressure_col, density_col : str
        Columns holding air pressure (Pa) and air density (kg m^-3).
    min_valid_percent : float
        Minimum percentage of slow-data samples required within each flux averaging window.
    min_dthetav_dz, min_dq_dz : float
        Gradients smaller in magnitude than these are set to NaN. Near-neutral or well-mixed
        windows drive the gradient through zero, where K is unidentifiable and would
        otherwise return enormous values of arbitrary sign. The defaults are rough sensor
        noise floors for the 10/26 m pair (~2% RH and ~0.1 K over 16 m); re-derive them for
        other pairs, since a dz of a few metres raises the floor proportionally.

    Returns:
    pd.DataFrame
        Indexed on resample_time windows, with columns Kh, Kq (m^2 s^-1), dthetav_dz
        (K m^-1), dq_dz (kg kg^-1 m^-1), wq_kinematic (m s^-1) and Kq_over_Kh. The log-mean
        height is stored in .attrs['z_log_mean'].
    """
    if z_low <= 0 or z_up <= 0:
        raise ValueError("z_low and z_up must be > 0")
    if z_up <= z_low:
        raise ValueError("z_up must be above z_low")

    def _rs(series):
        """Average onto the common resample_time grid, dropping under-sampled windows."""
        series = series[~series.index.duplicated(keep='first')].sort_index()
        return resample_with_threshold(
            series, resample_time, interpolate=interpolate,
            max_gap=max_gap, min_valid_percent=min_valid_percent)

    def _qc(series, flag_col):
        """Drop poor-quality averaging windows before they enter an average."""
        if qc_level is None or flag_col not in eddypro_data.columns:
            return series
        return series.where(eddypro_data[flag_col] < qc_level)

    # A sonic without an IRGA yields no water vapour covariance, so Kq is simply not
    # available for that mast and only the heat side is returned.
    has_h2o = wh2o_cov_col in eddypro_data.columns

    # Building one DataFrame aligns fluxes and slow data on the union of their windows, so
    # the two sources do not have to share a native frequency or a time span.
    columns = {
        'wts': _rs(_qc(eddypro_data[wts_cov_col], qc_wts_col)),
        'pressure': _rs(eddypro_data[pressure_col]),
        'rho_air': _rs(eddypro_data[density_col]),
        'T_low': _rs(T_low),
        'T_up': _rs(T_up),
        'RH_low': _rs(RH_low),
        'RH_up': _rs(RH_up),
    }
    if has_h2o:
        columns['wh2o'] = _rs(_qc(eddypro_data[wh2o_cov_col], qc_wh2o_col))
    data = pd.DataFrame(columns)
    idx = data.index

    def _specific_humidity(RH_liquid, T):
        # RH is liquid-referenced; RH_to_specific_humidity uses liquid saturation directly.
        q = RH_to_specific_humidity(RH_liquid, T, data['pressure'])
        return pd.Series(q, index=idx)

    q_low = _specific_humidity(data['RH_low'], data['T_low'])
    q_up = _specific_humidity(data['RH_up'], data['T_up'])

    # Potential temperature, then virtual potential temperature to match <w'Ts'>
    theta_low = (data['T_low'] + 273.15) + GAMMA_D * z_low
    theta_up = (data['T_up'] + 273.15) + GAMMA_D * z_up
    thetav_low = theta_low * (1 + 0.61 * q_low)
    thetav_up = theta_up * (1 + 0.61 * q_up)

    dz = z_up - z_low
    dthetav_dz = (thetav_up - thetav_low) / dz
    dq_dz = (q_up - q_low) / dz

    Kh = -data['wts'] / dthetav_dz.where(dthetav_dz.abs() >= min_dthetav_dz)

    result = pd.DataFrame({
        'Kh': Kh,
        'dthetav_dz': dthetav_dz,
        'dq_dz': dq_dz,
    }, index=idx)
    if has_h2o:
        # mmol m^-2 s^-1 -> mol m^-2 s^-1 -> kg m^-2 s^-1 -> kinematic m s^-1 (kg kg^-1)
        wq_kin = data['wh2o'] * 1e-3 * M_W / data['rho_air']
        Kq = -wq_kin / dq_dz.where(dq_dz.abs() >= min_dq_dz)
        result['Kq'] = Kq
        result['wq_kinematic'] = wq_kin
        result['Kq_over_Kh'] = Kq / Kh
    result.attrs['z_log_mean'] = log_mean_height(z_low, z_up)
    result.attrs['has_h2o'] = has_h2o
    return result


def Kh_Kq_profile_fit(eddypro_data, levels, z_eval,
                      resample_time='15min', qc_level=1, min_valid_percent=80,
                      interpolate=True, max_gap='5min',
                      wts_cov_col='w/ts_cov', wh2o_cov_col='w/h2o_cov',
                      qc_wts_col='qc_H', qc_wh2o_col='qc_LE',
                      pressure_col='air_pressure', density_col='air_density',
                      min_dthetav_dz=1e-2, min_dq_dz=1e-6):
    """
    Kh and Kq with the gradient from a log-linear profile fit through three or more levels.

    Each averaging window is fitted with

        x(z) = a + b * ln(z) + c * z      ->      dx/dz = b / z + c

    for x = theta_v and x = q, and the gradient is evaluated at z_eval. Compared with
    Kh_Kq_gradient this removes the height-attribution error: a two-level difference
    attributes to the log-mean height of the pair, which generally is not the sonic height,
    whereas this evaluates the derivative at z_eval exactly. The ln(z) + z form also carries
    curvature, which a difference across a deep layer smears out.

    With four levels and three parameters the fit is overdetermined by one, so the residuals
    are informative: in near-neutral windows the levels should collapse onto the fit, and a
    residual that persists at one arm across many windows is that sensor's offset rather than
    noise. De-offset before trusting stable-bin gradients. With exactly three levels the fit
    is exactly determined and the residuals are identically zero, which buys no such check.

    Only windows where every level reports are fitted, so one dead arm drops the window.

    Parameters:
    eddypro_data : pd.DataFrame
        EddyPro full output, from read_eddypro_data.
    levels : sequence of (T, RH, z)
        One tuple per measurement level: air temperature (degC), relative humidity (%, with
        respect to liquid water) and surveyed height (m). Three levels minimum.
    z_eval : float
        Height at which to evaluate the gradient, i.e. the sonic height that produced the
        covariances (get_sensor_info(sensor, year)[2]['sonic']).
    resample_time, qc_level, *_col, min_valid_percent, min_dthetav_dz, min_dq_dz
        As for Kh_Kq_gradient.

    Returns:
    pd.DataFrame
        Kh, Kq (m^2 s^-1), dthetav_dz (K m^-1), dq_dz (kg kg^-1 m^-1), wq_kinematic (m s^-1),
        Kq_over_Kh, and per-level fit residuals resid_thetav_<z> (K) and resid_q_<z>
        (kg kg^-1). .attrs holds 'z_eval', 'z_levels' and 'fit_dof'.
    """
    if len(levels) < 3:
        raise ValueError("need at least 3 levels to fit a + b*ln(z) + c*z")
    z = np.asarray([lv[2] for lv in levels], dtype=float)
    if np.any(z <= 0):
        raise ValueError("all level heights must be > 0")
    if z_eval <= 0:
        raise ValueError("z_eval must be > 0")

    def _rs(series):
        series = series[~series.index.duplicated(keep='first')].sort_index()
        return resample_with_threshold(
            series, resample_time, interpolate=interpolate,
            max_gap=max_gap, min_valid_percent=min_valid_percent)

    def _qc(series, flag_col):
        if qc_level is None or flag_col not in eddypro_data.columns:
            return series
        return series.where(eddypro_data[flag_col] < qc_level)

    # A sonic without an IRGA yields no water vapour covariance, so Kq is simply not
    # available for that mast and only the heat side is returned.
    has_h2o = wh2o_cov_col in eddypro_data.columns

    columns = {
        'wts': _rs(_qc(eddypro_data[wts_cov_col], qc_wts_col)),
        'pressure': _rs(eddypro_data[pressure_col]),
        'rho_air': _rs(eddypro_data[density_col]),
    }
    if has_h2o:
        columns['wh2o'] = _rs(_qc(eddypro_data[wh2o_cov_col], qc_wh2o_col))
    for i, (T, RH, _) in enumerate(levels):
        columns[f'T{i}'] = _rs(T)
        columns[f'RH{i}'] = _rs(RH)
    data = pd.DataFrame(columns)
    idx = data.index

    # theta_v and q at every level, on the common grid
    thetav = np.empty((len(idx), len(levels)))
    q = np.empty((len(idx), len(levels)))
    for i, z_i in enumerate(z):
        T_i = data[f'T{i}']
        # RH is liquid-referenced; RH_to_specific_humidity uses liquid saturation directly.
        q_i = RH_to_specific_humidity(data[f'RH{i}'], T_i, data['pressure'])
        theta_i = (T_i + 273.15) + GAMMA_D * z_i
        q[:, i] = q_i
        thetav[:, i] = theta_i * (1 + 0.61 * q_i)

    # The design matrix is the same for every window, so one pseudo-inverse solves them all.
    X = np.column_stack([np.ones_like(z), np.log(z), z])
    X_pinv = np.linalg.pinv(X)
    # d/dz of [1, ln z, z] at z_eval, so that grad = coefficients . dX
    dX = np.array([0.0, 1.0 / z_eval, 1.0])

    def _fit_gradient(Y):
        complete = np.isfinite(Y).all(axis=1)
        coeff = np.full((len(Y), X.shape[1]), np.nan)
        coeff[complete] = Y[complete] @ X_pinv.T
        residual = np.full_like(Y, np.nan)
        residual[complete] = Y[complete] - coeff[complete] @ X.T
        return coeff @ dX, residual

    dthetav_dz, resid_thetav = _fit_gradient(thetav)
    dq_dz, resid_q = _fit_gradient(q)
    dthetav_dz = pd.Series(dthetav_dz, index=idx)
    dq_dz = pd.Series(dq_dz, index=idx)

    Kh = -data['wts'] / dthetav_dz.where(dthetav_dz.abs() >= min_dthetav_dz)

    result = pd.DataFrame({
        'Kh': Kh,
        'dthetav_dz': dthetav_dz,
        'dq_dz': dq_dz,
    }, index=idx)
    if has_h2o:
        # mmol m^-2 s^-1 -> mol m^-2 s^-1 -> kg m^-2 s^-1 -> kinematic m s^-1 (kg kg^-1)
        wq_kin = data['wh2o'] * 1e-3 * M_W / data['rho_air']
        Kq = -wq_kin / dq_dz.where(dq_dz.abs() >= min_dq_dz)
        result['Kq'] = Kq
        result['wq_kinematic'] = wq_kin
        result['Kq_over_Kh'] = Kq / Kh
    for i, z_i in enumerate(z):
        result[f'resid_thetav_{z_i:g}'] = resid_thetav[:, i]
        result[f'resid_q_{z_i:g}'] = resid_q[:, i]
    result.attrs['z_eval'] = z_eval
    result.attrs['has_h2o'] = has_h2o
    result.attrs['z_levels'] = z.tolist()
    result.attrs['fit_dof'] = len(levels) - X.shape[1]
    return result


