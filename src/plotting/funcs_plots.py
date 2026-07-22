import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.dates as mdates
from ec import func_read_data
from utils.utils import (convert_RH_liquid_to_ice, resample_with_threshold,
                         RH_to_specific_humidity,
                         vapor_pressure_liquid_MK2005, vapor_pressure_ice_MK2005)
from utils.constants import epsilon


# One fixed colour per measurement height, matching the rest of this module.
HEIGHT_COLORS = ['royalblue', 'mediumseagreen', 'gold', 'tomato', 'orchid']


# --- manufacturer accuracy specifications ------------------------------------
# HygroVUE10 (Campbell Scientific), manual rev. 07/2021:
#   RH  +-1.5% (0-80% RH) / +-2% (80-100% RH) at 25 degC, plus <+-1% RH temperature
#       dependence over -40 to +60 degC
#   T   +-0.2 degC over -40 to 70 degC
# ATMOS 14 (METER Group), manual Figure 4/Figure 5:
#   RH  read off the accuracy grid; the grid's coldest column is 0 degC
#   T   from the accuracy curve, about +-0.7 degC at 0 degC rising to +-0.95 at -40 degC
_ATMOS14_RH_NODES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
_ATMOS14_RH_ACC_0C = [12, 8, 8, 5, 4, 4, 4, 4, 4, 4, 4,
                      4, 4, 4, 4, 4, 6, 5, 5, 5, 5]
_ATMOS14_T_NODES = [-40, -20, 0, 20, 40, 60, 80]
_ATMOS14_T_ACC = [0.95, 0.85, 0.70, 0.42, 0.42, 0.48, 0.80]


def sensor_accuracy(model, RH, T):
    """
    Manufacturer accuracy for an RH/T probe, as (sigma_RH in % points, sigma_T in degC).

    Note that the ATMOS 14 RH grid stops at 0 degC, so every Antarctic value is an
    extrapolation off the cold end of what METER specifies. Its 0 degC column is used here
    unchanged, which is optimistic: capacitive RH accuracy does not usually improve as the
    sensor gets colder.

    Parameters:
    model : {'HygroVUE10', 'ATMOS14'}
    RH : array-like
        Relative humidity (%).
    T : array-like
        Air temperature (degC).

    Returns:
    (sigma_RH, sigma_T)
    """
    RH = np.asarray(RH, dtype=float)
    T = np.asarray(T, dtype=float)
    if model == 'HygroVUE10':
        base = np.where(RH <= 80, 1.5, 2.0)
        # the temperature dependence is a separate, independent term in the manual
        return np.hypot(base, 1.0), np.full_like(T, 0.2)
    if model == 'ATMOS14':
        sig_rh = np.interp(RH, _ATMOS14_RH_NODES, _ATMOS14_RH_ACC_0C)
        sig_t = np.interp(T, _ATMOS14_T_NODES, _ATMOS14_T_ACC)
        return sig_rh, sig_t
    raise ValueError(f"unknown sensor model {model!r}")


def plot_mixing_ratio_with_spec_error(levels, pressure, start, end, models=None,
                                      resample_time=None, min_valid_percent=80,
                                      interpolate=False, max_gap='1h', ax=None):
    """
    Temperature, relative humidity and mixing ratio per arm, banded by manufacturer accuracy.

    Each panel carries the manufacturer's own accuracy for that quantity: sigma_T and
    sigma_RH straight off the spec sheet (see sensor_accuracy), and for specific humidity the
    two propagated together. Specific humidity uses RH_to_specific_humidity from utils, the
    same routine the flux/gradient code uses, so this figure and the K estimates are on one
    definition; its band is obtained by pushing +-sigma_RH and +-sigma_T through that routine
    and adding the two responses in quadrature.

    That band is the irreducible uncertainty of each point before any field
    cross-calibration, so it is the honest yardstick for whether a difference between two
    arms means anything at all.

    Parameters:
    levels : sequence of (T, RH, z) or (T, RH, z, model)
        Air temperature (degC), relative humidity (% wrt liquid) and surveyed height (m), so
        that the same list serves Kh_Kq_profile_fit and plot_RH_intercomparison. The sensor
        model may be appended per level, or supplied separately via models.
    pressure : pd.Series
        Air pressure (Pa).
    start, end : str or Timestamp
        Window to plot.
    models : sequence of str or None
        Sensor model per level, in the order given, as accepted by sensor_accuracy. Required
        unless every level already carries its model.
    resample_time : str or None
        Averaging window passed to resample_with_threshold, e.g. '15min'. None (the default)
        plots the data at its native resolution without resampling.
    min_valid_percent, interpolate, max_gap
        Forwarded to resample_with_threshold; ignored when resample_time is None.
    ax : sequence of 3 matplotlib axes, or None

    Returns:
    (fig, ax)
    """
    if models is not None:
        if len(models) != len(levels):
            raise ValueError(f"got {len(levels)} levels but {len(models)} models")
        levels = [(lv[0], lv[1], lv[2], m) for lv, m in zip(levels, models)]
    short = [i for i, lv in enumerate(levels) if len(lv) < 4]
    if short:
        raise ValueError(
            f"level(s) {short} carry no sensor model: pass models=[...] alongside the "
            f"3-element levels used by Kh_Kq_profile_fit, or append the model to each level")
    levels = sorted(levels, key=lambda lv: lv[2])
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    else:
        fig = ax[0].figure

    def _prep(series):
        s = pd.to_numeric(series, errors='coerce')
        s = s[~s.index.duplicated(keep='first')].sort_index().loc[start:end]
        if resample_time is None:
            return s
        return resample_with_threshold(s, resample_time, interpolate=interpolate,
                                       max_gap=max_gap, min_valid_percent=min_valid_percent)

    # No dropna anywhere: a NaN has to survive so a gap breaks the line instead of being
    # bridged by a straight segment that looks like data.
    frame = {}
    for i, (T, RH, _, _) in enumerate(levels):
        frame[f'T{i}'] = _prep(T)
        frame[f'RH{i}'] = _prep(RH)
    d = pd.DataFrame(frame)
    # Pressure comes off the flux grid (3 or 30 min) while the arms are on the slow grid
    # (1 min). Resampling puts both on one grid; without it, P would be absent on most slow
    # timestamps and would take r down with it, so it is interpolated onto the arms' index.
    # P varies slowly and enters r only as a 1/P scaling, so this is harmless either way.
    p = _prep(pressure)
    d['P'] = p.reindex(p.index.union(d.index)).interpolate(
        method='time', limit_direction='both').reindex(d.index)

    def _q(RH, T, P):
        # RH is liquid-referenced; RH_to_specific_humidity uses liquid saturation directly.
        return RH_to_specific_humidity(RH, T, P)

    def _q_and_sigma(RH, T, P, model):
        q = _q(RH, T, P)
        sig_rh, sig_t = sensor_accuracy(model, RH, T)
        # Propagate the spec accuracy through the very routine used for q, by a symmetric
        # perturbation on each input; the two 1-sigma responses add in quadrature.
        dq_rh = 0.5 * np.abs(_q(RH + sig_rh, T, P) - _q(RH - sig_rh, T, P))
        dq_t = 0.5 * np.abs(_q(RH, T + sig_t, P) - _q(RH, T - sig_t, P))
        sigma = np.hypot(dq_rh, dq_t)
        return (pd.Series(q, index=RH.index),
                pd.Series(sigma, index=RH.index))

    def _band(a, x, sigma, color):
        a.plot(x.index, x, color=color, linewidth=1.6)
        a.fill_between(x.index, x - sigma, x + sigma, color=color, alpha=0.18, linewidth=0)

    for i, (_, _, z_i, model) in enumerate(levels):
        T_i, RH_i = d[f'T{i}'], d[f'RH{i}']
        q_i, sig_q = _q_and_sigma(RH_i, T_i, d['P'], model)
        sig_rh, sig_t = sensor_accuracy(model, RH_i, T_i)
        c = HEIGHT_COLORS[i % len(HEIGHT_COLORS)]
        ax[0].plot(T_i.index, T_i, color=c, linewidth=1.6, label=f'{z_i:.2f} m  {model}')
        ax[0].fill_between(T_i.index, T_i - sig_t, T_i + sig_t, color=c, alpha=0.18,
                           linewidth=0)
        _band(ax[1], RH_i, pd.Series(sig_rh, index=RH_i.index), c)
        _band(ax[2], q_i * 1e3, sig_q * 1e3, c)

    ax[0].set_ylabel('air temperature [degC]')
    ax[0].set_title('Humidity arms with manufacturer accuracy bands'
                    + ('' if resample_time is None else f'  ({resample_time} means)'))
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].set_ylabel('RH wrt water [%]')
    ax[2].set_ylabel('specific humidity [g/kg]')
    for a in ax:
        a.grid(alpha=0.25)
    for lbl in ax[2].get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_horizontalalignment('right')

    fig.tight_layout()
    return fig, ax


def plot_RH_intercomparison(levels, pressure, ref_level=None, pair=None, trusted_pair=None,
                            offsets=np.arange(-8, 8.1, 0.5)):
    """
    Compare the humidity arms as distributions, and show how sensitive the moisture
    gradient is to a calibration offset on any one of them.

    The arms all measure RH with respect to liquid water, but Antarctic air commonly sits at
    saturation with respect to ice, which for a water-referenced sensor is a ceiling below
    100% that moves with temperature (100 * es_ice/es_liq, about 92% at -8 degC and 84% at
    -18 degC). Each probe levels off at its own value near that ceiling, and the spread
    between arms is a calibration difference rather than real structure in the air. Because
    Kq = -<w'q'> / (dq/dz) divides by a gradient, an offset of the same size as the true
    difference across the layer does not merely add scatter, it removes the signal.

    Panels:
        A  RH wrt water, per arm      - the different ceilings are visible directly
        B  RH wrt ice, per arm        - a well-behaved arm piles up at 100%, no further
        C  mixing ratio, per arm      - the wet/dry bias each offset produces
        D  mixing ratio difference against the reference arm, split by humidity - a span
           (multiplicative) error grows with humidity, a fixed offset does not
        E  the pair gradient as a function of an offset applied to the top arm
        F  Kq/Kh against the same offset, if fluxes are supplied

    Parameters:
    levels : sequence of (T, RH, z)
        Air temperature (degC), relative humidity (% wrt liquid) and surveyed height (m) for
        each arm, lowest first. Everything is aligned on the intersection of their indices.
    pressure : pd.Series
        Air pressure (Pa), e.g. eddypro_data['air_pressure'].
    ref_level : int or None
        Index into levels (after sorting by height) used as the reference arm that panels D
        and E difference against. Defaults to the second arm.
    pair : (int, int) or None
        Indices (lower, upper) of the arm pair whose gradient panel F sweeps. Use the pair
        that brackets the sonic, since that is the gradient Kq actually divides by. Defaults
        to the lowest and highest arms.
    trusted_pair : (int, int) or None
        Indices of two arms sharing one logger, whose gradient carries no cross-mast
        calibration difference. Drawn on panel F as the target the swept curve should meet,
        and the crossing gives the correction that would reconcile the two.
    offsets : array-like
        Hypothetical RH corrections (percentage points) swept in panel F.

    Returns:
    (fig, ax)
    """
    z = [lv[2] for lv in levels]
    order = np.argsort(z)
    levels = [levels[i] for i in order]
    z = np.asarray([lv[2] for lv in levels], dtype=float)
    if ref_level is None:
        ref_level = min(1, len(levels) - 1)

    # Align every arm and the pressure on a common index. Each series is de-duplicated first:
    # the slow files occasionally repeat a timestamp, and pandas cannot align on a duplicated
    # index ("cannot reindex on an axis with duplicate labels").
    def _dedup(series):
        series = pd.to_numeric(series, errors='coerce')
        return series[~series.index.duplicated(keep='first')].sort_index()

    frame = {'P': _dedup(pressure)}
    for i, (T, RH, _) in enumerate(levels):
        frame[f'T{i}'] = _dedup(T)
        frame[f'RH{i}'] = _dedup(RH)
    d = pd.DataFrame(frame).dropna()
    for i in range(len(levels)):
        d = d[(d[f'RH{i}'] > 1) & (d[f'RH{i}'] <= 100)]

    def _mixing_ratio(RH_liquid, T, P):
        """Mixing ratio (kg/kg) from RH with respect to liquid water."""
        e = RH_liquid / 100 * vapor_pressure_liquid_MK2005(T)
        return epsilon * e / (P - e)

    labels = [f'{zi:.2f} m' for zi in z]
    colors = [HEIGHT_COLORS[i % len(HEIGHT_COLORS)] for i in range(len(levels))]
    rh_ice = [convert_RH_liquid_to_ice(d[f'RH{i}'], d[f'T{i}']) for i in range(len(levels))]
    r = [_mixing_ratio(d[f'RH{i}'], d[f'T{i}'], d['P']) for i in range(len(levels))]

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()

    # --- A: RH wrt water -----------------------------------------------------
    for i in range(len(levels)):
        ax[0].hist(d[f'RH{i}'], bins=np.arange(0, 101, 2), histtype='step', linewidth=2,
                   density=True, color=colors[i], label=labels[i])
    ax[0].set_xlabel('RH wrt water [%]')
    ax[0].set_ylabel('density')
    ax[0].set_title('A  RH wrt water: each arm levels off\nat its own ceiling')
    ax[0].legend(fontsize=8)

    # --- B: RH wrt ice -------------------------------------------------------
    for i in range(len(levels)):
        ax[1].hist(rh_ice[i], bins=np.arange(0, 131, 2), histtype='step', linewidth=2,
                   density=True, color=colors[i], label=labels[i])
    ax[1].axvline(100, color='0.3', linestyle='--', linewidth=1.5)
    ax[1].annotate('ice saturation', xy=(100, ax[1].get_ylim()[1] * 0.9), xytext=(4, 0),
                   textcoords='offset points', fontsize=8, color='0.3')
    ax[1].set_xlabel('RH wrt ice [%]')
    ax[1].set_ylabel('density')
    ax[1].set_title('B  RH wrt ice: a sound arm stops at 100%,\none reading high overshoots')
    ax[1].legend(fontsize=8)

    # --- C: mixing ratio -----------------------------------------------------
    for i in range(len(levels)):
        ax[2].hist(r[i] * 1e3, bins=np.linspace(0, 3, 61), histtype='step', linewidth=2,
                   density=True, color=colors[i], label=f'{labels[i]}  med {r[i].median()*1e3:.3f}')
        ax[2].axvline(r[i].median() * 1e3, color=colors[i], linestyle=':', linewidth=1.5)
    ax[2].set_xlabel('mixing ratio [g/kg]')
    ax[2].set_ylabel('density')
    ax[2].set_title('C  derived mixing ratio: the arms are\nnearly indistinguishable in bulk')
    ax[2].legend(fontsize=8)

    # --- D: difference against the reference arm, as a distribution -----------
    ref = r[ref_level]
    rh_ref = d[f'RH{ref_level}']
    for i in range(len(levels)):
        if i == ref_level:
            continue
        diff = (r[i] - ref) * 1e3
        ax[3].hist(diff, bins=np.linspace(-0.4, 0.4, 81), histtype='step', linewidth=2,
                   density=True, color=colors[i],
                   label=f'{labels[i]} - {labels[ref_level]}   med {diff.median():+.3f}')
    ax[3].axvline(0, color='0.3', linestyle='--', linewidth=1.5)
    ax[3].set_xlabel(f'mixing ratio difference from {labels[ref_level]} [g/kg]')
    ax[3].set_ylabel('density')
    ax[3].set_title('D  arm-to-arm difference: wider than the\nreal vertical signal')
    ax[3].legend(fontsize=8)

    # --- E: is the difference a span error or a fixed offset? -----------------
    bins = [0, 40, 55, 70, 100]
    centres = [(bins[k] + bins[k + 1]) / 2 for k in range(len(bins) - 1)]
    for i in range(len(levels)):
        if i == ref_level:
            continue
        diff = (r[i] - ref) * 1e3
        med = diff.groupby(pd.cut(rh_ref, bins), observed=True).median()
        ax[4].plot(centres, med.values, 'o-', color=colors[i], linewidth=2, markersize=7,
                   label=f'{labels[i]} - {labels[ref_level]}')
    ax[4].axhline(0, color='0.3', linestyle='--', linewidth=1.5)
    ax[4].set_xlabel(f'RH at reference arm {labels[ref_level]} [%]')
    ax[4].set_ylabel('median mixing ratio difference [g/kg]')
    ax[4].set_title('E  difference grows with humidity =\nspan error, not a fixed offset')
    ax[4].legend(fontsize=8)

    # --- F: sensitivity of the pair gradient to an offset on the top arm ------
    # Sweep a hypothetical calibration correction on the top arm and read off the gradient it
    # would produce. x=0 is the data as it stands. The honest target is the gradient measured
    # by the two reference-mast arms, which share a logger and so carry no cross-mast offset.
    i_top, i_bot = pair if pair is not None else (0, len(levels) - 1)
    dz = z[i_top] - z[i_bot]
    grad = []
    for off in offsets:
        r_top = _mixing_ratio(d[f'RH{i_top}'] + off, d[f'T{i_top}'], d['P'])
        grad.append(((r_top - r[i_bot]) / dz).median())
    grad = np.asarray(grad) * 1e3
    ax[5].plot(offsets, grad, color=colors[i_top], linewidth=2, label='swept pair gradient')
    ax[5].axhline(0, color='0.3', linestyle='--', linewidth=1.5)
    ax[5].axvline(0, color='0.5', linestyle=':', linewidth=1.5)
    ax[5].annotate('your data\n(no correction)', xy=(0, grad[np.argmin(np.abs(offsets))]),
                   xytext=(6, -30), textcoords='offset points', fontsize=8, color='0.35',
                   arrowprops=dict(arrowstyle='->', color='0.5', linewidth=1))

    if trusted_pair is not None:
        a, b = trusted_pair
        g_trust = ((r[b] - r[a]) / (z[b] - z[a])).median() * 1e3
        ax[5].axhline(g_trust, color='0.2', linestyle='-.', linewidth=1.5,
                      label=f'within-mast {labels[a]}->{labels[b]}')
        need = np.interp(g_trust, grad, offsets) if grad[0] < grad[-1] else \
            np.interp(g_trust, grad[::-1], offsets[::-1])
        ax[5].plot([need], [g_trust], 'o', color='0.2', markersize=9, zorder=5)
        ax[5].annotate(f'{need:+.1f}% would reconcile\nwith the within-mast gradient',
                       xy=(need, g_trust), xytext=(8, 14), textcoords='offset points',
                       fontsize=8, color='0.2',
                       arrowprops=dict(arrowstyle='-', color='0.5', linewidth=1))

    # where the measured gradient passes through zero, Kq = -<w'q'>/(dr/dz) diverges
    sign_change = np.where(np.diff(np.sign(grad)))[0]
    if len(sign_change):
        x0 = np.interp(0, grad, offsets) if grad[0] < grad[-1] else \
            np.interp(0, grad[::-1], offsets[::-1])
        ax[5].plot([x0], [0], 'o', color='crimson', markersize=9, zorder=5)
        ax[5].annotate(f'gradient = 0 at {x0:+.1f}%\nKq diverges, sign flips',
                       xy=(x0, 0), xytext=(-4, 24), textcoords='offset points', fontsize=8,
                       color='crimson', ha='right',
                       arrowprops=dict(arrowstyle='-', color='crimson', linewidth=1))
    ax[5].set_xlabel(f'hypothetical RH correction on {labels[i_top]} [% points]')
    ax[5].set_ylabel(f'median d(r)/dz {labels[i_bot]}->{labels[i_top]} [g/kg/m]')
    title_f = f'F  what-if: how the {labels[i_bot]}->{labels[i_top]} gradient\nresponds to '\
              f'miscalibration of {labels[i_top]}'
    ax[5].set_title(title_f)
    ax[5].legend(fontsize=8, loc='lower right')

    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle('Humidity arm intercomparison: distributions and the sensitivity of the '
                 'moisture gradient to a calibration offset', fontsize=13)
    fig.tight_layout()
    return fig, ax



def plot_multilevel_slowdata_and_fluxes(
        slow_sfc, slow_bottom, slow_lower, slow_upper,
        fluxes_sfc, fluxes_bot, fluxes_low, fluxes_up,
        start, end,
        resample_time='10min', interpolate=False, interp_time='1h',
        bs_threshold=0.01):
    """
    Plots slow data and fluxes for all four measurement heights (SFC ~2m, BOTTOM ~5m,
    LOWER ~16m, UPPER ~26m) over a specified time range.

    Subplots:
        0 - Temperature at multiple heights
        1 - Relative Humidity at multiple heights
        2 - Wind Speed at multiple heights
        3 - Wind Direction (SFC only) + stability (z/L from SFC)
        4 - Net radiation at all available heights
        5 - FlowCapt blowing snow mass flux at all available heights
        6 - Sensible Heat Flux at 2, 5, 16, 26 m
        7 - Latent Heat Flux at 2, 16, 26 m

    Background highlights show periods where FlowCapt exceeds bs_threshold
    (g/m²/s), with a distinct colour per sensor height.
    """
    # ── color palette per measurement height ──────────────────────────────────
    c_2m  = 'deepskyblue'
    c_5m  = 'royalblue'
    c_10m = 'mediumseagreen'
    c_16m = 'gold'
    c_26m = 'tomato'

    # blowing-snow highlight colours (one per FlowCapt height)
    bs_colors = {
        'SFC':  ('skyblue',    0.20),   # ~surface
        '4m':   ('limegreen',  0.20),
        '11m':  ('orange',     0.18),
        '15m':  ('red',        0.15),
    }

    def _rs(series):
        return resample_with_threshold(series[start:end], resample_time)

    def _rs_flux(series):
        return resample_with_threshold(series[start:end], resample_time, interpolate, interp_time, min_valid_percent=65)

    def _col(df, col):
        return df[col] if col in df.columns else None

    def _bs_periods(series, threshold):
        """Return list of (t_start, t_end) where resampled series > threshold."""
        rs = resample_with_threshold(series[start:end], resample_time).fillna(0)
        mask = rs > threshold
        if not mask.any():
            return []
        changes = mask.astype(int).diff().fillna(0)
        t_starts = mask.index[changes == 1].tolist()
        t_ends   = mask.index[changes == -1].tolist()
        if mask.iloc[0]:
            t_starts = [mask.index[0]] + t_starts
        if mask.iloc[-1]:
            t_ends = t_ends + [mask.index[-1]]
        return list(zip(t_starts, t_ends))

    # ── collect FlowCapt series that exist ────────────────────────────────────
    fc_sources = []  # list of (label, series, highlight_key)
    fc_sources.append(('FC SFC (~0.5m)', slow_sfc['PF_FC4'], 'SFC'))
    for col, lbl, hkey in [
        ('FluxMean_4m',  'FC 4m',  '4m'),
        ('FluxMean_11m', 'FC 11m', '11m'),
        ('FluxMean_15m', 'FC 15m', '15m'),
    ]:
        s = _col(slow_lower, col)
        if s is not None:
            fc_sources.append((lbl, s, hkey))

    # ── build blowing-snow period spans for each source ───────────────────────
    bs_spans = {}   # hkey -> list of (t0, t1)
    for lbl, series, hkey in fc_sources:
        bs_spans[hkey] = _bs_periods(series, bs_threshold)

    # ── draw figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(8, 1, figsize=(13, 20), sharex=True)

    def _add_bs_highlights(a):
        for hkey, periods in bs_spans.items():
            color, alpha = bs_colors[hkey]
            for t0, t1 in periods:
                a.axvspan(t0, t1, color=color, alpha=alpha, linewidth=0)

    for a in ax:
        _add_bs_highlights(a)

    # ── 0: Temperature ────────────────────────────────────────────────────────
    ax[0].plot(_rs(slow_sfc['SFTempK'] - 273.15), label='T surf', color='darkblue', linestyle='dotted')
    ax[0].plot(_rs(slow_sfc['TA']),              label='T 2m',  color=c_2m)
    ax[0].plot(_rs(slow_bottom['Temp_5m_Avg']),  label='T 5m',  color=c_5m)
    for col, lbl, col_clr in [('Temp_10m_Avg', 'T 10m', c_10m),
                               ('Temp_16m_Avg', 'T 16m', c_16m)]:
        s = _col(slow_lower, col)
        if s is not None:
            ax[0].plot(_rs(s), label=lbl, color=col_clr)
    ax[0].plot(_rs(slow_upper['Temp_26m_Avg']), label='T 26m', color=c_26m)
    ax[0].set_ylabel(r'Temperature [$^o$C]')
    ax[0].legend(frameon=False, ncol=3)

    # ── 1: Relative Humidity ──────────────────────────────────────────────────
    ax[1].plot(_rs(convert_RH_liquid_to_ice(slow_sfc['RH'], slow_sfc['TA'])),
               label='RH 2m', color=c_2m)

    rh_5m = convert_RH_liquid_to_ice(slow_bottom['RH_5m_Avg'], slow_bottom['Temp_5m_Avg'])
    ax[1].plot(_rs(rh_5m), label='RH 5m', color=c_5m)

    for col, temp_col, lbl, col_clr in [('RH_10m_Avg', 'Temp_10m_Avg', 'RH 10m', c_10m),
                                        ('RH_16m_Avg', 'Temp_16m_Avg', 'RH 16m', c_16m)]:
        s = _col(slow_lower, col)
        t = _col(slow_lower, temp_col)
        if s is not None and t is not None:
            ax[1].plot(_rs(convert_RH_liquid_to_ice(s, t)), label=lbl, color=col_clr)

    rh_up = _col(slow_upper, 'RH_26m_Avg')
    temp_up = _col(slow_upper, 'Temp_26m_Avg')
    if rh_up is not None and temp_up is not None:
        ax[1].plot(_rs(convert_RH_liquid_to_ice(rh_up, temp_up)), label='RH 26m', color=c_26m)
    ax[1].set_ylabel('RH wrt ice [%]')
    ax[1].set_ylim(0, 115)
    ax[1].legend(frameon=False, ncol=3)

    # ── 2: Wind Speed ─────────────────────────────────────────────────────────
    ax[2].plot(_rs(slow_sfc['WS2_Avg']),        label='WS 2m',  color=c_2m)
    ax[2].plot(_rs(slow_bottom['WS_5m_Avg']),   label='WS 5m',  color=c_5m)
    for col, lbl, col_clr in [('WS_10m_Avg', 'WS 10m', c_10m),
                               ('WS_16m_Avg', 'WS 16m', c_16m)]:
        s = _col(slow_lower, col)
        if s is not None:
            ax[2].plot(_rs(s), label=lbl, color=col_clr)
    ax[2].plot(_rs(slow_upper['WS_26m_Avg']),   label='WS 26m', color=c_26m)
    ax[2].set_ylabel(r'Wind Speed [ms$^{-1}$]')
    ax[2].legend(frameon=False, ncol=3)

    # ── 3: Wind Direction (SFC) + stability ───────────────────────────────────
    wd = _rs(slow_sfc['WD1'])
    ax[3].scatter(wd[wd.between(45,  90)].index,  wd[wd.between(45,  90)],  s=10, color=c_2m, marker='s', label='WD (45-90)')
    ax[3].scatter(wd[wd.between(90, 180)].index,  wd[wd.between(90, 180)],  s=10, color=c_2m, marker='o', facecolors='none', label='WD (90-180)')
    ax[3].scatter(wd[~wd.between(45, 180)].index, wd[~wd.between(45, 180)], s=10, color=c_2m, marker='x')
    ax[3].scatter(_rs(slow_lower['WD_16m']).index, _rs(slow_lower['WD_16m']), s=10, color=c_16m, marker='x', label='WD 16m')
    ax[3].set_ylabel('Wind Direction')
    ax[3].set_ylim(0, 360)
    ax3r = ax[3].twinx()
    ax3r.scatter(_rs_flux(fluxes_sfc['(z-d)/L']).index,
                 _rs_flux(fluxes_sfc['(z-d)/L']), color='darkorange', s=5, marker='^', label='z/L SFC')
    ax3r.set_ylabel('z/L', color='darkorange')
    ax3r.set_ylim(-0.4, 2)
    ax3r.tick_params(axis='y', colors='darkorange')
    ax3r.yaxis.label.set_color('darkorange')
    ax3r.spines['right'].set_color('darkorange')
    ax[3].legend(frameon=False, ncol=2)
    ax3r.legend(frameon=False, loc='upper right')

    # ── 4: Net Radiation ──────────────────────────────────────────────────────
    ax[4].plot(_rs(-(slow_sfc['SWdown1'] - slow_sfc['SWup1'])),
               label='SW_net SFC', color=c_2m)
    ax[4].plot(_rs(-(slow_sfc['LWdown1'] - slow_sfc['LWup1'])),
               label='LW_net SFC', color=c_2m, linestyle='dashed')
    # BOTTOM: check Incoming/Outgoing style (same convention as LOWER) then fallback
    for sw_dn, sw_up, lw_dn, lw_up, lbl, col_clr in [
        ('Incoming_SW_5m_Avg',  'Outgoing_SW_5m_Avg',  'Incoming_LW_5m_Avg',  'Outgoing_LW_5m_Avg',  '5m',  c_5m),
        ('SWdown_5m_Avg', 'SWup_5m_Avg', 'LWdown_5m_Avg', 'LWup_5m_Avg', '5m', c_5m),
    ]:
        if all(c in slow_bottom.columns for c in [sw_dn, sw_up, lw_dn, lw_up]):
            ax[4].plot(_rs(-(slow_bottom[sw_dn] - slow_bottom[sw_up])),
                       label=f'SW_net {lbl}', color=col_clr)
            ax[4].plot(_rs(-(slow_bottom[lw_dn] - slow_bottom[lw_up])),
                       label=f'LW_net {lbl}', color=col_clr, linestyle='dashed')
            break
    # LOWER: known column names from logger header
    if all(c in slow_lower.columns for c in ['Incoming_SW_16m_Avg', 'Outgoing_SW_16m_Avg',
                                              'Incoming_LW_16m_Avg', 'Outgoing_LW_16m_Avg']):
        ax[4].plot(_rs(-(slow_lower['Incoming_SW_16m_Avg'] - slow_lower['Outgoing_SW_16m_Avg'])),
                   label='SW_net 16m', color=c_16m)
        ax[4].plot(_rs(-(slow_lower['Incoming_LW_16m_Avg'] - slow_lower['Outgoing_LW_16m_Avg'])),
                   label='LW_net 16m', color=c_16m, linestyle='dashed')
    # UPPER: same naming convention
    for sw_dn, sw_up, lw_dn, lw_up, lbl, col_clr in [
        ('Incoming_SW_26m_Avg', 'Outgoing_SW_26m_Avg', 'Incoming_UW_26m_Avg', 'Outgoing_UW_26m_Avg', '26m', c_26m),
        ('SWdown_26m_Avg', 'SWup_26m_Avg', 'LWdown_26m_Avg', 'LWup_26m_Avg', '26m', c_26m),
    ]:
        if all(c in slow_upper.columns for c in [sw_dn, sw_up, lw_dn, lw_up]):
            sw_net = _rs(-(slow_upper[sw_dn] - slow_upper[sw_up]))
            lw_net = _rs(-(slow_upper[lw_dn] - slow_upper[lw_up]))
            sw_net = sw_net[sw_net.between(-500, 500)]
            lw_net = lw_net[lw_net.between(-500, 500)]
            ax[4].plot(sw_net, label=f'SW_net {lbl}', color=col_clr)
            ax[4].plot(lw_net, label=f'LW_net {lbl}', color=col_clr, linestyle='dashed')
            break
    ax[4].set_ylabel(r'Net Radiation [Wm$^{-2}$]')
    ax[4].legend(frameon=False, ncol=3)

    # ── 5: FlowCapt blowing snow mass flux ────────────────────────────────────
    fc_line_colors = {'SFC': c_2m, '4m': c_5m, '11m': c_10m, '15m': c_16m}
    for lbl, series, hkey in fc_sources:
        ax[5].plot(_rs(series), label=lbl, color=fc_line_colors.get(hkey, 'grey'))
    ax[5].axhline(bs_threshold, color='black', linewidth=0.8, linestyle=':', label=f'threshold {bs_threshold}')
    ax[5].set_yscale('log')
    ax[5].set_ylabel(r'Mass Flux [gm$^{-2}$s$^{-1}$]')
    ax[5].legend(frameon=False, ncol=3)

    # ── 6: Sensible Heat Flux ─────────────────────────────────────────────────
    ax[6].plot(_rs_flux(fluxes_sfc['H']), label='H 2m',  color=c_2m)
    if fluxes_bot is not None:
        ax[6].plot(_rs_flux(fluxes_bot['H']), label='H 5m',  color=c_5m)
    if fluxes_low is not None:
        ax[6].plot(_rs_flux(fluxes_low['H']), label='H 16m', color=c_16m)
    if fluxes_up is not None:
        ax[6].plot(_rs_flux(fluxes_up['H']),  label='H 26m', color=c_26m)
    ax[6].set_ylabel(r'SHF [Wm$^{-2}$]')    
    ax[6].axhline(0, color='grey', linestyle='dashed', alpha=0.5)
    ax[6].legend(frameon=False, ncol=2)

    # ── 7: Latent Heat Flux ───────────────────────────────────────────────────
    ax[7].plot(_rs_flux(fluxes_sfc['LE']), label='LE 2m',  color=c_2m)
    if fluxes_low is not None:
        ax[7].plot(_rs_flux(fluxes_low['LE']), label='LE 16m', color=c_16m)
    if fluxes_up is not None:
        ax[7].plot(_rs_flux(fluxes_up['LE']),  label='LE 26m', color=c_26m)
    ax[7].set_ylabel(r'LE [Wm$^{-2}$]')
    ax[7].axhline(0, color='grey', linestyle='dashed', alpha=0.5)
    ax[7].legend(frameon=False, ncol=2)

    # ── legend for blowing-snow highlights ────────────────────────────────────
    from matplotlib.patches import Patch
    bs_legend = [Patch(facecolor=col, alpha=alpha, label=f'BS > {bs_threshold} ({hkey})')
                 for hkey, (col, alpha) in bs_colors.items()
                 if hkey in bs_spans and len(bs_spans[hkey]) > 0]
    if bs_legend:
        fig.legend(handles=bs_legend, loc='lower center', ncol=len(bs_legend),
                   frameon=True, title='Blowing snow highlights', fontsize=9,
                   title_fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f'{resample_time} resampled  {start} – {end}', y=0.995, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig, ax


def find_consecutive_periods(slowdata, SPC,  threshold=1, duration='4h', noBS=False):
    """
    Finds periods where both slowdata['PF_FC4'] and SPC['Corrected Mass Flux(kg/m^2/s)']
    are greater than a threshold for consecutive hours, while removing occurrences where
    slowdata['HS_Cor'] decreases by more than 1 per hour.

    Parameters:
        slowdata (pd.DataFrame): DataFrame containing 'PF_FC4' and 'HS_Cor' columns.
        SPC (pd.DataFrame): DataFrame containing 'Corrected Mass Flux(kg/m^2/s)' column.
        threshold (float): The threshold value to check against.
        duration (str): Minimum duration of consecutive periods (e.g., '3H').

    Returns:
        list: A list of tuples containing the start and end times of consecutive periods.
    """
    # Remove occurrences where 'HS_Cor' decreases by more than 1 per hour
    if noBS == False:
        # Find periods WITH blowing snow (values above threshold)
        hs_cor_diff = slowdata['HS_Cor'].resample('1h').mean().diff()
        slowdata = slowdata[hs_cor_diff.reindex(slowdata.index, method='ffill') >= -0.02/60] ### 2cm per hour
        slowdata = slowdata[slowdata['WS1_Avg'] > 3]
        # Create masks for values greater than the threshold
        mask_slowdata = slowdata['PF_FC4'] > threshold
        if SPC is not None:
            # mask_SPC = SPC['Corrected Mass Flux(kg/m^2/s)']  >= threshold /1000
            mask_SPC = SPC['Corrected Mass Flux(kg/m^2/s)']  >= 0
            # Combine masks to find periods where both conditions are met
            combined_mask = mask_slowdata & mask_SPC
        else:
            combined_mask = mask_slowdata
        resampled_mask = combined_mask.resample(duration).mean() > 0.7  # At least some values meet criteria
    else:  # noBS == True
        # Find periods WITHOUT blowing snow (values consistently below threshold)
        mask_slowdata = slowdata['PF_FC4'] < threshold
        if SPC is not None:
            mask_SPC = SPC['Corrected Mass Flux(kg/m^2/s)'] <= threshold / 1000
            # Combine masks to find periods where both conditions are met
            combined_mask = mask_slowdata & mask_SPC
        else:
            combined_mask = mask_slowdata
        # For no-BS periods, we want ALL values to be below threshold (mean close to 1.0)
        resampled_mask = combined_mask.resample(duration).mean() > 0.98  # 90% of values must meet criteria
    # Identify consecutive periods
    consecutive_periods = resampled_mask.astype(int).diff().fillna(0)
    start_times = resampled_mask[consecutive_periods == 1].index
    end_times = resampled_mask[consecutive_periods == -1].index

    # Ensure start and end times align correctly
    if len(end_times) < len(start_times):
        end_times = end_times.append(pd.Index([resampled_mask.index[-1]]))

    # Filter periods based on duration
    valid_periods = []
    for starts, ends in zip(start_times, end_times):
        if (ends - starts) >= pd.Timedelta(duration):
            valid_periods.append((starts, ends))

    return valid_periods




def plot_SFC_slowdata_and_fluxes(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, sensor, start, end, SPC=None, MO=None, resample_time='10min', interpolate=False, interp_time='1h'):
    """
    Plots slowdata and fluxes for a given sensor over a specified time range.

        
    """

    if SPC is not None:
        consecutive_periods = find_consecutive_periods(slowdata, SPC, threshold=1, duration='4h', noBS=False)
        filtered_periods = [period for period in consecutive_periods if period[0] >= pd.Timestamp(start) and period[1] <= pd.Timestamp(end)]

    fig, ax = plt.subplots(8, 1, figsize=(13, 18), sharex=True)

    for a in ax:
        if SPC is not None:
            for starts, ends in filtered_periods:
                a.axvspan(starts, ends, color='grey', alpha=0.2)

    ax[0].plot(resample_with_threshold(slowdata['SFTempK'][start:end] - 273.15, resample_time),
               label='TSurface', color='darkblue', alpha=0.8, linestyle='dashed')
    ax[0].plot(resample_with_threshold(slowdata['TA'][start:end], resample_time),
               label='TA', color='deepskyblue')
    # ax[0].plot(resample_with_threshold(fluxes_SFC['sonic_temperature'][start:end] - 273.15, resample_time),
    #            label='T_sonic_SFC', color='deepskyblue', alpha=0.8, linestyle='-.')
    ax[0].set_ylabel(r'Temperature [$^o$C]')
    # ax[0].plot(resample_with_threshold(fluxes_16m['sonic_temperature'][start:end] - 273.15, resample_time),
    #            label='T_16m', color='limegreen')
    # ax[0].plot(resample_with_threshold(fluxes_26m['sonic_temperature'][start:end] - 273.15, resample_time),
    #            label='T_26m', color='gold')
    ax[0].legend(frameon=False)

    ax[1].plot(resample_with_threshold(convert_RH_liquid_to_ice(slowdata['RH'], slowdata['TA'])[start:end],
                                       resample_time),
               label='RH', color='deepskyblue')
    ax[1].set_ylabel('RH wrt ice [%]')
    ax[1].legend(frameon=False)
    ax[1].set_ylim(0, 115)

    wd1 = resample_with_threshold(slowdata['WD1'][start:end], resample_time)
    wd2 = resample_with_threshold(slowdata['WD2'][start:end], resample_time)

    # WD1 markers
    ax[2].scatter(wd1[wd1.between(45, 90)].index, wd1[wd1.between(45, 90)],
                  label='WD1 (0-90)', s=10, color='deepskyblue', marker='s')
    ax[2].scatter(wd1[wd1.between(90, 180)].index, wd1[wd1.between(90, 180)],
                  label='WD1 (90-180)', s=10, color='deepskyblue',  marker='o', facecolors='none')
    ax[2].scatter(wd1[~wd1.between(45, 180)].index, wd1[~wd1.between(45, 180)],
                   s=10, color='deepskyblue', marker='x')

    # # WD2 markers
    # ax[2].scatter(wd2[wd2.between(45, 90)].index, wd2[wd2.between(45, 90)],
    #               label='WD2 (0-90)', s=10, color='darkblue', marker='s')
    # ax[2].scatter(wd2[wd2.between(90, 180)].index, wd2[wd2.between(90, 180)],
    #               label='WD2 (90-180)', s=10, color='darkblue', marker='o', facecolors='none')
    # ax[2].scatter(wd2[~wd2.between(45, 180)].index, wd2[~wd2.between(45, 180)],
    #               s=10, color='darkblue', marker='x')

    # Add a secondary y-axis (twinx) on the right
    ax2_right = ax[2].twinx()
    ax2_right.scatter(resample_with_threshold(fluxes_SFC['(z-d)/L'][start:end], resample_time).index, resample_with_threshold(fluxes_SFC['(z-d)/L'][start:end], resample_time, interpolate, interp_time),color='darkorange', label='z/L', s=5, marker='^')
    ax2_right.set_ylabel('z/L', color='darkorange')
    ax2_right.set_ylim(-0.4, 2)
    ax2_right.tick_params(axis='y', colors='darkorange')
    ax2_right.yaxis.label.set_color('darkorange')
    ax2_right.spines['right'].set_color('darkorange')
    ax2_right.legend(frameon=False, loc='upper right')

    ax[2].set_ylabel('Wind Direction')
    ax[2].legend(frameon=False)
    ax[2].set_ylim(0, 360)

    ax[3].plot(resample_with_threshold(slowdata['WS2_Avg'][start:end], resample_time),
               label='WS_2m', color='darkblue')
    ax[3].plot(resample_with_threshold(fluxes_SFC['wind_speed'][start:end], resample_time),
               label='WS_2m_sonic', color='royalblue', alpha=0.8, linestyle='dashed')
    # ax[3].plot(resample_with_threshold(slowdata['WS1_Avg'][start:end], resample_time),
            #    label='WS1_Avg', color='deepskyblue')
    if fluxes_16m is not None:
        ax[3].plot(resample_with_threshold(fluxes_16m['wind_speed'][start:end], resample_time),
                   label='WS_16m', color='limegreen')
    if fluxes_26m is not None:
        ax[3].plot(resample_with_threshold(fluxes_26m['wind_speed'][start:end], resample_time),
                   label='WS_26m', color='gold')
    ax[3].set_ylabel(r'Wind Speed [ms$^{-1}$]')
    ax[3].legend(frameon=False)

    ax[4].plot(resample_with_threshold(-(slowdata['SWdown1'] - slowdata['SWup1'])[start:end], resample_time),
               label='SW_net1', color='gold')
    ax[4].plot(resample_with_threshold(-(slowdata['LWdown1'] - slowdata['LWup1'])[start:end], resample_time),
               label='LW_net1', color='limegreen')
    ax[4].plot(resample_with_threshold(-(slowdata['SWdown2'] - slowdata['SWup2'])[start:end], resample_time),
               label='SW_net2', color='gold', linestyle='dashed', alpha=0.8)
    ax[4].plot(resample_with_threshold(-(slowdata['LWdown2'] - slowdata['LWup2'])[start:end], resample_time),
               label='LW_net2', color='limegreen', linestyle='dashed', alpha=0.8)
    ax[4].set_ylabel(r'Net Radiation [Wm$^{-2}$]')
    ax[4].legend(frameon=False)

    # ax[5].plot(resample_with_threshold(slowdata['HS_Cor'][start]-slowdata['HS_Cor'][start:end], resample_time),
    #            label='Snow height', color='deepskyblue')
    # ax[5].set_ylabel('Relative snow height [m]')
    # ax[5].legend(frameon=False)
    if SPC is not None:
        ax[5].plot(resample_with_threshold(SPC['Corrected Mass Flux(kg/m^2/s)'][start:end], resample_time)*1000, 
                    label='SPC Mass Flux', color='darkblue')
    ax[5].plot(resample_with_threshold(slowdata['PF_FC4'][start:end], resample_time),
               label='FlowCapt Mass Flux', color='deepskyblue')
    ax[5].legend(frameon=False)
    ax[5].set_ylabel(r'Mass flux [gm$^{-2}$s$^{-1}$]')

    ax[6].plot(resample_with_threshold(fluxes_SFC['H'][start:end], resample_time, interpolate, interp_time),
               label='H 2m', color='deepskyblue')
    if fluxes_16m is not None:
        ax[6].plot(resample_with_threshold(fluxes_16m['H'][start:end], resample_time, interpolate, interp_time),
                   label='H 16m', color='limegreen')
    if fluxes_26m is not None:
        ax[6].plot(resample_with_threshold(fluxes_26m['H'][start:end], resample_time, interpolate, interp_time),
                   label='H 26m', color='gold')
    # ax[6].plot(resample_with_threshold(MO['H'][start:end], resample_time, interpolate, interp_time),
            #    label='H MOST', color='red', alpha=0.8)
    ax[6].set_ylabel(r'SHF [Wm$^{-2}$]')
    # ax[6].set_ylim(-180, 80)
    ax[6].legend(frameon=False)

    ax[7].plot(resample_with_threshold(fluxes_SFC['LE'][start:end], resample_time, interpolate, interp_time),
               label='LE 2m', color='deepskyblue')
    if MO is not None:
        ax[7].plot(resample_with_threshold(MO['LE'][start:end], resample_time, interpolate, interp_time),
                  label='LE MOST', color='red', alpha=0.8)
    ax[7].set_ylabel(r'LE [Wm$^{-2}$]')
    ax[7].legend(frameon=False)


    fig.suptitle(f'{resample_time} resampled {start} - {end}', y=0.92, fontsize=16)
    # plt.savefig(f'./plots_months/{sensor}_{start}_slowdata_and_fluxes.png', bbox_inches='tight')
    return fig, ax

def check_log_profile(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, start, end, heights=[0,1.5,1.9,3.5,16,26], log=False):
    """
    Check the log profile for the slow data and fluxes.
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 6), sharey=True)
    time_diff=pd.Timestamp(end) - pd.Timestamp(start)
    # Wind Speed Profile
    wind_speeds = [0, resample_with_threshold(slowdata['WS2_Avg'][start:end], time_diff, True).mean(), 
                   resample_with_threshold(fluxes_SFC['wind_speed'][start:end], time_diff, True).mean(),
                   resample_with_threshold(slowdata['WS1_Avg'][start:end], time_diff, True).mean(), 
                   resample_with_threshold(fluxes_16m['wind_speed'][start:end], time_diff, True).mean(), 
                   resample_with_threshold(fluxes_26m['wind_speed'][start:end], time_diff, True).mean()]
    axes[0].scatter(wind_speeds, heights, label='Wind Speed Data Points')
    if log:
        log_wind_speeds = np.log(wind_speeds[1:])  # Exclude the first zero value
        log_heights = np.log(heights[1:])  # Exclude the first zero value
        slope, intercept = np.polyfit(log_wind_speeds, log_heights, 1)
        fitted_heights = np.exp(intercept) * np.array(wind_speeds[1:])**slope
        axes[0].plot(wind_speeds[1:], fitted_heights, label=f'Fit: slope={slope:.2f}', color='red')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel('Wind Speed (m/s)')
    axes[0].set_ylabel('Height (m)')
    # axes[0].legend()
    axes[0].set_title('Wind Speed Profile')

    # Temperature Profile
    temperatures = [
        resample_with_threshold(slowdata['SFTempK'][start:end] - 273.15, time_diff, True).mean(),
        resample_with_threshold(slowdata['TA'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_SFC['sonic_temperature'][start:end]- 273.15, time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['sonic_temperature'][start:end] - 273.15, time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['sonic_temperature'][start:end] - 273.15, time_diff, True).mean()
    ]
    axes[1].scatter(temperatures, heights[:3] + heights[4:], label='Temperature Data Points')
    axes[1].set_xlabel('Temperature (°C)')
    # axes[1].legend()
    axes[1].set_title('Temperature Profile')

    # Sensible Heat Flux Profile
    sensible_heat_fluxes = [
        resample_with_threshold(fluxes_SFC['H'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['H'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['H'][start:end], time_diff, True).mean()
    ]
    axes[2].scatter(sensible_heat_fluxes, [heights[2]] + heights[4:], label='Sensible Heat Flux Data Points')
    axes[2].set_xlabel('Sensible Heat Flux (W/m²)')
    # axes[2].legend()
    axes[2].set_title('Sensible Heat Flux Profile')

    # TKE Profile
    tke_fluxes = [
        resample_with_threshold(fluxes_SFC['TKE'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['TKE'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['TKE'][start:end], time_diff, True).mean()
    ]
    axes[3].scatter(tke_fluxes, [heights[2]] + heights[4:], label='TKE Data Points')
    axes[3].set_xlabel('TKE')
    # axes[3].legend()
    axes[3].set_title('TKE Profile')

    # TKE Profile
    tke_fluxes = [
        resample_with_threshold(fluxes_SFC['(z-d)/L'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_16m['(z-d)/L'][start:end], time_diff, True).mean(),
        resample_with_threshold(fluxes_26m['(z-d)/L'][start:end], time_diff, True).mean()
    ]
    axes[4].scatter(tke_fluxes, [heights[2]] + heights[4:], label='stability Data Points')
    axes[4].set_xlabel('stability')
    # axes[4].legend()
    axes[4].set_title('stability Profile')

    plt.suptitle(f'Wind, Temperature, and Sensible Heat Flux Profiles from {start} to {end}', fontsize=16, y=0.97)
    plt.tight_layout()
    # plt.savefig(f'./plots/log_profile_{start}_to_{end}.png', bbox_inches='tight')
    # plt.show()

def check_log_profiles(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m,consecutive_days, heights=[0,2,3,16,26], log=False):
    """
    Check the log profile for the slow data and fluxes.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
    for start,end in consecutive_days:
        # Wind Speed Profile
        wind_speeds = [0, slowdata['WS2_Avg'][start:end], slowdata['WS1_Avg'][start:end].mean(), fluxes_16m['wind_speed'][start:end].mean(), fluxes_26m['wind_speed'][start:end].mean()]
        axes[0].scatter(wind_speeds, heights, label='Wind Speed Data Points')
        if log:
            log_wind_speeds = np.log(wind_speeds[1:])  # Exclude the first zero value
            log_heights = np.log(heights[1:])  # Exclude the first zero value
            slope, intercept = np.polyfit(log_wind_speeds, log_heights, 1)
            fitted_heights = np.exp(intercept) * np.array(wind_speeds[1:])**slope
            axes[0].plot(wind_speeds[1:], fitted_heights, label=f'Fit: slope={slope:.2f}', color='red')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
        axes[0].set_xlabel('Wind Speed (m/s)')
        axes[0].set_ylabel('Height (m)')
        # axes[0].legend()
        axes[0].set_title('Wind Speed Profile')

        # Temperature Profile
        temperatures = [slowdata['SFTempK'][start:end].mean() - 273.15, slowdata['TA'][start:end].mean(), fluxes_16m['sonic_temperature'][start:end].mean() - 273.15, fluxes_26m['sonic_temperature'][start:end].mean() - 273.15]
        axes[1].scatter(temperatures, heights[:2] + heights[3:], label='Temperature Data Points')
        axes[1].set_xlabel('Temperature (°C)')
        # axes[1].legend()
        axes[1].set_title('Temperature Profile')

        # Sensible Heat Flux Profile
        sensible_heat_fluxes = [fluxes_SFC['H'][start:end].mean(), fluxes_16m['H'][start:end].mean(), fluxes_26m['H'][start:end].mean()]
        axes[2].scatter(sensible_heat_fluxes, [heights[1]] + heights[3:], label='Sensible Heat Flux Data Points')
        axes[2].set_xlabel('Sensible Heat Flux (W/m²)')
        # axes[2].legend()
        axes[2].set_title('Sensible Heat Flux Profile')

        # Sensible Heat Flux Profile
        heat_fluxes = [fluxes_SFC['TKE'][start:end].mean(), fluxes_16m['TKE'][start:end].mean(), fluxes_26m['TKE'][start:end].mean()]
        axes[3].scatter(heat_fluxes, [heights[1]] + heights[3:], label='Sensible Heat Flux Data Points')
        axes[3].set_xlabel('TKE')
        # axes[3].legend()
        axes[3].set_title('TKE') 

    plt.suptitle(f'Wind, Temperature, and Sensible Heat Flux Profiles from {start} to {end}', fontsize=16, y=0.97)
    plt.tight_layout()
    # plt.savefig(f'./plots/log_profile_{start}_to_{end}.png', bbox_inches='tight')
    # plt.show()


def plot_bi_monthly_mean_H(fluxes_SFC, fluxes_16m, fluxes_26m, heights, variable):
    """
    Plots the bi-monthly mean H with 25th and 75th percentiles for different heights.

    Parameters:
        fluxes_SFC (pd.DataFrame): DataFrame containing SFC flux data.
        fluxes_16m (pd.DataFrame): DataFrame containing 16m flux data.
        fluxes_26m (pd.DataFrame): DataFrame containing 26m flux data.
        heights (list): List of heights corresponding to SFC, 16m, and 26m.
        variable (str): The variable to plot.
    """
    # Add 'Month' and 'Day' columns to group data
    fluxes_SFC['Month'] = fluxes_SFC.index.month
    fluxes_SFC['Day'] = fluxes_SFC.index.day
    fluxes_16m['Month'] = fluxes_16m.index.month
    fluxes_16m['Day'] = fluxes_16m.index.day
    fluxes_26m['Month'] = fluxes_26m.index.month
    fluxes_26m['Day'] = fluxes_26m.index.day

    # Add 'Month_Name' column for labelingfluxes_SFC['TI']= np.sqrt(fluxes_SFC['u_var']**2 + fluxes_SFC['v_var']**2 + fluxes_SFC['w_var']**2) / fluxes_SFC['wind_speed']
    fluxes_SFC['Month_Name'] = fluxes_SFC.index.month_name()
    fluxes_16m['Month_Name'] = fluxes_16m.index.month_name()
    fluxes_26m['Month_Name'] = fluxes_26m.index.month_name()

    # Initialize the figure
    fig, axes = plt.subplots(2, 12, figsize=(30, 10), sharey=True, sharex=True)
    axes = axes.flatten()

    # Loop through each month and create subplots for the first and second halves
    for month in range(1, 13):
        for part in [1, 2]:  # 1 for first half, 2 for second half
            ax = axes[(month - 1) * 2 + (part - 1)]

            # Filter data for the current month and part
            if part == 1:
                sfc_part = fluxes_SFC[(fluxes_SFC['Month'] == month) & (fluxes_SFC['Day'] <= 15)][variable]
                m16_part = fluxes_16m[(fluxes_16m['Month'] == month) & (fluxes_16m['Day'] <= 15)][variable]
                m26_part = fluxes_26m[(fluxes_26m['Month'] == month) & (fluxes_26m['Day'] <= 15)][variable]
                title_suffix = " (1st Half)"
            else:
                sfc_part = fluxes_SFC[(fluxes_SFC['Month'] == month) & (fluxes_SFC['Day'] > 15)][variable]
                m16_part = fluxes_16m[(fluxes_16m['Month'] == month) & (fluxes_16m['Day'] > 15)][variable]
                m26_part = fluxes_26m[(fluxes_26m['Month'] == month) & (fluxes_26m['Day'] > 15)][variable]
                title_suffix = " (2nd Half)"

            # Calculate mean, 25th, and 75th percentiles
            means = [
                resample_with_threshold(sfc_part, '15D', True, '30min', 60).mean() if not sfc_part.empty else np.nan,
                resample_with_threshold(m16_part, '15D', True, '3h', 60).mean() if not m16_part.empty else np.nan,
                resample_with_threshold(m26_part, '15D', True, '3h', 60).mean() if not m26_part.empty else np.nan
            ]
            percentiles_25 = [
                resample_with_threshold(sfc_part, '1h', True, '30min', 60).quantile(0.25) if not sfc_part.empty else np.nan,
                resample_with_threshold(m16_part, '1h', True, '1h', 60).quantile(0.25) if not m16_part.empty else np.nan,
                resample_with_threshold(m26_part, '1h', True, '1h', 60).quantile(0.25) if not m26_part.empty else np.nan
            ]
            percentiles_75 = [
                resample_with_threshold(sfc_part, '1h', True, '30min', 60).quantile(0.75) if not sfc_part.empty else np.nan,
                resample_with_threshold(m16_part, '1h', True, '1h', 60).quantile(0.75) if not m16_part.empty else np.nan,
                resample_with_threshold(m26_part, '1h', True, '1h', 60).quantile(0.75) if not m26_part.empty else np.nan
            ]
            # Plot the means with whiskers
            # Ensure error bars are non-negative
            lower_error = np.maximum(0, np.array(means) - np.array(percentiles_25))
            upper_error = np.maximum(0, np.array(percentiles_75) - np.array(means))

            # Plot the means with whiskers
            ax.errorbar(
                means, heights,
                xerr=[lower_error, upper_error],
                fmt='o-', capsize=5, label='H'
            )

            # Set titles and labels
            if not fluxes_SFC['Month_Name'][fluxes_SFC['Month'] == month].empty:
                ax.set_title(f"{fluxes_SFC['Month_Name'][fluxes_SFC['Month'] == month].iloc[0]}{title_suffix}", fontsize=10)
            else:
                ax.set_title(f"Month {month}{title_suffix}", fontsize=10)
            if (month - 1) * 2 + (part - 1) % 4 == 0:  # First column
                ax.set_ylabel("Height (m)")
            if (month - 1) * 2 + (part - 1) >= 20:  # Last row
                ax.set_xlabel(f"{variable} (Mean ± IQR)")
            ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(f"HalfMonthly Mean {variable} with 25th and 75th Percentiles", fontsize=16, y=1.02)
    plt.savefig(f'./plots/BiMonthly_Mean_{variable}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_wind_speed_binned(fluxes_SFC, fluxes_16m, fluxes_26m, slowdata, heights, variable):
    """
    Plots the mean H with 25th and 75th percentiles for different heights, grouped into bins based on wind speed.

    Parameters:
        fluxes_SFC (pd.DataFrame): DataFrame containing SFC flux data.
        fluxes_16m (pd.DataFrame): DataFrame containing 16m flux data.
        fluxes_26m (pd.DataFrame): DataFrame containing 26m flux data.
        slowdata (pd.DataFrame): DataFrame containing wind speed data.
        heights (list): List of heights corresponding to SFC, 16m, and 26m.
        variable (str): The variable to plot.
    """
    # Define wind speed bins (0-20 m/s in steps of 2.5 m/s)
    bins = np.arange(0, 18, 3)
    bin_labels = [f"{bins[i]}-{bins[i+1]} m/s" for i in range(len(bins) - 1)]
    slowdata_mean = slowdata.resample('30min').mean()  # Resample slowdata to 30-minute intervals

    slowdata_mean['Wind_Speed_Bin'] = pd.cut(slowdata_mean['WS1_Avg'], bins=bins, labels=bin_labels, include_lowest=True)

    # Add wind speed bins to flux data
    fluxes_SFC['Wind_Speed_Bin'] = slowdata_mean['Wind_Speed_Bin']
    fluxes_16m['Wind_Speed_Bin'] = slowdata_mean['Wind_Speed_Bin']
    fluxes_26m['Wind_Speed_Bin'] = slowdata_mean['Wind_Speed_Bin']

    # Initialize the figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 8), sharey=True, sharex=True)
    axes = axes.flatten()

    # Loop through each wind speed bin and create subplots
    for i, wind_bin in enumerate(bin_labels):
        ax = axes[i]

        # Filter data for the current wind speed bin
        sfc_bin = fluxes_SFC[fluxes_SFC['Wind_Speed_Bin'] == wind_bin][variable]
        m16_bin = fluxes_16m[fluxes_16m['Wind_Speed_Bin'] == wind_bin][variable]
        m26_bin = fluxes_26m[fluxes_26m['Wind_Speed_Bin'] == wind_bin][variable]
        # Calculate mean, 25th, and 75th percentiles
        means = [
            fluxes_SFC.median() if not sfc_bin.empty else np.nan,
            fluxes_16m.median() if not m16_bin.empty else np.nan,
            fluxes_26m.median() if not m26_bin.empty else np.nan
        ]
        percentiles_25 = [
            fluxes_SFC.quantile(0.25) if not sfc_bin.empty else np.nan,
            fluxes_16m.quantile(0.25) if not m16_bin.empty else np.nan,
            fluxes_26m.quantile(0.25) if not m26_bin.empty else np.nan
        ]
        percentiles_75 = [
            fluxes_SFC.quantile(0.75) if not sfc_bin.empty else np.nan,
            fluxes_16m.quantile(0.75) if not m16_bin.empty else np.nan,
            fluxes_26m.quantile(0.75) if not m26_bin.empty else np.nan
        ]

        # Ensure error bars are non-negative
        lower_error = np.maximum(0, np.array(means) - np.array(percentiles_25))
        upper_error = np.maximum(0, np.array(percentiles_75) - np.array(means))

        # Plot the means with whiskers
        ax.errorbar(
            means, heights,
            xerr=[lower_error, upper_error],
            fmt='o-', capsize=5, label='H'
        )

        # Set titles and labels
        ax.set_title(f"Wind Speed Bin: {wind_bin}", fontsize=10)
        if i == 0:  # First column
            ax.set_ylabel("Height (m)")
        ax.set_xlabel(f"{variable} (Mean ± IQR)")
        ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(f"{variable} by Wind Speed Bins with 25th and 75th Percentiles", fontsize=16, y=1.02)
    plt.savefig(f'./plots/wind_speed_binned_{variable}.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_filtered_wind_speed(fluxes_SFC, fluxes_16m, fluxes_26m, slowdata, heights, variable):
    """
    Plots two subplots for filtered cases:
    1. When slowdata['PF_FC4'] > 0.1 and slowdata['WS1_Avg'] > 5.
    2. When slowdata['PF_FC4'] == 0 and slowdata['WS1_Avg'] > 5.

    Parameters:
        fluxes_SFC (pd.DataFrame): DataFrame containing SFC flux data.
        fluxes_16m (pd.DataFrame): DataFrame containing 16m flux data.
        fluxes_26m (pd.DataFrame): DataFrame containing 26m flux data.
        slowdata (pd.DataFrame): DataFrame containing slow data.
        heights (list): List of heights corresponding to SFC, 16m, and 26m.
        variable (str): The variable to plot.
    """
    # Filter data for the two cases
    # Define wind speed bins (0-20 m/s in steps of 2.5 m/s)

    # Case 1: Filter for PF_FC4 > 0.1 and WS1_Avg > 5
    slowdata_mean = slowdata.resample('30min').mean()  # Resample slowdata to 30-minute intervals
    case_BS_condition = (slowdata_mean['PF_FC4'] > 0.1) & (slowdata_mean['WS1_Avg'] > 5) & (slowdata_mean['WS1_Avg'] < 10)
    case_noBS_condition = (slowdata_mean['PF_FC4'] <= 0.00001) & (slowdata_mean['WS1_Avg'] > 5) & (slowdata_mean['WS1_Avg'] < 10)

    # Add bins to flux data based on conditions using .loc to avoid SettingWithCopyWarning
    fluxes_SFC.loc[:, 'BS_bin'] = np.where(case_BS_condition, 'BS', np.where(case_noBS_condition, 'no_BS', 'else'))
    fluxes_16m.loc[:, 'BS_bin'] = np.where(case_BS_condition, 'BS', np.where(case_noBS_condition, 'no_BS', 'else'))
    fluxes_26m.loc[:, 'BS_bin'] = np.where(case_BS_condition, 'BS', np.where(case_noBS_condition, 'no_BS', 'else'))


    # Initialize the figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    axes = axes.flatten()
    bin_labels = ['BS', 'no_BS', 'else']
    # Loop through each wind speed bin and create subplots
    for i, BS_bin in enumerate(bin_labels):
        
        ax = axes[i]

        # Filter data for the current wind speed bin
        sfc_bin = fluxes_SFC[fluxes_SFC['BS_bin'] == BS_bin][variable]
        m16_bin = fluxes_16m[fluxes_16m['BS_bin'] == BS_bin][variable]
        m26_bin = fluxes_26m[fluxes_26m['BS_bin'] == BS_bin][variable]
        # Calculate mean, 25th, and 75th percentiles
        means = [
            resample_with_threshold(sfc_bin, '30min', False, '30min', 50).mean() if not sfc_bin.empty else np.nan,
            resample_with_threshold(m16_bin, '30min', False, '30min', 50).mean() if not m16_bin.empty else np.nan,
            resample_with_threshold(m26_bin, '30min', False, '30min', 50).mean() if not m26_bin.empty else np.nan
        ]
        percentiles_25 = [
            resample_with_threshold(sfc_bin, '30min', False, '30min', 50).quantile(0.25) if not sfc_bin.empty else np.nan,
            resample_with_threshold(m16_bin, '30min', False, '30min', 50).quantile(0.25) if not m16_bin.empty else np.nan,
            resample_with_threshold(m26_bin, '30min', False, '30min', 50).quantile(0.25) if not m26_bin.empty else np.nan
        ]
        percentiles_75 = [
            resample_with_threshold(sfc_bin, '30min', False, '30min', 50).quantile(0.75) if not sfc_bin.empty else np.nan,
            resample_with_threshold(m16_bin, '30min', False, '30min', 50).quantile(0.75) if not m16_bin.empty else np.nan,
            resample_with_threshold(m26_bin, '30min', False, '30min', 50).quantile(0.75) if not m26_bin.empty else np.nan
        ]

        # Ensure error bars are non-negative
        lower_error = np.maximum(0, np.array(means) - np.array(percentiles_25))
        # lower_error = np.array(percentiles_25)
        upper_error = np.maximum(0, np.array(percentiles_75) - np.array(means))
        # upper_error = np.array(percentiles_75)

        # Plot the means with whiskers
        ax.errorbar(
            means, heights,
            xerr=[lower_error, upper_error],
            fmt='o-', capsize=5, label='H'
        )

        # Set titles and labels
        ax.set_title(f"{BS_bin}", fontsize=10)
        if i % 4 == 0:  # First column
            ax.set_ylabel("Height (m)")
        if i >= 4:  # Last row
            ax.set_xlabel(f"{variable} (Mean ± IQR)")
        ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.suptitle(f"{variable} by BS Bins with 25th and 75th Percentiles", fontsize=16, y=1.02)
    plt.savefig(f'./plots/BS_binned_{variable}.png', bbox_inches='tight', dpi=300)
    plt.show()
    # return case_BS, case_noBS