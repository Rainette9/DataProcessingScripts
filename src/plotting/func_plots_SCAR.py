"""
Plotting functions for the SCAR latent-heat-flux-divergence analysis.

Two figures:
  - plot_fluxes_and_conditions : multi-panel time series of MO/EC LE plus
    temperature, wind, RH (w.r.t. ice) and blowing-snow particle flux for a
    chosen period.
  - plot_lhf_and_blsn           : the same figure reduced to two panels, LHF and
    FlowCapt mass flux.
  - plot_le_divergence_signs   : LE divergence vs blowing-snow particle flux,
    coloured by the sign combination of LE_SFC and the upper-level LE.
  - plot_shf_divergence_wind_stability : median SHF divergence vs wind speed,
    binned by SFC stability z/L, as three panels (26-2, 16-2, 26-16 m).

Both are lifted from notebooks/MO_fluxes.ipynb and parameterised so the driver
notebook (notebooks/LE_div_SCAR.ipynb) only has to supply the already-read data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cmcrameri import cm as cmc

from utils.utils import resample_with_threshold, convert_RH_liquid_to_ice


# Default resampling settings (match MO_fluxes.ipynb).
KW_MO = dict(resample_time='30min', interpolate=True, max_gap='10min', min_valid_percent=80)
KW_EC = dict(resample_time='30min', min_valid_percent=90, max_gap='5min', interpolate=True)
KW_TEMP = dict(resample_time='30min', interpolate=True, min_valid_percent=80, max_gap='10min')
KW_BS = dict(resample_time='30min', interpolate=True, min_valid_percent=80, max_gap='10min')

# One fixed colour per measurement height.
HEIGHT_COLORS = {'SFC': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}

# Named cluster-regime colours.
REGIME_COLORS = {
    'Cold Katabatic':        '#2E86AB',
    'Transitional Synoptic': '#E67E22',
    'Warm Synoptic':         '#E74C3C',
}


# Stability classes from the SFC z/L (|z/L| <= ZL_NEUTRAL is neutral), Okabe-Ito colours.
ZL_NEUTRAL = 0.1
STAB_COLORS = {'unstable': '#D55E00', 'neutral': '#7F7F7F', 'stable': '#0072B2'}


def stability_series(zL, zl_neutral=ZL_NEUTRAL, kw=None):
    """Resampled z/L series -> 'unstable' / 'neutral' / 'stable' labels (NaN where z/L is missing)."""
    zl = resample_with_threshold(zL, **(kw or KW_EC))
    lab = pd.Series(np.select([zl < -zl_neutral, zl > zl_neutral, zl.notna()],
                              ['unstable', 'stable', 'neutral'], default=None), index=zl.index, dtype=object)
    lab[zl.isna()] = np.nan
    return lab


def _regime_spans(series, step='6h'):
    """Collapse a regime label series into ``(start, end, regime)`` spans."""
    spans = []
    current = None
    t_start = None
    for t, regime in series.items():
        if regime != current:
            if current is not None:
                spans.append((t_start, t, current))
            current = regime
            t_start = t
    if current is not None and t_start is not None:
        spans.append((t_start, series.index[-1] + pd.Timedelta(step), current))
    return spans


def _shade_regimes(fig, axes, regime_series, regime_colors, fs_leg, step='6h', alpha=0.2):
    """Shade every panel by the class in ``regime_series`` (cluster regime or stability class)
    and add a shared legend on top. ``step`` is the series' time step, ``alpha`` the opacity of
    the spans."""
    for ax in axes:
        for t_start, t_end, regime in _regime_spans(regime_series, step):
            if isinstance(regime, str):
                ax.axvspan(t_start, t_end, color=regime_colors.get(regime, '#cccccc'),
                           alpha=alpha, linewidth=0)

    regime_handles = [Patch(facecolor=regime_colors[r], edgecolor='none', alpha=max(alpha, 0.6), label=r)
                      for r in regime_colors if r in regime_series.values]
    if regime_handles:
        fig.legend(handles=regime_handles, loc='upper center', ncol=len(regime_handles),
                   fontsize=fs_leg, frameon=False, bbox_to_anchor=(0.5, 0.98))


def _plot_le_panel(ax, plot_start, plot_end, mo_results, ec_results,
                   colors, kw_mo, kw_ec, fs_leg, var='LE', ylabel='LHF [W/m²]'):
    """MO (dotted) and EC (solid) flux ``var`` (LE or H) for every level."""
    for res, lbl, height_key in mo_results:
        ax.plot(resample_with_threshold(res[var], **kw_mo)[plot_start:plot_end],
                label=lbl, color=colors[height_key], ls=':', lw=2)

    for res, lbl, height_key in ec_results:
        ax.plot(resample_with_threshold(res[var], **kw_ec)[plot_start:plot_end],
                label=lbl, color=colors[height_key], ls='-', alpha=0.8, lw=1.8)

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(ylabel)
    ax.legend(ncol=4, fontsize=fs_leg)
    ax.grid(True)


def _plot_blsn_panel(ax, plot_start, plot_end, slow_sfc, blsn_alg,
                     colors, kw_bs, kw_temp, fs_tick, fs_label, fs_leg):
    """FlowCapt particle mass flux (log scale) with BLSN cloud height on a twin axis."""
    ax.plot(resample_with_threshold(slow_sfc['PF_FC4'], **kw_bs)[plot_start:plot_end],
            label='PF_FC4', color=colors['SFC'])
    ax.set_ylabel('FlowCapt Mass flux [g/m²/s]')
    ax.legend(fontsize=fs_leg)
    ax.grid(True)
    ax.set_yscale('log')

    ax_bs = ax.twinx()
    bs_col = 'top_of_detected_BLSN'
    if bs_col in blsn_alg.columns:
        bs_series = resample_with_threshold(blsn_alg[bs_col], **kw_temp)[plot_start:plot_end]
        ax_bs.plot(bs_series, color='tab:purple', lw=1.5, label='BLSN cloud height')
        ax_bs.set_ylabel('BLSN cloud height [m]', color='tab:purple', fontsize=fs_label)
        ax_bs.tick_params(axis='y', labelcolor='tab:purple', labelsize=fs_tick)
        ax_bs.legend(loc='upper right', fontsize=fs_leg)
    else:
        ax.text(0.99, 0.95, 'BLSN height column missing', transform=ax.transAxes,
                ha='right', va='top', color='tab:purple', fontsize=fs_leg)
    return ax_bs


def plot_fluxes_and_conditions(
        plot_start, plot_end,
        result_sfc, result_up,
        fluxes_sfc, fluxes_low, fluxes_up,
        slow_sfc, slow_lower, slow_upper,
        cluster, blsn_alg,
        colors=HEIGHT_COLORS, regime_colors=REGIME_COLORS,
        kw_mo=None, kw_ec=None, kw_temp=None, kw_bs=None,
        shade_by='regime', stab_colors=STAB_COLORS, zl_neutral=ZL_NEUTRAL, show_H=True,
        drift_threshold=0.001, drift_ws_min=None, drift_color='0.45', drift_alpha=0.35, figsize=None):
    """Multi-panel overview of fluxes and conditions over ``[plot_start, plot_end]``.

    Panels (top to bottom): LE (MO dotted, EC solid), H (same, if ``show_H``), air/skin temperature,
    wind speed, RH w.r.t. ice and blowing-snow particle flux (log scale) with
    the detected BLSN cloud height on a twin axis. Background shading marks the
    cluster regime (``shade_by='regime'``, 6-hourly), the SFC stability class
    (``shade_by='stability'``: unstable / neutral / stable from the 2 m z/L,
    30 min), or blowing snow (``shade_by='drift'``: grey where the FlowCapt mass
    flux, resampled with ``kw_bs``, exceeds ``drift_threshold`` g m-2 s-1 and, if ``drift_ws_min`` is
    set, the 2 m wind speed of the same mean exceeds ``drift_ws_min`` m/s; ``drift_color`` and
    ``drift_alpha`` set the shade).
    ``figsize`` overrides the default ``(14, 3 * n_panels)``. Returns ``(fig, axes)``.
    """
    kw_mo = kw_mo or KW_MO
    kw_ec = kw_ec or KW_EC
    kw_temp = kw_temp or KW_TEMP
    kw_bs = kw_bs or KW_BS

    # Poster-friendly font sizes.
    FS_TICK, FS_LABEL, FS_LEG, FS_TITLE = 14, 16, 13, 18

    n_pan = 6 if show_H else 5
    fig, axes = plt.subplots(n_pan, 1, figsize=figsize or (14, 3 * n_pan), sharex=True)
    # the panels after the flux panel(s) shift down by one when H is shown
    ax_T, ax_WS, ax_RH, ax_BS = axes[n_pan - 4], axes[n_pan - 3], axes[n_pan - 2], axes[n_pan - 1]

    mo_results = [
        (result_sfc, 'MO ~2m',  'SFC'),
        (result_up,  'MO ~26m', '26m'),
    ]
    ec_results = [
        (fluxes_sfc, 'EC 2m',  'SFC'),
        (fluxes_low, 'EC 16m', '16m'),
        (fluxes_up,  'EC 26m', '26m'),
    ]

    # --- Background shading: cluster regime (6-hourly) or SFC stability class (30 min) ---
    if shade_by == 'stability':
        stab = stability_series(fluxes_sfc['(z-d)/L'], zl_neutral, kw_ec).loc[plot_start:plot_end]
        _shade_regimes(fig, axes, stab, stab_colors, FS_LEG, step=kw_ec['resample_time'])
    elif shade_by == 'regime':
        regime_series = cluster['regime'].loc[plot_start:plot_end]
        _shade_regimes(fig, axes, regime_series, regime_colors, FS_LEG)
    elif shade_by == 'drift':
        pf = resample_with_threshold(slow_sfc['PF_FC4'], **kw_bs).loc[plot_start:plot_end]
        lbl = f'blowing snow ({kw_bs["resample_time"]} mean mass flux > {drift_threshold:g} g m$^{{-2}}$ s$^{{-1}}$'
        drift = pf > drift_threshold
        if drift_ws_min is not None:      # drift only counts with the 2 m wind of the same mean above drift_ws_min
            ws = resample_with_threshold(slow_sfc['WS2_Avg'], **kw_bs).reindex(pf.index)
            drift &= ws > drift_ws_min
            lbl += f' and 2 m wind > {drift_ws_min:g} m s$^{{-1}}$'
        lbl += ')'
        drift_series = pd.Series(np.where(drift, lbl, np.nan), index=pf.index, dtype=object)
        _shade_regimes(fig, axes, drift_series, {lbl: drift_color}, FS_LEG, step=kw_bs['resample_time'],
                       alpha=drift_alpha)
    elif shade_by is not None:
        raise ValueError("shade_by must be 'regime', 'stability', 'drift' or None")

    # --- LE ---
    _plot_le_panel(axes[0], plot_start, plot_end, mo_results, ec_results,
                   colors, kw_mo, kw_ec, FS_LEG)
    if show_H:
        _plot_le_panel(axes[1], plot_start, plot_end, mo_results, ec_results,
                       colors, kw_mo, kw_ec, FS_LEG, var='H', ylabel='SHF [W/m²]')

    # --- Temperatures: air temp at all heights + skin temperature ---
    ax_T.plot(resample_with_threshold(slow_sfc['TA'], **kw_temp)[plot_start:plot_end],
                 label='T 2m', color=colors['SFC'])
    if 'Temp_16m_Avg' in slow_lower.columns:
        ax_T.plot(resample_with_threshold(slow_lower['Temp_16m_Avg'], **kw_temp)[plot_start:plot_end],
                     label='T 16m', color=colors['16m'])
    elif 'Temp_16m' in slow_lower.columns:
        ax_T.plot(resample_with_threshold(slow_lower['Temp_16m'], **kw_temp)[plot_start:plot_end],
                     label='T 16m', color=colors['16m'])
    if 'Temp_26m_Avg' in slow_upper.columns:
        ax_T.plot(resample_with_threshold(slow_upper['Temp_26m_Avg'], **kw_temp)[plot_start:plot_end],
                     label='T 26m', color=colors['26m'])
    if 'Tskin_comp' in slow_sfc.columns:
        ax_T.plot(resample_with_threshold(slow_sfc['Tskin_comp'] - 273.15, **kw_temp)[plot_start:plot_end],
                     label='Tskin [°C]', color=colors['SFC'], ls='--', lw=2)

    ax_T.set_ylabel('Temperature [°C]')
    ax_T.legend(ncol=3, fontsize=FS_LEG)
    ax_T.grid(True)

    # --- Wind speeds (all heights) ---
    if 'WS2_Avg' in slow_sfc.columns:
        ax_WS.plot(resample_with_threshold(slow_sfc['WS2_Avg'], **kw_temp)[plot_start:plot_end],
                     label='WS 2m', color=colors['SFC'])
    if 'wind_speed' in fluxes_low.columns:
        ax_WS.plot(resample_with_threshold(fluxes_low['wind_speed'], **kw_ec)[plot_start:plot_end],
                     label='WS 16m', color=colors['16m'])
    if 'WS_26m_Avg' in slow_upper.columns:
        ax_WS.plot(resample_with_threshold(slow_upper['WS_26m_Avg'], **kw_temp)[plot_start:plot_end],
                     label='WS 26m', color=colors['26m'])
    elif 'wind_speed' in fluxes_up.columns:
        ax_WS.plot(resample_with_threshold(fluxes_up['wind_speed'], **kw_ec)[plot_start:plot_end],
                     label='WS 26m', color=colors['26m'])

    ax_WS.set_ylabel('Wind speed [m/s]')
    ax_WS.legend(ncol=3, fontsize=FS_LEG)
    ax_WS.grid(True)

    # --- Relative humidity w.r.t. ice (all heights) ---
    # Capacitive probes report RH w.r.t. liquid water; convert to ice-referenced
    # RH using the co-located air temperature before resampling.
    if 'RH' in slow_sfc.columns:
        rh_ice_2m = convert_RH_liquid_to_ice(slow_sfc['RH'], slow_sfc['TA'])
        ax_RH.plot(resample_with_threshold(rh_ice_2m, **kw_temp)[plot_start:plot_end],
                     label='RH 2m', color=colors['SFC'])
    if 'RH_16m_Avg' in slow_lower.columns and 'Temp_16m_Avg' in slow_lower.columns:
        rh_ice_16m = convert_RH_liquid_to_ice(slow_lower['RH_16m_Avg'], slow_lower['Temp_16m_Avg'])
        ax_RH.plot(resample_with_threshold(rh_ice_16m, **kw_temp)[plot_start:plot_end],
                     label='RH 16m', color=colors['16m'])
    if 'RH_26m_Avg' in slow_upper.columns and 'Temp_26m_Avg' in slow_upper.columns:
        rh_ice_26m = convert_RH_liquid_to_ice(slow_upper['RH_26m_Avg'], slow_upper['Temp_26m_Avg'])
        ax_RH.plot(resample_with_threshold(rh_ice_26m, **kw_temp)[plot_start:plot_end],
                     label='RH 26m', color=colors['26m'])

    ax_RH.axhline(100, color='k', lw=0.8, ls='--')
    ax_RH.set_ylabel('RH w.r.t. ice [%]')
    ax_RH.legend(ncol=3, fontsize=FS_LEG)
    ax_RH.grid(True)

    # --- PF_FC4 (particle flux) at SFC, with BLSN cloud height on a twin axis ---
    _plot_blsn_panel(ax_BS, plot_start, plot_end, slow_sfc, blsn_alg,
                     colors, kw_bs, kw_temp, FS_TICK, FS_LABEL, FS_LEG)

    # Enlarge y-axis labels and tick labels on every panel for poster legibility.
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=FS_LABEL)
        ax.tick_params(axis='both', labelsize=FS_TICK)

    plt.tight_layout()
    return fig, axes


def plot_lhf_and_blsn(
        plot_start, plot_end,
        result_sfc, result_up,
        fluxes_sfc, fluxes_low, fluxes_up,
        slow_sfc, blsn_alg, cluster,
        colors=HEIGHT_COLORS, regime_colors=REGIME_COLORS,
        kw_mo=None, kw_ec=None, kw_temp=None, kw_bs=None,
        figsize=(14, 7)):
    """Two-panel version of :func:`plot_fluxes_and_conditions`.

    Same styling and regime shading, but only the LHF panel (MO dotted, EC
    solid) and the FlowCapt mass flux panel (log scale, with the detected BLSN
    cloud height on a twin axis). Returns ``(fig, axes)``.
    """
    kw_mo = kw_mo or KW_MO
    kw_ec = kw_ec or KW_EC
    kw_temp = kw_temp or KW_TEMP
    kw_bs = kw_bs or KW_BS

    # Poster-friendly font sizes (match plot_fluxes_and_conditions).
    FS_TICK, FS_LABEL, FS_LEG = 14, 16, 13

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    mo_results = [
        (result_sfc, 'MO ~2m',  'SFC'),
        (result_up,  'MO ~26m', '26m'),
    ]
    ec_results = [
        (fluxes_sfc, 'EC 2m',  'SFC'),
        (fluxes_low, 'EC 16m', '16m'),
        (fluxes_up,  'EC 26m', '26m'),
    ]

    regime_series = cluster['regime'].loc[plot_start:plot_end]
    _shade_regimes(fig, axes, regime_series, regime_colors, FS_LEG)

    _plot_le_panel(axes[0], plot_start, plot_end, mo_results, ec_results,
                   colors, kw_mo, kw_ec, FS_LEG)

    _plot_blsn_panel(axes[1], plot_start, plot_end, slow_sfc, blsn_alg,
                     colors, kw_bs, kw_temp, FS_TICK, FS_LABEL, FS_LEG)

    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=FS_LABEL)
        ax.tick_params(axis='both', labelsize=FS_TICK)

    plt.tight_layout()
    return fig, axes


def plot_le_divergence_signs(
        fluxes_sfc, fluxes_low, fluxes_up, slow_sfc,
        colors=HEIGHT_COLORS,
        start_scatter='2025-02-20', end_scatter='2025-06-26',
        exclude_start='2025-04-19', exclude_end='2025-04-20',
        kw_ec=None, kw_bs=None, layout='row', figsize=None):
    """LE divergence vs blowing-snow particle flux, coloured by sign combination.

    Heights: SFC = 2 m, LOWER = 16 m, UPPER = 26 m. Three panels of the LE
    difference between the upper and lower level (26 m − 2 m, 16 m − 2 m,
    26 m − 16 m), each plotted against PF_FC4 (log x). Points are coloured by
    the sign combination of the lower- and upper-level LE; a binned median with
    an interquartile band is overlaid per category. The window
    ``[exclude_start, exclude_end]`` is masked out.

    ``layout='row'`` (default) puts the three panels side by side, ``'column'``
    stacks them with a shared x axis. ``figsize`` overrides the per-layout
    default. Returns ``(fig, axes)``.
    """
    kw_ec = kw_ec or KW_EC
    kw_bs = kw_bs or KW_BS

    # Poster-friendly font sizes.
    FS_TICK, FS_LABEL, FS_LEG, FS_TITLE = 12, 13, 12, 15

    def _mask_period(series, t0, t1):
        mask = (series.index >= t0) & (series.index <= t1)
        out = series.copy()
        out[mask] = np.nan
        return out

    # --- Data ---
    fc_raw = resample_with_threshold(slow_sfc['PF_FC4'], **kw_bs)
    le_sfc = resample_with_threshold(fluxes_sfc['LE'], **kw_ec)
    le_26  = resample_with_threshold(fluxes_up['LE'],  **kw_ec)
    le_16  = resample_with_threshold(fluxes_low['LE'], **kw_ec)

    fc_bs_filt = _mask_period(fc_raw[start_scatter:end_scatter], exclude_start, exclude_end)

    def _mask(series):
        return _mask_period(series[start_scatter:end_scatter], exclude_start, exclude_end)

    # (LE difference, upper height, lower height, lower-level LE, colour).
    # SFC = 2 m (le_sfc), LOWER = 16 m (le_16), UPPER = 26 m (le_26).
    panels_filt = [
        (_mask(le_26 - le_sfc), '26 m', '2 m',  _mask(le_sfc), colors['26m']),
        (_mask(le_16 - le_sfc), '16 m', '2 m',  _mask(le_sfc), colors['16m']),
        (_mask(le_26 - le_16),  '26 m', '16 m', _mask(le_16),  colors['26m']),
    ]

    # le_upper is recovered as le_lower + le_diff since le_diff = le_upper - le_lower.
    cat_colors = {
        ('neg', 'pos'): ('#E69F00', r'LHF$_{lower}$<0, LHF$_{upper}$>0'),
        ('neg', 'neg'): ('#d62728', r'LHF$_{lower}$<0, LHF$_{upper}$<0'),
        ('pos', 'pos'): ('#1f77b4', r'LHF$_{lower}$>0, LHF$_{upper}$>0'),
        # ('pos', 'neg'): ('#999999', r'LHF$_{lower}$>0, LHF$_{upper}$<0'),
    }

    if layout not in ('row', 'column'):
        raise ValueError("layout must be 'row' or 'column'")
    column = layout == 'column'

    nrows, ncols = (3, 1) if column else (1, 3)
    figsize = figsize or ((7, 14) if column else (14, 6))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)

    for i, (ax, (diff_series, upper_lbl, lower_lbl, low_series, _color)) in enumerate(zip(axes, panels_filt)):
        title = f'{upper_lbl} − {lower_lbl}'
        up_tex, low_tex = upper_lbl.replace(' ', r'\,'), lower_lbl.replace(' ', r'\,')

        df = pd.concat([fc_bs_filt, diff_series, low_series.rename('le_low')], axis=1).dropna()
        df.columns = ['fc', 'le_diff', 'le_low']
        df = df[df['fc'] > 0]
        df['le_up'] = df['le_low'] + df['le_diff']

        df['cat'] = [
            ('neg' if s < 0 else 'pos') + '_' + ('neg' if u < 0 else 'pos')
            for s, u in zip(df['le_low'], df['le_up'])
        ]

        for (low_sign, up_sign), (c, base_lbl) in cat_colors.items():
            key = f'{low_sign}_{up_sign}'
            grp = df[df['cat'] == key]
            if grp.empty:
                continue
            lbl = base_lbl.replace('lower', low_tex).replace('upper', up_tex)
            ax.scatter(grp['fc'], grp['le_diff'],
                       color=c, alpha=0.45, s=14, linewidths=0, zorder=2, label=lbl)

            x = grp['fc'].values
            y = grp['le_diff'].values

            # logarithmic bins
            n_bins = 10
            bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)

            bin_centers = []
            medians = []
            q25 = []
            q75 = []

            for left, right in zip(bins[:-1], bins[1:]):
                if right == bins[-1]:
                    mask = (x >= left) & (x <= right)
                else:
                    mask = (x >= left) & (x < right)

                if np.sum(mask) < 3:   # skip sparsely populated bins
                    continue

                xx = x[mask]
                yy = y[mask]

                # x-position = median x in the bin (better than geometric center)
                bin_centers.append(np.median(xx))
                medians.append(np.median(yy))
                q25.append(np.percentile(yy, 25))
                q75.append(np.percentile(yy, 75))

            bin_centers = np.asarray(bin_centers)
            medians = np.asarray(medians)
            q25 = np.asarray(q25)
            q75 = np.asarray(q75)

            # shaded interquartile range
            ax.fill_between(bin_centers, q25, q75, color=c, alpha=0.20, zorder=1)

            # median line
            ax.plot(bin_centers, medians, '-o', color=c, lw=2, ms=5, zorder=3)

        ax.axhline(0, color='k', lw=0.8, ls='--', zorder=0)
        ax.set_xscale('log')
        # Stacked panels share the x axis, so only label the bottom one.
        if not column or i == len(panels_filt) - 1:
            ax.set_xlabel('Blowing snow mass flux [g m$^{-2}$ s$^{-1}$]', fontsize=FS_LABEL)
        if column:
            ax.set_ylabel('LHF difference  [W m$^{-2}$]', fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_LABEL)
        ax.grid(True, which='both', alpha=0.25, lw=0.6)
        ax.legend(fontsize=FS_LEG, framealpha=0.7, markerscale=2.2)
        ax.annotate(f'n = {len(df)}', xy=(0.03, 0.96), xycoords='axes fraction',
                    ha='left', va='top', fontsize=FS_LEG, color='0.4')
        ax.tick_params(axis='both', which='both', labelsize=FS_TICK)
        ax.set_ylim(-30, 110)

    if not column:
        axes[0].set_ylabel('LHF difference  [W m$^{-2}$]', fontsize=FS_LABEL)
    fig.suptitle(
        f'Latent heat flux (LHF) divergence vs blowing snow mass flux\n'
        f'{start_scatter}  –  {end_scatter} ',
        fontsize=FS_TITLE, y=0.995 if column else 0.96,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97] if column else None)
    return fig, axes


def plot_shf_divergence_wind_stability(
        fluxes_sfc, fluxes_low, fluxes_up,
        ws_bins=None, stability_bins=None,
        xlim=(0, 12), ylim=(-30, 70)):
    """Median SHF divergence vs wind speed, binned by SFC stability z/L.

    Three panels laid out like the LE-divergence figure: 26 m − 2 m, 16 m − 2 m
    and 26 m − 16 m. In each, the median sensible-heat-flux difference is plotted
    against wind speed (`wind_speed` of the lower level) for a set of SFC
    stability bins (`(z-d)/L`), coloured with the roma colormap. A single shared
    stability legend sits centred below the panels. Returns ``(fig, axes)``.
    """
    ws_bins = np.arange(0, 18, 0.5) if ws_bins is None else np.asarray(ws_bins)
    stability_bins = np.arange(-1, 1.5, 0.2) if stability_bins is None else np.asarray(stability_bins)
    bin_centers = (ws_bins[:-1] + ws_bins[1:]) / 2

    stability_labels = [f'{stability_bins[i]:.1f} to {stability_bins[i + 1]:.1f}'
                        for i in range(len(stability_bins) - 1)]
    colors = cmc.roma(np.linspace(0, 1, len(stability_labels)))

    # Poster-friendly font sizes (match the other figures).
    FS_TICK, FS_LABEL, FS_LEG, FS_TITLE = 14, 16, 13, 18

    def _dedup(s):
        # EddyPro output can carry duplicate timestamps; keep the first.
        return s[~s.index.duplicated(keep='first')]

    # SFC stability is the reference regime for every panel.
    stab_sfc = _dedup(fluxes_sfc['(z-d)/L'])

    def _median_profile(ax, df_low, df_high):
        # Align wind, both SHFs and SFC stability on a common (deduplicated) index.
        df = pd.concat([
            _dedup(df_low['wind_speed']).rename('ws'),
            _dedup(df_low['H']).rename('h_low'),
            _dedup(df_high['H']).rename('h_high'),
            stab_sfc.rename('stab'),
        ], axis=1)
        x = df['ws']
        y = df['h_high'] - df['h_low']
        stab = df['stab']
        for i, (left, right) in enumerate(zip(stability_bins[:-1], stability_bins[1:])):
            s_mask = (stab >= left) & (stab < right)
            medians = []
            for bl, br in zip(ws_bins[:-1], ws_bins[1:]):
                vals = y[(x >= bl) & (x < br) & s_mask].dropna()
                medians.append(np.median(vals) if len(vals) else np.nan)
            ax.plot(bin_centers, medians, marker='o', ms=5, lw=2.2,
                    color=colors[i], label=stability_labels[i])

    # (lower-level frame, upper-level frame, title); SFC = 2 m.
    panels = [
        (fluxes_sfc, fluxes_up,  '26 m − 2 m'),
        (fluxes_sfc, fluxes_low, '16 m − 2 m'),
        (fluxes_low, fluxes_up,  '26 m − 16 m'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, sharex=True)
    for ax, (df_low, df_high, title) in zip(axes, panels):
        _median_profile(ax, df_low, df_high)
        ax.axhline(0, color='k', lw=0.8, ls='--', zorder=0)
        ax.set_title(title, fontsize=FS_LABEL)
        ax.set_xlabel(r'Wind Speed [m s$^{-1}$]', fontsize=FS_LABEL)
        ax.grid(True, which='both', alpha=0.3, linestyle='--', lw=0.6)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel('SHF difference  [W m$^{-2}$]', fontsize=FS_LABEL)

    # Single shared stability legend, centred below the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Stability  z/L', loc='lower center',
               ncol=(len(labels) + 1) // 2, fontsize=FS_LEG, title_fontsize=FS_LABEL,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Sensible-heat flux (SHF) divergence vs wind speed by stability',
                 fontsize=FS_TITLE, y=1.0)

    plt.tight_layout(rect=[0, 0.12, 1, 0.97])
    return fig, axes
