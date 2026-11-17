"""Latent-heat flux divergence between EC levels: table, categories, composites, driver ranking.

Used by ``notebooks/LE_divergence.ipynb``. The workflow is

1. :func:`build_divergence_table`  - one 30 min table of flux differences and candidate drivers
2. :func:`classify_divergence`     - divergent / none / convergent against an empirical noise floor
3. ``plot_*``                      - composites that separate blowing snow from its correlates
4. :func:`rank_divergence_drivers` - gradient-boosted model, time-blocked CV, grouped permutation
                                     importance

Sign convention: ``dLE = LE(upper) - LE(lower)``. ``dLE > 0`` ("divergent") means the upward
vapour flux grows with height, i.e. a vapour source in the layer, which is what sublimation of
suspended snow produces. ``dLE < 0`` is "convergent" (a vapour sink, e.g. deposition).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import (resample_with_threshold, convert_RH_liquid_to_ice, RH_to_specific_humidity,
                         vapor_pressure_liquid_MK2005, vapor_pressure_ice_MK2005, Lsubl)
from utils.constants import epsilon

# Floor used for log10(PF_FC4) when the FlowCapt reads exactly 0 (no drift)
LOG_PF_FLOOR = -6.0

CATEGORY_ORDER = ['convergent', 'none', 'divergent']
CATEGORY_COLORS = {'convergent': '#0072B2', 'none': '#BBBBBB', 'divergent': '#D55E00'}
BS_COLORS = {True: '#D55E00', False: '#7F7F7F'}   # blowing snow / no blowing snow

DRIVER_LABELS = {
    'logPF': r'log$_{10}$ mass flux [g m$^{-2}$ s$^{-1}$]',
    'blsn_top': 'BLSN layer top [m]',
    'blsn_cat': 'BLSN category',
    'WS_2m': r'wind speed 2 m [m s$^{-1}$]',
    'WS_26m': r'wind speed 26 m [m s$^{-1}$]',
    'ustar_2m': r'$u_*$ 2 m [m s$^{-1}$]',
    'RHi_2m': 'RH wrt ice 2 m [%]',
    'RHi_26m': 'RH wrt ice 26 m [%]',
    'qdef_2m': r'saturation deficit wrt ice 2 m [g kg$^{-1}$]',
    'qdef_26m': r'saturation deficit wrt ice 26 m [g kg$^{-1}$]',
    'TA_2m': 'air temperature 2 m [°C]',
    'Tskin': 'skin temperature [°C]',
    'dT_26_2': 'T(26 m) − T(2 m) [K]',
    'zL_2m': 'z/L at 2 m',
    'SWnet': r'SW net [W m$^{-2}$]',
    'LWnet': r'LW net [W m$^{-2}$]',
    'WD_sin': 'sin(wind direction)',
    'WD_cos': 'cos(wind direction)',
    'WD': 'wind direction [°]',
    'regime_code': 'regime (code)',
}

DEFAULT_DRIVER_GROUPS = {
    'blowing snow': ['logPF', 'blsn_top', 'blsn_cat'],
    'wind / turbulence': ['WS_2m', 'WS_26m', 'ustar_2m'],
    'humidity': ['RHi_2m', 'RHi_26m', 'qdef_2m', 'qdef_26m'],
    'temperature': ['TA_2m', 'Tskin', 'dT_26_2'],
    'stability': ['zL_2m'],
    'radiation': ['SWnet', 'LWnet'],
    'synoptic': ['WD_sin', 'WD_cos', 'regime_code'],
}


# ---------------------------------------------------------------------------
# 1. table
# ---------------------------------------------------------------------------
def _circular_mean_deg(series, resample_time):
    """Vector-mean wind direction [deg] per resample window."""
    rad = np.deg2rad(series)
    s = np.sin(rad).resample(resample_time).mean()
    c = np.cos(rad).resample(resample_time).mean()
    return np.rad2deg(np.arctan2(s, c)) % 360


def build_divergence_table(fluxes, sonic_heights, slow_sfc, slow_upper, blsn_alg, cluster,
                           start, end, kw_ec, kw_met, pairs,
                           bs_threshold=0.0, wd_exclude=None, kw_bs=None, kw_blsn=None, drift_ws_min=None):
    """One table of flux differences between EC levels plus candidate drivers.

    Parameters
    ----------
    fluxes : dict
        ``{'2m': fluxes_sfc, '16m': fluxes_low, '26m': fluxes_up}`` from ``read_eddypro_data``
        (QC flags are already applied there).
    sonic_heights : dict
        Sonic height [m] per level key. Only height *differences* are used, so the snow-depth
        correction (the same offset at every level) cancels.
    slow_sfc, slow_upper : DataFrame
        Slow logger data. ``slow_sfc`` needs ``PF_FC4, TA, RH, WS2_Avg, WD1, Tskin_comp`` and
        the radiation columns; ``slow_upper`` the 26 m T/RH/wind/pressure.
    blsn_alg : DataFrame
        Ceilometer blowing-snow algorithm output (``top_of_detected_BLSN``, ``category``).
    cluster : DataFrame
        6-hourly regime table with a ``regime`` column.
    start, end : str
        Analysis window.
    kw_ec, kw_met, kw_bs, kw_blsn : dict
        ``resample_with_threshold`` kwargs for the EddyPro fluxes, the slow met variables
        (T, RH, wind, radiation, pressure), the FlowCapt mass flux and the ceilometer BLSN
        output. ``kw_bs`` and ``kw_blsn`` default to ``kw_met``. All must share the same
        ``resample_time``, because the columns are joined on their timestamps.
    pairs : list of (upper_key, lower_key)
    bs_threshold : float
        ``PF_FC4 > bs_threshold`` marks a half-hour as blowing snow (``bs`` column).
    drift_ws_min : float or None
        If set, drift only counts when the 2 m wind speed of the same interval is above this [m/s]:
        where PF_FC4 > 0 in weaker (or unknown) wind, ``PF_FC4``, ``logPF`` and ``bs`` are set to NaN, so
        those intervals are neither drift nor no drift. The unmasked flux is kept in ``PF_FC4_raw``.
    wd_exclude : (lo, hi) or None
        Wind-direction sector [deg] to drop (mast flow distortion). ``lo > hi`` wraps over north.

    Returns
    -------
    DataFrame indexed by time. Columns: ``LE_<lvl>``, ``H_<lvl>``, ``ustar_<lvl>``, per pair ``dLE_<up>_<lo>``,
    ``dH_<up>_<lo>``, ``dLEdz_<up>_<lo>``; the driver columns named in ``DRIVER_LABELS``;
    ``PF_FC4``, ``bs``, ``regime``, ``stability``; humidity: ``q_2m``, ``q_26m``, ``qsat_ice_sfc`` and
    ``dq_2m``, ``dq_26m`` (= q − q_sat,ice(T_skin)) in g/kg.
    """
    kw_bs = kw_bs or kw_met
    kw_blsn = kw_blsn or kw_met
    steps = {k: pd.Timedelta(v['resample_time']) for k, v in
             [('kw_ec', kw_ec), ('kw_met', kw_met), ('kw_bs', kw_bs), ('kw_blsn', kw_blsn)]}
    if len(set(steps.values())) > 1:
        raise ValueError(f'all kw_* must share one resample_time, got {steps}; '
                         'set `resample_time` once in the config cell and build every kw_* from it')
    rt = kw_ec['resample_time']
    ec = lambda s: resample_with_threshold(s, **kw_ec)
    met = lambda s: resample_with_threshold(s, **kw_met)
    bs_ = lambda s: resample_with_threshold(s, **kw_bs)
    blsn = lambda s: resample_with_threshold(s, **kw_blsn)

    cols = {}
    for lvl, fl in fluxes.items():
        cols[f'LE_{lvl}'] = ec(fl['LE'])
        cols[f'H_{lvl}'] = ec(fl['H'])
        cols[f'ustar_{lvl}'] = ec(fl['u*'])

    f2 = fluxes['2m']
    # --- blowing snow
    pf = bs_(slow_sfc['PF_FC4'])
    cols['PF_FC4'] = pf
    cols['logPF'] = np.log10(pf.where(pf > 0)).fillna(LOG_PF_FLOOR).where(pf.notna())
    cols['blsn_top'] = blsn(blsn_alg['top_of_detected_BLSN'])
    cols['blsn_cat'] = blsn_alg['category'].resample(rt).median()   # categorical: median, not a threshold mean
    # --- wind / turbulence
    cols['WS_2m'] = met(slow_sfc['WS2_Avg'])
    cols['WS_26m'] = met(slow_upper['WS_26m_Avg'])
    # --- humidity (RH is liquid-referenced on the loggers)
    TA2, RH2 = met(slow_sfc['TA']), met(slow_sfc['RH'])
    TA26, RH26 = met(slow_upper['Temp_26m_Avg']), met(slow_upper['RH_26m_Avg'])
    P2 = ec(f2['air_pressure'])                      # Pa
    P26 = met(slow_upper['AP_26m_Avg'] * 1000)       # kPa -> Pa
    for tag, T, RH, P in [('2m', TA2, RH2, P2), ('26m', TA26, RH26, P26)]:
        d = pd.concat([T.rename('T'), RH.rename('RH'), P.rename('P')], axis=1)
        rhi = convert_RH_liquid_to_ice(d['RH'], d['T'])
        q = pd.Series(RH_to_specific_humidity(d['RH'], d['T'], d['P']), index=d.index)
        cols[f'RHi_{tag}'] = rhi
        cols[f'q_{tag}'] = q * 1e3                       # specific humidity [g/kg]
        # q_sat,ice - q, with q_sat,ice ~ q * 100 / RH_ice  [g/kg]
        cols[f'qdef_{tag}'] = q * (100.0 / rhi - 1.0) * 1e3
    # saturation specific humidity over ice at the radiometric surface temperature [g/kg], and the
    # vapour excess of the air over it: dq > 0 means the air holds more vapour than the surface can
    Tskin_C = met(slow_sfc['Tskin_comp']) - 273.15
    e_i = vapor_pressure_ice_MK2005(Tskin_C)
    cols['qsat_ice_sfc'] = epsilon * e_i / (P2 - (1 - epsilon) * e_i) * 1e3
    for tag in ['2m', '26m']:
        cols[f'dq_{tag}'] = cols[f'q_{tag}'] - cols['qsat_ice_sfc']
    # --- temperature
    cols['TA_2m'] = TA2
    cols['Tskin'] = met(slow_sfc['Tskin_comp']) - 273.15
    cols['dT_26_2'] = TA26 - TA2
    # --- stability
    cols['zL_2m'] = ec(f2['(z-d)/L'])
    # --- radiation (positive = gain for the surface)
    cols['SWnet'] = met(slow_sfc['SWdown1'] - slow_sfc['SWup1'])
    cols['LWnet'] = met(slow_sfc['LWdown1'] - slow_sfc['LWup1'])
    # --- synoptic
    wd = _circular_mean_deg(slow_sfc['WD1'], rt)
    cols['WD'] = wd
    cols['WD_sin'] = np.sin(np.deg2rad(wd))
    cols['WD_cos'] = np.cos(np.deg2rad(wd))

    tab = pd.concat(cols, axis=1)[start:end]

    regimes = list(pd.unique(cluster['regime'].dropna()))
    tab['regime'] = cluster['regime'].reindex(tab.index, method='ffill', limit=11)  # 6 h = 12 x 30 min
    tab['regime_code'] = tab['regime'].map({r: i for i, r in enumerate(sorted(regimes))})

    for up, lo in pairs:
        dz = sonic_heights[up] - sonic_heights[lo]
        tag = f'{up[:-1]}_{lo[:-1]}'
        tab[f'dLE_{tag}'] = tab[f'LE_{up}'] - tab[f'LE_{lo}']
        tab[f'dH_{tag}'] = tab[f'H_{up}'] - tab[f'H_{lo}']
        tab[f'dLEdz_{tag}'] = tab[f'dLE_{tag}'] / dz

    if drift_ws_min is not None:
        tab['PF_FC4_raw'] = tab['PF_FC4']
        weak = (tab['PF_FC4'] > 0) & ~(tab['WS_2m'] > drift_ws_min)
        tab.loc[weak, ['PF_FC4', 'logPF']] = np.nan
    tab['bs'] = (tab['PF_FC4'] > bs_threshold).where(tab['PF_FC4'].notna())

    if wd_exclude is not None:
        lo_, hi_ = wd_exclude
        in_sector = (tab['WD'].between(lo_, hi_) if lo_ <= hi_
                     else (tab['WD'] >= lo_) | (tab['WD'] <= hi_))
        flux_cols = [c for c in tab.columns if c.startswith(('LE_', 'H_', 'dLE', 'dH_'))]
        tab.loc[in_sector, flux_cols] = np.nan

    tab.attrs['pairs'] = [f'{up[:-1]}_{lo[:-1]}' for up, lo in pairs]
    return tab


def add_stability_class(tab, zL_neutral=0.1, col='zL_2m'):
    """Add a ``stability`` column (unstable / neutral / stable) from z/L."""
    zl = tab[col]
    tab['stability'] = np.select([zl < -zL_neutral, zl > zL_neutral, zl.notna()],
                                 ['unstable', 'stable', 'neutral'], default=None)
    tab.loc[zl.isna(), 'stability'] = np.nan
    return tab


# ---------------------------------------------------------------------------
# 1b. LI-7500 signal-strength screen
# ---------------------------------------------------------------------------
def li7500_diag_halfhourly(folder_hf, sensor, start, end, cache_file=None, verbose=True):
    """Half-hourly statistics of the LI-7500 diagnostic value from the Peddy fast files.

    The LI-7500 diagnostic byte holds four status flags in the upper bits (240 when all are OK)
    and the signal strength (AGC) in the lower four bits, in steps of 6.25 %. Peddy removes
    records with a value below 240 (hard failures) but does not look at the AGC, which rises when
    snow, rime or droplets obstruct the optical path.

    Reads ``<folder_hf>/<sensor>/<yyyymm>/<sensor>_Fastdata_proc_<date>_<HHMM>.dat`` (one complete
    hour of 100 ms data per file, so each half of a file is one half-hour). Returns a DataFrame
    indexed by half-hour start with

    - ``share_above``: share of records with a diagnostic value above the baseline
    - ``share_fail``:  share of records below 240 (already removed and interpolated by Peddy)
    - ``diag_mode``:   most frequent diagnostic value in the half-hour
    - ``h2o_std``, ``h2o_skew``: standard deviation and skewness of the processed H2O signal
    - ``n``: records in the half-hour

    The baseline is the most frequent diagnostic value of the whole period (``attrs['baseline']``).
    Reading takes a few minutes per sensor, so the result is cached in ``cache_file`` (CSV) and
    extended only by the hours that are not in it yet.
    """
    import glob, os, re
    cached = None
    if cache_file is not None and os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cached = cached[~cached.index.duplicated(keep='last')].sort_index()
    files = sorted(glob.glob(os.path.join(folder_hf, sensor, '*', f'{sensor}_Fastdata_proc_*.dat')))
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta('1D')
    rows, todo = [], []
    for f in files:
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.dat$', f)
        if not m:
            continue
        t = pd.Timestamp(f'{m.group(1)} {m.group(2)}:{m.group(3)}')
        if t0 <= t < t1 and (cached is None or t not in cached.index):
            todo.append((t, f))
    if verbose and todo:
        print(f'{sensor}: reading the diagnostic value from {len(todo)} hourly fast files ...')
    for t, f in todo:
        x = pd.read_csv(f, usecols=['H2O', 'diag_gas'])
        half = len(x) // 2
        for k, g in enumerate((x.iloc[:half], x.iloc[half:])):
            dg = g['diag_gas'].dropna()
            if dg.empty:   # keep a row anyway, so the hour counts as read and is not re-read next time
                rows.append(dict(time=t + pd.Timedelta(minutes=30 * k), n=len(g)))
                continue
            vals, counts = np.unique(dg.values.astype(int), return_counts=True)
            rows.append(dict(time=t + pd.Timedelta(minutes=30 * k), n=len(g), diag_mode=int(vals[counts.argmax()]),
                             share_fail=float((dg < 240).mean()),
                             **{f'c{int(v)}': int(c) for v, c in zip(vals, counts) if v >= 240},
                             h2o_std=g['H2O'].std(), h2o_skew=g['H2O'].skew()))
    out = pd.DataFrame(rows).set_index('time') if rows else None
    if cached is not None:
        out = cached if out is None else pd.concat([cached, out]).sort_index()
    if out is not None:
        out = out[~out.index.duplicated(keep='last')]
    if out is None:
        raise FileNotFoundError(f'no fast files for {sensor} between {start} and {end} under {folder_hf}')
    if cache_file is not None and todo:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        out.to_csv(cache_file)
    out = out[t0:t1].copy()
    ccols = [c for c in out.columns if re.fullmatch(r'c\d+', c)]
    out[ccols] = out[ccols].fillna(0)
    baseline = int(out[ccols].sum().idxmax()[1:])
    above = [c for c in ccols if int(c[1:]) > baseline]
    out['share_above'] = (out[above].sum(axis=1) / out['n']).where(out['diag_mode'].notna())
    out.attrs['baseline'] = baseline
    return out


def _align_diag(series, index):
    """Half-hourly diagnostic series on the (possibly finer) index of the table: each step takes
    the value of the half-hour it falls in."""
    return series.sort_index().reindex(index, method='ffill', tolerance=pd.Timedelta('30min'))


def apply_agc_screen(tab, diag, max_share, level='2m'):
    """Copy of ``tab`` with H and LE of one level set to NaN where the LI-7500 signal strength was
    above its baseline in more than ``max_share`` of the records, and the flux differences that
    involve that level recomputed. Only LE depends on the LI-7500, but H is removed too so that
    H + LE comparisons keep using the same half-hours."""
    out = tab.copy()
    flagged = (_align_diag(diag['share_above'], out.index) > max_share).fillna(False)
    out.loc[flagged, [f'LE_{level}', f'H_{level}']] = np.nan
    for pair in out.attrs.get('pairs', []):
        up, lo = (x + 'm' for x in pair.split('_'))
        if level in (up, lo):
            dz = out[f'dLE_{pair}'] / out[f'dLEdz_{pair}']
            out[f'dLE_{pair}'] = out[f'LE_{up}'] - out[f'LE_{lo}']
            out[f'dH_{pair}'] = out[f'H_{up}'] - out[f'H_{lo}']
            out[f'dLEdz_{pair}'] = out[f'dLE_{pair}'] / dz.median()
    out.attrs = dict(tab.attrs)
    return out


def agc_screen_summary(tab, diag, thresholds=(0.01, 0.05, 0.10, 0.25), level='2m'):
    """How many half-hours each ``max_share`` threshold would remove: overall, among half-hours
    with a valid LE at ``level``, per drift class, and among half-hours with negative LE."""
    share = diag['share_above'].reindex(tab.index)
    le = tab[f'LE_{level}']
    valid = le.notna() & share.notna()
    cls = drift_class(tab)
    rows = {}
    for thr in thresholds:
        fl = share > thr
        r = {'half-hours flagged': int((fl & valid).sum()),
             'share of valid LE': (fl & valid).sum() / valid.sum()}
        for c in DRIFT_CLASS_LABELS:
            m = valid & (cls == c)
            r[f'drift {c}'] = (fl & m).sum() / m.sum() if m.sum() else np.nan
        neg = valid & (le < 0)
        r['of LE < 0'] = (fl & neg).sum() / neg.sum() if neg.sum() else np.nan
        r['median LE flagged'] = le[fl & valid].median()
        r['median LE kept'] = le[~fl & valid].median()
        rows[f'> {thr:.0%} of records'] = r
    return pd.DataFrame(rows).T


def plot_agc_screen_example(tab, diag, start, end, max_share, level='2m', ref_level='16m',
                            height_colors=None, remove_flagged=False):
    """Time series of one period: drift flux, share of records with raised signal strength,
    H2O standard deviation, and LE at ``level``. With ``remove_flagged=False`` the flagged
    half-hours are marked on the LE line; with ``True`` they are taken out of the line and the
    removed periods are shaded grey; with ``'both'`` the original line and the screened line are
    drawn together."""
    height_colors = height_colors or {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}
    t, d = tab[start:end], diag[start:end]
    share = d['share_above'].reindex(t.index)
    flagged = share > max_share
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1.6]))
    axes[0].plot(t['PF_FC4'], color='k', lw=1.3)
    axes[0].set_ylabel('mass flux\n[g m$^{-2}$ s$^{-1}$]')
    axes[1].fill_between(share.index, 0, share * 100, step='mid', color='#D55E00', alpha=0.6)
    axes[1].axhline(max_share * 100, color='k', ls='--', lw=1, label=f'threshold {max_share:.0%}')
    axes[1].set_ylabel(f'records above\nbaseline {diag.attrs.get("baseline", "")} [%]')
    axes[1].legend(fontsize=8, frameon=False, loc='upper right')
    axes[2].plot(d['h2o_std'].reindex(t.index), color='0.3', lw=1.3)
    axes[2].set_ylabel('H$_2$O std\n[mmol m$^{-3}$]')
    le = t[f'LE_{level}']
    axes[3].plot(t[f'LE_{ref_level}'], color=height_colors.get(ref_level, '0.5'), lw=1.3, alpha=0.9,
                 label=f'LE {ref_level[:-1]} m')
    n_fl = int((flagged & le.notna()).sum())
    if remove_flagged:
        if remove_flagged == 'both':   # original line underneath, screened line on top
            axes[3].plot(le, color='#D55E00', lw=1.3, label=f'LE {level[:-1]} m, all data')
        axes[3].plot(le.where(~flagged), color=height_colors.get(level, '0.2'), lw=1.8,
                     label=f'LE {level[:-1]} m, flagged data removed')
        step = pd.Timedelta(tab.index[1] - tab.index[0])
        runs = flagged.ne(flagged.shift()).cumsum()[flagged]
        for k, (_, seg) in enumerate(runs.groupby(runs)):
            axes[3].axvspan(seg.index[0], seg.index[-1] + step, color='0.85', linewidth=0, zorder=0,
                            label=f'removed ({n_fl} half-hours)' if k == 0 else None)
    else:
        axes[3].plot(le, color=height_colors.get(level, '0.2'), lw=1.5, label=f'LE {level[:-1]} m')
        axes[3].scatter(le.index[flagged], le[flagged], s=28, color='#D55E00', zorder=3,
                        label=f'LE {level[:-1]} m, would be removed ({n_fl} half-hours)')
    axes[3].axhline(0, color='k', lw=0.6)
    axes[3].set_ylabel('LE [W m$^{-2}$]')
    axes[3].legend(fontsize=8, frameon=False, ncol=3, loc='upper left')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f'{start} → {end}: LI-7500 signal-strength screen at {level[:-1]} m', fontsize=10)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2. categories
# ---------------------------------------------------------------------------
def classify_divergence(tab, pair, noise_mask, quantile=0.9):
    """Label each half-hour divergent / none / convergent for one pair.

    The noise floor is the ``quantile`` of ``|dLE|`` inside ``noise_mask`` (periods where no
    divergence is expected: no drift, near-neutral). Returns ``(threshold, category Series)``.
    """
    d = tab[f'dLE_{pair}']
    ref = d[noise_mask].dropna()
    if len(ref) < 30:
        raise ValueError(f'only {len(ref)} reference samples for the noise floor of pair {pair}')
    thr = float(ref.abs().quantile(quantile))
    cat = pd.Series(np.select([d > thr, d < -thr], ['divergent', 'convergent'], default='none'),
                    index=tab.index, dtype=object)
    cat[d.isna()] = np.nan
    return thr, cat


def occurrence_table(tab, cat, by):
    """Share of each divergence category within each class of ``by`` (row-normalised), plus n."""
    d = pd.DataFrame({'cat': cat, 'by': tab[by]}).dropna()
    out = pd.crosstab(d['by'], d['cat'], normalize='index').reindex(columns=CATEGORY_ORDER, fill_value=0)
    out['n'] = d.groupby('by').size()
    return out


def plot_category_strip(tab, cats, regime_colors=None, figsize=(14, None)):
    """Time strip of the divergence category per pair, with dLE on top of it.

    ``cats`` is ``{pair: category Series}``.
    """
    n = len(cats)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] or 2.6 * n), sharex=True, squeeze=False)
    for ax, (pair, cat) in zip(axes[:, 0], cats.items()):
        d = tab[f'dLE_{pair}']
        for name in CATEGORY_ORDER:
            m = cat == name
            ax.scatter(d.index[m], d[m], s=5, color=CATEGORY_COLORS[name], linewidths=0,
                       label=f'{name} ({m.sum() / cat.notna().sum():.0%})')
        ax.axhline(0, color='k', lw=0.6)
        up, lo = pair.split('_')
        ax.set_ylabel(f'LE({up} m) − LE({lo} m)\n[W m$^{{-2}}$]')
        ax.legend(ncol=3, fontsize=8, frameon=False, loc='upper right', markerscale=2.5)
        ax.grid(True, alpha=0.25)
        if regime_colors is not None and 'regime' in tab:
            r = tab['regime']
            change = r.ne(r.shift()).cumsum()
            for _, seg in r.groupby(change):
                if isinstance(seg.iloc[0], str):
                    ax.axvspan(seg.index[0], seg.index[-1], color=regime_colors.get(seg.iloc[0], '#cccccc'),
                               alpha=0.10, linewidth=0)
    fig.tight_layout()
    return fig, axes


def plot_event_timeseries(tab, start, end, height_colors=None, cats=None, pair=None, figsize=(12, 17)):
    """Time series of the variables behind the divergence for one event window.

    Panels (one quantity per axis, shared time axis): LE and H per level, FlowCapt drift
    mass flux, ceilometer blowing-snow top, RH wrt ice at 2 m and 26 m (100 % marked),
    wind speed at 2 m and 26 m, air temperatures at 2 m and 26 m plus skin temperature, and
    z/L at 2 m. Steps with ``RHi_26m > 100`` are shaded on every panel. If ``cats`` and
    ``pair`` are given, convergent and divergent steps of that pair are marked as ticks
    under the LE panel.
    """
    hc = height_colors or {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}
    t = tab[start:end]
    levels = [lvl for lvl in ['2m', '16m', '26m'] if f'LE_{lvl}' in t]
    fig, axes = plt.subplots(8, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1.6, 1.2, 1, 1, 1.2, 1, 1.2, 1]})

    # LE and H per level
    for ax, var, lab in [(axes[0], 'LE', 'LE [W m$^{-2}$]'), (axes[1], 'H', 'H [W m$^{-2}$]')]:
        for lvl in levels:
            ax.plot(t[f'{var}_{lvl}'], color=hc[lvl], lw=1.6, label=f'{lvl[:-1]} m')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylabel(lab)
    if cats is not None and pair is not None:
        c = cats[pair][start:end]
        y = axes[0].get_ylim()[0]
        for name in ['convergent', 'divergent']:
            ix = c.index[c == name]
            axes[0].plot(ix, np.full(len(ix), y), '|', color=CATEGORY_COLORS[name], ms=8, mew=1.5,
                         label=f'{name} ({pair.replace("_", "−")} m)')
    axes[0].legend(ncol=5, fontsize=8, frameon=False, loc='upper left')

    # drift: FlowCapt mass flux and ceilometer blowing-snow top
    axes[2].plot(t['PF_FC4'], color='k', lw=1.4)
    axes[2].set_ylabel('mass flux\n[g m$^{-2}$ s$^{-1}$]')
    axes[2].set_ylim(bottom=0)
    if 'blsn_top' in t and t['blsn_top'].notna().any():
        axes[3].plot(t['blsn_top'], color='k', lw=1.4)
        axes[3].set_ylim(bottom=0)
    else:
        axes[3].text(0.5, 0.5, 'no ceilometer BLSN data in this window', transform=axes[3].transAxes,
                     ha='center', va='center', fontsize=9, color='0.4')
        axes[3].set_yticks([])
    axes[3].set_ylabel('ceilometer\nBLSN top [m]')

    # humidity wrt ice
    for lvl in ['2m', '26m']:
        if f'RHi_{lvl}' in t:
            axes[4].plot(t[f'RHi_{lvl}'], color=hc[lvl], lw=1.6, label=f'{lvl[:-1]} m')
    axes[4].axhline(100, color='k', lw=0.8, ls='--')
    axes[4].set_ylabel('RH wrt ice [%]')
    axes[4].legend(ncol=2, fontsize=9, frameon=False, loc='lower left')

    # wind
    for lvl in ['2m', '26m']:
        if f'WS_{lvl}' in t:
            axes[5].plot(t[f'WS_{lvl}'], color=hc[lvl], lw=1.6, label=f'{lvl[:-1]} m')
    axes[5].set_ylabel('wind speed\n[m s$^{-1}$]')
    axes[5].set_ylim(bottom=0)
    axes[5].legend(ncol=2, fontsize=9, frameon=False, loc='upper left')

    # temperatures
    axes[6].plot(t['TA_2m'], color=hc['2m'], lw=1.6, label='air 2 m')
    if 'dT_26_2' in t:
        axes[6].plot(t['TA_2m'] + t['dT_26_2'], color=hc['26m'], lw=1.6, label='air 26 m')
    if 'Tskin' in t:
        axes[6].plot(t['Tskin'], color='0.3', lw=1.4, ls=':', label='skin')
    axes[6].set_ylabel('temperature [°C]')
    axes[6].legend(ncol=3, fontsize=9, frameon=False, loc='upper left')

    # stability
    axes[7].plot(t['zL_2m'], color='k', lw=1.4)
    axes[7].axhline(0, color='k', lw=0.6)
    if t['zL_2m'].abs().max() > 0.3:          # symlog only when the range needs it
        axes[7].set_yscale('symlog', linthresh=0.1)
    axes[7].set_ylabel('z/L (2 m)')

    # shade ice supersaturation at 26 m
    sup = (t['RHi_26m'] > 100).values
    if sup.any():
        for ax in axes:
            ax.fill_between(t.index, 0, 1, where=sup, transform=ax.get_xaxis_transform(),
                            color='0.75', alpha=0.4, lw=0, step='post')

    axes[0].set_title(f'{start} → {end}   (shaded: RH$_{{ice}}$ at 26 m > 100 %)', fontsize=10)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes

def load_mo_fluxes(paths, require_converged=True):
    """Read the MO bulk-flux CSVs written by ``LE_div_SCAR.ipynb`` (10 min ``time`` index, columns
    ``u_star, Tw_flux, qw_flux, zeta, psi_m, psi_s, converged, LE, H``).

    ``paths`` is ``{level: csv_path}`` with the level tags used in the divergence table (``'2m'``
    for SFC, ``'26m'`` for UPPER). Rows whose MO iteration did not converge are set to NaN when
    ``require_converged`` is on. Returns ``{level: DataFrame}``.
    """
    out = {}
    for lvl, path in paths.items():
        d = pd.read_csv(path, parse_dates=['time'], index_col='time').sort_index()
        if require_converged and 'converged' in d:
            d.loc[d['converged'] != 1, ['LE', 'H', 'u_star']] = np.nan
        out[lvl] = d
    return out


def plot_event_zoom(tab, start, end, phases, sonic_heights, height_colors=None, phase_colors=None,
                    smooth='30min', le_ylim=None, mo=None, figsize=(13, 7.5)):
    """Zoom on a short window: LE per level with a running mean, drift flux and RH wrt ice on
    the left; mean LE profile (± 1 std) per phase on the right.

    ``phases`` is ``{label: (start, end)}``; each phase is shaded in the time series and drawn
    as one profile. ``smooth`` is the running-mean window laid over the raw values (the raw
    resolution of ``tab`` is kept as a thin line). ``sonic_heights`` as in
    :func:`build_divergence_table`.

    ``mo`` (optional) is ``{level: DataFrame}`` as returned by :func:`load_mo_fluxes`: the MO
    bulk LE of each level is laid over the EC series as a dashed line of the same colour, and
    its phase means are added to the profile panel as open markers joined by a dashed line.
    """
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    mo = mo or {}
    hc = height_colors or {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}
    default_pc = ['#D55E00', '#0072B2', '#7F7F7F', '#009E73', '#CC79A7']
    pc = phase_colors or {name: default_pc[i % len(default_pc)] for i, name in enumerate(phases)}
    levels = [lvl for lvl in ['2m', '16m', '26m'] if f'LE_{lvl}' in tab]
    t = tab[start:end]
    sm = lambda s: s.rolling(smooth, center=True, min_periods=3).mean()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.6, 1], height_ratios=[2, 1, 1], hspace=0.08, wspace=0.25)
    ax_le = fig.add_subplot(gs[0, 0])
    ax_pf = fig.add_subplot(gs[1, 0], sharex=ax_le)
    ax_rh = fig.add_subplot(gs[2, 0], sharex=ax_le)
    ax_prof = fig.add_subplot(gs[:, 1])

    for lvl in levels:
        ax_le.plot(t[f'LE_{lvl}'], color=hc[lvl], lw=0.7, alpha=0.35)
        ax_le.plot(sm(t[f'LE_{lvl}']), color=hc[lvl], lw=2, label=f'{lvl[:-1]} m')
    for lvl, d in mo.items():
        ax_le.plot(d['LE'][start:end], color=hc.get(lvl, 'k'), lw=1.6, ls='--')
    ax_le.axhline(0, color='k', lw=0.6)
    ax_le.set_ylabel('LE [W m$^{-2}$]')
    if le_ylim is not None:
        ax_le.set_ylim(*le_ylim)
    handles, labels = ax_le.get_legend_handles_labels()
    if mo:
        handles.append(Line2D([], [], color='k', lw=1.6, ls='--'))
        labels.append('MO bulk')
    ax_le.legend(handles, labels, ncol=len(handles), frameon=False, fontsize=9, loc='lower left',
                 title=f'thin: raw EC, thick: {smooth} mean EC', title_fontsize=8)
    for name, (a, b) in phases.items():
        for ax in (ax_le, ax_pf, ax_rh):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=pc[name], alpha=0.12, lw=0)

    ax_pf.plot(t['PF_FC4'], color='k', lw=1.3)
    ax_pf.set_ylabel('mass flux\n[g m$^{-2}$ s$^{-1}$]')
    ax_pf.set_ylim(bottom=0)

    for lvl in ['2m', '26m']:
        if f'RHi_{lvl}' in t:
            ax_rh.plot(t[f'RHi_{lvl}'], color=hc[lvl], lw=1.6, label=f'{lvl[:-1]} m')
    ax_rh.axhline(100, color='k', lw=0.8, ls='--')
    ax_rh.set_ylabel('RH wrt ice [%]')
    ax_rh.legend(ncol=2, frameon=False, fontsize=9, loc='lower left')
    ax_rh.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    for ax in (ax_le, ax_pf):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (ax_le, ax_pf, ax_rh):
        ax.grid(True, alpha=0.25)

    zs = [sonic_heights[lvl] for lvl in levels]
    for name, (a, b) in phases.items():
        e = tab[a:b]
        m = [e[f'LE_{lvl}'].mean() for lvl in levels]
        sd = [e[f'LE_{lvl}'].std() for lvl in levels]
        n = int(e[[f'LE_{lvl}' for lvl in levels]].notna().all(axis=1).sum())
        ax_prof.errorbar(m, zs, xerr=sd, color=pc[name], marker='o', ms=7, lw=2, capsize=3,
                         label=f'{name}, n={n}')
        if mo:
            mo_lvls = [lvl for lvl in levels if lvl in mo]
            mo_m = [mo[lvl]['LE'][a:b].mean() for lvl in mo_lvls]
            ax_prof.plot(mo_m, [sonic_heights[lvl] for lvl in mo_lvls], color=pc[name], ls='--', lw=1.5,
                         marker='o', ms=7, mfc='white', mew=1.5)
    if mo:
        ax_prof.plot([], [], color='k', ls='--', lw=1.5, marker='o', ms=7, mfc='white', mew=1.5,
                     label='MO bulk, phase mean')
    ax_prof.axvline(0, color='k', lw=0.6)
    ax_prof.set_xlabel('mean LE [W m$^{-2}$] (± 1 std)')
    ax_prof.set_ylabel('height [m]')
    ax_prof.set_yticks(zs)
    ax_prof.set_yticklabels([f'{z:.1f}' for z in zs])
    ax_prof.grid(True, alpha=0.25)
    ax_prof.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax_prof.set_title('LE profile per phase', fontsize=10)
    ax_le.set_title(f'{start} → {end}', fontsize=10)
    return fig, (ax_le, ax_pf, ax_rh, ax_prof)

def _fit_line(x, y):
    """Least-squares slope, intercept, r² and n of y against x (NaNs dropped)."""
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    slope, intercept = np.polyfit(x[m], y[m], 1)
    r2 = np.corrcoef(x[m], y[m])[0, 1] ** 2
    return slope, intercept, r2, int(m.sum())


def plot_surface_flux_law(tab, q_levels=('2m', '26m'), le_col='LE_2m', ustar_col='ustar_2m',
                          ustar_min=0.1, by_drift=True, height_colors=None, figsize=(12, 5.5)):
    """Observation version of the CRYOWRF "surface flux law" scatter.

    y = surface LE / u* [W m⁻² per m s⁻¹] (LE < 0 is deposition), x = q(z) − q_sat,ice(T_skin)
    [g/kg], one panel per humidity level in ``q_levels`` (``dq_<lvl>`` columns of ``tab``).
    Positive x: the air holds more vapour than the surface can, so the surface should deposit.
    Half-hours with u* below ``ustar_min`` are dropped (LE/u* blows up). With ``by_drift`` the
    points are split by the FlowCapt ``bs`` flag and a least-squares line is fitted per group;
    the fit over all points is always drawn.

    Returns ``(fig, axes, fits)`` with ``fits`` a DataFrame of slope, intercept, r² and n per
    panel and group.
    """
    hc = height_colors or {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}
    ok = tab[ustar_col] >= ustar_min
    y = (tab[le_col] / tab[ustar_col]).where(ok)
    groups = [('all', tab.index == tab.index, 'k', '-')]
    if by_drift and 'bs' in tab:
        groups += [('drift', tab['bs'] == True, BS_COLORS[True], '--'),
                   ('no drift', tab['bs'] == False, BS_COLORS[False], '--')]
    fig, axes = plt.subplots(1, len(q_levels), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, lvl in zip(axes, q_levels):
        x = tab[f'dq_{lvl}']
        if by_drift and 'bs' in tab:
            for name, m, col, _ in groups[1:]:
                ax.scatter(x[m], y[m], s=8, color=col, alpha=0.45, lw=0, label=name)
        else:
            ax.scatter(x, y, s=8, color=hc.get(lvl, '0.3'), alpha=0.45, lw=0)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        for name, m, col, ls in groups:
            slope, icpt, r2, n = _fit_line(x[m], y[m])
            rows.append({'q level': lvl, 'group': name, 'slope [W m-2 / (m s-1) per g/kg]': slope,
                         'intercept': icpt, 'r2': r2, 'n': n})
            if np.isfinite(slope):
                ax.plot(xs, slope * xs + icpt, color=col, ls=ls, lw=1.8,
                        label=f'{name}: slope {slope:.0f}, r² {r2:.2f}, n={n}')
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel(f'q({lvl[:-1]} m) − q$_{{sat,ice}}$(T$_{{skin}}$) [g kg$^{{-1}}$]')
        ax.set_title(f'vapour excess at {lvl[:-1]} m', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f'{le_col} / {ustar_col}  [W m$^{{-2}}$ per m s$^{{-1}}$]')
    fig.suptitle(f'surface flux law from observations (u* ≥ {ustar_min} m s$^{{-1}}$)', fontsize=11)
    fig.tight_layout()
    return fig, axes, pd.DataFrame(rows)

MODEL_RUN_COLORS = {'fmprog': '#D55E00', 'v202': '#0072B2'}
MODEL_CSV_COLUMNS = ['time', 'run', 'Q2_FM', 'QSAT_SFC', 'LH_SFC', 'LH_1.9m', 'UST', 'BS_DRIFT_FC']
MODEL_CSV_OPTIONAL = ['T2', 'T_FM2', 'TSK', 'QSAT_ICE_T2', 'LH_PART_1.9m']   # v2/v3 export; carried when present


def load_model_flux_law(csv_path, kw, drift_threshold=0.0, ustar_min=0.1):
    """Read the CRYOWRF column export for the surface-flux-law comparison and bring it onto the
    observation footing: 30 min ``resample_with_threshold`` per run, then
    x = (Q2_FM − QSAT_SFC)·1e3 [g/kg], y19 = LH_1.9m / UST, ysfc = LH_SFC / UST, and a drift flag
    ``BS_DRIFT_FC > drift_threshold``.

    The CSV holds 5 min values with columns ``MODEL_CSV_COLUMNS``: ``time`` (UTC), ``run`` (e.g.
    fmprog, v202), ``Q2_FM`` fine-mesh q at 2 m [kg/kg], ``QSAT_SFC`` = qsat_noah(TSK, PSFC)
    [kg/kg], ``LH_SFC`` surface latent heat flux and ``LH_1.9m`` = surface LH + blowing-snow
    sublimation below 1.9 m [W m⁻², positive up], ``UST`` [m/s], ``BS_DRIFT_FC`` FlowCapt-like
    drift mass flux [g m⁻² s⁻¹]. Returns a DataFrame indexed by time with columns
    ``run, x, y19, ysfc, drift``, plus ``ypart`` = particle sublimation below the sensor per unit
    u*: ``LH_PART_1.9m``/UST when the v3 export carries it, else (LH_1.9m − LH_SFC)/UST. Note
    that in v3 ``LH_1.9m`` is the flux the run's own mesh carries at 1.9 m: for fmprog that is
    surface + particles below, for v202 (constant-flux diagnostic mesh) it equals ``LH_SFC``, its
    particle vapour being injected at the first WRF level instead. When the v2 columns ``MODEL_CSV_OPTIONAL`` are
    present (``T2``, ``T_FM2``, ``TSK`` [K], ``QSAT_ICE_T2`` = qsat_ice(T_FM2, PSFC) [kg/kg]) it also
    returns ``x_air`` = (Q2_FM − QSAT_ICE_T2)·1e3 [g/kg], the air's excess over ice saturation at
    its own temperature (negative = undersaturated, what drives particle sublimation), and
    ``dT_air_sfc`` = T_FM2 − TSK [K].
    """
    # literal copies of MODEL_CSV_COLUMNS / MODEL_CSV_OPTIONAL: no module-global lookup at call
    # time, so the function also works in a kernel where autoreload swapped the code but not
    # the module namespace
    required = ['time', 'run', 'Q2_FM', 'QSAT_SFC', 'LH_SFC', 'LH_1.9m', 'UST', 'BS_DRIFT_FC']
    optional_all = ['T2', 'T_FM2', 'TSK', 'QSAT_ICE_T2', 'LH_PART_1.9m']
    raw = pd.read_csv(csv_path)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'model CSV lacks columns {missing}; expected {required}')
    # UTC, tz-naive like the observations (the export writes an ISO 'Z' suffix)
    raw['time'] = pd.to_datetime(raw['time'], utc=True).dt.tz_localize(None)
    optional = [c for c in optional_all if c in raw.columns]
    out = []
    for run, g in raw.groupby('run'):
        g = g.set_index('time').sort_index()
        rs = {c: resample_with_threshold(g[c], **kw) for c in required[2:] + optional}
        d = pd.DataFrame(rs)
        ok = d['UST'] >= ustar_min
        o = pd.DataFrame({
            'run': run,
            'x': (d['Q2_FM'] - d['QSAT_SFC']) * 1e3,
            'y19': (d['LH_1.9m'] / d['UST']).where(ok),
            'ysfc': (d['LH_SFC'] / d['UST']).where(ok),
            'ypart': ((d['LH_PART_1.9m'] if 'LH_PART_1.9m' in d else d['LH_1.9m'] - d['LH_SFC']) / d['UST']).where(ok),
            'drift': (d['BS_DRIFT_FC'] > drift_threshold).where(d['BS_DRIFT_FC'].notna()),
            'drift_flux': d['BS_DRIFT_FC'],
        }, index=d.index)
        if 'QSAT_ICE_T2' in d:
            o['x_air'] = (d['Q2_FM'] - d['QSAT_ICE_T2']) * 1e3
        if 'TSK' in d and ('T_FM2' in d or 'T2' in d):
            o['dT_air_sfc'] = d['T_FM2' if 'T_FM2' in d else 'T2'] - d['TSK']
        out.append(o)
    return pd.concat(out)


def model_particle_term_check(model, window):
    """Which variable sets the model's particle sublimation below the sensor? Per run, inside
    ``window``: median particle term (LH_1.9m − LH_SFC)/u*, and its correlation with the air's
    excess over ice saturation at the air temperature (``x_air``, if the v2 columns are there),
    with the surface excess (``x``) and with the drift mass flux. A tight negative correlation with
    ``x_air`` and a weak one with ``x`` means the particle term follows the air undersaturation,
    not the surface law, which is why the 1.9 m flux scatters about the line and the surface flux
    does not."""
    rows = {}
    for run, g in model.groupby('run'):
        g = g[window[0]:window[1]]
        r = {'particle term / u*, median': g['ypart'].median(), 'std': g['ypart'].std(),
             'corr with surface excess x': g['ypart'].corr(g['x']),
             'corr with drift flux': g['ypart'].corr(g['drift_flux'])}
        if 'x_air' in g:
            r['corr with air excess x_air'] = g['ypart'].corr(g['x_air'])
            r['median x_air [g/kg]'] = g['x_air'].median()
        if 'dT_air_sfc' in g:
            r['median T_air − T_skin [K]'] = g['dT_air_sfc'].median()
        rows[run] = r
    return pd.DataFrame(rows)


def plot_flux_law_model_vs_obs(tab, window, model=None, q_level='2m', le_col='LE_2m', ustar_col='ustar_2m',
                               ustar_min=0.1, obs_color='k', run_colors=None, zoom=None, show_surface=True,
                               figsize=(15, 6.5)):
    """Surface flux law for one event window: observations against the CRYOWRF column.

    Same axes as :func:`plot_surface_flux_law`: y = LE / u*, x = q(z) − q_sat,ice(T_skin).
    Two panels: the full range of the season on the left (context), and on the right a zoom on
    the window data (``zoom = (xlim, ylim)``; by default the 2nd–98th percentile range of the
    window observations and the model, with a margin). The full-season observations are a grey
    background, the observations inside ``window`` are highlighted in ``obs_color`` (circle =
    drift, triangle = no drift); the model runs use the directory colours ``MODEL_RUN_COLORS``,
    so the observations must not share them. ``model`` is the output of
    :func:`load_model_flux_law` (``None`` draws the observations only): per run, filled markers
    show the flux at 1.9 m (surface LH + blowing-snow sublimation below the sensor), open
    markers the surface LH alone (``show_surface=False`` drops them); the vertical gap between the
    two is the latent heat exchanged on particles below the sensor, which the observations cannot
    separate. One least-squares
    line per series with slope, r² and n (legend on the zoom panel). Returns ``(fig, axes, fits)``.
    """
    rc = run_colors or MODEL_RUN_COLORS
    x_all = tab[f'dq_{q_level}']
    y_all = (tab[le_col] / tab[ustar_col]).where(tab[ustar_col] >= ustar_min)
    w = tab[window[0]:window[1]]
    x_w, y_w = x_all[w.index], y_all[w.index]
    runs = {} if model is None else {run: g[window[0]:window[1]] for run, g in model.groupby('run')}

    # zoom limits from the window data
    if zoom is None:
        xs_ = [x_w.dropna()] + [g['x'].dropna() for g in runs.values()]
        ys_ = [y_w.dropna()] + [(pd.concat([g['y19'], g['ysfc']]) if show_surface else g['y19']).dropna()
                                for g in runs.values()]
        xs_, ys_ = pd.concat(xs_), pd.concat(ys_)
        xlo, xhi = xs_.quantile([0.02, 0.98]); ylo, yhi = ys_.quantile([0.02, 0.98])
        dx, dy = 0.15 * (xhi - xlo) or 0.02, 0.15 * (yhi - ylo) or 5
        zoom = ((xlo - dx, xhi + dx), (min(ylo - dy, -5), max(yhi + dy, 5)))

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1.25]})
    rows = []
    fits_done = set()

    def draw(ax, big):
        ms = 1.6 if big else 1.0   # marker scale on the zoom panel
        ax.scatter(x_all, y_all, s=6 * ms, color='0.78', alpha=0.5, lw=0, label='obs, season 2025', zorder=1)

        def add_fit(x, y, label, color, ls='-', zorder=5):
            slope, icpt, r2, n = _fit_line(x, y)
            if label not in fits_done:
                rows.append({'series': label, 'slope': slope, 'intercept': icpt, 'r2': r2, 'n': n})
                fits_done.add(label)
            if np.isfinite(slope):
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 30)
                ax.plot(xs, slope * xs + icpt, color=color, ls=ls, lw=2 if big else 1.5, zorder=zorder,
                        label=f'{label}: slope {slope:.0f}, r² {r2:.2f}, n={n}')

        if 'bs' in w:
            m = w['bs'] == True
            if m.any():
                ax.scatter(x_w[m], y_w[m], s=30 * ms, color=obs_color, marker='o', lw=0, alpha=0.9, zorder=3,
                           label='obs, window, drift')
            if (~m).any():
                ax.scatter(x_w[~m], y_w[~m], s=36 * ms, color=obs_color, marker='^', lw=0, alpha=0.9, zorder=3,
                           label='obs, window, no drift')
        else:
            ax.scatter(x_w, y_w, s=30 * ms, color=obs_color, lw=0, alpha=0.9, zorder=3, label='obs, window')
        add_fit(x_w, y_w, f'obs {le_col} ({window[0][5:]} → {window[1][5:]})', obs_color)

        for run, g in runs.items():
            col = rc.get(run, '0.3')
            d = g['drift'] == True
            same = np.allclose(g['y19'].dropna(), g['ysfc'].reindex(g['y19'].dropna().index), atol=1e-6)
            lab19 = f'{run} LH(1.9 m)/u* = surface LH/u* (constant-flux mesh)' if same else f'{run} LH(1.9 m)/u*'
            if d.any():
                ax.scatter(g['x'][d], g['y19'][d], s=34 * ms, color=col, marker='o', lw=0, alpha=0.9, zorder=4,
                           label=f'{lab19}, drift')
            if (~d).any():
                ax.scatter(g['x'][~d], g['y19'][~d], s=40 * ms, color=col, marker='^', lw=0, alpha=0.9, zorder=4,
                           label=f'{lab19}, no drift')
            add_fit(g['x'], g['y19'], f'{run} 1.9 m', col, '-')
            if show_surface and not same:
                ax.scatter(g['x'], g['ysfc'], s=34 * ms, facecolors='none', edgecolors=col, marker='o', lw=1.0,
                           alpha=0.9, zorder=4, label=f'{run} surface LH/u*')
                add_fit(g['x'], g['ysfc'], f'{run} surface', col, ':')
            n_nd = int((~d).sum())
            if n_nd < 10 and f'{run} no-drift steps in window' not in fits_done:
                rows.append({'series': f'{run} no-drift steps in window', 'slope': np.nan, 'intercept': np.nan,
                             'r2': np.nan, 'n': n_nd})
                fits_done.add(f'{run} no-drift steps in window')
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel(f'q({q_level[:-1]} m) − q$_{{sat,ice}}$(T$_{{skin}}$) [g kg$^{{-1}}$]')
        ax.grid(True, alpha=0.25)

    draw(axes[0], big=False)
    draw(axes[1], big=True)
    axes[0].set_ylabel('LE / u*  [W m$^{-2}$ per m s$^{-1}$]')
    axes[0].set_title('full range (season context)', fontsize=10)
    axes[1].set_xlim(*zoom[0]); axes[1].set_ylim(*zoom[1])
    axes[1].set_title('zoom on the window', fontsize=10)
    axes[1].legend(fontsize=7.5, frameon=False, loc='lower left', ncol=1)
    # zoom box on the overview
    (x0, x1), (y0, y1) = zoom
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='0.3', lw=1, ls='--', zorder=6))
    fig.suptitle(f'surface flux law, {window[0]} → {window[1]} (u* ≥ {ustar_min} m s$^{{-1}}$)', fontsize=11)
    fig.tight_layout()
    return fig, axes, pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. composites
# ---------------------------------------------------------------------------
def _binned(x, y, n_bins=8, min_n=10):
    """Quantile-binned median and IQR of y against x."""
    d = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(d) < min_n * 2 or d['x'].nunique() < 3:
        return None
    try:
        b = pd.qcut(d['x'], n_bins, duplicates='drop')
    except ValueError:
        return None
    g = d.groupby(b, observed=True)
    res = pd.DataFrame({'x': g['x'].median(), 'med': g['y'].median(),
                        'q25': g['y'].quantile(0.25), 'q75': g['y'].quantile(0.75), 'n': g.size()})
    return res[res['n'] >= min_n]


def plot_driver_composites(tab, pair, drivers, n_bins=8, ncols=4, var='LE'):
    """Binned median + IQR of the flux difference (``var`` = 'LE' or 'H') against each driver,
    split by blowing snow / no blowing snow."""
    y = tab[f'd{var}_{pair}']
    nrows = int(np.ceil(len(drivers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), sharey=True, squeeze=False)
    for ax, drv in zip(axes.ravel(), drivers):
        for bs, lbl in [(False, 'no blowing snow'), (True, 'blowing snow')]:
            m = tab['bs'] == bs
            r = _binned(tab.loc[m, drv], y[m], n_bins)
            if r is None:
                continue
            ax.fill_between(r['x'], r['q25'], r['q75'], color=BS_COLORS[bs], alpha=0.2, linewidth=0)
            ax.plot(r['x'], r['med'], 'o-', color=BS_COLORS[bs], lw=2, ms=4, label=lbl)
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel(DRIVER_LABELS.get(drv, drv), fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(drivers):]:
        ax.set_visible(False)
    up, lo = pair.split('_')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'{var}({up} m) − {var}({lo} m) [W m$^{{-2}}$]', fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BS_COLORS[b], marker='o', ms=4, lw=2, label=l)
               for b, l in [(False, 'no blowing snow'), (True, 'blowing snow')]]
    fig.legend(handles=handles, loc='upper right', ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f'd{var} {up} m − {lo} m: binned median and IQR per driver', y=1.0)
    fig.tight_layout()
    return fig, axes


def plot_matched_wind(tab, pairs, ws_col='WS_2m', bins=None, min_n=10, bin_width=2.0, pf_thresholds=None):
    """dLE for different drift groups inside the same wind-speed bins.

    Wind and drift are collinear; comparing within a wind bin isolates the drift effect.
    Groups: with ``pf_thresholds=None`` the two groups of the ``bs`` column (no blowing snow /
    blowing snow). With thresholds, e.g. ``(0.01,)``, the groups are no blowing snow
    (PF_FC4 = 0), 0 < PF_FC4 < 0.01 and PF_FC4 >= 0.01 [g m-2 s-1]; more thresholds give more
    groups. ``bin_width`` sets the wind-speed bin size [m/s] when ``bins`` is not given; bins
    with fewer than ``min_n`` half-hours are not drawn.
    Returns ``(fig, axes, table)`` where table holds the medians and counts per bin and group.
    """
    if bins is None:   # bins of `bin_width` m/s from 0 up to the 99th percentile
        hi = np.nanpercentile(tab[ws_col], 99)
        bins = np.arange(0, hi + bin_width, bin_width)
    width = float(np.median(np.diff(bins)))

    # drift groups: list of (label, mask, colour)
    pf = tab['PF_FC4']
    if pf_thresholds is None:
        groups = [('no blowing snow', tab['bs'] == False, BS_COLORS[False]),
                  ('blowing snow', tab['bs'] == True, BS_COLORS[True])]
    else:
        thr = sorted(pf_thresholds)
        edges = [0.0] + list(thr) + [np.inf]
        shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(edges) - 1))
        groups = [('no blowing snow (PF = 0)', pf == 0, BS_COLORS[False])]
        for k, (lo_, hi_) in enumerate(zip(edges[:-1], edges[1:])):
            if k == 0:
                lbl, m = f'0 < PF < {hi_:g}', (pf > 0) & (pf < hi_)
            elif np.isinf(hi_):
                lbl, m = f'PF ≥ {lo_:g}', pf >= lo_
            else:
                lbl, m = f'{lo_:g} ≤ PF < {hi_:g}', (pf >= lo_) & (pf < hi_)
            groups.append((lbl, m, shades[k]))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * 0.5 * width / len(groups)

    wsb = pd.cut(tab[ws_col], bins)
    centres = np.array([iv.mid for iv in wsb.cat.categories])
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.6), sharey=True, squeeze=False)
    rows = []
    for ax, pair in zip(axes[0], pairs):
        y = tab[f'dLE_{pair}']
        for (lbl, m, color), off in zip(groups, offsets):
            m = m.fillna(False).astype(bool)
            g = pd.DataFrame({'b': wsb[m], 'y': y[m]}).dropna().groupby('b', observed=False)['y']
            med, q25, q75, n = g.median(), g.quantile(0.25), g.quantile(0.75), g.size()
            ok = (n >= min_n).values
            ax.errorbar(centres[ok] + off, med.values[ok],
                        yerr=[(med - q25).values[ok], (q75 - med).values[ok]],
                        fmt='o', color=color, capsize=3, lw=1.5, ms=6, label=lbl)
            for c, md, nn, k in zip(centres, med.values, n.values, ok):
                rows.append(dict(pair=pair, ws_bin=c, group=lbl, median=md if k else np.nan, n=int(nn)))
        ax.axhline(0, color='k', lw=0.6, ls='--')
        up, lo = pair.split('_')
        ax.set_title(f'LE({up} m) − LE({lo} m)')
        ax.set_xlabel(DRIVER_LABELS.get(ws_col, ws_col))
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel(r'dLE median and IQR [W m$^{-2}$]')
    axes[0, 0].legend(fontsize=9, frameon=False, title='mass flux [g m$^{-2}$ s$^{-1}$]' if pf_thresholds is not None else None,
                      title_fontsize=8)
    fig.suptitle('Same wind speed, by drift group', y=1.0)
    fig.tight_layout()
    table = pd.DataFrame(rows)
    table['group'] = pd.Categorical(table['group'], [g[0] for g in groups], ordered=True)
    return fig, axes, table


def plot_matched_drift(tab, pairs, log_bin_width=0.5, ws_thresholds=None, ws_col='WS_2m', min_n=10,
                       ustar=None, ustar_min=0.1, pf_min=None):
    """dLE against the blowing-snow mass flux PF_FC4, for different wind-speed groups.

    The mirror image of :func:`plot_matched_wind`: inside one drift-flux bin, does dLE still
    differ between wind-speed groups? PF_FC4 is binned in steps of ``log_bin_width`` decades.
    Half-hours without drift (PF_FC4 = 0) cannot sit on a log axis and are drawn at the far
    left, labelled "0". ``ws_thresholds``, e.g. ``(6, 9)``, gives the groups WS < 6, 6 <= WS < 9
    and WS >= 9 m/s; ``None`` gives a single group. Bins with fewer than ``min_n`` half-hours
    are not drawn. ``pf_min`` [g m-2 s-1], e.g. the drift threshold 0.001, makes the first bin start
    there and leaves out half-hours with 0 < PF_FC4 <= pf_min; ``None`` starts at the smallest flux.

    ``ustar`` divides every LE by the measured friction velocity of the same half-hour before the
    difference is taken: ``'local'`` uses each level's own sonic (``ustar_<lvl>``, so
    y = LE(up)/u*(up) − LE(lo)/u*(lo)), ``'2m'`` uses the 2 m value for both levels
    (y = dLE/u*(2 m)); ``None`` leaves LE in W m⁻². Half-hours with any u* used below
    ``ustar_min`` are dropped. Returns ``(fig, axes, table)``.
    """
    pf, ws = tab['PF_FC4'], tab[ws_col]
    if ustar not in (None, 'local', '2m'):
        raise ValueError("ustar must be None, 'local' or '2m'")
    lp = np.log10(pf.where(pf > (pf_min or 0)))       # drift below pf_min: NaN, left out
    lo_edge = (np.log10(pf_min) if pf_min else np.floor(np.nanmin(lp) / log_bin_width) * log_bin_width)
    hi_edge = np.ceil(np.nanpercentile(lp.dropna(), 99.5) / log_bin_width) * log_bin_width
    edges = np.arange(lo_edge, hi_edge + log_bin_width / 2, log_bin_width)
    centres = (edges[:-1] + edges[1:]) / 2
    x_zero = edges[0] - 1.5 * log_bin_width            # position of the PF = 0 group
    pfb = pd.cut(lp, edges, labels=False, include_lowest=True)
    pfb = pfb.where(pf != 0, -1)                         # -1 = no drift
    xpos = np.r_[x_zero, centres]

    if ws_thresholds is None:
        groups = [('all wind speeds', ws.notna(), '#0072B2')]
    else:
        thr = sorted(ws_thresholds)
        e = [-np.inf] + list(thr) + [np.inf]
        shades = plt.cm.viridis(np.linspace(0.85, 0.15, len(e) - 1))
        groups = []
        for k, (a_, b_) in enumerate(zip(e[:-1], e[1:])):
            lbl = (f'WS < {b_:g}' if np.isinf(a_) else f'WS ≥ {a_:g}' if np.isinf(b_) else f'{a_:g} ≤ WS < {b_:g}')
            groups.append((lbl, (ws >= a_) & (ws < b_), shades[k]))
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * 0.5 * log_bin_width / len(groups)

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.6), sharey=True, squeeze=False)
    rows = []
    for k_ax, (ax, pair) in enumerate(zip(axes[0], pairs)):
        ax.text(0.02, 0.97, f'{"abcdefgh"[k_ax]})', transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold')
        up, lo = pair.split('_')
        if ustar is None:
            y = tab[f'dLE_{pair}']
        elif ustar == 'local':
            u_up, u_lo = tab[f'ustar_{up}m'], tab[f'ustar_{lo}m']
            y = ((tab[f'LE_{up}m'] / u_up) - (tab[f'LE_{lo}m'] / u_lo)).where((u_up >= ustar_min) & (u_lo >= ustar_min))
        else:
            y = (tab[f'dLE_{pair}'] / tab['ustar_2m']).where(tab['ustar_2m'] >= ustar_min)
        for (lbl, m, color), off in zip(groups, offsets):
            d = pd.DataFrame({'b': pfb[m], 'y': y[m]}).dropna()
            g = d.groupby('b')['y']
            st = pd.DataFrame({'med': g.median(), 'q25': g.quantile(0.25), 'q75': g.quantile(0.75), 'n': g.size()})
            st = st.reindex(range(-1, len(centres)))
            ok = (st['n'] >= min_n).values
            ax.errorbar(xpos[ok] + off, st['med'].values[ok],
                        yerr=[(st['med'] - st['q25']).values[ok], (st['q75'] - st['med']).values[ok]],
                        fmt='o', color=color, capsize=3, lw=1.5, ms=6, label=lbl)
            for k in range(-1, len(centres)):
                lab = 'no drift' if k == -1 else f'{10 ** edges[k]:.2g}–{10 ** edges[k + 1]:.2g}'
                r = st.loc[k]
                rows.append(dict(pair=pair, pf_bin=lab, group=lbl,
                                 median=r['med'] if (r['n'] >= min_n) else np.nan, n=int(0 if np.isnan(r['n']) else r['n'])))
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.axvline(edges[0] - 0.5 * log_bin_width, color='0.6', lw=0.8, ls=':')
        ticks = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
        ax.set_xticks(np.r_[x_zero, ticks], ['0'] + [f'$10^{{{int(t)}}}$' for t in ticks])
        if ustar is None:
            ax.set_title(f'dLE = LE({up} m) − LE({lo} m)', fontsize=10)
        elif ustar == 'local':
            ax.set_title(f'LE({up} m)/u$_*$({up} m) − LE({lo} m)/u$_*$({lo} m)', fontsize=10)
        else:
            ax.set_title(f'[LE({up} m) − LE({lo} m)] / u$_*$(2 m)', fontsize=10)
        ax.set_xlabel('mass flux [g m$^{-2}$ s$^{-1}$]')
        ax.grid(True, alpha=0.25)
    if ustar is None:
        axes[0, 0].set_ylabel(r'dLE median and IQR [W m$^{-2}$]')
    else:
        axes[0, 0].set_ylabel('d(LE/u$_*$) median and IQR [W m$^{-2}$ per m s$^{-1}$]')
        fig.suptitle(f'u$_*$ ≥ {ustar_min} m s$^{{-1}}$', fontsize=9, y=0.995)
    axes[0, 0].legend(fontsize=9, frameon=False, title=f'{DRIVER_LABELS.get(ws_col, ws_col)}', title_fontsize=8,
                      loc='upper left', bbox_to_anchor=(0.08, 0.98))
    fig.tight_layout()
    table = pd.DataFrame(rows)
    table['group'] = pd.Categorical(table['group'], [g[0] for g in groups], ordered=True)
    table['pf_bin'] = pd.Categorical(table['pf_bin'], list(dict.fromkeys(table['pf_bin'])), ordered=True)
    return fig, axes, table


def plot_2d_medians(tab, pair, x, y, n_bins=7, min_n=8, only_bs=True, cmap='RdBu_r'):
    """Median dLE on a 2-D quantile grid of two drivers (default: blowing-snow samples only)."""
    d = tab.loc[tab['bs'] == True] if only_bs else tab
    d = d[[x, y, f'dLE_{pair}']].dropna()
    xb = pd.qcut(d[x], n_bins, duplicates='drop')
    yb = pd.qcut(d[y], n_bins, duplicates='drop')
    g = d.groupby([yb, xb], observed=False)[f'dLE_{pair}']
    med = g.median().where(g.size() >= min_n).unstack()
    v = np.nanpercentile(np.abs(med.values), 95) if np.isfinite(med.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(med.values, origin='lower', aspect='auto', cmap=cmap, vmin=-v, vmax=v)
    ax.set_xticks(range(med.shape[1]), [f'{iv.mid:.2g}' for iv in med.columns], rotation=45)
    ax.set_yticks(range(med.shape[0]), [f'{iv.mid:.2g}' for iv in med.index])
    ax.set_xlabel(DRIVER_LABELS.get(x, x)); ax.set_ylabel(DRIVER_LABELS.get(y, y))
    up, lo = pair.split('_')
    fig.colorbar(im, ax=ax, label=f'median LE({up} m) − LE({lo} m) [W m$^{{-2}}$]')
    ax.set_title(('blowing snow only: ' if only_bs else '') + f'bins with < {min_n} samples left blank', fontsize=10)
    fig.tight_layout()
    return fig, ax, med


def plot_energy_check(tab, pairs):
    """dH against dLE. Sublimation in the layer gives dLE > 0 with dH < 0 (slope near -1)."""
    from scipy.stats import linregress
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5), squeeze=False)
    out = []
    for ax, pair in zip(axes[0], pairs):
        d = tab[[f'dLE_{pair}', f'dH_{pair}', 'bs']].dropna()
        for bs, lbl in [(False, 'no blowing snow'), (True, 'blowing snow')]:
            g = d[d['bs'] == bs]
            ax.scatter(g[f'dLE_{pair}'], g[f'dH_{pair}'], s=8, alpha=0.35, color=BS_COLORS[bs], linewidths=0)
            if len(g) >= 10:
                fit = linregress(g[f'dLE_{pair}'], g[f'dH_{pair}'])
                xx = np.array(g[f'dLE_{pair}'].quantile([0.01, 0.99]))
                ax.plot(xx, fit.intercept + fit.slope * xx, color=BS_COLORS[bs], lw=2,
                        label=f'{lbl}: slope {fit.slope:.2f}, r² {fit.rvalue**2:.2f}, n {len(g)}')
                out.append(dict(pair=pair, bs=bs, slope=fit.slope, r2=fit.rvalue**2, n=len(g)))
        lim = np.nanpercentile(np.abs(d[[f'dLE_{pair}', f'dH_{pair}']].values), 99)
        ax.plot([-lim, lim], [lim, -lim], 'k:', lw=1, label='dH = −dLE')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        up, lo = pair.split('_')
        ax.set_xlabel(f'LE({up} m) − LE({lo} m) [W m$^{{-2}}$]')
        ax.set_ylabel(f'H({up} m) − H({lo} m) [W m$^{{-2}}$]')
        ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=1)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes, pd.DataFrame(out)


DRIFT_CLASS_EDGES = [-np.inf, LOG_PF_FLOOR + 0.5, -2, -1, 0, np.inf]
DRIFT_CLASS_LABELS = ['no drift', 'PF < 0.01', '0.01–0.1', '0.1–1', 'PF > 1']


def drift_class(tab):
    """Ordered drift-intensity class from log10(PF_FC4) [g m-2 s-1]."""
    return pd.cut(tab['logPF'], DRIFT_CLASS_EDGES, labels=DRIFT_CLASS_LABELS)


def plot_flux_profiles_by_drift(tab, sonic_heights, levels=('2m', '16m', '26m'), rh_heights=None):
    """Median LE, H and H + LE at each level, one profile per drift-intensity class.

    Only half-hours with H and LE valid at all levels are used, so every profile is built from
    the same samples. If suspended snow sublimates between two levels, LE grows and H falls by
    the same amount across that layer while H + LE stays constant with height.
    ``rh_heights`` (``{'2m': z, '26m': z}``, T/RH probe heights) adds a fourth panel with the
    median RH wrt ice per class, as in Schwat et al. (2025, their Fig. 5c).
    Returns ``(fig, axes, table of medians)``.
    """
    need = [f'{v}_{l}' for v in ('LE', 'H') for l in levels]
    d = tab.dropna(subset=need + ['logPF']).copy()
    d['drift'] = drift_class(d)
    for l in levels:
        d[f'H+LE_{l}'] = d[f'H_{l}'] + d[f'LE_{l}']
    g = d.groupby('drift', observed=True)
    med = g[[f'{v}_{l}' for v in ('LE', 'H', 'H+LE') for l in levels]].median()
    med.insert(0, 'WS_2m', g['WS_2m'].median())
    med.insert(0, 'n', g.size())

    z = [sonic_heights[l] for l in levels]
    cls_colors = ['#7F7F7F'] + [plt.cm.Oranges(v) for v in np.linspace(0.35, 1.0, len(DRIFT_CLASS_LABELS) - 1)]
    n_pan = 4 if rh_heights else 3
    fig, axes = plt.subplots(1, n_pan, figsize=(4.7 * n_pan, 5), sharey=True)
    for ax, v, title in zip(axes, ('LE', 'H', 'H+LE'), ('latent heat flux LE', 'sensible heat flux H', 'H + LE')):
        for cls, c in zip(DRIFT_CLASS_LABELS, cls_colors):
            if cls not in med.index:
                continue
            ax.plot([med.loc[cls, f'{v}_{l}'] for l in levels], z, 'o-', color=c, lw=2, ms=6,
                    label=f'{cls} (n={int(med.loc[cls, "n"])})')
        ax.axvline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel(f'median {title} [W m$^{{-2}}$]')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
    if rh_heights:
        rh_cols = [f'RHi_{l}' for l in rh_heights]
        rh_med = d.groupby('drift', observed=True)[rh_cols].median()
        med = med.join(rh_med)
        for cls, c in zip(DRIFT_CLASS_LABELS, cls_colors):
            if cls in rh_med.index:
                axes[3].plot(rh_med.loc[cls, rh_cols].values, list(rh_heights.values()), 'o-', color=c, lw=2, ms=6)
        axes[3].axvline(100, color='k', lw=0.6, ls='--')
        axes[3].set_xlabel('median RH wrt ice [%]')
        axes[3].set_title('relative humidity wrt ice (T/RH probes)', fontsize=10)
        axes[3].grid(True, alpha=0.25)
    axes[0].set_ylabel('height [m]')
    axes[0].legend(title='mass flux [g m$^{-2}$ s$^{-1}$]', fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle('Flux profiles by drift intensity (same half-hours at all levels)', y=1.0)
    fig.tight_layout()
    return fig, axes, med

def plot_le_profile_by_drift(tab, sonic_heights, levels=('2m', '16m', '26m'), var='LE', pf_edges=None,
                             log_bin_width=0.5, stat='median', spread='iqr', min_n=20, cmap='batlow',
                             cmap_range=(0.8, 0.0), dodge=0.3, figsize=(3.5, 3.6), fontsize=8, title=False,
                             sign_labels=True, show_n=True, legend_loc='upper left'):
    """One panel: the ``var`` profile (LE, H or H+LE) at the sonic levels, one line per drift-flux bin.

    A finer version of the first panel of :func:`plot_flux_profiles_by_drift`. The bins are either
    ``pf_edges``, a list of PF_FC4 edges in g m-2 s-1 (e.g. ``[0.001, 0.01, 0.1, 1, 10]``), or,
    when ``pf_edges`` is None, log bins of ``log_bin_width`` decades spanning the data. Half-hours
    with PF_FC4 = 0 form the "no drift" profile (grey, open markers). Only half-hours valid at all
    levels are used, so each profile is built from the same samples at every height.

    ``stat`` is ``'median'`` or ``'mean'``. ``spread`` adds horizontal bars: ``'iqr'`` (25th to
    75th percentile, the spread of the half-hours), ``'std'``, or ``'sem'`` (standard error of
    the mean, std/sqrt(n): the uncertainty of the plotted mean itself); ``None`` draws none. The
    profiles are shifted vertically by ``dodge`` metres per bin so that the bars do not overlap.
    Bins with fewer than ``min_n`` half-hours are skipped.

    Styled for a paper figure: ``cmap`` is a Scientific Colour Map (Crameri) name from
    ``cmcrameri`` (falls back to a matplotlib name), sampled from ``cmap_range[0]`` (lowest drift
    bin) to ``cmap_range[1]`` (highest), so stronger drift is darker. ``figsize`` defaults to one
    journal column (~8.5 cm) and ``fontsize`` sets all text. No title unless ``title`` is True or a
    string (the caption carries it); ``sign_labels`` writes "deposition" / "sublimation" on either
    side of LE = 0. Returns ``(fig, ax, table)`` with the statistic, the spread bounds, n and the
    median PF_FC4 per bin.
    """
    cols = [f'{var}_{l}' for l in levels] if var != 'H+LE' else [f'LE_{l}' for l in levels] + [f'H_{l}' for l in levels]
    d = tab.dropna(subset=cols + ['PF_FC4']).copy()
    if var == 'H+LE':
        for l in levels:
            d[f'H+LE_{l}'] = d[f'H_{l}'] + d[f'LE_{l}']
    vcols = [f'{var}_{l}' for l in levels]
    pf = d['PF_FC4']
    if pf_edges is None:
        _, e, _, _ = _pf_log_bins(pf, log_bin_width)
        pf_edges = 10.0 ** e
    pf_edges = np.asarray(pf_edges, dtype=float)
    labels = ['no drift'] + [f'{a:.3g}–{b:.3g}' for a, b in zip(pf_edges[:-1], pf_edges[1:])]
    b = pd.cut(pf.where(pf > 0), pf_edges, labels=labels[1:], include_lowest=True).astype(object)
    b[pf == 0] = 'no drift'
    d['bin'] = pd.Categorical(b, labels, ordered=True)
    g = d.groupby('bin', observed=True)
    agg = g[vcols].mean() if stat == 'mean' else g[vcols].median()
    out = agg.copy()
    out.insert(0, 'PF_median', g['PF_FC4'].median())
    out.insert(0, 'n', g.size())
    if spread == 'iqr':
        lo, hi = g[vcols].quantile(0.25), g[vcols].quantile(0.75)
    elif spread == 'std':
        sd = g[vcols].std(); lo, hi = agg - sd, agg + sd
    elif spread == 'sem':
        se = g[vcols].std() / np.sqrt(g.size().values[:, None]); lo, hi = agg - se, agg + se
    else:
        lo = hi = None
    if lo is not None:
        out = out.join(lo.add_suffix('_lo')).join(hi.add_suffix('_hi'))

    z = [sonic_heights[l] for l in levels]
    n_bins = len(labels) - 1
    try:
        from cmcrameri import cm as cmc
        cm_ = getattr(cmc, cmap) if isinstance(cmap, str) and hasattr(cmc, cmap) else plt.get_cmap(cmap)
    except ImportError:
        cm_ = plt.get_cmap(cmap)
    colors = ['#8C8C8C'] + list(cm_(np.linspace(cmap_range[0], cmap_range[1], n_bins)))
    shown = [lab for lab in labels if lab in out.index and out.loc[lab, 'n'] >= min_n]

    rc = {'font.size': fontsize, 'axes.labelsize': fontsize, 'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize, 'legend.fontsize': fontsize - 1, 'axes.linewidth': 0.6,
          'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(0, color='0.3', lw=0.6, zorder=1)
        for k, lab in enumerate(shown):
            c = colors[labels.index(lab)]
            x = out.loc[lab, vcols].values.astype(float)
            zz = np.asarray(z) + (k - (len(shown) - 1) / 2) * dodge
            if lo is not None:
                ax.errorbar(x, zz, xerr=[x - lo.loc[lab].values, hi.loc[lab].values - x], color=c, lw=0,
                            elinewidth=0.7, capsize=0, alpha=0.55, zorder=2)
            txt = lab + (f' (n = {int(out.loc[lab, "n"])})' if show_n else '')
            ax.plot(x, zz, 'o-', color=c, lw=1.2, ms=3.5, mfc='white' if lab == 'no drift' else c, mec=c,
                    mew=0.9, zorder=3, label=txt)
        ax.set_yticks(z, [f'{v:.1f}' for v in z])
        ax.set_ylabel('Height (m)')
        unit = 'W m$^{-2}$'
        name = {'LE': 'LE', 'H': 'H', 'H+LE': 'H + LE'}.get(var, var)
        ax.set_xlabel(f'{stat.capitalize()} {name} ({unit})')
        lim = max(abs(v) for v in ax.get_xlim())
        ax.set_xlim(-lim, lim)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis='x', color='0.88', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        if sign_labels and var == 'LE':
            kw_t = dict(transform=ax.get_xaxis_transform(), fontsize=fontsize - 1, color='0.35', va='bottom')
            ax.text(-0.03 * 2 * lim, 1.0, '← deposition', ha='right', **kw_t)
            ax.text(0.03 * 2 * lim, 1.0, 'sublimation →', ha='left', **kw_t)
        if title:
            ax.set_title(title if isinstance(title, str) else f'{name} profile by drift flux ({stat})',
                         fontsize=fontsize, pad=14 if sign_labels else 6)
        leg_title = 'Drift flux (g m$^{-2}$ s$^{-1}$)'
        ax.legend(title=leg_title, frameon=False, loc=legend_loc, handlelength=1.0, handletextpad=0.5,
                  labelspacing=0.3, borderaxespad=0.2, title_fontsize=fontsize - 1, alignment='left')
        fig.tight_layout()
    return fig, ax, out


def _pf_log_bins(pf, log_bin_width):
    """Log10 drift-flux bins as in :func:`plot_matched_drift`: (bin index per half-hour with -1 = no
    drift, bin edges, bin centres, x position of the no-drift group)."""
    lp = np.log10(pf.where(pf > 0))
    lo_edge = np.floor(np.nanmin(lp) / log_bin_width) * log_bin_width
    hi_edge = np.ceil(np.nanpercentile(lp.dropna(), 99.5) / log_bin_width) * log_bin_width
    edges = np.arange(lo_edge, hi_edge + log_bin_width / 2, log_bin_width)
    pfb = pd.cut(lp, edges, labels=False, include_lowest=True).where(pf != 0, -1)
    return pfb, edges, (edges[:-1] + edges[1:]) / 2, edges[0] - 1.5 * log_bin_width


def _ws_groups(ws, ws_thresholds):
    """[(label, mask, colour)] wind-speed groups as in :func:`plot_matched_drift`."""
    if ws_thresholds is None:
        return [('all wind speeds', ws.notna(), '#0072B2')]
    e = [-np.inf] + sorted(ws_thresholds) + [np.inf]
    shades = plt.cm.viridis(np.linspace(0.85, 0.15, len(e) - 1))
    out = []
    for k, (a_, b_) in enumerate(zip(e[:-1], e[1:])):
        lbl = (f'WS < {b_:g}' if np.isinf(a_) else f'WS ≥ {a_:g}' if np.isinf(b_) else f'{a_:g} ≤ WS < {b_:g}')
        out.append((lbl, (ws >= a_) & (ws < b_), shades[k]))
    return out


def sign_profile_class(tab, lower, upper, le_floor):
    """Sign pattern of LE at two levels per half-hour, with |LE| above ``le_floor`` [W m-2] at both
    levels required: ``'dep. below, subl. above'`` (LE(lower) < 0 < LE(upper)), ``'subl. below, dep.
    above'``, ``'same sign'``, or ``'below floor'``. NaN where either LE is missing."""
    lo, up = tab[f'LE_{lower}'], tab[f'LE_{upper}']
    out = pd.Series(np.nan, index=tab.index, dtype=object)
    valid = lo.notna() & up.notna()
    strong = valid & (lo.abs() > le_floor) & (up.abs() > le_floor)
    out[valid] = 'below floor'
    out[strong] = 'same sign'
    out[strong & (lo < 0) & (up > 0)] = 'dep. below, subl. above'
    out[strong & (lo > 0) & (up < 0)] = 'subl. below, dep. above'
    return out


def agc_pass(tab, diag, levels, max_share):
    """True where the LI-7500 signal strength of every analyser in ``levels`` was above its baseline
    in at most ``max_share`` of the records (``diag`` = ``{level: diag table}``); False where it was
    not; NaN where the diagnostic is missing for a level that is in ``diag``. Levels without a
    diagnostic table are not screened."""
    ok = pd.Series(True, index=tab.index)
    for lvl in levels:
        if lvl in diag:
            share = _align_diag(diag[lvl]['share_above'], tab.index)
            ok = ok & (share <= max_share).where(share.notna())
    return ok


def two_signed_profiles(tab, pairs, diag, le_floor, max_share=0.1, log_bin_width=0.5, ws_thresholds=None,
                        ws_col='WS_2m', min_n=10, rh_levels=('2m', '26m'), figsize=None):
    """Profiles with deposition at the lower level and sublimation at the upper one, per drift bin.

    A sign change between two levels needs a vapour source in between: no scaling of one flux
    with wind or u* can produce it. For each ``(lower, upper)`` pair in ``pairs`` the fraction of
    half-hours in which LE(lower) < −floor and LE(upper) > +floor is drawn against the FlowCapt
    drift flux (log bins of ``log_bin_width`` decades, no-drift group at the left), per wind-speed
    group as in :func:`plot_matched_drift`. Filled markers use all half-hours; open markers keep
    only those that pass the LI-7500 signal-strength screen at both analysers of the pair
    (``diag`` = ``{level: li7500_diag_halfhourly(...)}``, share above baseline ≤ ``max_share``).
    The last panel gives the median RH over ice at ``rh_levels`` per drift bin for the two-signed
    half-hours of the first pair (solid) against the same-signed ones (dashed), both after the
    screen. ``le_floor`` is one value [W m-2] or ``{pair_tag: value}``.

    Returns ``(fig, axes, table)``; the table holds, per pair, drift bin and screen state, the
    number of valid half-hours, the number and fraction of two-signed ones, and median LE and RH
    over ice of the two-signed cases.
    """
    pf, ws = tab['PF_FC4'], tab[ws_col]
    pfb, edges, centres, x_zero = _pf_log_bins(pf, log_bin_width)
    xpos = np.r_[x_zero, centres]
    groups = _ws_groups(ws, ws_thresholds)
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * 0.5 * log_bin_width / len(groups)
    bin_labels = ['no drift'] + [f'{10 ** a:.2g}–{10 ** b:.2g}' for a, b in zip(edges[:-1], edges[1:])]

    n_ax = len(pairs) + 1
    fig, axes = plt.subplots(1, n_ax, figsize=figsize or (5.2 * n_ax, 4.6), squeeze=False)
    axes = axes[0]
    rows, flip_first = [], None
    for k_ax, (ax, (lower, upper)) in enumerate(zip(axes, pairs)):
        tag = f'{upper[:-1]}_{lower[:-1]}'
        floor = le_floor[tag] if isinstance(le_floor, dict) else le_floor
        cls = sign_profile_class(tab, lower, upper, floor)
        flip = cls == 'dep. below, subl. above'
        valid = cls.notna()
        passed = agc_pass(tab, diag, (lower, upper), max_share)
        if k_ax == 0:
            flip_first = (flip, valid, passed)
        ax.text(0.02, 0.97, f'{"abcdefgh"[k_ax]})', transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold')
        for (lbl, m, color), off in zip(groups, offsets):
            for screened, fmt, face in [(False, 'o', color), (True, 'o', 'white')]:
                keep = m & valid & ((passed == True) if screened else True)
                d = pd.DataFrame({'b': pfb[keep], 'flip': flip[keep]}).dropna(subset=['b'])
                g = d.groupby('b')['flip']
                st = pd.DataFrame({'n': g.size(), 'n_flip': g.sum()}).reindex(range(-1, len(centres)))
                st['frac'] = st['n_flip'] / st['n']
                ok = (st['n'] >= min_n).values
                ax.plot(xpos[ok] + off, st['frac'].values[ok] * 100, marker=fmt, ls='-' if not screened else '--',
                        color=color, mfc=face, mew=1.5, ms=6, lw=1.2,
                        label=f'{lbl}{", screened" if screened else ""}')
                for b_, r in st.iterrows():
                    sel = keep & (pfb == b_) & flip
                    rows.append(dict(pair=tag, pf_bin=bin_labels[b_ + 1], group=lbl, screened=screened,
                                     n=int(0 if np.isnan(r['n']) else r['n']), n_flip=int(0 if np.isnan(r['n_flip']) else r['n_flip']),
                                     frac=r['frac'] if r['n'] >= min_n else np.nan,
                                     LE_lower=tab.loc[sel, f'LE_{lower}'].median(), LE_upper=tab.loc[sel, f'LE_{upper}'].median(),
                                     **{f'RHi_{lv}': tab.loc[sel, f'RHi_{lv}'].median() for lv in rh_levels if f'RHi_{lv}' in tab}))
        ax.axvline(edges[0] - 0.5 * log_bin_width, color='0.6', lw=0.8, ls=':')
        ticks = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
        ax.set_xticks(np.r_[x_zero, ticks], ['0'] + [f'$10^{{{int(t)}}}$' for t in ticks])
        ax.set_title(f'LE({lower[:-1]} m) < −{floor:.1f}, LE({upper[:-1]} m) > +{floor:.1f} W m$^{{-2}}$', fontsize=10)
        ax.set_xlabel('mass flux [g m$^{-2}$ s$^{-1}$]')
        ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.3))
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel('share of half-hours with deposition below,\nsublimation above [%]')
    axes[0].legend(fontsize=8, frameon=False, title=f'{DRIVER_LABELS.get(ws_col, ws_col)}; dashed/open: screened',
                   title_fontsize=8, loc='upper left', bbox_to_anchor=(0.06, 0.97))

    # RH over ice of the two-signed vs same-signed half-hours (first pair, screened)
    ax = axes[-1]
    ax.text(0.02, 0.97, f'{"abcdefgh"[n_ax - 1]})', transform=ax.transAxes, ha='left', va='top',
            fontsize=12, fontweight='bold')
    flip, valid, passed = flip_first
    hc = {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}
    for lv in rh_levels:
        if f'RHi_{lv}' not in tab:
            continue
        for is_flip, ls, lbl in [(True, '-', 'two-signed'), (False, '--', 'same sign')]:
            sel = valid & (passed == True) & (flip == is_flip) & (tab[f'LE_{pairs[0][0]}'].abs() > 0)
            d = pd.DataFrame({'b': pfb[sel], 'rh': tab.loc[sel, f'RHi_{lv}']}).dropna()
            g = d.groupby('b')['rh']
            st = pd.DataFrame({'med': g.median(), 'n': g.size()}).reindex(range(-1, len(centres)))
            ok = (st['n'] >= min_n).values
            ax.plot(xpos[ok], st['med'].values[ok], marker='o', ls=ls, color=hc.get(lv, 'k'), ms=5, lw=1.5,
                    label=f'{lv[:-1]} m, {lbl}')
    ax.axhline(100, color='k', lw=0.8, ls=':')
    ax.axvline(edges[0] - 0.5 * log_bin_width, color='0.6', lw=0.8, ls=':')
    ticks = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
    ax.set_xticks(np.r_[x_zero, ticks], ['0'] + [f'$10^{{{int(t)}}}$' for t in ticks])
    ax.set_xlabel('mass flux [g m$^{-2}$ s$^{-1}$]')
    ax.set_ylabel('median RH wrt ice [%]')
    ax.set_title(f'RH of two-signed vs same-signed profiles ({pairs[0][0][:-1]}→{pairs[0][1][:-1]} m, screened)', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.grid(True, alpha=0.25)
    fig.suptitle(f'two-signed LE profiles; screen: LI-7500 records above baseline ≤ {max_share:.0%} at both analysers',
                 fontsize=10)
    fig.tight_layout()
    table = pd.DataFrame(rows)
    table['pf_bin'] = pd.Categorical(table['pf_bin'], bin_labels, ordered=True)
    table['group'] = pd.Categorical(table['group'], [g[0] for g in groups], ordered=True)
    return fig, axes, table


def two_signed_summary(table):
    """Counts of two-signed profiles per pair and drift bin, all wind groups together, before and
    after the screen."""
    g = table.groupby(['pair', 'pf_bin', 'screened'], observed=True)[['n', 'n_flip']].sum()
    g['frac'] = g['n_flip'] / g['n']
    return g.unstack('screened')


# ---------------------------------------------------------------------------
# 3a. two-signed profiles (above) and layer energy (below)

def plot_layer_energy_by_wind(tab, pair, ws_col='WS_2m', bins=None, min_n=20, only_bs=True, bin_width=2.0):
    """Median dLE, dH and d(H + LE) across one layer per wind-speed bin.

    Sublimation in the layer predicts dLE = −dH in every bin, i.e. d(H + LE) = 0.
    Also returns, per bin, the slope of dH on dLE *inside* the bin, and the slope after removing
    the common wind dependence, to show how much of the half-hourly anti-correlation is only the
    shared wind dependence. ``bin_width`` sets the wind-speed bin size [m/s] when ``bins`` is not
    given; bins with fewer than ``min_n`` half-hours are left out. Returns ``(fig, ax, table)``.
    """
    from scipy.stats import linregress
    d = tab[[f'dLE_{pair}', f'dH_{pair}', ws_col, 'ustar_2m', 'bs']].dropna()
    if only_bs:
        d = d[d['bs'] == True]
    d = d.rename(columns={f'dLE_{pair}': 'dLE', f'dH_{pair}': 'dH'})
    d['dSum'] = d['dLE'] + d['dH']
    if bins is None:   # bins of `bin_width` m/s from 2 m/s up to the 99th percentile
        bins = np.arange(2, np.ceil(np.nanpercentile(d[ws_col], 99)) + bin_width, bin_width)
    shift = 0.075 * float(np.median(np.diff(bins)))   # side-by-side offset of dLE and dH
    d['wsb'] = pd.cut(d[ws_col], bins)
    rows = []
    for iv, g in d.groupby('wsb', observed=True):
        if len(g) < min_n:
            continue
        fit = linregress(g['dLE'], g['dH'])
        rows.append(dict(ws=iv.mid, n=len(g), dLE=g['dLE'].median(), dH=g['dH'].median(), dSum=g['dSum'].median(),
                         dLE_q25=g['dLE'].quantile(.25), dLE_q75=g['dLE'].quantile(.75),
                         dH_q25=g['dH'].quantile(.25), dH_q75=g['dH'].quantile(.75),
                         slope_in_bin=fit.slope, r2_in_bin=fit.rvalue ** 2))
    t = pd.DataFrame(rows).set_index('ws')

    # half-hourly slope after removing the common wind dependence (WS, WS², u*)
    X = np.c_[np.ones(len(d)), d[ws_col], d[ws_col] ** 2, d['ustar_2m']]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    raw, part = linregress(d['dLE'], d['dH']), linregress(res(d['dLE'].values), res(d['dH'].values))

    up, lo = pair.split('_')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(t.index - shift, t['dLE'], yerr=[t['dLE'] - t['dLE_q25'], t['dLE_q75'] - t['dLE']], fmt='o-',
                color='#0072B2', capsize=3, lw=1.8, label=f'dLE = LE({up} m) − LE({lo} m)')
    ax.errorbar(t.index + shift, t['dH'], yerr=[t['dH'] - t['dH_q25'], t['dH_q75'] - t['dH']], fmt='s-',
                color='#D55E00', capsize=3, lw=1.8, label=f'dH = H({up} m) − H({lo} m)')
    ax.plot(t.index, t['dSum'], 'D--', color='k', lw=1.5, ms=5, label='d(H + LE)')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel(DRIVER_LABELS.get(ws_col, ws_col))
    ax.set_ylabel('median and IQR across the layer [W m$^{-2}$]')
    ax.set_title(('blowing snow only. ' if only_bs else '')
                 + f'Half-hourly slope dH/dLE: {raw.slope:.2f} (r² {raw.rvalue**2:.2f}); '
                 + f'with wind removed: {part.slope:.2f} (r² {part.rvalue**2:.2f})', fontsize=9)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax, t[['n', 'dLE', 'dH', 'dSum', 'slope_in_bin', 'r2_in_bin']]


def plot_layer_energy_by_drift(tab, pair, log_bin_width=0.5, min_n=20):
    """Median dLE, dH and d(H + LE) across one layer per bin of the FlowCapt mass flux PF_FC4.

    The counterpart of :func:`plot_layer_energy_by_wind` with drift intensity on the x-axis.
    PF_FC4 is binned in steps of ``log_bin_width`` decades; half-hours without drift
    (PF_FC4 = 0) are drawn at the far left, labelled "0". Sublimation in the layer predicts
    dLE = −dH, i.e. d(H + LE) = 0, in every drift bin. Bins with fewer than ``min_n`` half-hours
    are left out. Returns ``(fig, ax, table)``.
    """
    d = tab[[f'dLE_{pair}', f'dH_{pair}', 'PF_FC4', 'WS_2m']].dropna(subset=[f'dLE_{pair}', f'dH_{pair}', 'PF_FC4'])
    d = d.rename(columns={f'dLE_{pair}': 'dLE', f'dH_{pair}': 'dH'})
    d['dSum'] = d['dLE'] + d['dH']
    lp = np.log10(d['PF_FC4'].where(d['PF_FC4'] > 0))
    lo_edge = np.floor(np.nanmin(lp) / log_bin_width) * log_bin_width
    hi_edge = np.ceil(np.nanpercentile(lp.dropna(), 99.5) / log_bin_width) * log_bin_width
    edges = np.arange(lo_edge, hi_edge + log_bin_width / 2, log_bin_width)
    centres = (edges[:-1] + edges[1:]) / 2
    x_zero = edges[0] - 1.5 * log_bin_width
    d['b'] = pd.cut(lp, edges, labels=False, include_lowest=True)
    d.loc[d['PF_FC4'] == 0, 'b'] = -1

    rows = []
    for k, g in d.dropna(subset=['b']).groupby('b'):
        if len(g) < min_n:
            continue
        k = int(k)
        rows.append(dict(x=x_zero if k == -1 else centres[k],
                         pf_bin='no drift' if k == -1 else f'{10 ** edges[k]:.2g}–{10 ** edges[k + 1]:.2g}',
                         n=len(g), WS_2m=g['WS_2m'].median(),
                         dLE=g['dLE'].median(), dH=g['dH'].median(), dSum=g['dSum'].median(),
                         dLE_q25=g['dLE'].quantile(.25), dLE_q75=g['dLE'].quantile(.75),
                         dH_q25=g['dH'].quantile(.25), dH_q75=g['dH'].quantile(.75)))
    t = pd.DataFrame(rows)
    drift = t['pf_bin'] != 'no drift'      # connect only the drift bins with a line
    shift = 0.075 * log_bin_width

    up, lo = pair.split('_')
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, q25, q75, color, mk, off, lbl in [
            ('dLE', 'dLE_q25', 'dLE_q75', '#0072B2', 'o', -shift, f'dLE = LE({up} m) − LE({lo} m)'),
            ('dH', 'dH_q25', 'dH_q75', '#D55E00', 's', shift, f'dH = H({up} m) − H({lo} m)')]:
        ax.errorbar(t['x'] + off, t[col], yerr=[t[col] - t[q25], t[q75] - t[col]], fmt=mk, color=color,
                    capsize=3, lw=1.8, label=lbl)
        ax.plot(t.loc[drift, 'x'] + off, t.loc[drift, col], '-', color=color, lw=1.8)
    ax.plot(t['x'], t['dSum'], 'D', color='k', ms=5, label='d(H + LE)')
    ax.plot(t.loc[drift, 'x'], t.loc[drift, 'dSum'], '--', color='k', lw=1.5)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(edges[0] - 0.5 * log_bin_width, color='0.6', lw=0.8, ls=':')
    ticks = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
    ax.set_xticks(np.r_[x_zero, ticks], ['0'] + [f'$10^{{{int(v)}}}$' for v in ticks])
    ax.set_xlabel('mass flux [g m$^{-2}$ s$^{-1}$]')
    ax.set_ylabel('median and IQR across the layer [W m$^{-2}$]')
    ax.set_title(f'Energy balance of the {lo}–{up} m layer per drift-flux bin', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax, t.set_index('pf_bin')[['n', 'WS_2m', 'dLE', 'dH', 'dSum']]


# ---------------------------------------------------------------------------
# 3b. water vapour budget (Schwat et al. 2025, J. Hydrometeor., Eq. 4)
# ---------------------------------------------------------------------------
BUDGET_COLORS = {'storage': '#7F7F7F', 'vtfd': '#0072B2', 'vadv': '#E69F00', 'res': '#D55E00'}
BUDGET_LABELS = {'storage': r'storage  $\partial\rho_v/\partial t$',
                 'vtfd': r"turbulent flux divergence  $\partial\overline{w'\rho_v'}/\partial z$",
                 'vadv': r'vertical advection  $\overline{w}\,\partial\rho_v/\partial z$',
                 'res': 'residual (source if > 0)'}


def absolute_humidity(T_C, RH):
    """Water vapour density [g m-3] from T [degC] and liquid-referenced RH [%]."""
    R_v = 461.5
    return RH / 100.0 * vapor_pressure_liquid_MK2005(T_C) / (R_v * (T_C + 273.15)) * 1e3


def planar_fit_w(u, v, w, n_sectors=10, by_month=True, min_n=200):
    """Mean vertical wind after a sectorial planar fit (Wilczak et al. 2001).

    ``u, v, w`` are mean wind components in the sonic frame (e.g. 1 min logger averages).
    Inside each wind-direction sector (and month, to follow the changing snow surface) the plane
    ``w = b0 + b1 u + b2 v`` is fitted by least squares and removed. What is left is the vertical
    wind relative to the mean streamline plane of that sector. Returns a Series like ``w``.
    Sectors with fewer than ``min_n`` samples are left NaN.
    """
    d = pd.DataFrame({'u': u, 'v': v, 'w': w}).dropna()
    sector = (np.rad2deg(np.arctan2(d['v'], d['u'])) % 360 // (360 / n_sectors)).astype(int)
    keys = [sector, d.index.to_period('M')] if by_month else [sector]
    out = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby(keys):
        if len(g) < min_n:
            continue
        A = np.c_[np.ones(len(g)), g['u'], g['v']]
        b = np.linalg.lstsq(A, g['w'].values, rcond=None)[0]
        out.loc[g.index] = g['w'].values - A @ b
    return out.reindex(w.index)


def vapour_budget(tab, slow_sfc, slow_upper, sonic_heights, th_heights, kw_met, pairs=None,
                  n_sectors=10):
    """Terms of the water vapour budget for each layer between two EC levels  [mg m-3 s-1]:

        residual = d(rho_v)/dt + d(w'rho_v')/dz + w * d(rho_v)/dz

    after Schwat et al. (2025, Eq. 4). A positive residual is a vapour source in the layer
    (blowing-snow sublimation), or unresolved horizontal advection, or measurement error.

    - storage: time derivative (central difference) of the mean of the 2 m and 26 m vapour
      density from the T/RH probes. The same value is used for every layer.
    - turbulent flux divergence (vtfd): (E_upper - E_lower) / dz with E = LE / Ls.
    - vertical advection (vadv): mean vertical wind from a sectorial planar fit of the 2 m and
      26 m logger sonic means, interpolated linearly in height (0 at the surface), times the
      vapour gradient of a log profile through the 2 m and 26 m probes, both at the layer
      mid-height. With only two humidity heights this term is a rough estimate.

    ``th_heights`` = ``{'2m': z, '26m': z}`` are the T/RH probe heights. Returns a DataFrame
    indexed like ``tab`` with ``rho_2m, rho_26m, w_2m, w_26m, E_<lvl>, storage`` and per pair
    ``vtfd_<pair>, vadv_<pair>, res_<pair>``.
    """
    rt = kw_met['resample_time']
    met = lambda x: resample_with_threshold(x, **kw_met).reindex(tab.index)
    b = pd.DataFrame(index=tab.index)

    # vapour density [g m-3]
    b['rho_2m'] = met(absolute_humidity(slow_sfc['TA'], slow_sfc['RH']))
    b['rho_26m'] = met(absolute_humidity(slow_upper['Temp_26m_Avg'], slow_upper['RH_26m_Avg']))

    # storage [mg m-3 s-1]
    dt = pd.Timedelta(rt).total_seconds()
    rho_mean = b[['rho_2m', 'rho_26m']].mean(axis=1, skipna=False)
    b['storage'] = (rho_mean.shift(-1) - rho_mean.shift(1)) / (2 * dt) * 1e3

    # mean vertical wind [m s-1] from the planar fit
    w2 = planar_fit_w(slow_sfc['Ux_Avg'], slow_sfc['Uy_Avg'], slow_sfc['Uz_Avg'], n_sectors)
    w26 = planar_fit_w(slow_upper['Ux_26m_Avg'], slow_upper['Uy_26m_Avg'], slow_upper['Uz_26m_Avg'], n_sectors)
    b['w_2m'], b['w_26m'] = met(w2), met(w26)

    # vapour flux [mg m-2 s-1]
    Ls = Lsubl(tab['TA_2m'])
    for lvl in sonic_heights:
        b[f'E_{lvl}'] = tab[f'LE_{lvl}'] / Ls * 1e6

    z_lo, z_hi = th_heights['2m'], th_heights['26m']
    dlnz = np.log(z_hi / z_lo)
    zs2, zs26 = sonic_heights['2m'], sonic_heights['26m']
    pairs = pairs or [tuple(p.split('_')) for p in tab.attrs.get('pairs', [])]
    for up, lo in pairs:
        up_k, lo_k = (up if up.endswith('m') else up + 'm'), (lo if lo.endswith('m') else lo + 'm')
        tag = f'{up_k[:-1]}_{lo_k[:-1]}'
        zu, zl = sonic_heights[up_k], sonic_heights[lo_k]
        zm = 0.5 * (zu + zl)
        b[f'vtfd_{tag}'] = (b[f'E_{up_k}'] - b[f'E_{lo_k}']) / (zu - zl)
        # w at the layer mid-height: linear between 0 (surface), 2 m and 26 m
        w_m = (b['w_2m'] + (b['w_26m'] - b['w_2m']) * (zm - zs2) / (zs26 - zs2)) if zm >= zs2 else b['w_2m'] * zm / zs2
        drho_dz = (b['rho_26m'] - b['rho_2m']) / dlnz / zm          # g m-4
        b[f'vadv_{tag}'] = w_m * drho_dz * 1e3
        b[f'res_{tag}'] = b['storage'] + b[f'vtfd_{tag}'] + b[f'vadv_{tag}']
    return b


def layer_energy_terms(tab, budget, slow_sfc, slow_upper, sonic_heights, th_heights, kw_met, pairs,
                       rho_air=1.16, cp=1004.0):
    """Energy budget terms of each layer between two EC levels [W m-2]:

        residual = dH + dLE + heat storage + heat advection + Ls (vapour storage + vapour advection)

    the energy counterpart of :func:`vapour_budget` (Schwat et al. 2025, Eq. 4, with the heat terms
    added). Without a source or sink in the layer and without horizontal advection the residual is 0.
    Uses ``budget`` from :func:`vapour_budget` for the mean vertical wind and the vapour terms.

    - heat storage: rho cp dz dT/dt (central difference) of T at the layer mid-height, from a log
      profile through the 2 m and 26 m T/RH probes.
    - heat advection: rho cp dz w dtheta/dz at the layer mid-height, with dtheta/dz = dT/dz + g/cp from
      the same log profile and w interpolated between the planar-fit w at 2 and 26 m.
    - vapour storage and advection: the ``storage`` and ``vadv_<pair>`` terms of ``budget``, times
      Ls dz.

    ``rho_air`` [kg m-3] is a constant (about 1.16 at PEA, 850 hPa and −18 °C). Returns a DataFrame
    indexed like ``tab`` with, per pair, ``dSum, Hstor, Hadv, Estor, Eadv, stor, adv, res``.
    """
    met = lambda x: resample_with_threshold(x, **kw_met).reindex(tab.index)
    T2, T26 = met(slow_sfc['TA']), met(slow_upper['Temp_26m_Avg'])
    dt = pd.Timedelta(kw_met['resample_time']).total_seconds()
    z2, z26 = th_heights['2m'], th_heights['26m']
    dlnz = np.log(z26 / z2)
    zs2, zs26 = sonic_heights['2m'], sonic_heights['26m']
    Ls = Lsubl(tab['TA_2m'])
    out = pd.DataFrame(index=tab.index)
    for up, lo in pairs:
        tag = f'{up[:-1]}_{lo[:-1]}'
        zu, zl = sonic_heights[up], sonic_heights[lo]
        dz, zm = zu - zl, 0.5 * (zu + zl)
        Tm = T2 + (T26 - T2) * np.log(zm / z2) / dlnz
        dth_dz = (T26 - T2) / dlnz / zm + 9.81 / cp
        w_m = budget['w_2m'] + (budget['w_26m'] - budget['w_2m']) * (zm - zs2) / (zs26 - zs2)
        out[f'dSum_{tag}'] = tab[f'dLE_{tag}'] + tab[f'dH_{tag}']
        out[f'Hstor_{tag}'] = rho_air * cp * dz * (Tm.shift(-1) - Tm.shift(1)) / (2 * dt)
        out[f'Hadv_{tag}'] = rho_air * cp * dz * w_m * dth_dz
        out[f'Estor_{tag}'] = Ls * dz * budget['storage'] * 1e-6
        out[f'Eadv_{tag}'] = Ls * dz * budget[f'vadv_{tag}'] * 1e-6
        out[f'stor_{tag}'] = out[f'Hstor_{tag}'] + out[f'Estor_{tag}']
        out[f'adv_{tag}'] = out[f'Hadv_{tag}'] + out[f'Eadv_{tag}']
        out[f'res_{tag}'] = out[f'dSum_{tag}'] + out[f'stor_{tag}'] + out[f'adv_{tag}']
        out[f'w_{tag}'] = w_m
    return out


def plot_budget_by_drift(budget, tab, pair):
    """Median budget terms per drift-intensity class for one layer. Returns ``(fig, ax, table)``."""
    terms = {'storage': 'storage', 'vtfd': f'vtfd_{pair}', 'vadv': f'vadv_{pair}', 'res': f'res_{pair}'}
    d = budget[list(terms.values())].dropna()
    d['drift'] = drift_class(tab.loc[d.index])
    d = d.dropna(subset=['drift'])
    g = d.groupby('drift', observed=True)
    med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
    n = g.size()
    x = np.arange(len(med)); wdt = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (name, col) in enumerate(terms.items()):
        ax.bar(x + (k - 1.5) * wdt, med[col], wdt * 0.92, color=BUDGET_COLORS[name], label=BUDGET_LABELS[name],
               yerr=[med[col] - q25[col], q75[col] - med[col]], error_kw=dict(lw=1, capsize=2, ecolor='0.3'))
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x, [f'{c}\n(n={n[c]})' for c in med.index])
    ax.set_xlabel('drift class, mass flux [g m$^{-2}$ s$^{-1}$]')
    ax.set_ylabel('median and IQR [mg m$^{-3}$ s$^{-1}$]')
    up, lo = pair.split('_')
    ax.set_title(f'Water vapour budget of the {lo}–{up} m layer', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    out = med.rename(columns={v: k for k, v in terms.items()}); out.insert(0, 'n', n)
    return fig, ax, out


def plot_budget_event(budget, tab, start, end, pair, smooth='2h', height_colors=None):
    """Event time series after Schwat et al. (2025, Fig. 6): drift flux, vapour flux per level,
    and the budget terms of one layer (rolling mean over ``smooth``)."""
    height_colors = height_colors or {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}
    sm = lambda x: x.rolling(smooth, center=True, min_periods=2).mean()
    b, t = budget[start:end], tab[start:end]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t['PF_FC4'], color='k', lw=1.5)
    axes[0].set_ylabel('mass flux [g m$^{-2}$ s$^{-1}$]')
    for lvl in [c[2:] for c in b.columns if c.startswith('E_')]:
        axes[1].plot(sm(b[f'E_{lvl}']), color=height_colors.get(lvl, '0.4'), lw=1.8, label=f'{lvl[:-1]} m')
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set_ylabel(r"$\overline{w'\rho_v'}$ [mg m$^{-2}$ s$^{-1}$]")
    axes[1].legend(ncol=3, fontsize=9, frameon=False)
    for name, col, ls in [('storage', 'storage', ':'), ('vtfd', f'vtfd_{pair}', '--'), ('vadv', f'vadv_{pair}', '-')]:
        axes[2].plot(sm(b[col]), color=BUDGET_COLORS[name], ls=ls, lw=1.8, label=BUDGET_LABELS[name])
    r = sm(b[f'res_{pair}'])
    axes[2].fill_between(r.index, 0, r, color=BUDGET_COLORS['res'], alpha=0.25, label=BUDGET_LABELS['res'])
    axes[2].axhline(0, color='k', lw=0.6)
    axes[2].set_ylabel('[mg m$^{-3}$ s$^{-1}$]')
    axes[2].legend(fontsize=8, frameon=False, ncol=2)
    up, lo = pair.split('_')
    axes[0].set_title(f'{start} → {end}: budget of the {lo}–{up} m layer ({smooth} rolling mean)', fontsize=10)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 4. driver ranking
# ---------------------------------------------------------------------------
def driver_correlation(tab, driver_groups):
    """Spearman correlation matrix of all driver columns."""
    cols = [c for g in driver_groups.values() for c in g]
    return tab[cols].corr(method='spearman')


def rank_divergence_drivers(tab, target, driver_groups, block='7D', n_splits=5, n_repeats=5,
                            min_samples=300, random_state=0):
    """Rank driver groups for one target with a gradient-boosted model.

    - model: ``HistGradientBoostingRegressor`` (handles NaN in the drivers natively)
    - cross-validation: ``GroupKFold`` on consecutive time blocks of length ``block``, because
      30 min samples are autocorrelated and random folds would inflate the skill
    - skill: R² of the pooled out-of-fold predictions (``r2``). Per-fold R² (``r2_folds``) is
      kept for reference but is unstable when a fold contains only quiet weeks.
    - importance: all columns of a group are permuted together in the held-out data and the
      drop in pooled out-of-fold R² is recorded (grouped permutation importance), so correlated
      variables inside a group do not share out the importance arbitrarily. ``std`` is the
      spread over ``n_repeats`` permutations.

    Returns a dict with ``r2``, ``r2_folds``, ``importance`` (DataFrame: mean, std, sorted),
    ``n``, ``model`` (fitted on all samples, for partial dependence), ``X``, ``target``.
    Returns ``None`` if fewer than ``min_samples`` valid targets exist.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    from threadpoolctl import threadpool_limits

    cols = [c for g in driver_groups.values() for c in g]
    d = tab[cols + [target]].dropna(subset=[target])
    d = d.loc[:, [c for c in d.columns if d[c].notna().any()]]
    cols = [c for c in cols if c in d.columns]
    if len(d) < min_samples:
        return None
    X, y = d[cols], d[target]
    blocks = ((d.index - d.index.min()) // pd.Timedelta(block)).values
    n_splits = min(n_splits, len(np.unique(blocks)))
    make = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                 min_samples_leaf=20, l2_regularization=1.0,
                                                 random_state=random_state)
    groups = {g: [c for c in gc if c in cols] for g, gc in driver_groups.items()}
    groups = {g: gc for g, gc in groups.items() if gc}
    rng = np.random.default_rng(random_state)
    oof = np.full(len(y), np.nan)
    oof_perm = {g: np.full((n_repeats, len(y)), np.nan) for g in groups}
    r2_folds = []
    # one thread: the data are small, and OpenMP oversubscription makes the fit ~15x slower
    with threadpool_limits(limits=1):
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, blocks):
            model = make().fit(X.iloc[tr], y.iloc[tr])
            Xte = X.iloc[te]
            oof[te] = model.predict(Xte)
            r2_folds.append(r2_score(y.iloc[te], oof[te]))
            for g, gcols in groups.items():
                for k in range(n_repeats):
                    Xp = Xte.copy()
                    Xp[gcols] = Xp[gcols].values[rng.permutation(len(Xp))]
                    oof_perm[g][k, te] = model.predict(Xp)
        final = make().fit(X, y)
    base = r2_score(y, oof)
    drops = {g: [base - r2_score(y, p[k]) for k in range(n_repeats)] for g, p in oof_perm.items()}
    imp = pd.DataFrame({'mean': {g: np.mean(v) for g, v in drops.items()},
                        'std': {g: np.std(v) for g, v in drops.items()}}).sort_values('mean', ascending=False)
    return dict(r2=base, r2_folds=np.array(r2_folds), importance=imp, n=len(d), model=final, X=X,
                target=target, oof=pd.Series(oof, index=y.index), y=y)


MIN_SKILL_R2 = 0.1   # below this out-of-fold R² the importances are not interpretable


def has_skill(res, min_r2=MIN_SKILL_R2):
    """True if a rank result exists and its out-of-fold R² reaches ``min_r2``."""
    return res is not None and float(np.mean(res['r2'])) >= min_r2


def plot_importance(results, title=None, min_r2=MIN_SKILL_R2):
    """Bar chart of grouped permutation importance. ``results`` is ``{label: rank result}``.
    Models with out-of-fold R² below ``min_r2`` are drawn in grey and marked as not interpretable."""
    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        print('No model was fitted (too few valid samples), so there is nothing to plot.')
        return None, None
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2), squeeze=False)
    for ax, (lbl, res) in zip(axes[0], results.items()):
        imp = res['importance'].iloc[::-1]
        ok = has_skill(res, min_r2)
        ax.barh(imp.index, imp['mean'], xerr=imp['std'], color='#0072B2' if ok else '#BBBBBB', height=0.6,
                error_kw=dict(lw=1, capsize=2, ecolor='0.3'))
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel('drop in out-of-fold R² when the group is permuted')
        ax.set_title(f'{lbl}\nout-of-fold R² = {float(np.mean(res["r2"])):.2f}, n = {res["n"]}'
                     + ('' if ok else '\nno skill: ranking not interpretable'), fontsize=10)
        ax.grid(True, axis='x', alpha=0.25)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_partial_dependence(res, driver_groups, n_groups=3, grid_resolution=30):
    """Partial-dependence curves for the first driver of each of the top ``n_groups`` groups
    plus every further driver in those groups."""
    from sklearn.inspection import partial_dependence
    from threadpoolctl import threadpool_limits
    top = list(res['importance'].index[:n_groups])
    feats = [c for g in top for c in driver_groups[g] if c in res['X'].columns]
    ncols = min(4, len(feats)); nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        with threadpool_limits(limits=1):
            pd_ = partial_dependence(res['model'], res['X'], [f], grid_resolution=grid_resolution,
                                     percentiles=(0.02, 0.98))
        ax.plot(pd_['grid_values'][0], pd_['average'][0], color='#0072B2', lw=2)
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel(DRIVER_LABELS.get(f, f), fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('modelled dLE [W m$^{-2}$]', fontsize=9)
    fig.suptitle(f'Partial dependence of {res["target"]} on the top {n_groups} driver groups', y=1.0)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. numbers quoted in the paper text
# ---------------------------------------------------------------------------
def drift_events(pf, threshold=0.001, max_gap='2h'):
    """Blowing-snow events from a regular mass-flux series: runs of ``pf > threshold``, merged when
    the gap between two runs is at most ``max_gap``. Missing values count as no drift. Returns a
    DataFrame with ``start, end, hours, pf_max``."""
    step = pd.Timedelta(pd.infer_freq(pf.index[:50]) or (pf.index[1] - pf.index[0]))
    on = (pf > threshold).values
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if runs and pf.index[i] - runs[-1][1] <= pd.Timedelta(max_gap) + step:
                runs[-1][1] = pf.index[j]
            else:
                runs.append([pf.index[i], pf.index[j]])
            i = j + 1
        else:
            i += 1
    ev = pd.DataFrame(runs, columns=['start', 'end'])
    ev['hours'] = (ev['end'] - ev['start'] + step) / pd.Timedelta('1h')
    ev['pf_max'] = [pf.loc[a:b].max() for a, b in zip(ev['start'], ev['end'])]
    return ev


def paper_numbers(tab, noise_thr, diag, energy=None, flip_pairs=(('2m', '16m'), ('2m', '26m')),
                  drift_thr=0.001, strong_thr=1.0, very_strong_thr=10.0, event_gap='2h',
                  profile_edges=(0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100), ws_edges=(2, 4, 6, 8, 10, 12, 14),
                  ws_min=8.0, max_share=0.1, zL_neutral=0.1, drift_ws_min=5.0, ws_col='WS_2m'):
    """Every number quoted in the measurement results of the paper, from one regular table ``tab``
    (30 min in the paper, which speaks of half-hours).

    "No drift" is PF_FC4 = 0. "Drift" is PF_FC4 > ``drift_thr`` with the wind speed ``ws_col`` above
    ``drift_ws_min`` in the same interval; "strong" drift has PF_FC4 > ``strong_thr`` and the same wind
    condition. Intervals with PF_FC4 > 0 but weaker wind count as neither drift nor no drift.
    ``noise_thr`` is ``{pair: noise floor}`` from :func:`classify_divergence`, ``diag``
    ``{level: li7500_diag_halfhourly(...)}`` for the screen, ``energy`` the output of
    :func:`layer_energy_terms` (optional). The sublimated mass is computed separately from the
    5 min fluxes, by :func:`sublimated_mass_5min`.

    Returns ``(numbers, tables)``: ``numbers`` has one row per quoted value (section, quantity, value,
    unit); ``tables`` holds the per-bin tables behind them (``events``, ``profile``, ``energy``).
    """
    from scipy.stats import spearmanr
    rows = []
    add = lambda sec, q, v, u='': rows.append(dict(section=sec, quantity=q, value=v, unit=u))
    pairs = [p for p in ('16_2', '26_2', '26_16') if f'dLE_{p}' in tab]
    pf = tab['PF_FC4']
    windy = tab[ws_col] > drift_ws_min
    pf_d = pf.where(windy | (pf == 0))         # mass flux, NaN where the FlowCapt reads drift in weak wind
    nodrift, drift = pf == 0, pf_d > drift_thr
    wtxt = f', {ws_col} > {drift_ws_min:g} m/s'
    step = pd.Timedelta(pd.infer_freq(tab.index[:50]) or (tab.index[1] - tab.index[0]))
    hh = step.total_seconds()
    W = 'W m-2'

    # 1. noise floor and window
    add('window', 'start', str(tab.index[0])); add('window', 'end', str(tab.index[-1]))
    add('window', 'interval', str(step))
    for p in pairs:
        add('noise floor', f'90th pct |dLE| {p}, no drift and neutral', noise_thr[p], W)

    # 2. drift occurrence and events
    v = tab.get('PF_FC4_raw', pf).dropna()     # all intervals with FlowCapt data, whatever the wind
    add('drift occurrence', 'intervals with FlowCapt data', len(v))
    add('drift occurrence', f'share PF > {drift_thr:g} (any wind)', (v > drift_thr).mean(), '-')
    add('drift occurrence', f'share PF > {drift_thr:g}{wtxt}', (pf_d.loc[v.index] > drift_thr).mean(), '-')
    add('drift occurrence', f'share PF > {strong_thr:g}{wtxt}', (pf_d.loc[v.index] > strong_thr).mean(), '-')
    add('drift occurrence', f'share PF > 0 in wind <= {drift_ws_min:g} m/s (neither drift nor no drift)',
        ((v > 0) & ~windy.loc[v.index]).mean(), '-')
    ev = drift_events(pf_d, drift_thr, event_gap)
    add('events', f'events (PF > {drift_thr:g}{wtxt}, gaps <= {event_gap} merged)', len(ev))
    for h in (12, 24):
        add('events', f'events longer than {h} h', int((ev['hours'] > h).sum()))

    # 3. no drift reference (all three levels valid)
    lv = ['2m', '16m', '26m']
    all3 = tab[[f'LE_{l}' for l in lv]].notna().all(axis=1)
    nd = tab[nodrift & all3]
    add('no drift', 'intervals (PF = 0, LE valid at 3 levels)', len(nd))
    for l in lv:
        add('no drift', f'mean LE {l}', nd[f'LE_{l}'].mean(), W)
    for p in pairs:
        add('no drift', f'mean dLE {p}', nd[f'dLE_{p}'].mean(), W)

    # 4. relation with the mass flux
    for p in pairs:
        d = tab.loc[drift, ['logPF', f'dLE_{p}']].dropna()
        add('dLE vs mass flux', f'Spearman r dLE {p} vs log PF (drift, n={len(d)})',
            spearmanr(d['logPF'], d[f'dLE_{p}'])[0], '-')
    for thr in (strong_thr, very_strong_thr):
        for p in pairs:
            add('dLE vs mass flux', f'median dLE {p}, PF > {thr:g}{wtxt}', tab.loc[pf_d > thr, f'dLE_{p}'].median(), W)

    # 5. profiles per mass-flux bin (intervals valid at all three levels)
    d = tab[all3].copy()
    labels = (['0', 'trace'] + [f'{a:g}–{b:g}' for a, b in zip(profile_edges[:-1], profile_edges[1:])]
              + [f'> {profile_edges[-1]:g}'])
    d['bin'] = pd.cut(pf_d.loc[d.index], [-np.inf, 0] + list(profile_edges) + [np.inf], labels=labels)   # weak-wind drift: NaN, left out
    d = d[d['bin'] != 'trace']          # 0 < PF <= the first edge: below the drift definition, not shown
    g = d.groupby('bin', observed=True)
    prof = g[[f'LE_{l}' for l in lv] + ['RHi_2m', 'RHi_26m', 'WS_2m']].mean()
    prof.insert(0, 'n', g.size())
    prof['share LE_2m < 0'] = g['LE_2m'].apply(lambda x: (x < 0).mean())
    prof['share LE_26m > 0'] = g['LE_26m'].apply(lambda x: (x > 0).mean())
    for b in prof.index:
        add('profile (mean per bin)', f'PF {b}: LE 2 / 16 / 26 m',
            ' / '.join(f'{prof.loc[b, f"LE_{l}"]:.1f}' for l in lv), W)
        add('profile (mean per bin)', f'PF {b}: share LE 2 m < 0 | RHi 2 m', 
            f'{prof.loc[b, "share LE_2m < 0"]:.2f} | {prof.loc[b, "RHi_2m"]:.0f} %')
    add('profile (mean per bin)', 'share LE 26 m > 0 for PF > 0.1',
        (d.loc[pf_d.loc[d.index] > 0.1, 'LE_26m'] > 0).mean(), '-')

    # 6. two-signed profiles
    for lo, up in flip_pairs:
        tag = f'{up[:-1]}_{lo[:-1]}'
        cls = sign_profile_class(tab, lo, up, noise_thr[tag])
        flip = cls == 'dep. below, subl. above'
        ok = agc_pass(tab, diag, [lo, up], max_share) == True
        sec = f'two-signed {lo}/{up} (floor {noise_thr[tag]:.2f})'
        add(sec, 'no drift: valid intervals', int((nodrift & cls.notna()).sum()))
        add(sec, 'no drift: two-signed intervals', int((nodrift & flip).sum()))
        m = (pf_d > strong_thr) & cls.notna()
        add(sec, f'PF > {strong_thr:g}: valid | two-signed | share', f'{m.sum()} | {(m & flip).sum()} | {(m & flip).sum() / m.sum():.2f}')
        ms = m & ok
        add(sec, f'PF > {strong_thr:g}, LI-7500 screen: valid | two-signed | share',
            f'{ms.sum()} | {(ms & flip).sum()} | {(ms & flip).sum() / max(ms.sum(), 1):.2f}')
        md = drift & ok
        for name, sel in (('two-signed', md & flip), ('same sign', md & (cls == 'same sign'))):
            add(sec, f'drift, screened, {name}: n | median RHi 2 m | 26 m',
                f'{sel.sum()} | {tab.loc[sel, "RHi_2m"].median():.0f} | {tab.loc[sel, "RHi_26m"].median():.0f}', '%')

    # 7. stability during strong drift
    s = tab[(pf_d > strong_thr) & tab['zL_2m'].notna()]
    neu = s[s['zL_2m'].abs() <= zL_neutral]
    add('strong drift, stability', f'PF > {strong_thr:g}: neutral | all (|z/L| <= {zL_neutral:g})', f'{len(neu)} | {len(s)}')
    for p in ('16_2', '26_2'):
        add('strong drift, stability', f'neutral: median dLE {p}', neu[f'dLE_{p}'].median(), W)
    add('strong drift, stability', 'neutral: share dLE 26_2 above noise floor', (neu['dLE_26_2'] > noise_thr['26_2']).mean(), '-')

    # 8. layer energy per wind bin (drift only)
    d = tab[drift].copy()
    d['wsb'] = pd.cut(d['WS_2m'], list(ws_edges))
    en_rows = []
    for p in ('16_2', '26_16', '26_2'):
        x = d.dropna(subset=[f'dLE_{p}', f'dH_{p}'])
        gg = x.groupby('wsb', observed=True)
        t = pd.DataFrame({'n': gg.size(), 'dLE': gg[f'dLE_{p}'].median(), 'dH': gg[f'dH_{p}'].median(),
                          'dSum': (x[f'dLE_{p}'] + x[f'dH_{p}']).groupby(x['wsb'], observed=True).median()})
        if energy is not None:
            e = energy.loc[x.index]
            for k in ('stor', 'adv', 'res'):
                t[k] = e[f'{k}_{p}'].groupby(x['wsb'], observed=True).median()
        t['pair'] = p
        en_rows.append(t.reset_index())
        hi = t[[iv.left >= ws_min for iv in t.index]]
        add(f'layer energy {p}, WS > {ws_min:g}', 'dLE range of bin medians', f'{hi["dLE"].min():.1f} to {hi["dLE"].max():.1f}', W)
        add(f'layer energy {p}, WS > {ws_min:g}', 'dH range of bin medians', f'{hi["dH"].min():.1f} to {hi["dH"].max():.1f}', W)
        add(f'layer energy {p}, WS > {ws_min:g}', 'd(H+LE) range of bin medians', f'{hi["dSum"].min():.1f} to {hi["dSum"].max():.1f}', W)
        if energy is not None:
            xs = x[x['WS_2m'] > ws_min]
            e = energy.loc[xs.index]
            add(f'layer energy {p}, WS > {ws_min:g}', 'median storage (heat + vapour)', e[f'stor_{p}'].median(), W)
            add(f'layer energy {p}, WS > {ws_min:g}', 'median advection (heat + vapour) | median w [cm/s]',
                f'{e[f"adv_{p}"].median():.1f} | {e[f"w_{p}"].median() * 100:.2f}', W)
            add(f'layer energy {p}, WS > {ws_min:g}', 'median residual d(H+LE) + storage + advection', e[f'res_{p}'].median(), W)
    energy_tab = pd.concat(en_rows, ignore_index=True)

    numbers = pd.DataFrame(rows)
    numbers['value'] = [round(x, 3) if isinstance(x, (float, np.floating)) else x for x in numbers['value']]
    return numbers.set_index(['section', 'quantity']), dict(events=ev, profile=prof, energy=energy_tab)


def sublimated_mass_5min(fluxes, drift30, ta, start, end, pairs=(('26m', '2m'), ('16m', '2m'), ('26m', '16m'))):
    """Sublimated mass [mm w.e. = kg m-2] from the 5 min EddyPro fluxes as they are: no resampling and
    no gap filling, so every sum uses only intervals in which the fluxes were measured.

    ``fluxes`` is ``{level: read_eddypro_data(...)}`` (QC mask applied there), ``drift30`` the 30 min
    drift flag of the analysis table (1 = drift, 0 = not, NaN = unknown); a 5 min interval takes the
    flag of its half-hour, so drift follows the paper's definition. ``ta`` is air temperature [°C] for the latent heat of
    sublimation (any resolution, matched to the nearest earlier value within 30 min; where it is
    missing the window median is used, Ls changes by < 0.2 % over the season).

    Per level and per pair (upper, lower) the sums run over the intervals valid at that level or at
    both levels of the pair. Returns a DataFrame with the same (section, quantity) index as
    :func:`paper_numbers`.
    """
    step = pd.Timedelta(pd.infer_freq(next(iter(fluxes.values())).index[:50]) or '5min')
    slots = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta('1D') - step, freq=step)
    dt = step.total_seconds()
    le = pd.DataFrame({l: f['LE'][~f.index.duplicated()].reindex(slots) for l, f in fluxes.items()})
    T = ta.sort_index().reindex(slots, method='ffill', tolerance=pd.Timedelta('30min'))
    Ls = Lsubl(T).fillna(Lsubl(T).median())
    E = le.div(Ls, axis=0) * dt                                   # kg m-2 per interval
    flag = drift30.astype(float).sort_index().reindex(slots, method='ffill', tolerance=pd.Timedelta('30min') - step)
    drift, nodrift = flag == 1, flag == 0
    n = len(slots)
    share = lambda m: f'{int(m.sum())} of {n} = {m.sum() / n:.1%}'
    rows = []
    add = lambda q, v, u='': rows.append(dict(section='sublimated mass (5 min)', quantity=q, value=v, unit=u))
    add('window', f'{slots[0]} to {slots[-1] + step}, {n} intervals of {step}')
    add('drift intervals (drift flag of the half-hour)', share(drift))
    for l in le:
        v = le[l].notna()
        add(f'valid LE {l}', share(v))
        add(f'valid LE {l}: sum LE dt / Ls', E[l][v].sum(), 'mm w.e.')
        add(f'valid LE {l}, drift: share of drift intervals', f'{(v & drift).sum() / drift.sum():.1%}')
    if len(le.columns) > 2:
        add('valid LE at all levels', share(le.notna().all(axis=1)))
    for up, lo in pairs:
        if up not in le or lo not in le:
            continue
        both = le[up].notna() & le[lo].notna()
        tag = f'{up}/{lo}'
        add(f'{tag} both valid', share(both))
        add(f'{tag} both valid, drift: share of drift intervals', f'{(both & drift).sum() / drift.sum():.1%}')
        for name, m in (('all conditions', both), ('drift', both & drift), ('outside drift', both & nodrift)):
            add(f'{tag} both valid, {name}: sum LE {lo} | LE {up} | difference',
                f'{E[lo][m].sum():.2f} | {E[up][m].sum():.2f} | {(E[up] - E[lo])[m].sum():.2f}', 'mm w.e.')
    out = pd.DataFrame(rows)
    out['value'] = [round(x, 3) if isinstance(x, (float, np.floating)) else x for x in out['value']]
    return out.set_index(['section', 'quantity'])


def sublimation_per_level_5min(fluxes, ta, start, end, drift30=None, levels=('2m', '16m', '26m'),
                               positive_only=True):
    """Sublimated mass [mm w.e. = kg m-2] per level from the positive 5 min EddyPro LE only, summed over
    the intervals in which LE is valid at ALL ``levels``, so the levels are compared on the same samples.

    Inputs as in :func:`sublimated_mass_5min`: ``fluxes`` = ``{level: read_eddypro_data(...)}``, ``ta``
    air temperature [°C] for Ls, ``drift30`` the optional 30 min drift flag (1 / 0 / NaN), taken over by
    the 5 min intervals of its half-hour. Returns ``(table, info)``: ``table`` has one row per level
    and the columns 'all' and, with ``drift30``, 'drift' and 'no drift'; ``info`` holds the number of
    intervals, the share of the window they cover and the share of the drift intervals they cover.

    ``positive_only=True`` sums sublimation only (negative LE counted as zero); ``False`` sums the net
    vapour flux (deposition subtracted). Differences between levels are a layer source only for the
    net sums: the source in the layer is F(upper) - F(lower) with both fluxes signed.
    """
    step = pd.Timedelta(pd.infer_freq(next(iter(fluxes.values())).index[:50]) or '5min')
    slots = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta('1D') - step, freq=step)
    le = pd.DataFrame({l: fluxes[l]['LE'][~fluxes[l].index.duplicated()].reindex(slots) for l in levels})
    T = ta.sort_index().reindex(slots, method='ffill', tolerance=pd.Timedelta('30min'))
    Ls = Lsubl(T).fillna(Lsubl(T).median())
    E = (le.clip(lower=0) if positive_only else le).div(Ls, axis=0) * step.total_seconds()   # kg m-2 per interval
    valid = le.notna().all(axis=1)
    masks = {'all': valid}
    info = {'positive_only': positive_only, 'levels': list(levels), 'step': step, 'n_window': len(slots), 'n_valid': int(valid.sum()), 'share_window': valid.mean()}
    if drift30 is not None:
        flag = drift30.astype(float).sort_index().reindex(slots, method='ffill',
                                                          tolerance=pd.Timedelta('30min') - step)
        masks['drift'], masks['no drift'] = valid & (flag == 1), valid & (flag == 0)
        info['share_drift'] = masks['drift'].sum() / (flag == 1).sum()
        info['n_drift'] = int(masks['drift'].sum())
    table = pd.DataFrame({k: E[m].sum() for k, m in masks.items()})
    table.index.name = 'level'
    return table, info


def sublimation_latex(table, info, heights=None, decimals=1, label='tab:sublimation', diffs=(('26m', '2m'),)):
    """LaTeX (booktabs) code for the table of :func:`sublimation_per_level_5min`. ``heights``
    (``{level: m}``) puts the measured sonic height in the first column instead of the level name;
    the caption states the sample (number of intervals, share of the period and of the drift).
    ``diffs`` adds one row per (upper, lower) pair below a rule: the upper minus the lower level, i.e.
    the sublimation within the layer between them."""
    pct = lambda x: rf'{100 * x:.0f}\,\%'
    cols = list(table.columns)
    head = {'all': 'All', 'drift': 'Drift', 'no drift': 'No drift'}
    what = ('Sublimated mass (mm w.e.) from the positive latent heat flux' if info.get('positive_only', True)
            else 'Net vapour flux (mm w.e.; positive upward, deposition subtracted)')
    caption = (what + ' at each level, summed over the '
               rf'{info["n_valid"]} 5\,min intervals ({pct(info["share_window"])} of the period) in which the '
               'flux is valid at ' + {2: 'both', 3: 'all three'}.get(len(info.get('levels', [0] * 3)), 'all')
               + ' levels')
    if 'share_drift' in info:
        caption += f'; these contain {pct(info["share_drift"])} of the intervals with drift'
    lines = [r'\begin{table}[t]', r'\centering', rf'\caption{{{caption}.}}', rf'\label{{{label}}}',
             r'\begin{tabular}{l' + 'r' * len(cols) + '}', r'\toprule',
             'Height & ' + ' & '.join(head.get(c, c) for c in cols) + r' \\', r'\midrule']
    name = lambda lvl: rf'{heights[lvl]:.1f}\,m' if heights else lvl.replace('m', r'\,m')
    num = lambda v: f'{v + 0.0 if round(v, decimals) != 0 else 0.0:.{decimals}f}'.replace('-', '$-$')   # no '-0.0'
    row_tex = lambda h, row: h + ' & ' + ' & '.join(num(v) for v in row) + r' \\'
    for lvl, row in table.iterrows():
        lines.append(row_tex(name(lvl), row))
    if diffs:
        lines.append(r'\midrule')
        for up, lo in diffs:
            lines.append(row_tex(f'{name(up)} $-$ {name(lo)}', table.loc[up] - table.loc[lo]))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)
