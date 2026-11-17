"""Constant-flux layer check between EC levels.

Monin-Obukhov bulk theory relates the surface to one measurement height through a single set
of fluxes, which assumes those fluxes do not change with height in between (the constant-flux
layer). With EC at several heights this can be tested directly:

- momentum:  u*(upper) / u*(reference)
- heat:      H(upper) / H(reference)
- moisture:  LE(upper) / LE(reference)
- Obukhov length: L(upper) / L(reference), which combines the first two (L ~ u*^3 / H)

A half-hour passes the check for an upper level either if L there is within a tolerance of the
reference level (``criterion='L'``, the default), or if u* and H are both within their own
tolerances (``criterion='ustar_H'``). LE is reported but is not part of the criterion by default,
because LE is the quantity the MO evaluation is about.

Used by section 11 of ``notebooks/MO_fluxes.ipynb``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import resample_with_threshold

RATIO_LABELS = {'ustar': r'$u_*$', 'H': 'H', 'LE': 'LE', 'L': 'L'}
RATIO_COLORS = {'ustar': '#0072B2', 'H': '#D55E00', 'LE': '#009E73', 'L': '#7F7F7F'}
CF_COLORS = {True: '#0072B2', False: '#D55E00'}
STAB_ORDER = ['unstable', 'neutral', 'stable']


def constant_flux_table(fluxes, kw_ec, start, end, ref='2m', min_ustar=0.05, min_H=5.0, min_LE=2.0,
                        zL_neutral=0.1, pf=None, pf_edges=(0.0, 0.1)):
    """Fluxes per level and their ratios to the reference level.

    Parameters
    ----------
    fluxes : dict
        ``{'2m': fluxes_sfc, '16m': fluxes_low, '26m': fluxes_up}`` from ``read_eddypro_data``.
    kw_ec : dict
        ``resample_with_threshold`` kwargs for the EddyPro series.
    ref : str
        Reference (lowest) level.
    min_ustar, min_H, min_LE : float
        A ratio is only formed where the reference-level value exceeds this magnitude
        (u* in m/s, H and LE in W m-2); ratios of near-zero fluxes are meaningless.
    zL_neutral : float
        ``|z/L| <= zL_neutral`` at the reference level counts as neutral.
    pf : Series or None
        FlowCapt mass flux, already resampled; adds a ``drift`` class column.
    pf_edges : (lo, hi)
        ``PF <= lo`` = no drift, ``lo < PF < hi`` = weak drift, ``PF >= hi`` = strong drift.

    Returns
    -------
    DataFrame with ``ustar_<lvl>, H_<lvl>, LE_<lvl>, L_<lvl>``, ratios ``r_<var>_<lvl>`` for
    every level except the reference, ``zL``, ``stability`` and (optionally) ``drift``.
    """
    rs = lambda s: resample_with_threshold(s, **kw_ec)
    cols = {}
    for lvl, fl in fluxes.items():
        cols[f'ustar_{lvl}'] = rs(fl['u*'])
        cols[f'H_{lvl}'] = rs(fl['H'])
        cols[f'LE_{lvl}'] = rs(fl['LE']) if 'LE' in fl.columns else pd.Series(np.nan, index=cols[f'H_{lvl}'].index)
        # L is built from u* and H, but read_eddypro_data only removes H (not L) where the H
        # quality flag fails. Drop L there too, so the L ratio rests on the same data as H.
        L_ok = fl['L'].where((fl['L'] != 0) & fl['H'].notna())
        # L from the averaged fluxes is not the average of L (L -> infinity near neutral),
        # so average 1/L and invert
        cols[f'L_{lvl}'] = 1.0 / rs(1.0 / L_ok)
    cols['zL'] = rs(fluxes[ref]['(z-d)/L'])
    t = pd.concat(cols, axis=1)[start:end]

    mins = {'ustar': min_ustar, 'H': min_H, 'LE': min_LE}
    for lvl in fluxes:
        if lvl == ref:
            continue
        for v in ('ustar', 'H', 'LE'):
            ok = t[f'{v}_{ref}'].abs() >= mins[v]
            t[f'r_{v}_{lvl}'] = (t[f'{v}_{lvl}'] / t[f'{v}_{ref}']).where(ok)   # all-NaN where a level has no LE
        okL = (t[f'ustar_{ref}'] >= min_ustar) & (t[f'H_{ref}'].abs() >= min_H)
        t[f'r_L_{lvl}'] = (t[f'L_{lvl}'] / t[f'L_{ref}']).where(okL)

    zl = t['zL']
    t['stability'] = np.select([zl < -zL_neutral, zl > zL_neutral, zl.notna()],
                               ['unstable', 'stable', 'neutral'], default=None)
    t.loc[zl.isna(), 'stability'] = np.nan
    if pf is not None:
        p = pf.reindex(t.index)
        lo, hi = pf_edges
        t['PF_FC4'] = p
        t['drift'] = np.select([p <= lo, p < hi, p >= hi],
                               ['no drift', f'PF < {hi:g}', f'PF ≥ {hi:g}'], default=None)
        t.loc[p.isna(), 'drift'] = np.nan
        t.attrs['drift_order'] = ['no drift', f'PF < {hi:g}', f'PF ≥ {hi:g}']
    t.attrs['ref'] = ref
    t.attrs['levels'] = [l for l in fluxes if l != ref]
    return t


def classify_constant_flux(cf, criterion='L', tol_L=1.0, tol_ustar=0.2, tol_H=0.3, tol_LE=None):
    """Add ``cf_<lvl>`` (True / False / NaN) per upper level.

    A ratio ``r`` is within tolerance ``tol`` if ``1 / (1 + tol) <= r <= 1 + tol``; this is
    symmetric on a log scale and requires the same sign.

    ``criterion='L'``: only the Obukhov-length ratio L(upper)/L(ref) is tested against ``tol_L``
    (1.0 = within a factor of 2). L ~ u*^3 / H, so this is one number for momentum and heat
    together. It is simple, but u* and H errors can cancel in L (e.g. u* 20 % higher and H 70 %
    higher give the same L), and L is poorly defined near neutral, where H is close to zero.

    ``criterion='ustar_H'``: u* and H are tested separately against ``tol_ustar`` and ``tol_H``;
    both must pass.

    With either criterion, LE must also pass if ``tol_LE`` is given. Half-hours where a required
    ratio is undefined stay NaN.
    """
    if criterion not in ('L', 'ustar_H'):
        raise ValueError("criterion must be 'L' or 'ustar_H'")
    cf = cf.copy()
    inside = lambda r, tol: (r >= 1.0 / (1.0 + tol)) & (r <= 1.0 + tol)
    for lvl in cf.attrs['levels']:
        need = ([(f'r_L_{lvl}', tol_L)] if criterion == 'L'
                else [(f'r_ustar_{lvl}', tol_ustar), (f'r_H_{lvl}', tol_H)])
        if tol_LE is not None:
            need.append((f'r_LE_{lvl}', tol_LE))
        ok = pd.Series(True, index=cf.index)
        valid = pd.Series(True, index=cf.index)
        for col, tol in need:
            ok &= inside(cf[col], tol)
            valid &= cf[col].notna()
        cf[f'cf_{lvl}'] = ok.where(valid)
    cf.attrs['criterion'] = criterion
    cf.attrs['tol'] = (dict(L=tol_L, LE=tol_LE) if criterion == 'L'
                       else dict(ustar=tol_ustar, H=tol_H, LE=tol_LE))
    return cf


def constant_flux_shares(cf, by):
    """Share of half-hours that pass the check per class of ``by``, per upper level, with n."""
    out = {}
    for lvl in cf.attrs['levels']:
        d = cf[[f'cf_{lvl}', by]].dropna()
        g = d.groupby(by)[f'cf_{lvl}']
        out[(lvl, 'share constant flux')] = g.mean()
        out[(lvl, 'n')] = g.size()
    t = pd.DataFrame(out)
    order = cf.attrs.get('drift_order') if by == 'drift' else STAB_ORDER
    return t.reindex([o for o in order if o in t.index])


def plot_flux_ratios(cf, by='stability', variables=('ustar', 'H', 'LE', 'L'), min_n=15):
    """Median and IQR of the flux ratios upper/reference per class of ``by``, one panel per
    upper level, on a log axis. The grey band marks the tolerances of the criterion."""
    levels = cf.attrs['levels']
    order = cf.attrs.get('drift_order') if by == 'drift' else STAB_ORDER
    order = [o for o in order if o in set(cf[by].dropna())]
    tol = cf.attrs.get('tol', {})
    fig, axes = plt.subplots(1, len(levels), figsize=(6.2 * len(levels), 4.8), sharey=True, squeeze=False)
    width = 0.8 / len(variables)
    for ax, lvl in zip(axes[0], levels):
        for k, v in enumerate(variables):
            xs, med, lo, hi = [], [], [], []
            for i, cls in enumerate(order):
                r = cf.loc[cf[by] == cls, f'r_{v}_{lvl}'].dropna()
                r = r[r > 0]          # a log axis cannot show sign changes; their share is in the table
                if len(r) < min_n:
                    continue
                xs.append(i + (k - (len(variables) - 1) / 2) * width)
                med.append(r.median()); lo.append(r.quantile(0.25)); hi.append(r.quantile(0.75))
            med, lo, hi = map(np.array, (med, lo, hi))
            if len(xs):
                ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt='o', color=RATIO_COLORS[v], capsize=3,
                            lw=1.6, ms=6, label=RATIO_LABELS[v])
        if tol.get('L') is not None:
            ax.axhspan(1 / (1 + tol['L']), 1 + tol['L'], color='0.85', zorder=0, linewidth=0,
                       label=f"L tolerance (factor {1 + tol['L']:g})")
        elif tol.get('ustar') is not None:
            ax.axhspan(1 / (1 + tol['ustar']), 1 + tol['ustar'], color='0.85', zorder=0, linewidth=0,
                       label=f"$u_*$ tolerance ±{tol['ustar']:.0%}")
        ax.axhline(1, color='k', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_yticks([0.25, 0.5, 1, 2, 4], ['0.25', '0.5', '1', '2', '4'])
        ax.set_xticks(range(len(order)), order)
        ax.set_title(f'{lvl[:-1]} m / {cf.attrs["ref"][:-1]} m')
        ax.grid(True, axis='y', alpha=0.25)
    axes[0, 0].set_ylabel('ratio upper / reference level\n(median and IQR, same-sign half-hours)')
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f'Flux ratios between EC levels by {"drift class" if by == "drift" else "SFC stability"}', y=1.0)
    fig.tight_layout()
    return fig, axes


def sign_change_shares(cf, by):
    """Share of half-hours where H (and LE) at the upper level has the opposite sign to the
    reference level, per class of ``by``."""
    out = {}
    for lvl in cf.attrs['levels']:
        for v in ('H', 'LE'):
            d = cf[[f'r_{v}_{lvl}', by]].dropna()
            out[(lvl, f'{v} opposite sign')] = (d[f'r_{v}_{lvl}'] < 0).groupby(d[by]).mean()
    t = pd.DataFrame(out)
    order = cf.attrs.get('drift_order') if by == 'drift' else STAB_ORDER
    return t.reindex([o for o in order if o in t.index])


def plot_mo_vs_ec_by_cf(cf, ec, mo, level, var='LE', kw_ec=None, kw_mo=None, lim=None):
    """EC against MO flux at one level, split by whether the constant-flux check passes
    between the reference level and that level. Returns ``(fig, ax, stats DataFrame)``."""
    from scipy.stats import linregress
    e = resample_with_threshold(ec, **kw_ec) if kw_ec else ec
    m = resample_with_threshold(mo, **kw_mo) if kw_mo else mo
    d = pd.concat([e.rename('ec'), m.rename('mo'), cf[f'cf_{level}'].rename('cf')], axis=1).dropna()
    d['cf'] = d['cf'].astype(bool)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    rows = []
    for flag, lbl in [(False, 'not constant flux'), (True, 'constant flux')]:
        g = d[d['cf'] == flag]
        if len(g) < 5:
            continue
        ax.scatter(g['ec'], g['mo'], s=12, alpha=0.4, color=CF_COLORS[flag], linewidths=0)
        fit = linregress(g['ec'], g['mo'])
        xx = np.array(g['ec'].quantile([0.005, 0.995]))
        err = g['mo'] - g['ec']
        ax.plot(xx, fit.intercept + fit.slope * xx, color=CF_COLORS[flag], lw=2.2,
                label=f'{lbl}: b = {fit.slope:.2f}, r² = {fit.rvalue**2:.2f}, n = {len(g)}')
        rows.append(dict(group=lbl, n=len(g), slope=fit.slope, r2=fit.rvalue ** 2,
                         bias=err.mean(), median_abs_error=err.abs().median(), rmse=np.sqrt((err ** 2).mean())))
    if lim is None:
        lim = np.nanpercentile(np.abs(d[['ec', 'mo']].values), 99.5)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='1:1')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color='k', lw=0.5, alpha=0.5); ax.axvline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel(f'{var} EC {level[:-1]} m [W m$^{{-2}}$]')
    ax.set_ylabel(f'{var} MO {level[:-1]} m [W m$^{{-2}}$]')
    ax.set_title(f'{var} at {level[:-1]} m: EC vs MO, split by the constant-flux check '
                 f'({cf.attrs["ref"][:-1]}–{level[:-1]} m)', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax, pd.DataFrame(rows).set_index('group') if rows else pd.DataFrame()


def plot_mo_error_vs_ratio(cf, ec, mo, level, ratio='ustar', var='LE', kw_ec=None, kw_mo=None, n_bins=10,
                           min_n=15):
    """Median and IQR of the MO − EC difference against a flux ratio (quantile bins)."""
    e = resample_with_threshold(ec, **kw_ec) if kw_ec else ec
    m = resample_with_threshold(mo, **kw_mo) if kw_mo else mo
    d = pd.concat([(m - e).rename('err'), cf[f'r_{ratio}_{level}'].rename('r')], axis=1).dropna()
    d = d[d['r'] > 0]
    b = pd.qcut(d['r'], n_bins, duplicates='drop')
    g = d.groupby(b, observed=True)
    s = pd.DataFrame({'x': g['r'].median(), 'med': g['err'].median(), 'q25': g['err'].quantile(.25),
                      'q75': g['err'].quantile(.75), 'n': g.size()})
    s = s[s['n'] >= min_n]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(s['x'], s['q25'], s['q75'], color=RATIO_COLORS[ratio], alpha=0.2, linewidth=0)
    ax.plot(s['x'], s['med'], 'o-', color=RATIO_COLORS[ratio], lw=2)
    tol = cf.attrs.get('tol', {}).get(ratio)
    if tol is not None:
        ax.axvspan(1 / (1 + tol), 1 + tol, color='0.85', zorder=0, linewidth=0, label=f'tolerance ±{tol:.0%}')
        ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color='k', lw=0.6, ls='--'); ax.axvline(1, color='k', lw=0.6, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(f'{RATIO_LABELS[ratio]}({level[:-1]} m) / {RATIO_LABELS[ratio]}({cf.attrs["ref"][:-1]} m)')
    ax.set_ylabel(f'{var} MO − EC at {level[:-1]} m [W m$^{{-2}}$]\n(median and IQR)')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax, s


# ---------------------------------------------------------------------------
# coupling: the "hockey stick" of Sun et al. (2012)
# ---------------------------------------------------------------------------
def hockey_stick_fit(U, sw, weights=None, n_grid=80):
    """Two-segment linear fit of sigma_w against wind speed with a free breakpoint.

    Below the breakpoint V_s the fit is a (nearly flat) line, above it a steeper line; both meet
    at V_s. The breakpoint is found by a grid search between the 5th and 95th percentile of U,
    keeping only solutions where the upper slope is steeper than the lower one. Pass binned
    medians with their counts as ``weights`` for a fit that is robust to outliers.
    Returns ``dict(V_s, slope_lo, slope_hi, intercept, sse)`` or ``None``.
    """
    U, sw = np.asarray(U, float), np.asarray(sw, float)
    w = np.ones_like(U) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(U) & np.isfinite(sw) & np.isfinite(w)
    U, sw, w = U[ok], sw[ok], w[ok]
    sq = np.sqrt(w)
    best = None
    for vs in np.linspace(np.percentile(U, 5), np.percentile(U, 95), n_grid):
        # sw = a + b_lo * U + (b_hi - b_lo) * max(U - vs, 0)
        X = np.c_[np.ones_like(U), U, np.maximum(U - vs, 0)]
        coef, *_ = np.linalg.lstsq(X * sq[:, None], sw * sq, rcond=None)
        if coef[2] <= 0:          # upper segment must be steeper
            continue
        sse = float(np.sum(w * (sw - X @ coef) ** 2))
        if best is None or sse < best['sse']:
            best = dict(V_s=vs, intercept=coef[0], slope_lo=coef[1], slope_hi=coef[1] + coef[2], sse=sse)
    return best


STAB_COLORS = {'unstable': '#D55E00', 'neutral': '#7F7F7F', 'stable': '#0072B2'}


def plot_hockey_stick(fluxes, kw_ec, start, end, pf=None, bs_threshold=0.0, n_bins=30, min_n=15,
                      fit_on='all', sw_max=None, color_by='drift', zL=None, zl_neutral=0.1,
                      stab_colors=STAB_COLORS, fit_per_class=False, min_n_class=200):
    """sigma_w against wind speed per EC level (Sun et al. 2012).

    One panel per level in ``fluxes`` (``{'2m': df, ...}``), using EddyPro ``w_var`` and
    ``wind_speed`` at that level, averaged with ``kw_ec``. Points are half-hours (grey: no drift,
    orange: drift, if ``pf`` is given), lines are binned medians, and the two-segment fit gives
    the threshold wind speed V_s: below it turbulence is weak and not driven by surface shear
    (decoupled), above it sigma_w grows linearly with wind (coupled).
    ``color_by='drift'`` colours the points by the ``pf`` groups; ``color_by='stability'`` colours
    them by the class of ``zL`` (unstable / neutral / stable with ``zl_neutral``). Pass the 2 m
    z/L for all panels: a level's own z/L is built from its own u* and H, so weak turbulence
    means large z/L by construction and the colouring would partly show sigma_w itself.
    ``fit_per_class=True`` (with ``color_by='stability'``) draws binned medians and a separate
    two-segment fit for every stability class with at least ``min_n_class`` half-hours, in the
    class colour, instead of one fit for all data; the table then has one row per level and class.
    ``fit_on`` selects the half-hours for the fit: 'all' (default; the no-drift half-hours alone
    do not reach high wind speeds at the upper levels), 'no drift' or 'drift'. The fit uses the
    binned medians weighted by their counts, so outliers (e.g. sonic-fault periods) do not pull
    it. ``sw_max`` clips the y-axis (default: 99.5th percentile).
    Returns ``(fig, axes, table of fits)``.
    """
    rs = lambda x: resample_with_threshold(x, **kw_ec)[start:end]
    bs = None
    if pf is not None:
        bs = (pf.reindex(rs(fluxes[next(iter(fluxes))]['w_var']).index) > bs_threshold)
    stab = None
    if color_by == 'stability':
        if zL is None:
            raise ValueError("color_by='stability' needs zL (e.g. fluxes_sfc['(z-d)/L'])")
        z = rs(zL)
        stab = pd.Series(np.select([z < -zl_neutral, z > zl_neutral, z.notna()],
                                   ['unstable', 'stable', 'neutral'], default=None), index=z.index, dtype=object)
        stab[z.isna()] = np.nan
    levels = list(fluxes)
    fig, axes = plt.subplots(1, len(levels), figsize=(5.4 * len(levels), 4.6), sharey=True, squeeze=False)
    rows = []
    for ax, lvl in zip(axes[0], levels):
        d = pd.DataFrame({'U': rs(fluxes[lvl]['wind_speed']),
                          'sw': np.sqrt(rs(fluxes[lvl]['w_var']))}).dropna()
        if bs is not None:
            d['bs'] = bs.reindex(d.index)
        if stab is not None:
            d['stab'] = stab.reindex(d.index)
            for cls in ['stable', 'neutral', 'unstable']:   # stable first: it is the biggest class
                g = d[d['stab'] == cls]
                if len(g):
                    ax.scatter(g['U'], g['sw'], s=5, alpha=0.3, color=stab_colors[cls], linewidths=0,
                               label=f'{cls} (n={len(g)})')
        elif bs is not None:
            for flag, c, lbl in [(False, '0.55', 'no drift'), (True, '#D55E00', 'drift')]:
                g = d[d['bs'] == flag]
                ax.scatter(g['U'], g['sw'], s=5, alpha=0.25, color=c, linewidths=0, label=f'{lbl} (n={len(g)})')
        else:
            ax.scatter(d['U'], d['sw'], s=5, alpha=0.25, color='0.55', linewidths=0, label=f'all (n={len(d)})')
        sel = d if (fit_on == 'all' or bs is None) else d[d['bs'] == (fit_on == 'drift')]
        # one fit set (label, data, colour), or one per stability class
        if fit_per_class and stab is not None:
            fit_sets = [(cls, d[d['stab'] == cls], stab_colors[cls]) for cls in ['unstable', 'neutral', 'stable']]
            fit_sets = [(c, g, col) for c, g, col in fit_sets if len(g) >= min_n_class]
        else:
            fit_sets = [(fit_on, sel, '#0072B2')]
        for k_set, (lbl, sel_k, col) in enumerate(fit_sets):
            b = pd.qcut(sel_k['U'], n_bins, duplicates='drop')
            g = sel_k.groupby(b, observed=True)
            med = pd.DataFrame({'U': g['U'].median(), 'sw': g['sw'].median(), 'n': g.size()})
            med = med[med['n'] >= min_n]
            med_col = 'k' if len(fit_sets) == 1 else col
            ax.plot(med['U'], med['sw'], 'o-', color=med_col, ms=4, lw=1.5, label=f'binned median ({lbl})')
            fit = hockey_stick_fit(med['U'], med['sw'], weights=med['n'])
            if fit is not None:
                xx = np.linspace(sel_k['U'].min(), sel_k['U'].max(), 100)
                yy = fit['intercept'] + fit['slope_lo'] * xx + (fit['slope_hi'] - fit['slope_lo']) * np.maximum(xx - fit['V_s'], 0)
                ax.plot(xx, yy, color=col, lw=2.2, label=f"fit {lbl}: V$_s$ = {fit['V_s']:.1f} m/s")
                ax.axvline(fit['V_s'], color=col, ls=':', lw=1)
            else:
                fit = dict(V_s=np.nan, intercept=np.nan, slope_lo=np.nan, slope_hi=np.nan)
                ax.text(0.98, 0.05 + 0.06 * k_set, f'{lbl}: no breakpoint (no steeper upper segment)',
                        transform=ax.transAxes, ha='right', fontsize=8, color=col)
            rows.append(dict(level=lvl, fit=lbl, n_fit=len(sel_k), V_s=fit['V_s'], slope_below=fit['slope_lo'],
                             slope_above=fit['slope_hi'], sigma_w_at_Vs=fit['intercept'] + fit['slope_lo'] * fit['V_s'],
                             share_below_Vs=float((sel_k['U'] < fit['V_s']).mean())))
        ax.set_ylim(0, sw_max or np.nanpercentile(d['sw'], 99.5))
        ax.set_title(f'{lvl[:-1]} m sonic', fontsize=11)
        ax.set_xlabel(f'wind speed at {lvl[:-1]} m [m s$^{{-1}}$]')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc='upper left', markerscale=3)
    axes[0, 0].set_ylabel(r'$\sigma_w$ [m s$^{-1}$]')
    fig.tight_layout()
    return fig, axes, pd.DataFrame(rows).set_index(['level', 'fit'])


def plot_ratios_vs_drift(cf, levels=None, only_class='neutral', log_bin_width=0.5, min_n=15,
                         variables=('ustar', 'H', 'LE')):
    """Flux ratios upper / 2 m against the FlowCapt mass flux, for coupled half-hours only.

    One panel per variable (u*, H, LE), one line per upper level, median and IQR per PF_FC4 bin
    of ``log_bin_width`` decades; half-hours without drift are drawn at the far left ("0").
    ``only_class`` restricts the half-hours to one surface stability class (default neutral, i.e.
    the coupled regime right of the hockey-stick break); ``None`` uses all. The grey band marks
    the criterion tolerances where set. If u* stays near 1 while LE and H move away, the layer is
    a constant-flux layer for momentum but not for the scalars. Returns ``(fig, axes, table)``.
    """
    levels = levels or cf.attrs['levels']
    ref = cf.attrs.get('ref', '2m')
    d = cf if only_class is None else cf[cf['stability'] == only_class]
    pf = d['PF_FC4']
    lp = np.log10(pf.where(pf > 0))
    lo_edge = np.floor(np.nanmin(lp) / log_bin_width) * log_bin_width
    hi_edge = np.ceil(np.nanpercentile(lp.dropna(), 99.5) / log_bin_width) * log_bin_width
    edges = np.arange(lo_edge, hi_edge + log_bin_width / 2, log_bin_width)
    centres = (edges[:-1] + edges[1:]) / 2
    x_zero = edges[0] - 1.5 * log_bin_width
    b = pd.cut(lp, edges, labels=False, include_lowest=True).where(pf != 0, -1)
    tol = cf.attrs.get('tol', {})
    lvl_colors = {'5m': '#CC79A7', '16m': '#E69F00', '26m': '#009E73'}
    offsets = (np.arange(len(levels)) - (len(levels) - 1) / 2) * 0.25 * log_bin_width

    fig, axes = plt.subplots(1, len(variables), figsize=(5.4 * len(variables), 4.8), sharex=True, squeeze=False)
    rows = []
    for ax, v in zip(axes[0], variables):
        for lvl, off in zip(levels, offsets):
            r = d[f'r_{v}_{lvl}']
            r = r.where(r > 0)   # log axis: same-sign half-hours only; sign changes are counted in the table
            g = pd.DataFrame({'b': b, 'r': r}).dropna().groupby('b')['r']
            st = pd.DataFrame({'med': g.median(), 'q25': g.quantile(.25), 'q75': g.quantile(.75), 'n': g.size()})
            st = st[st['n'] >= min_n]
            idx = st.index.values.astype(int)
            x = np.where(idx == -1, x_zero, centres[np.clip(idx, 0, len(centres) - 1)]) + off
            ax.errorbar(x, st['med'], yerr=[st['med'] - st['q25'], st['q75'] - st['med']], fmt='o', ms=5,
                        capsize=3, lw=1.5, color=lvl_colors.get(lvl, '0.3'), label=f'{lvl[:-1]} m / {ref[:-1]} m')
            drift = idx != -1
            ax.plot(x[drift], st['med'][drift], '-', color=lvl_colors.get(lvl, '0.3'), lw=1.5)
            neg = pd.DataFrame({'b': b, 'r': d[f'r_{v}_{lvl}']}).dropna()
            for k, st_k in st.iterrows():
                k = int(k)
                lab = 'no drift' if k == -1 else f'{10 ** edges[k]:.2g}–{10 ** edges[k + 1]:.2g}'
                rows.append(dict(variable=v, level=lvl, pf_bin=lab, n=int(st_k['n']), median_ratio=st_k['med'],
                                 share_opposite_sign=float((neg.loc[neg['b'] == k, 'r'] < 0).mean())))
        t_ = tol.get(v)
        if t_ is not None:
            ax.axhspan(1 / (1 + t_), 1 + t_, color='0.88', zorder=0, linewidth=0, label=f'tolerance ±{t_:.0%}')
        ax.axhline(1, color='k', lw=0.8, ls='--')
        ax.axvline(edges[0] - 0.5 * log_bin_width, color='0.6', lw=0.8, ls=':')
        ax.set_yscale('log')
        ax.set_yticks([0.25, 0.5, 1, 2, 4], ['0.25', '0.5', '1', '2', '4'])
        ticks = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
        ax.set_xticks(np.r_[x_zero, ticks], ['0'] + [f'$10^{{{int(t)}}}$' for t in ticks])
        ax.set_xlabel(cf.attrs.get('pf_label', 'PF_FC4') + ' [g m$^{-2}$ s$^{-1}$]')
        ax.set_title(f'{RATIO_LABELS[v]}(upper) / {RATIO_LABELS[v]}({ref[:-1]} m)', fontsize=11)
        ax.grid(True, axis='y', alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    axes[0, 0].set_ylabel(f'ratio upper / {ref[:-1]} m (median and IQR)')
    sub = f'{only_class} half-hours at {ref[:-1]} m only' if only_class else 'all half-hours'
    fig.suptitle(f'Flux ratios between levels against drift intensity ({sub})', y=1.0)
    fig.tight_layout()
    out = pd.DataFrame(rows)
    order = ['no drift'] + [f'{10 ** a:.2g}–{10 ** b:.2g}' for a, b in zip(edges[:-1], edges[1:])]
    out['pf_bin'] = pd.Categorical(out['pf_bin'], [o for o in order if o in set(out['pf_bin'])], ordered=True)
    return fig, axes, out
