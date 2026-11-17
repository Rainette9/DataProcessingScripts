"""Extra quality screens for EddyPro fluxes, built from the Peddy fast files.

EddyPro's quality flags do not see everything that went wrong upstream. Two known cases at PEA:

- **Sonic fault periods.** When the CSAT3 raises its error bits (diagnostic value >= 4096), Peddy
  sets those samples to NaN, but the samples just before and after a fault are corrupt without a
  flag, and Peddy's gap filling draws straight lines across the removed samples between those
  corrupt end points. The ramps are smooth, so neither the spike filter nor EddyPro's
  stationarity flags catch them, and the resulting fluxes (hundreds of W m-2) pass with qc = 0.
  :func:`sonic_fault_intervals` finds every averaging interval that contains flagged samples,
  straight from the fast files, and :func:`apply_sonic_screen` removes the fluxes there.

- LI-7500 signal-strength (AGC) periods: see ``func_flux_divergence.li7500_diag_halfhourly``.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

CSAT_FAULT = 4096          # CSAT3 diagnostic value at or above which an error bit is set
FLUX_COLS = ['H', 'LE', 'Tau', 'u*', 'h2o_flux', 'w_var', 'ts_var', 'u_var', 'v_var',
             'L', '(z-d)/L', 'w/ts_cov', 'w/h2o_cov', 'wind_speed']


def sonic_fault_intervals(folder_hf, sensor, start, end, step='5min', fault=CSAT_FAULT,
                          cache_file=None, verbose=True):
    """Per averaging interval: number of fast samples with a sonic fault, and the NaN share.

    Reads ``<folder_hf>/<sensor>/<yyyymm>/<sensor>_Fastdata_proc_<date>_<HHMM>.dat`` (one hour of
    100 ms data per file). Returns a DataFrame indexed by interval start with ``n`` (samples),
    ``n_fault`` (samples with ``diag_sonic >= fault``), ``nan_share`` (share of NaN ``Uz``, i.e.
    samples Peddy removed and could not fill) and ``diag_max``.

    Reading takes a minute or two per sensor; the result is cached in ``cache_file`` (CSV) and
    only hours missing from the cache are read.
    """
    cached = None
    if cache_file is not None and os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cached = cached[~cached.index.duplicated(keep='last')].sort_index()
    files = sorted(glob.glob(os.path.join(folder_hf, sensor, '*', f'{sensor}_Fastdata_proc_*.dat')))
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta('1D')
    todo = []
    for f in files:
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.dat$', f)
        if not m:
            continue
        t = pd.Timestamp(f'{m.group(1)} {m.group(2)}:{m.group(3)}')
        if t0 <= t < t1 and (cached is None or t not in cached.index):
            todo.append((t, f))
    if verbose and todo:
        print(f'{sensor}: reading the sonic diagnostic from {len(todo)} hourly fast files ...')
    rows = []
    for t, f in todo:
        x = pd.read_csv(f, usecols=['timestamp', 'Uz', 'diag_sonic'], parse_dates=['timestamp']).set_index('timestamp')
        g = x.groupby(pd.Grouper(freq=step))
        r = pd.DataFrame({'n': g.size(), 'n_fault': g['diag_sonic'].apply(lambda s: int((s >= fault).sum())),
                          'nan_share': g['Uz'].apply(lambda s: float(s.isna().mean())),
                          'diag_max': g['diag_sonic'].max()})
        r = r[r['n'] > 0]
        if t not in r.index:   # make sure the file's hour itself is recorded, so it is not re-read
            r.loc[t] = [0, 0, np.nan, np.nan]
        rows.append(r)
    out = pd.concat(rows) if rows else None
    if cached is not None:
        out = cached if out is None else pd.concat([cached, out])
    if out is None:
        raise FileNotFoundError(f'no fast files for {sensor} between {start} and {end} under {folder_hf}')
    out = out[~out.index.duplicated(keep='last')].sort_index()
    if cache_file is not None and todo:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        out.to_csv(cache_file)
    return out[t0:t1]


def apply_sonic_screen(fluxes, faults, margin=1, min_fault=1, cols=FLUX_COLS):
    """Copy of an EddyPro DataFrame with the fluxes removed in every averaging interval that
    contains at least ``min_fault`` sonic-fault samples, plus ``margin`` intervals on each side
    (the corrupt unflagged samples sit next to the flagged ones, and interpolated ramps can reach
    into the neighbouring interval).

    ``faults`` is the table from :func:`sonic_fault_intervals`, on the same averaging step as
    ``fluxes``. Returns ``(screened DataFrame, boolean Series of removed rows)``.
    """
    step = pd.infer_freq(fluxes.index[:50]) or pd.Timedelta(fluxes.index[1] - fluxes.index[0])
    bad = (faults['n_fault'] >= min_fault)
    bad = bad[bad].index
    removed = pd.Series(False, index=fluxes.index)
    for k in range(-margin, margin + 1):
        removed |= fluxes.index.isin(bad + k * pd.Timedelta(step))
    out = fluxes.copy()
    out.loc[removed, [c for c in cols if c in out.columns]] = np.nan
    return out, removed
