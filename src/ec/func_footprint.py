"""Flux footprint climatologies after Kljun et al. (2015, FFP) for the PEA EC levels.

The FFP code itself is not part of this repo: pass the folder that holds
``calc_footprint_FFP_climatology.py`` (``~/Documents/Github/FFP_Python(2)``) as ``ffp_dir``.
Inputs come from the EddyPro full_output as read by ``read_eddypro_data``.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OMEGA = 7.2921e-5          # Earth's angular velocity [rad s-1]
PEA_LAT = -71.95           # Princess Elisabeth Antarctica [deg]
HEIGHT_COLORS = {'2m': '#0072B2', '16m': '#E69F00', '26m': '#009E73'}


def boundary_layer_height(ustar, L, lat=PEA_LAT):
    """Boundary-layer height [m] from u* and the Obukhov length, as recommended for FFP when no
    measurement exists: the stable-layer equation of Nieuwstadt (1981),
    h = L / 3.8 * (-1 + sqrt(1 + 2.28 u* / (f L))), for L > 0, and its neutral limit
    h = 0.3 u* / |f| otherwise (no convective estimate; unstable cases are rare at PEA in winter)."""
    f = 2 * OMEGA * abs(np.sin(np.deg2rad(lat)))
    ustar, L = np.asarray(ustar, float), np.asarray(L, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        h_stable = L / 3.8 * (-1 + np.sqrt(1 + 2.28 * ustar / (f * L)))
    return np.where(L > 0, h_stable, 0.3 * ustar / f)


def vector_mean_direction(wd, step):
    """Vector-mean wind direction [deg] of ``wd`` per ``step`` (labelled by interval start)."""
    r = np.deg2rad(wd)
    return np.rad2deg(np.arctan2(np.sin(r).resample(step).mean(), np.cos(r).resample(step).mean())) % 360


def ffp_inputs(fluxes, wind_dir, start, end, lat=PEA_LAT):
    """FFP input table from one EddyPro full_output DataFrame for ``start``–``end``.

    zm is taken from EddyPro itself, (z-d)/L × L, so that it matches the L used; umean is the
    EddyPro wind speed at zm; sigma_v = sqrt(v_var); h from :func:`boundary_layer_height`.
    The wind direction must come from a wind vane (``wind_dir``, slow data, any resolution), not
    from EddyPro: the fast data are double-rotated in Peddy before EddyPro, so EddyPro's
    ``wind_dir`` is only the orientation of the rotated axes (about constant). ``wind_dir`` is
    vector-averaged over each EddyPro interval. Only intervals with a valid momentum flux (Tau
    passed the QC mask of ``read_eddypro_data``) and complete inputs are kept.
    """
    d = fluxes.loc[start:end]
    d = d[~d.index.duplicated()]
    step = d.index.to_series().diff().mode()[0]
    wd = vector_mean_direction(wind_dir.loc[start:end], step).reindex(d.index)
    out = pd.DataFrame({
        'zm': d['(z-d)/L'] * d['L'],
        'umean': d['wind_speed'],
        'ol': d['L'],
        'sigmav': np.sqrt(d['v_var']),
        'ustar': d['u*'],
        'wind_dir': wd,
    }, index=d.index)
    out['h'] = boundary_layer_height(out['ustar'], out['ol'], lat)
    if 'Tau' in d:
        out = out[d['Tau'].notna()]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def footprint_climatology(inputs, ffp_dir, domain=(-3000, 3000, -3000, 3000), dx=10.0,
                          rs=(0.5, 0.8), verbosity=0):
    """FFP climatology of all intervals in ``inputs`` (from :func:`ffp_inputs`). Intervals outside
    the validity range of FFP (u* <= 0.1 m/s, zm/L <= -15.5, h <= zm) are skipped by FFP itself.
    Returns the FFP output dict plus ``n_in`` (intervals passed in)."""
    if ffp_dir not in sys.path:
        sys.path.insert(0, ffp_dir)
    from calc_footprint_FFP_climatology import FFP_climatology
    zm = inputs['zm'].median()          # constant per sonic; the median avoids rounding noise
    res = FFP_climatology(zm=float(zm), z0=None, umean=inputs['umean'].tolist(), h=inputs['h'].tolist(),
                          ol=inputs['ol'].tolist(), sigmav=inputs['sigmav'].tolist(),
                          ustar=inputs['ustar'].tolist(), wind_dir=inputs['wind_dir'].tolist(),
                          domain=list(domain), dx=dx, dy=dx, rs=list(rs), smooth_data=1, crop=False,
                          verbosity=verbosity, fig=False)
    res['n_in'] = len(inputs)
    res['zm'] = zm
    return res


def footprint_summary(res, inputs):
    """Peak distance and the largest upwind distance of each r% contour [m] of one climatology."""
    f = res['fclim_2d']
    i, j = np.unravel_index(np.nanargmax(f), f.shape)
    row = {'zm [m]': res['zm'], 'n used': res['n'], 'n in': res['n_in'],
           'peak distance [m]': float(np.hypot(res['x_2d'][i, j], res['y_2d'][i, j])),
           'median WD [deg]': float(np.rad2deg(np.arctan2(np.sin(np.deg2rad(inputs['wind_dir'])).mean(),
                                                          np.cos(np.deg2rad(inputs['wind_dir'])).mean())) % 360),
           'median u* [m/s]': inputs['ustar'].median(), 'median L [m]': inputs['ol'].median(),
           'median h [m]': inputs['h'].median()}
    for r, xr, yr in zip(res.get('rs', []), res.get('xr', []), res.get('yr', [])):
        row[f'{int(round(r * 100))} % extent [m]'] = (float(np.nanmax(np.hypot(xr, yr))) if xr is not None else np.nan)
    return row


def dem_local_grid(dem_path, lat, lon, half_width=4000.0, res=None):
    """Elevation model resampled onto a local grid around a point: x east and y north of
    (``lat``, ``lon``) [m], from -``half_width`` to +``half_width``, the frame of the FFP footprints.

    Works for any projected GeoTIFF (e.g. the REMA mosaic in EPSG:3031, whose grid axes are not
    aligned with north): every local grid point is mapped through an azimuthal equidistant
    projection centred on the point into the DEM projection and interpolated bilinearly. ``res``
    defaults to the DEM resolution. Values below -1000 m are treated as missing.
    Returns ``(X, Y, Z)`` as 2-D arrays.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import CRS, Transformer
    from scipy.ndimage import map_coordinates
    with rasterio.open(dem_path) as src:
        crs = src.crs
        res = res or abs(src.res[0])
        x0, y0 = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
        pad = 1.5 * half_width                       # covers the rotation between the two frames
        win = from_bounds(x0 - pad, y0 - pad, x0 + pad, y0 + pad, src.transform).round_offsets().round_lengths()
        z = src.read(1, window=win, masked=True).astype(float).filled(np.nan)
        wt = src.window_transform(win)
    z[z < -1000] = np.nan
    local = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    xs = np.arange(-half_width, half_width + res / 2, res)
    X, Y = np.meshgrid(xs, xs)
    DX, DY = Transformer.from_crs(local, crs, always_xy=True).transform(X, Y)
    col, row = ~wt * (DX, DY)                        # fractional pixel coordinates (corner-based)
    Z = map_coordinates(z, [row - 0.5, col - 0.5], order=1, mode='constant', cval=np.nan)
    return X, Y, Z


def plot_footprints(results, rs_plot=(0.5, 0.8), colors=None, figsize=None, inset_extent=250.0, inset_pos=None,
                    dem=None, dem_interval=10.0, dem_label_every=50.0):
    """One map panel per entry of ``results`` = ``{panel title: {level: (ffp result, inputs)}}``:
    the r% source-area contours of every level (solid = largest r, dashed = smaller ones), the tower
    at the origin, x east and y north of the tower [m], and the mean wind direction in the title. All
    panels share one view cropped to the contours; an inset zooms on ±``inset_extent`` m around the
    tower so that the short 2 m footprint is visible (None = no inset). ``inset_pos`` gives the inset
    position per panel as ``[x0, y0, width, height]`` in axes fractions (one list, or one per panel).
    ``dem`` = ``(X, Y, Z)`` from :func:`dem_local_grid` draws elevation contours every ``dem_interval``
    m in grey under the footprints, labelled every ``dem_label_every`` m.
    Returns ``(fig, axes, summary table)``."""
    colors = colors or HEIGHT_COLORS
    styles = {r: ('-' if r == max(rs_plot) else '--') for r in rs_plot}
    contours, rows = {}, []
    xs, ys = [0.0], [0.0]
    for title, levels in results.items():
        contours[title] = []
        for lvl, (res, inp) in levels.items():
            rows.append({'panel': title, 'level': lvl, **footprint_summary(res, inp)})
            for r, xr, yr in zip(res.get('rs', []), res.get('xr', []), res.get('yr', [])):
                if xr is None or not any(np.isclose(r, rs_plot)):
                    continue
                style = styles[min(rs_plot, key=lambda q: abs(q - r))]
                contours[title].append((lvl, style, np.asarray(xr), np.asarray(yr)))
                xs += [np.nanmin(xr), np.nanmax(xr)]
                ys += [np.nanmin(yr), np.nanmax(yr)]
    pad = 0.08 * max(max(xs) - min(xs), max(ys) - min(ys))
    xlim, ylim = (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)
    summary = pd.DataFrame(rows).set_index(['panel', 'level'])

    n = len(results)
    w, h_ = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fig, axes = plt.subplots(1, n, figsize=figsize or (5.2 * n, 5.2 * h_ / w + 0.8), squeeze=False)
    axes = axes[0]
    if inset_pos is None or np.ndim(inset_pos) == 1:
        inset_pos = [inset_pos or [0.14, 0.53, 0.40, 0.40]] * n
    for k, (ax, (title, levels)) in enumerate(zip(axes, results.items())):
        targets = [ax]
        if inset_extent:
            ins = ax.inset_axes(inset_pos[k])
            ins.set_xlim(-inset_extent, inset_extent)
            ins.set_ylim(-inset_extent, inset_extent)
            ins.set_aspect('equal')
            ins.tick_params(labelsize=7)
            ins.set_title(f'±{inset_extent:g} m', fontsize=7, pad=2)
            ins.grid(True, alpha=0.25)
            targets.append(ins)
        if dem is not None:
            X, Y, Z = dem
            lev = np.arange(np.floor(np.nanmin(Z) / dem_interval) * dem_interval, np.nanmax(Z) + dem_interval, dem_interval)
            for tgt in targets:
                cs = tgt.contour(X, Y, Z, levels=lev, colors='0.6', linewidths=0.5, zorder=0)
                if tgt is ax and dem_label_every:
                    lab = [l for l in cs.levels if np.isclose(l % dem_label_every, 0) or np.isclose(l % dem_label_every, dem_label_every)]
                    tgt.clabel(cs, levels=lab, fontsize=6, fmt='%d', inline=True)
        for tgt in targets:
            for lvl, style, xr, yr in contours[title]:
                tgt.plot(xr, yr, style, color=colors.get(lvl, 'k'), lw=1.6)
            tgt.plot(0, 0, marker='^', color='k', ms=6 if tgt is not ax else 8, zorder=5, mfc='none' if tgt is not ax else 'k')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        wdeg = summary.loc[title]['median WD [deg]'].iloc[-1]       # highest level
        ax.set_title(f'{title}, mean wind direction {wdeg:.0f}°', fontsize=10)
        ax.set_xlabel('distance east of the tower [m]')
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel('distance north of the tower [m]')
    handles = [plt.Line2D([], [], color=colors.get(l, 'k'), lw=2, label=l) for l in
               dict.fromkeys(l for lv in results.values() for l in lv)]
    handles += [plt.Line2D([], [], color='0.3', ls=styles[r], lw=1.5, label=f'{int(round(r * 100))} % source area')
                for r in sorted(rs_plot)]
    handles += [plt.Line2D([], [], color='k', marker='^', ls='', label='tower')]
    axes[-1].legend(handles=handles, fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    return fig, axes, summary
