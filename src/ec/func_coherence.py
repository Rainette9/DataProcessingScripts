import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, csd
from scipy.optimize import curve_fit


def compute_coherence(u1, u2, dz, fs=10):
    u1_clean, u2_clean = u1.align(u2, join='inner')
    mask = (~u1_clean.isna()) & (~u2_clean.isna())
    u1_clean = u1_clean[mask]
    u2_clean = u2_clean[mask]

    f, S11 = welch(u1_clean, fs=fs, nperseg= 4096)
    _, S22 = welch(u2_clean, fs=fs, nperseg= 4096)
    _, S12 = csd(u1_clean, u2_clean, fs=fs, nperseg= 4096)
    coh = np.abs(S12)**2 / (S11 * S22)

    u_mean = ((u1_clean + u2_clean) / 2).mean()
    def model(f, a):
        return np.exp(-a * dz * f / u_mean)

    mask_fit = np.isfinite(coh) & np.isfinite(f)
    popt, _ = curve_fit(model, f[mask_fit], coh[mask_fit], p0=[1.0])
    a_best = popt[0]
    y = model(f, a_best)

    return f, coh, y, a_best

def compute_coherence_fit(u_mean, coh, dz, f):
    def model(f, a):
        return np.exp(-a * dz * f / u_mean)
    popt, _ = curve_fit(model, f, coh, p0=[1.0])
    a_best = popt[0]
    y = model(f, a_best)

    return y, a_best

def plot_coherence(u1, u2, dz, label, color=None, fs=10):
    u1_clean, u2_clean = u1.align(u2, join='inner')
    mask = (~u1_clean.isna()) & (~u2_clean.isna())
    u1_clean = u1_clean[mask]
    u2_clean = u2_clean[mask]

    f, S11 = welch(u1_clean, fs=fs, nperseg= 4096)
    _, S22 = welch(u2_clean, fs=fs, nperseg= 4096)
    _, S12 = csd(u1_clean, u2_clean, fs=fs, nperseg= 4096)
    coh = np.abs(S12)**2 / (S11 * S22)

    u_mean = ((u1_clean + u2_clean) / 2).mean()
    def model(f, a):
        return np.exp(-a * dz * f / u_mean)

    mask_fit = np.isfinite(coh) & np.isfinite(f)
    popt, _ = curve_fit(model, f[mask_fit], coh[mask_fit], p0=[1.0])
    a_best = popt[0]
    y = model(f, a_best)

    # Convert frequency to period in seconds, avoid division by zero
    period = np.divide(1, f, out=np.full_like(f, np.nan), where=(f > 0))
    plt.semilogx(period, coh, label=fr'{label} $|\mathrm{{S}}_{{ij}}|^2/(\mathrm{{S}}_i \mathrm{{S}}_j)$', color=color)
    plt.semilogx(period, y, '--', label=f'{label} Fit: a={a_best:.2f}', color=color)
    plt.gca().invert_xaxis()