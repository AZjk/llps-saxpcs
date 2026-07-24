"""SI figure: 2x2 grid of XPCS g2(tau) for the 4 non-primary q bins.

The main figure (saxpcs.py) shows q index 0; this shows q indices 1-4.  Each
subplot plots the first B0147 file (frames 1-200) and the last five B0147 files
(801-1313), coloured by elapsed time on the SAME scale as the main figure, with
double-exponential fits (baseline fixed at 1.0) overlaid.
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- PARAMETERS ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
XPCS_HEADER = 'B0147'
N_LAST = 5
grid_q_indices = [1, 2, 3, 4]        # the 4 "other" q bins (main figure = 0)
# XPCS elapsed-time colours: same scale as saxpcs.py -- the N_LAST times are
# mapped to evenly-spaced positions of this colormap for maximum separation.
XPCS_CMAP = plt.cm.plasma
CMAP_LO, CMAP_HI = 0.10, 0.88

# --- FIT MODEL (baseline fixed at 1.0) ---
p1, p2, contrast = 0.5, 0.5, 0.135
BASELINE = 1.0


def double_exp_eq9(tau, tau_fast, f, tau_slow):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + BASELINE


p0 = [1e-3, 0.5, 100.0]
bounds = ([1e-6, 0.0, 1.0], [10.0, 1.0, 10000.0])

# --- HDF paths ---
START_TIME_PATH = '/entry/start_time'
TIME_FORMAT     = '%Y-%m-%d %H:%M:%S'
FRAME_TIME_PATH = '/entry/instrument/detector_1/frame_time'
DELAY_PATH      = '/xpcs/multitau/delay_list'
G2_PATH         = '/xpcs/multitau/normalized_g2'
G2_ERR_PATH     = '/xpcs/multitau/normalized_g2_err'
DYN_Q_PATH      = '/xpcs/qmap/dynamic_v_list_dim0'

_name_re = re.compile(r'Average_([A-Za-z]\d+)_.*?_(\d+)_(\d+)_results')


def parse_name(fname):
    m = _name_re.search(os.path.basename(fname))
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, -1, -1)


def read_start_time(hf):
    raw = hf[START_TIME_PATH][()]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return datetime.strptime(str(raw).strip(), TIME_FORMAT)


def read_g2(hf):
    t0 = hf[FRAME_TIME_PATH][()]
    t0 = t0.item() if isinstance(t0, np.ndarray) else t0
    tau = hf[DELAY_PATH][()] * t0
    tau = tau[:, 0] if tau.ndim > 1 else tau
    return tau, hf[G2_PATH][()], hf[G2_ERR_PATH][()], hf[DYN_Q_PATH][()]


def fit_g2(tau, g2, g2_err):
    valid = (tau > 0) & ~np.isnan(g2) & ~np.isnan(g2_err) & (g2_err > 0)
    if valid.sum() < 5:
        return None
    try:
        popt, _ = curve_fit(double_exp_eq9, tau[valid], g2[valid], sigma=g2_err[valid],
                            absolute_sigma=True, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except RuntimeError:
        return None


# --- DISCOVER FILES ---
file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf')))
b0147, start_times = [], {}
for fp in file_paths:
    if parse_name(fp)[0] != XPCS_HEADER:
        continue
    with h5py.File(fp, 'r') as hf:
        start_times[fp] = read_start_time(hf)
    b0147.append(fp)
b0147.sort(key=lambda p: parse_name(p)[1])

xpcs_files = b0147[-N_LAST:]                    # last N only (NOT the 1-200 file)
t_ref = start_times[b0147[0]]                   # time origin = first B0147 file
elapsed = {fp: (start_times[fp] - t_ref).total_seconds() for fp in xpcs_files}

# Elapsed-time colour, matching saxpcs.py: N_LAST times mapped to evenly spaced
# colormap positions (sorted by time) so consecutive times are easy to tell apart.
_xpcs_sorted = sorted(xpcs_files, key=lambda p: elapsed[p])
_xpcs_pos = np.linspace(CMAP_LO, CMAP_HI, len(_xpcs_sorted))
xpcs_color = {fp: XPCS_CMAP(p) for fp, p in zip(_xpcs_sorted, _xpcs_pos)}

# Pre-load g2 once per file.
data = {}
for fp in xpcs_files:
    with h5py.File(fp, 'r') as hf:
        data[fp] = read_g2(hf)

# --- FIGURE ---
plt.rcParams.update({'font.size': 14, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

for ax, q_idx in zip(axes.flat, grid_q_indices):
    for fp in xpcs_files:
        tau, g2, g2_err, q_vals = data[fp]
        color = xpcs_color[fp]
        m = tau > 0
        ax.errorbar(tau[m], g2[m, q_idx], yerr=g2_err[m, q_idx], fmt='o', color=color,
                    markersize=4, capsize=2, elinewidth=1, alpha=0.85, linestyle='none')
        res = fit_g2(tau, g2[:, q_idx], g2_err[:, q_idx])
        if res is not None:
            t_fit = np.logspace(np.log10(tau[m].min()), np.log10(tau[m].max()), 200)
            ax.plot(t_fit, double_exp_eq9(t_fit, *res), color=color, lw=2.0)
    ax.set_xscale('log')
    ax.set_ylim(0.98, 1.18)                 # slightly below 1 so error bars aren't clipped
    ax.set_yticks([1.0, 1.05, 1.10, 1.15])
    ax.set_title(f'$Q = {q_vals[q_idx]:.5f}\ \AA^{{-1}}$', fontsize=14)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(which='major', ls='-', lw=0.5, color='0.80')
    ax.grid(which='minor', ls='-', lw=0.3, color='0.90')

for ax in axes[-1, :]:
    ax.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=16)
for ax in axes[:, 0]:
    ax.set_ylabel('g$_2$', fontsize=16)

plt.tight_layout()
plt.savefig('g2_grid_SI.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('wrote g2_grid_SI.pdf')
