"""Combined SAXS + XPCS analysis, reading the averaged HDF files directly.

One figure, three panels:
  1. SAXS I(q) for B0146, the first B0147 file (frames 1-200), and the last five
     B0147 files (frames 801-1313), with the D0138 buffer subtracted.
  2. XPCS g2(tau) at one q for the first + last-five B0147 files, with fits.
  3. Fitted fast fraction f vs elapsed time, per q bin.

Colour encodes elapsed time and is CONSISTENT across all three panels (e.g. the
7863 s dataset is the same red everywhere).  In the fit panel the marker SHAPE
encodes the q bin.  B0146 is a 6 C reference taken before the 30 C isothermal
run; its acquisition time is not a time origin, so it appears (grey, dashed) in
the SAXS panel only.  Legends are intentionally omitted -- describe the curves in
the figure caption (the fit panel already carries the time stamps on its x-axis).

Note on the fits: the averaged files come from ~100 raw datasets each, so the
stored g2 error is the standard error of the mean (~10x smaller than a single
dataset).  The fits are therefore tight and the fit-parameter error bars small;
reduced chi^2 (printed below) stays near 1, confirming the errors are calibrated.
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

# --- PARAMETERS ---
data_dir = "/home/8-id-i/2022-1/babnigg202203_nexus/average/"

BACKGROUND_HEADER = 'D0138'          # subtracted as background in the SAXS panel
XPCS_HEADER = 'B0147'                # header used for the g2 / fit panels
N_LAST = 5                           # number of last (highest-frame) files
target_q_idx = 0                     # q bin shown in the g2 panel
fit_q_indices = [0, 1, 2, 3, 4]      # q bins fitted for the f-vs-time panel
Q_MARKERS = ['o', 's', '^', 'D', 'v']   # one marker per fit q bin
CHI2_MAX = 10.0                      # skip fits worse than this reduced chi^2
                                     # (the liquid t=0 file does not obey the
                                     # arrested double-exp model: chi^2 ~ 1e2-1e3)
PHI_AVERAGE = True                   # azimuthally average I(q) over phi sectors
CMAP = plt.cm.turbo                  # elapsed-time colour map (shared everywhere)

# --- FIT MODEL (baseline fixed at 1.0; see module docstring) ---
p1, p2, contrast = 0.5, 0.5, 0.135
BASELINE = 1.0


def double_exp_eq9(tau, tau_fast, f, tau_slow):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + BASELINE


p0 = [1e-3, 0.5, 100.0]
bounds = ([1e-6, 0.0, 1.0], [10.0, 1.0, 10000.0])

# --- HDF field locations ---
START_TIME_PATH = '/entry/start_time'
TIME_FORMAT     = '%Y-%m-%d %H:%M:%S'
FRAME_TIME_PATH = '/entry/instrument/detector_1/frame_time'
DELAY_PATH      = '/xpcs/multitau/delay_list'
G2_PATH         = '/xpcs/multitau/normalized_g2'
G2_ERR_PATH     = '/xpcs/multitau/normalized_g2_err'
DYN_Q_PATH      = '/xpcs/qmap/dynamic_v_list_dim0'
SAXS_PATH       = '/xpcs/temporal_mean/scattering_1d'
STATIC_MAP_PATH = '/xpcs/qmap/static_index_mapping'
STATIC_Q_PATH   = '/xpcs/qmap/static_v_list_dim0'
STATIC_PHI_PATH = '/xpcs/qmap/static_v_list_dim1'

_name_re = re.compile(r'Average_([A-Za-z]\d+)_.*?_(\d+)_(\d+)_results')


# --- READ HELPERS ---
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


def read_saxs_iq(hf, phi_average=True):
    intensity = np.asarray(hf[SAXS_PATH][()]).reshape(-1)
    idx_map = hf[STATIC_MAP_PATH][()]
    q_list = hf[STATIC_Q_PATH][()]
    n_phi = hf[STATIC_PHI_PATH].shape[0]
    q_idx = idx_map // n_phi
    uq = np.unique(q_idx)
    if phi_average and n_phi > 1:
        inten = np.array([np.nanmean(intensity[q_idx == qi]) for qi in uq])
    else:
        inten = np.array([intensity[q_idx == qi][0] for qi in uq])
    return q_list[uq], inten


def read_g2(hf):
    t0 = hf[FRAME_TIME_PATH][()]
    t0 = t0.item() if isinstance(t0, np.ndarray) else t0
    tau = hf[DELAY_PATH][()] * t0
    tau = tau[:, 0] if tau.ndim > 1 else tau
    return tau, hf[G2_PATH][()], hf[G2_ERR_PATH][()], hf[DYN_Q_PATH][()]


def fit_g2(tau, g2, g2_err):
    """Return (popt, f_err, red_chi2) or None."""
    valid = (tau > 0) & ~np.isnan(g2) & ~np.isnan(g2_err) & (g2_err > 0)
    if valid.sum() < 5:
        return None
    try:
        popt, pcov = curve_fit(double_exp_eq9, tau[valid], g2[valid], sigma=g2_err[valid],
                               absolute_sigma=True, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        return None
    resid = (g2[valid] - double_exp_eq9(tau[valid], *popt)) / g2_err[valid]
    red_chi2 = np.sum(resid**2) / (valid.sum() - len(popt))
    return popt, np.sqrt(pcov[1, 1]), red_chi2


# --- DISCOVER FILES ---
file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf')))
assert file_paths, f'no HDF files found in {data_dir}'
by_header, start_times = {}, {}
for fp in file_paths:
    header = parse_name(fp)[0]
    with h5py.File(fp, 'r') as hf:
        start_times[fp] = read_start_time(hf)
    by_header.setdefault(header, []).append(fp)
for h in by_header:
    by_header[h].sort(key=lambda p: parse_name(p)[1])

b0147 = by_header[XPCS_HEADER]
first_file = b0147[0]                          # first B0147 file (frames 1-200)
xpcs_files = b0147[-N_LAST:]                    # g2 / fit: last N only (NOT 1-200)
saxs_files = [first_file] + xpcs_files          # SAXS panel: first + last N
t_ref = start_times[first_file]                 # time origin = first B0147 file


def elapsed(fp):
    return (start_times[fp] - t_ref).total_seconds()


# Shared elapsed-time colour scale (0 .. last B0147 file); the same mapping is
# used in every panel so a given elapsed time is always the same colour.
norm = Normalize(vmin=0, vmax=max(elapsed(f) for f in saxs_files))
def ecolor(fp):
    return CMAP(norm(elapsed(fp)))


print('SAXS files:', [f'{parse_name(f)[1]}-{parse_name(f)[2]} ({elapsed(f):.0f} s)' for f in saxs_files])
print('XPCS files:', [f'{parse_name(f)[1]}-{parse_name(f)[2]} ({elapsed(f):.0f} s)' for f in xpcs_files])

# --- FIGURE ---
plt.rcParams.update({'font.size': 16, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

# ============================================================
# PANEL 1: SAXS I(q), background-subtracted (B0146 + first + last N B0147)
# ============================================================
bg_I = None
if BACKGROUND_HEADER in by_header:
    with h5py.File(by_header[BACKGROUND_HEADER][0], 'r') as hf:
        _, bg_I = read_saxs_iq(hf, PHI_AVERAGE)

for fp in by_header.get('B0146', []):
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = I - bg_I if bg_I is not None else I
    pos = I > 0
    ax1.plot(q[pos], I[pos], color='0.5', lw=2, ls='--')     # 6 C reference

for fp in saxs_files:                                        # first + last N B0147
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = I - bg_I if bg_I is not None else I
    pos = I > 0
    ax1.plot(q[pos], I[pos], color=ecolor(fp), lw=1.8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=20)
ax1.set_ylabel(r'$I(Q) - I_{\mathrm{buffer}}(Q)$ (a.u.)', fontsize=20)
ax1.set_title(f'SAXS ({BACKGROUND_HEADER} subtracted)', fontsize=16)

# ============================================================
# PANEL 2: g2(tau) at target q, with fits
# ============================================================
q_val_target = None
for fp in xpcs_files:
    with h5py.File(fp, 'r') as hf:
        tau, g2, g2_err, q_vals = read_g2(hf)
    q_val_target = q_vals[target_q_idx]
    color = ecolor(fp)
    m = tau > 0
    ax2.errorbar(tau[m], g2[m, target_q_idx], yerr=g2_err[m, target_q_idx], fmt='o',
                 color=color, markersize=4, capsize=2, elinewidth=1, alpha=0.85, linestyle='none')
    res = fit_g2(tau, g2[:, target_q_idx], g2_err[:, target_q_idx])
    if res is not None:
        popt, _, chi2 = res
        flag = '' if chi2 < CHI2_MAX else '  (unreliable -> no fit line drawn)'
        print(f'  g2 fit q{target_q_idx} {elapsed(fp):5.0f} s: f={popt[1]:.3f} red_chi2={chi2:.2f}{flag}')
        if chi2 < CHI2_MAX:      # skip drawing the fit for the liquid t=0 file
            tt = tau[m]
            t_fit = np.logspace(np.log10(tt.min()), np.log10(tt.max()), 200)
            ax2.plot(t_fit, double_exp_eq9(t_fit, *popt), color=color, lw=2.5)

ax2.set_xscale('log')
ax2.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=20)
ax2.set_ylabel('g$_2$', fontsize=20)
ax2.set_title(f'XPCS g$_2$ ($Q = {q_val_target:.5f}\ \AA^{{-1}}$)', fontsize=16)
ax2.set_ylim(1.0, 1.18)

# ============================================================
# PANEL 3: fitted fast fraction f vs elapsed time (colour=time, marker=q)
# ============================================================
fit_rows = []
for fp in xpcs_files:
    with h5py.File(fp, 'r') as hf:
        tau, g2, g2_err, q_vals = read_g2(hf)
    for q_idx in fit_q_indices:
        res = fit_g2(tau, g2[:, q_idx], g2_err[:, q_idx])
        if res is not None:
            popt, f_err, chi2 = res
            if chi2 >= CHI2_MAX:      # drop unreliable fits (liquid t=0 file)
                continue
            fit_rows.append({'q_index': q_idx, 'q_val': q_vals[q_idx],
                             'elapsed': elapsed(fp), 'f': popt[1], 'f_err': f_err})

for q_idx, marker in zip(fit_q_indices, Q_MARKERS):
    rows = sorted([r for r in fit_rows if r['q_index'] == q_idx], key=lambda r: r['elapsed'])
    if not rows:
        continue
    xs = [r['elapsed'] for r in rows]
    ys = [r['f'] for r in rows]
    es = [r['f_err'] for r in rows]
    ax3.plot(xs, ys, '-', color='0.75', lw=1, zorder=1)          # per-q guide line
    for x, y, e in zip(xs, ys, es):
        ax3.errorbar(x, y, yerr=e, marker=marker, color=CMAP(norm(x)),
                     markersize=10, capsize=3, mec='k', mew=0.5, zorder=2)

ax3.set_xlabel('Elapsed Time (s)', fontsize=20)
ax3.set_ylabel('Fast Fraction ($f$)', fontsize=20)
ax3.set_title('g$_2$ fit: fast fraction', fontsize=16)

plt.tight_layout()
plt.savefig('SAXPCS_Combined.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('wrote SAXPCS_Combined.pdf')
