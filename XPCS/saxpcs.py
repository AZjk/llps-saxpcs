"""Combined SAXS + XPCS analysis, reading the averaged HDF files directly.

One figure, three panels:
  1. SAXS I(q) for B0146, the first B0147 file (frames 1-200), and the last five
     B0147 files (frames 801-1313), with the D0138 buffer subtracted.
  2. XPCS g2(tau) at one q for the first + last-five B0147 files, with fits.
  3. Fitted fast fraction f vs elapsed time, per q bin.
A second figure (3 panels) plots the fit parameters: (a) shared exponents p1, p2
vs elapsed time, (b) tau_fast vs Q, and (c) tau_slow vs Q.

Colour encodes elapsed time and is CONSISTENT across all three panels (e.g. the
7863 s dataset is the same red everywhere).  In the fit panel the marker SHAPE
encodes the q bin.  B0146 is a 6 C reference taken before the 30 C isothermal
run; its acquisition time is not a time origin, so it appears (grey, dashed) in
the SAXS panel only.  Legends are intentionally omitted -- describe the curves in
the figure caption (the fit panel already carries the time stamps on its x-axis).

Fit model and shared exponents
------------------------------
The g2 model is a double stretched-exponential (Siegert form):

    g2 = contrast * ( f e^-(tau/tau_fast)^p1 + (1-f) e^-(tau/tau_slow)^p2 )^2 + 1

with contrast = 0.135 (read from g2) and baseline = 1 both fixed.  For each
elapsed time all fitted q bins are fit SIMULTANEOUSLY (a global fit): the
stretching exponents p1 (fast) and p2 (slow) are SHARED across q -- they depend
only on elapsed time -- while tau_fast, f and tau_slow are independent per q.
This is the physically motivated constraint that the relaxation-shape exponents
are a property of the sample state at a given age, not of the q bin, and it
stabilises the otherwise poorly-conditioned slow mode (tau_slow lies beyond the
~1.5 s delay window, so its shape cannot be pinned q-by-q).

Uncertainties
-------------
The fit minimises error-weighted residuals (g2_model - g2)/g2_err using the
g2_err stored in the averaged files directly (absolute_sigma convention).  The
parameter covariance is inv(J^T J) at the solution; 1-sigma errors on f, p1 and
p2 are the square roots of its diagonal.  Because p1/p2 are shared, their
uncertainty is correctly propagated into f (f errors are larger than a per-q
fixed-exponent fit would report -- that difference is the exponent systematic,
now folded in honestly).
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.optimize import least_squares

# --- PARAMETERS ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BACKGROUND_HEADER = 'D0138'          # subtracted as background in the SAXS panel
# Absolute scattering cross-section coefficients: convert the raw SAXS 1d to an
# absolute differential cross section (cm^-1) as
#     I_abs(q) = coef_sam * I_sample(q) - coef_buf * I_buffer(q)
coef_sam = 6.93e4                     # sample coefficient
coef_buf = 7.62e4                     # buffer coefficient
XPCS_HEADER = 'B0147'                # header used for the g2 / fit panels
N_LAST = 5                           # number of last (highest-frame) files
target_q_idx = 0                     # q bin shown in the g2 panel
fit_q_indices = [0, 1, 2, 3, 4]      # q bins fitted for the f-vs-time panel
Q_MARKERS = ['o', 's', '^', 'D', 'v']   # one marker per fit q bin
CHI2_MAX = 10.0                      # skip fits worse than this reduced chi^2
                                     # (the liquid t=0 file does not obey the
                                     # arrested double-exp model: chi^2 ~ 1e2-1e3)
PHI_AVERAGE = True                   # azimuthally average I(q) over phi sectors
# XPCS elapsed-time colours: the N_LAST times are mapped to evenly-spaced
# positions of this colormap (maximises separation so times are easy to tell
# apart), independent of the SAXS-only SI figure.  The 0 s and 6 C reference
# datasets get their own distinct colours (below), outside this scale.
XPCS_CMAP = plt.cm.plasma
CMAP_LO, CMAP_HI = 0.10, 0.88        # colormap span used for the N_LAST times
COLOR_0S = 'black'                   # first B0147 file (elapsed = 0 s)
COLOR_6C = '#1f77b4'                 # B0146 (6 C reference, before isothermal)

# --- FIT MODEL (contrast + baseline fixed; p1, p2 shared across q per time) ---
contrast = 0.135
BASELINE = 1.0

# per-q parameter bounds / start (tau_fast, f, tau_slow) and shared (p1, p2)
PQ_P0    = [1e-3, 0.5, 100.0]
PQ_LO    = [1e-6, 0.0, 1.0]
PQ_HI    = [10.0, 1.0, 10000.0]
P_EXP_P0 = [0.5, 0.5]
P_EXP_LO = [0.2, 0.2]
P_EXP_HI = [3.0, 3.0]


def double_exp(tau, tau_fast, f, tau_slow, p1, p2):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + BASELINE

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


def add_minor_grid(ax):
    """Light grid lines that line up with both major and minor ticks."""
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(which='major', ls='-', lw=0.5, color='0.80')
    ax.grid(which='minor', ls='-', lw=0.3, color='0.90')


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


def fit_g2_global(tau, g2, g2_err, q_indices):
    """Global fit of several q bins for one elapsed time, sharing p1 and p2.

    Parameter vector = [p1, p2, (tau_fast, f, tau_slow) x nq].  Minimises the
    error-weighted residual (model - g2) / g2_err over all q simultaneously,
    using the g2_err stored in the file directly (absolute_sigma convention).
    Parameter 1-sigma errors are sqrt(diag(inv(J^T J))).

    Returns a dict:
      {'p1','p1_err','p2','p2_err','red_chi2',
       'per_q': {q_idx: {'tau_fast','f','tau_slow','f_err'}}}
    or None if too few q bins have usable data.
    """
    data = []
    for qi in q_indices:
        v = (tau > 0) & ~np.isnan(g2[:, qi]) & ~np.isnan(g2_err[:, qi]) & (g2_err[:, qi] > 0)
        if v.sum() >= 5:
            data.append((qi, tau[v], g2[v, qi], g2_err[v, qi]))
    nq = len(data)
    if nq == 0:
        return None

    def residual(p):
        p1, p2 = p[0], p[1]
        parts = []
        for i, (qi, tv, gv, ev) in enumerate(data):
            tf, f, ts = p[2 + 3 * i: 5 + 3 * i]
            parts.append((double_exp(tv, tf, f, ts, p1, p2) - gv) / ev)
        return np.concatenate(parts)

    x0 = list(P_EXP_P0) + PQ_P0 * nq
    lo = list(P_EXP_LO) + PQ_LO * nq
    hi = list(P_EXP_HI) + PQ_HI * nq
    res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=40000)

    ndof = max(len(res.fun) - len(res.x), 1)
    red_chi2 = float(np.sum(res.fun**2) / ndof)
    # covariance from the Gauss-Newton Hessian of error-weighted residuals
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
        perr = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        perr = np.full(len(res.x), np.nan)

    out = {'p1': res.x[0], 'p1_err': perr[0],
           'p2': res.x[1], 'p2_err': perr[1],
           'red_chi2': red_chi2, 'per_q': {}}
    for i, (qi, tv, gv, ev) in enumerate(data):
        tf, f, ts = res.x[2 + 3 * i: 5 + 3 * i]
        out['per_q'][qi] = {'tau_fast': tf, 'tau_fast_err': perr[2 + 3 * i],
                            'f': f, 'f_err': perr[3 + 3 * i],
                            'tau_slow': ts, 'tau_slow_err': perr[4 + 3 * i]}
    return out


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


# XPCS elapsed-time colours: map the N_LAST files (sorted by time) to evenly
# spaced colormap positions so consecutive times are maximally distinguishable.
# The same mapping is reused in every XPCS panel (and in g2_grid_SI.py).
_xpcs_sorted = sorted(xpcs_files, key=elapsed)
_xpcs_pos = np.linspace(CMAP_LO, CMAP_HI, len(_xpcs_sorted))
_xpcs_color = {fp: XPCS_CMAP(p) for fp, p in zip(_xpcs_sorted, _xpcs_pos)}


def ecolor(fp):
    """Colour for an XPCS elapsed time; 0 s and the 6 C ref use distinct colours."""
    if fp == first_file:
        return COLOR_0S
    return _xpcs_color[fp]


# elapsed-time value -> colour (xpcs_files only; used where fp isn't in scope)
time_color = {elapsed(fp): ecolor(fp) for fp in xpcs_files}


print('SAXS files:', [f'{parse_name(f)[1]}-{parse_name(f)[2]} ({elapsed(f):.0f} s)' for f in saxs_files])
print('XPCS files:', [f'{parse_name(f)[1]}-{parse_name(f)[2]} ({elapsed(f):.0f} s)' for f in xpcs_files])

# --- GLOBAL FITS (one per elapsed time; p1, p2 shared across q) ---
# Fit each XPCS file once and reuse the result in every panel.
g2_data = {}   # fp -> (tau, g2, g2_err, q_vals)
fits = {}      # fp -> result dict from fit_g2_global
for fp in xpcs_files:
    with h5py.File(fp, 'r') as hf:
        g2_data[fp] = read_g2(hf)
    tau, g2, g2_err, q_vals = g2_data[fp]
    fits[fp] = fit_g2_global(tau, g2, g2_err, fit_q_indices)
    r = fits[fp]
    if r is not None:
        fmt = ' '.join(f'f{qi}={r["per_q"][qi]["f"]:.3f}' for qi in r['per_q'])
        print(f'  fit {elapsed(fp):5.0f} s: p1={r["p1"]:.3f}+/-{r["p1_err"]:.3f} '
              f'p2={r["p2"]:.3f}+/-{r["p2_err"]:.3f} chi2={r["red_chi2"]:.2f}  {fmt}')

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

saxs_handles = []
for fp in by_header.get('B0146', []):                        # 6 C reference (distinct)
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = coef_sam * I - coef_buf * bg_I if bg_I is not None else coef_sam * I
    pos = I > 0
    ax1.plot(q[pos], I[pos], color=COLOR_6C, marker='s', ls='none', ms=5,
             mfc='none', mew=1.2)
    saxs_handles.append(Line2D([], [], color=COLOR_6C, marker='s', ls='none',
                               mfc='none', mew=1.2, ms=6, label='6 $^\\circ$C ref'))

for fp in saxs_files:                                        # first (0 s) + last N B0147
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = coef_sam * I - coef_buf * bg_I if bg_I is not None else coef_sam * I
    pos = I > 0
    color = ecolor(fp)
    ax1.plot(q[pos], I[pos], color=color, marker='o', ls='none', ms=5,
             mfc='none', mew=1.2)
    saxs_handles.append(Line2D([], [], color=color, marker='o', ls='none',
                               mfc='none', mew=1.2, ms=6, label=f'{elapsed(fp):.0f} s'))

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=20)
ax1.set_ylabel(r'$d\Sigma/d\Omega$ (cm$^{-1}$)', fontsize=20)
ax1.set_title(f'SAXS ({BACKGROUND_HEADER} subtracted)', fontsize=16)
ax1.legend(handles=saxs_handles, loc='upper right', fontsize=13, frameon=False,
           handletextpad=0.4, labelspacing=0.3)
add_minor_grid(ax1)

# ============================================================
# PANEL 2: g2(tau) at target q, with fits
# ============================================================
q_val_target = None
for fp in xpcs_files:
    tau, g2, g2_err, q_vals = g2_data[fp]
    q_val_target = q_vals[target_q_idx]
    color = ecolor(fp)
    m = tau > 0
    ax2.errorbar(tau[m], g2[m, target_q_idx], yerr=g2_err[m, target_q_idx], fmt='o',
                 color=color, markersize=4, capsize=2, elinewidth=1, alpha=0.85, linestyle='none')
    r = fits[fp]
    if r is not None and target_q_idx in r['per_q'] and r['red_chi2'] < CHI2_MAX:
        pq = r['per_q'][target_q_idx]
        tt = tau[m]
        t_fit = np.logspace(np.log10(tt.min()), np.log10(tt.max()), 200)
        ax2.plot(t_fit, double_exp(t_fit, pq['tau_fast'], pq['f'], pq['tau_slow'],
                                   r['p1'], r['p2']), color=color, lw=2.5)

ax2.set_xscale('log')
ax2.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=20)
ax2.set_ylabel('g$_2$', fontsize=20)
ax2.set_title(f'XPCS g$_2$ ($Q = {q_val_target:.5f}\ \AA^{{-1}}$)', fontsize=16)
ax2.set_ylim(1.0, 1.18)
add_minor_grid(ax2)

# ============================================================
# PANEL 3: fitted fast fraction f vs elapsed time (colour=time, marker=q)
# ============================================================
fit_rows = []
q_val_of = {}                                    # q_idx -> Q value (for legends)
for fp in xpcs_files:
    r = fits[fp]
    if r is None or r['red_chi2'] >= CHI2_MAX:   # drop unreliable global fits
        continue
    _, _, _, q_vals = g2_data[fp]
    for q_idx, pq in r['per_q'].items():
        q_val_of[q_idx] = q_vals[q_idx]
        fit_rows.append({'q_index': q_idx, 'q_val': q_vals[q_idx],
                         'elapsed': elapsed(fp), 'f': pq['f'], 'f_err': pq['f_err']})

for q_idx, marker in zip(fit_q_indices, Q_MARKERS):
    rows = sorted([r for r in fit_rows if r['q_index'] == q_idx], key=lambda r: r['elapsed'])
    if not rows:
        continue
    xs = [r['elapsed'] for r in rows]
    ys = [r['f'] for r in rows]
    es = [r['f_err'] for r in rows]
    ax3.plot(xs, ys, '-', color='0.75', lw=1, zorder=1)          # per-q guide line
    for x, y, e in zip(xs, ys, es):
        ax3.errorbar(x, y, yerr=e, marker=marker, color=time_color[x],
                     markersize=10, capsize=3, mfc='none', mew=1.4, zorder=2)

# legend: marker shape -> q bin (open black markers, top-right)
q_handles = [Line2D([], [], marker=mk, ls='none', mfc='none', mec='k', mew=1.4,
                    markersize=10, label=rf'$Q = {q_val_of[qi]:.5f}\ \AA^{{-1}}$')
             for qi, mk in zip(fit_q_indices, Q_MARKERS) if qi in q_val_of]
ax3.legend(handles=q_handles, loc='upper right', fontsize=14, frameon=False,
           handletextpad=0.4, labelspacing=0.3)

ax3.set_xlabel('Elapsed Time (s)', fontsize=20)
ax3.set_ylabel('Fast Fraction ($f$)', fontsize=20)
ax3.set_title('g$_2$ fit: fast fraction', fontsize=16)
add_minor_grid(ax3)

plt.tight_layout()
plt.savefig('SAXPCS_30C_g2.pdf', dpi=300, bbox_inches='tight')
print('wrote SAXPCS_30C_g2.pdf')

# ============================================================
# FIGURE 2 (3 panels): (a) p1, p2 vs elapsed time;
#                      (b) tau_fast vs Q;  (c) tau_slow vs Q
# All XPCS colours use the same elapsed-time scale as figure 1.
# ============================================================
fig2, (axp, axf, axs) = plt.subplots(1, 3, figsize=(21, 6))

# --- (a) shared stretching exponents vs elapsed time (colour = time) ---
exp_rows = [{'elapsed': elapsed(fp), 'fp': fp,
             'p1': fits[fp]['p1'], 'p1_err': fits[fp]['p1_err'],
             'p2': fits[fp]['p2'], 'p2_err': fits[fp]['p2_err']}
            for fp in xpcs_files
            if fits[fp] is not None and fits[fp]['red_chi2'] < CHI2_MAX]
exp_rows.sort(key=lambda r: r['elapsed'])
xe = [r['elapsed'] for r in exp_rows]
axp.plot(xe, [r['p1'] for r in exp_rows], '-', color='0.75', lw=1, zorder=1)
axp.plot(xe, [r['p2'] for r in exp_rows], '-', color='0.75', lw=1, zorder=1)
for r in exp_rows:                                   # p1 = circle, p2 = square
    axp.errorbar(r['elapsed'], r['p1'], yerr=r['p1_err'], marker='o', color=ecolor(r['fp']),
                 markersize=10, capsize=3, mfc='none', mew=1.6, zorder=2)
    axp.errorbar(r['elapsed'], r['p2'], yerr=r['p2_err'], marker='s', color=ecolor(r['fp']),
                 markersize=10, capsize=3, mfc='none', mew=1.6, zorder=2)
axp.axhline(1.0, color='0.6', ls=':', lw=1)          # simple-exponential reference
p_handles = [Line2D([], [], marker='o', ls='none', mfc='none', mec='k', mew=1.6,
                    markersize=10, label=r'$p_1$ (fast)'),
             Line2D([], [], marker='s', ls='none', mfc='none', mec='k', mew=1.6,
                    markersize=10, label=r'$p_2$ (slow)')]
axp.legend(handles=p_handles, frameon=False, fontsize=17, loc='upper left')
axp.set_xlabel('Elapsed Time (s)', fontsize=20)
axp.set_ylabel('Stretching exponent', fontsize=20)
axp.set_title('Shared KWW exponents', fontsize=16)
add_minor_grid(axp)

# --- (b, c) relaxation times vs Q, one curve per elapsed time ---
for fp in xpcs_files:
    r = fits[fp]
    if r is None or r['red_chi2'] >= CHI2_MAX:
        continue
    _, _, _, q_vals = g2_data[fp]
    qs = sorted(r['per_q'])
    Q      = [q_vals[qi] for qi in qs]
    tf     = [r['per_q'][qi]['tau_fast'] for qi in qs]
    tf_err = [r['per_q'][qi]['tau_fast_err'] for qi in qs]
    ts     = [r['per_q'][qi]['tau_slow'] for qi in qs]
    ts_err = [r['per_q'][qi]['tau_slow_err'] for qi in qs]
    color  = ecolor(fp)
    lbl    = f'{elapsed(fp):.0f} s'
    axf.errorbar(Q, tf, yerr=tf_err, marker='o', ls='-', color=color,
                 markersize=10, capsize=3, mfc='none', mew=1.6, label=lbl)
    axs.errorbar(Q, ts, yerr=ts_err, marker='o', ls='-', color=color,
                 markersize=10, capsize=3, mfc='none', mew=1.6, label=lbl)

# Q ticks: label the mantissa (4..8) and show a single "x10^-3" at the corner,
# instead of repeating x10^-3 on every tick (elapsed time is keyed in figure 1).
_q_major = [4e-3, 5e-3, 6e-3, 7e-3, 8e-3]
_q_minor = [3.5e-3, 4.5e-3, 5.5e-3, 6.5e-3, 7.5e-3, 8.5e-3]
for a, name in ((axf, r'$\tau_{\mathrm{fast}}$'), (axs, r'$\tau_{\mathrm{slow}}$')):
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlim(3.5e-3, 8.8e-3)
    a.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=20)
    a.set_ylabel(f'{name} (s)', fontsize=20)
    add_minor_grid(a)                              # sets minorticks_on + grids
    a.xaxis.set_major_locator(FixedLocator(_q_major))
    a.xaxis.set_minor_locator(FixedLocator(_q_minor))
    a.xaxis.set_major_formatter(FixedFormatter(['4', '5', '6', '7', '8']))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.text(1.0, -0.055, r'$\times 10^{-3}$', transform=a.transAxes,
           ha='right', va='top', fontsize=13)     # single exponent, like the screenshot
axf.set_title(r'Fast relaxation time vs $Q$', fontsize=16)
axs.set_title(r'Slow relaxation time vs $Q$', fontsize=16)

fig2.tight_layout()
fig2.savefig('SAXPCS_30C_FitParams.pdf', dpi=300, bbox_inches='tight')
print('wrote SAXPCS_30C_FitParams.pdf')

plt.show()
