"""Combined SAXS + XPCS figure from the CSVs written by ``read_saxpcs.py``.

Three panels:
  1. SAXS I(q) for the B0146 file and every B0147 file, with the D0138 buffer
     subtracted as background.
  2. XPCS g2(tau) for the last 5 B0147 files (frames 801-1313) at a chosen q,
     with double-exponential fits overlaid.
  3. The fitted fast fraction f versus elapsed time for those 5 files, per q bin.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

# --- FIT MODEL (same as read_xpcs.py) ---
p1, p2, contrast = 0.5, 0.5, 0.135


def double_exp_eq9(tau, tau_fast, f, tau_slow, d):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + d


p0 = [1e-3, 0.5, 100.0, 1.0]
bounds = ([1e-6, 0.0, 1.0, 0.98], [10.0, 1.0, 10000.0, 1.02])

# --- SELECTION SETTINGS ---
BACKGROUND = 'D0138'       # header of the file subtracted as background
SAXS_HEADERS = ('B0146', 'B0147')   # headers shown in the SAXS panel
XPCS_HEADER = 'B0147'      # header used for the g2 panels
N_LAST = 5                 # number of last (highest-frame) files for g2/fit
target_q_idx = 0           # q bin shown in the g2 panel
fit_q_indices = [0, 1, 2, 3, 4]   # q bins fitted for the f-vs-time panel

_name_re = re.compile(r'Average_([A-Za-z]\d+)_.*?_(\d+)_(\d+)_results')


def parse_name(fname):
    """Return (header, start_frame, end_frame) parsed from a result file name."""
    m = _name_re.search(fname)
    if not m:
        return None, -1, -1
    return m.group(1), int(m.group(2)), int(m.group(3))


# --- LOAD DATA ---
df_g2 = pd.read_csv('g2_data.csv')
df_saxs = pd.read_csv('saxs_1d.csv')
# file_name / elapsed_s are written once per block -> forward-fill them.
for df in (df_g2, df_saxs):
    df['file_name'] = df['file_name'].ffill()
    df['elapsed_s'] = df['elapsed_s'].ffill()

for df in (df_g2, df_saxs):
    parsed = df['file_name'].apply(parse_name)
    df['header'] = parsed.apply(lambda t: t[0])
    df['start_frame'] = parsed.apply(lambda t: t[1])

# The last N B0147 files (highest start frame) drive the g2 / fit panels.
xpcs_files = (df_g2[df_g2['header'] == XPCS_HEADER]
              .sort_values('start_frame')['file_name'].unique())
sel_files = list(xpcs_files[-N_LAST:])
sel_elapsed = {f: df_g2[df_g2['file_name'] == f]['elapsed_s'].iloc[0] for f in sel_files}
t0 = min(sel_elapsed.values())     # elapsed origin for the gelation window
print('g2/fit files:', [f'{parse_name(f)[1]}-{parse_name(f)[2]}' for f in sel_files])

# --- FIGURE ---
plt.rcParams.update({'font.size': 16, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

# ============================================================
# PANEL 1: SAXS I(q), background-subtracted
# ============================================================
bg = df_saxs[df_saxs['header'] == BACKGROUND]
bg_by_q = bg.set_index('q_index')['intensity'] if not bg.empty else None

saxs_files = (df_saxs[df_saxs['header'].isin(SAXS_HEADERS)]
              .sort_values(['header', 'start_frame'])['file_name'].unique())

# Colour the B0147 series by elapsed time; B0146 gets a fixed colour.
b0147_elapsed = {f: df_saxs[df_saxs['file_name'] == f]['elapsed_s'].iloc[0]
                 for f in saxs_files if parse_name(f)[0] == 'B0147'}
norm = Normalize(vmin=min(b0147_elapsed.values()), vmax=max(b0147_elapsed.values()))
cmap = plt.cm.turbo

for f in saxs_files:
    d = df_saxs[df_saxs['file_name'] == f].sort_values('q_index')
    inten = d['intensity'].values.astype(float)
    if bg_by_q is not None:
        inten = inten - bg_by_q.reindex(d['q_index']).values
    q = d['q'].values
    pos = inten > 0     # log axis: keep positive points only
    header = parse_name(f)[0]
    if header == 'B0146':
        ax1.plot(q[pos], inten[pos], color='k', lw=2, ls='--', label='B0146 (60 °C)')
    else:
        ax1.plot(q[pos], inten[pos], color=cmap(norm(b0147_elapsed[f])), lw=1.5, alpha=0.9)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=20)
ax1.set_ylabel(r'$I(Q) - I_{\mathrm{buffer}}(Q)$ (a.u.)', fontsize=20)
ax1.set_title(f'SAXS ({BACKGROUND} subtracted)', fontsize=16)
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax1, pad=0.02)
cbar.set_label('B0147 elapsed (s)', fontsize=13)
ax1.legend(loc='lower left', fontsize=13)

# ============================================================
# PANEL 2: g2(tau) for the last 5 B0147 files, with fits
# ============================================================
g2_norm = Normalize(vmin=0, vmax=max(sel_elapsed[f] - t0 for f in sel_files))
fit_results = []     # (file, q_index, elapsed_rel, popt, f_err)

for f in sel_files:
    d = df_g2[(df_g2['file_name'] == f) & (df_g2['q_index'] == target_q_idx)]
    d = d[d['tau'] > 0].sort_values('tau')
    rel = sel_elapsed[f] - t0
    color = plt.cm.turbo(g2_norm(rel))
    ax2.errorbar(d['tau'], d['g2'], yerr=d['g2_err'], fmt='o', color=color,
                 label=f'{rel:.0f} s', markersize=5, capsize=2, alpha=0.8, linestyle='none')
    tau = d['tau'].values
    g2v = d['g2'].values
    err = d['g2_err'].values
    valid = (~np.isnan(g2v)) & (~np.isnan(err)) & (err > 0)
    try:
        popt, _ = curve_fit(double_exp_eq9, tau[valid], g2v[valid], sigma=err[valid],
                            absolute_sigma=True, p0=p0, bounds=bounds, maxfev=10000)
        t_fit = np.logspace(np.log10(tau[valid].min()), np.log10(tau[valid].max()), 200)
        ax2.plot(t_fit, double_exp_eq9(t_fit, *popt), color=color, lw=2.5)
    except RuntimeError:
        pass

q_val_target = df_g2[df_g2['q_index'] == target_q_idx]['q_val'].iloc[0]
ax2.set_xscale('log')
ax2.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=20)
ax2.set_ylabel('g$_2$', fontsize=20)
ax2.set_title(f'XPCS g$_2$ ($Q = {q_val_target:.5f}\ \AA^{{-1}}$)', fontsize=16)
ax2.legend(loc='lower left', fontsize=12, title='Elapsed')
ax2.set_ylim(1.0, 1.18)

# ============================================================
# PANEL 3: fitted fast fraction f vs elapsed time, per q bin
# ============================================================
for f in sel_files:
    rel = sel_elapsed[f] - t0
    for q_idx in fit_q_indices:
        d = df_g2[(df_g2['file_name'] == f) & (df_g2['q_index'] == q_idx)]
        d = d[d['tau'] > 0].sort_values('tau')
        tau = d['tau'].values
        g2v = d['g2'].values
        err = d['g2_err'].values
        valid = (~np.isnan(g2v)) & (~np.isnan(err)) & (err > 0)
        if valid.sum() < 5:
            continue
        try:
            popt, pcov = curve_fit(double_exp_eq9, tau[valid], g2v[valid], sigma=err[valid],
                                   absolute_sigma=True, p0=p0, bounds=bounds, maxfev=10000)
            fit_results.append({'q_index': q_idx, 'q_val': d['q_val'].iloc[0],
                                'elapsed': rel, 'f': popt[1], 'f_err': np.sqrt(pcov[1, 1])})
        except RuntimeError:
            pass

df_fit = pd.DataFrame(fit_results)
q_colors = plt.cm.viridis(np.linspace(0, 0.9, len(fit_q_indices)))
for i, q_idx in enumerate(fit_q_indices):
    q_data = df_fit[df_fit['q_index'] == q_idx].sort_values('elapsed')
    if q_data.empty:
        continue
    q_val = q_data['q_val'].iloc[0]
    ax3.errorbar(q_data['elapsed'], q_data['f'], yerr=q_data['f_err'], fmt='-s',
                 color=q_colors[i], label=f'$Q = {q_val:.4f}\ \AA^{{-1}}$',
                 markersize=8, lw=2, capsize=4)

ax3.set_xlabel('Elapsed Time (s)', fontsize=20)
ax3.set_ylabel('Fast Fraction ($f$)', fontsize=20)
ax3.set_title('g$_2$ fit: fast fraction', fontsize=16)
ax3.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('SAXPCS_Combined.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('wrote SAXPCS_Combined.pdf')
