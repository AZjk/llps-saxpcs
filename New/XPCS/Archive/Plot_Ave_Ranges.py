import sys
import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- PARAMETERS ---
group = 'B0147'
prefix = '/home/8-id-i/2022-1/babnigg202203_nexus/reprocess_results'

file_ranges = [
    (950,  1050),
    (1051, 1150),
    (1151, 1250),
    (1251, 1313),
]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['Wait: 0 min', 'Wait: 10 min', 'Wait: 20 min', 'Wait: 30 min']

q_indices    = [0, 1, 2, 3, 4]
target_q_idx = 1

p1 = 0.5
p2 = 0.5
contrast = 0.135

def double_exp_eq9(tau, tau_fast, f, tau_slow, d):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + d

p0     = [1e-3, 0.5, 100.0, 1.0]
bounds = ([1e-6, 0.0, 1.0, 0.98], [10.0, 1.0, 10000.0, 1.05])

# --- LOAD DATA ---
# Read the averaged HDF files produced by Average_Ranges.py (one per range).
def _load_average(start, end):
    """Load g2/g2_err and the delay/q axes from an averaged range file."""
    pattern = os.path.join(prefix, f'{group}*_Average_{start:05d}_{end:05d}_results.hdf')
    matches = sorted(glob.glob(pattern))
    assert matches, f'no averaged file found for range ({start}, {end}); run Average_Ranges.py first'
    with h5py.File(matches[0], 'r') as hf:
        t0 = float(np.asarray(hf['/entry/instrument/detector_1/frame_time'][()]).flat[0])
        delay_list = hf['/xpcs/multitau/delay_list'][()]
        tau = (delay_list[:, 0] if delay_list.ndim > 1 else delay_list) * t0
        g2 = hf['/xpcs/multitau/normalized_g2'][()]
        g2_err = hf['/xpcs/multitau/normalized_g2_err'][()]
        q_vals = hf['/xpcs/qmap/dynamic_v_list_dim0'][()]
    return {'g2': g2, 'g2_err': g2_err}, tau, q_vals

avg_all = []
t_el = ql_dyn = None
for start, end in file_ranges:
    avg_dict, tau, q_vals = _load_average(start, end)
    if t_el is None:
        t_el, ql_dyn = tau, q_vals
    avg_all.append(avg_dict)

# --- FIT ---
f_results = {idx: [] for idx in q_indices}

for i in range(4):
    g2     = avg_all[i]['g2']
    g2_err = avg_all[i]['g2_err']

    if i < 3:
        for idx in q_indices:
            valid = (
                (t_el > 0)
                & ~np.isnan(g2[:, idx])
                & ~np.isnan(g2_err[:, idx])
                & (g2_err[:, idx] > 0)
            )
            try:
                popt, pcov = curve_fit(
                    double_exp_eq9, t_el[valid], g2[valid, idx],
                    sigma=g2_err[valid, idx], absolute_sigma=True,
                    p0=p0, bounds=bounds, maxfev=10000
                )
                f_results[idx].append((popt[1], np.sqrt(pcov[1, 1])))
            except RuntimeError:
                f_results[idx].append((np.nan, np.nan))
    else:
        for idx in q_indices:
            f_results[idx].append((np.nan, np.nan))

# --- MAIN PLOT ---
plt.rcParams.update({'font.size': 18, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(4):
    g   = avg_all[i]['g2'][:, target_q_idx]
    err = avg_all[i]['g2_err'][:, target_q_idx]

    valid = (t_el > 0) & ~np.isnan(g)
    t_v   = t_el[valid]
    g_v   = g[valid]
    err_v = err[valid]

    ax.errorbar(t_v, g_v, yerr=err_v, fmt='o', color=colors[i],
                label=labels[i], markersize=5, capsize=2, alpha=0.8, linestyle='none')

    if i < 3:
        try:
            popt, _ = curve_fit(double_exp_eq9, t_v, g_v, sigma=err_v,
                                p0=p0, bounds=bounds, maxfev=10000)
            t_fit = np.logspace(np.log10(t_v.min()), np.log10(t_v.max()), 200)
            ax.plot(t_fit, double_exp_eq9(t_fit, *popt), color=colors[i], lw=3.0)
        except RuntimeError:
            pass

ax.set_xscale('log')
ax.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=22)
ax.set_ylabel('g$_2$', fontsize=22)
q_target = ql_dyn[target_q_idx]
ax.set_title(
    rf'Isothermal Gelation at 30 °C ($Q = {q_target:.5f}\ \AA^{{-1}}$)',
    fontsize=20
)
ax.legend(loc='lower left', fontsize=16)
ax.set_ylim(1.0, 1.25)

# --- INSET (fast fraction f vs Q) ---
ax_ins = ax.inset_axes([0.55, 0.55, 0.40, 0.40])
q_plot_vals = [ql_dyn[idx] for idx in q_indices]

for i in range(3):
    f_vals = [f_results[idx][i][0] for idx in q_indices]
    f_errs = [f_results[idx][i][1] for idx in q_indices]
    ax_ins.errorbar(q_plot_vals, f_vals, yerr=f_errs, fmt='-s', markersize=6, lw=2,
                    color=colors[i], label=labels[i], capsize=4)

ax_ins.set_xlabel(r'$Q \ (\AA^{-1})$', fontsize=14)
ax_ins.set_ylabel('Fast Fraction ($f$)', fontsize=14)
ax_ins.tick_params(axis='both', which='major', labelsize=12)
ax_ins.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('XPCS_Arrested_Dynamics_Eq9_Ranges.pdf', dpi=300, bbox_inches='tight')
plt.show()
