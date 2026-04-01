import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- 1. DEFINITIONS & PARAMETERS ---
data_dir = "/home/8-id-i/2022-1/babnigg202203_nexus/average/"

file_names = [
    'Average_B0147_S3_7_300C10p_att00_Rq0_00950-01050_results.hdf',
    'Average_B0147_S3_7_300C10p_att00_Rq0_01051-01150_results.hdf',
    'Average_B0147_S3_7_300C10p_att00_Rq0_01151-01250_results.hdf',
    'Average_B0147_S3_7_300C10p_att00_Rq0_01251-01313_results.hdf'
]

wait_times = [0, 10, 20, 30] 
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['Wait: 0 min', 'Wait: 10 min', 'Wait: 20 min', 'Wait: 30 min']

q_indices = [0, 1, 2, 3, 4]
target_q_idx = 1  

p1 = 0.5  
p2 = 0.5  
contrast = 0.135  

def double_exp_eq9(tau, tau_fast, f, tau_slow, d):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + d

p0 = [1e-3, 0.5, 100.0, 1.0]
bounds = ([1e-6, 0.0, 1.0, 0.98], [10.0, 1.0, 10000.0, 1.05])

# --- 2. LOAD DATA & PERFORM FITS ---
all_tau, all_g2, all_g2_err = [], [], []
f_results = {idx: [] for idx in q_indices}
q_values = {}

for i, fname in enumerate(file_names):
    file_path = os.path.join(data_dir, fname)
    if not os.path.exists(file_path):
        file_path = fname  
        
    with h5py.File(file_path, 'r') as hf:
        t0 = hf['/entry/instrument/detector_1/frame_time'][()]
        if isinstance(t0, np.ndarray):
            t0 = t0.item()
            
        tau = hf['/xpcs/multitau/delay_list'][()] * t0
        g2 = hf['/xpcs/multitau/normalized_g2'][()]
        g2_err = hf['/xpcs/multitau/normalized_g2_err'][()]
        q_vals = hf['/xpcs/qmap/dynamic_v_list_dim0'][()]
        
    if tau.ndim > 1:
        tau = tau[:, 0]
        
    all_tau.append(tau)
    all_g2.append(g2)
    all_g2_err.append(g2_err)
    
    for idx in q_indices:
        q_values[idx] = q_vals[idx]
        
    if i < 3:
        for idx in q_indices:
            valid = (tau > 0) & ~np.isnan(g2[:, idx]) & ~np.isnan(g2_err[:, idx]) & (g2_err[:, idx] > 0)
            x_data = tau[valid]
            y_data = g2[valid, idx]
            y_err = g2_err[valid, idx]
            
            try:
                popt, pcov = curve_fit(double_exp_eq9, x_data, y_data, sigma=y_err, absolute_sigma=True, 
                                       p0=p0, bounds=bounds, maxfev=10000)
                f_val = popt[1]
                f_err = np.sqrt(pcov[1, 1])
                f_results[idx].append((f_val, f_err))
            except RuntimeError:
                f_results[idx].append((np.nan, np.nan))
    else:
        for idx in q_indices:
            f_results[idx].append((np.nan, np.nan))

# --- 3. PLOT MAIN FIGURE ---
plt.rcParams.update({'font.size': 18, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(4):
    t = all_tau[i]
    g = all_g2[i][:, target_q_idx]
    err = all_g2_err[i][:, target_q_idx]
    
    valid = (t > 0) & ~np.isnan(g)
    t_valid = t[valid]
    g_valid = g[valid]
    err_valid = err[valid]
    
    ax.errorbar(t_valid, g_valid, yerr=err_valid, fmt='o', color=colors[i], 
                label=labels[i], markersize=5, capsize=2, alpha=0.8, linestyle='none')
    
    if i < 3:
        try:
            popt, _ = curve_fit(double_exp_eq9, t_valid, g_valid, sigma=err_valid, p0=p0, bounds=bounds, maxfev=10000)
            t_fit = np.logspace(np.log10(t_valid.min()), np.log10(t_valid.max()), 200)
            ax.plot(t_fit, double_exp_eq9(t_fit, *popt), color=colors[i], lw=3.0)
        except RuntimeError:
            pass

ax.set_xscale('log')
ax.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=22)
ax.set_ylabel('g$_2$', fontsize=22)
ax.set_title(f'Isothermal Gelation at 30 °C ($Q = {q_values[target_q_idx]:.5f} \ \AA^{{-1}}$)', fontsize=20)
ax.legend(loc='lower left', fontsize=16)
ax.set_ylim(1.0, 1.25)  

# --- 4. PLOT INSET (f vs. Q) ---
ax_ins = ax.inset_axes([0.55, 0.55, 0.40, 0.40]) 
q_plot_vals = [q_values[idx] for idx in q_indices]

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
plt.savefig('XPCS_Arrested_Dynamics_Eq9.pdf', dpi=300, bbox_inches='tight')
plt.show()