import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os

# --- PARAMETERS ---
data_dir = "/home/8-id-i/2022-1/babnigg202203_nexus/average/"
file_names = [
    'Average_B0147_S3_7_300C10p_att00_Rq0_00950-01050_results.hdf',
    'Average_B0147_S3_7_300C10p_att00_Rq0_01051-01150_results.hdf',
    'Average_B0147_S3_7_300C10p_att00_Rq0_01151-01250_results.hdf',
    'Average_B0147_S3_7_300C10p_att00_Rq0_01251-01313_results.hdf'
]
wait_times = [0, 10, 20, 30] 
q_indices = [0, 1, 2, 3, 4]

p1, p2, contrast = 0.5, 0.5, 0.135  

def double_exp_eq9(tau, tau_fast, f, tau_slow, d):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + d

p0 = [1e-3, 0.5, 100.0, 1.0]
bounds = ([1e-6, 0.0, 1.0, 0.98], [10.0, 1.0, 10000.0, 1.02])

# --- PROCESS DATA ---
g2_records = []
fit_records = []

for i, fname in enumerate(file_names):
    file_path = os.path.join(data_dir, fname)
    if not os.path.exists(file_path): file_path = fname  
        
    with h5py.File(file_path, 'r') as hf:
        t0 = hf['/entry/instrument/detector_1/frame_time'][()]
        t0 = t0.item() if isinstance(t0, np.ndarray) else t0
        tau = hf['/xpcs/multitau/delay_list'][()] * t0
        g2 = hf['/xpcs/multitau/normalized_g2'][()]
        g2_err = hf['/xpcs/multitau/normalized_g2_err'][()]
        q_vals = hf['/xpcs/qmap/dynamic_v_list_dim0'][()]
        
    tau = tau[:, 0] if tau.ndim > 1 else tau
    
    # BUG FIX: Extract ALL specified Q bins for the g2 data, not just the target
    for idx in q_indices:
        q_val = q_vals[idx]
        for t, g, err in zip(tau, g2[:, idx], g2_err[:, idx]):
            if t > 0 and not np.isnan(g):
                g2_records.append({
                    'wait_time': wait_times[i], 'q_index': idx, 'q_val': q_val, 
                    'tau': t, 'g2': g, 'g2_err': err
                })

    # Fit all Q bins for the first 3 files
    if i < 3:
        for idx in q_indices:
            valid = (tau > 0) & ~np.isnan(g2[:, idx]) & ~np.isnan(g2_err[:, idx]) & (g2_err[:, idx] > 0)
            try:
                popt, pcov = curve_fit(double_exp_eq9, tau[valid], g2[valid, idx], 
                                       sigma=g2_err[valid, idx], absolute_sigma=True, 
                                       p0=p0, bounds=bounds, maxfev=10000)
                f_err = np.sqrt(pcov[1, 1])
                fit_records.append({
                    'wait_time': wait_times[i], 'q_index': idx, 'q_val': q_vals[idx],
                    'tau_fast': popt[0], 'f': popt[1], 'tau_slow': popt[2], 'd': popt[3], 'f_err': f_err
                })
            except RuntimeError:
                pass

# Save to CSV
pd.DataFrame(g2_records).to_csv('g2_data.csv', index=False)
pd.DataFrame(fit_records).to_csv('fit_parameters.csv', index=False)