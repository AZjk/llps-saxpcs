import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- SETTINGS ---
colors = {0: 'tab:blue', 10: 'tab:orange', 20: 'tab:green', 30: 'tab:red'}
labels = {0: 'Wait: 0 min', 10: 'Wait: 10 min', 20: 'Wait: 20 min', 30: 'Wait: 30 min'}

p1, p2, contrast = 0.5, 0.5, 0.135  

def double_exp_eq9(tau, tau_fast, f, tau_slow, d):
    decay_fast = f * np.exp(-(tau / tau_fast)**p1)
    decay_slow = (1 - f) * np.exp(-(tau / tau_slow)**p2)
    return contrast * (decay_fast + decay_slow)**2 + d

# --- LOAD DATA ---
df_g2 = pd.read_csv('g2_data.csv')
df_fit = pd.read_csv('fit_parameters.csv')

plt.rcParams.update({'font.size': 16, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- LEFT PANEL: g2 vs tau ---
# CHANGE THE ROI INDEX HERE: 0 is the first ROI, 1 is the second, etc.
target_q_idx = 0 

df_g2_target = df_g2[df_g2['q_index'] == target_q_idx]
target_q_val = df_g2_target['q_val'].iloc[0] if not df_g2_target.empty else 0.0038

for wt in sorted(df_g2_target['wait_time'].unique()):
    data = df_g2_target[df_g2_target['wait_time'] == wt]
    ax1.errorbar(data['tau'], data['g2'], yerr=data['g2_err'], fmt='o', 
                 color=colors[wt], label=labels[wt], markersize=5, capsize=2, alpha=0.8, linestyle='none')
    
    # Regenerate the smooth fit line from saved parameters
    fit_data = df_fit[(df_fit['wait_time'] == wt) & (df_fit['q_index'] == target_q_idx)]
    if not fit_data.empty:
        popt = fit_data.iloc[0][['tau_fast', 'f', 'tau_slow', 'd']].values
        t_fit = np.logspace(np.log10(data['tau'].min()), np.log10(data['tau'].max()), 200)
        ax1.plot(t_fit, double_exp_eq9(t_fit, *popt), color=colors[wt], lw=3.0)

ax1.set_xscale('log')
ax1.set_xlabel(r'Delay Time, $\tau$ (s)', fontsize=20)
ax1.set_ylabel('g$_2$', fontsize=20) 
ax1.set_title(f'Isothermal Gelation at 30 °C ($Q = {target_q_val:.5f} \ \AA^{{-1}}$)', fontsize=16)
ax1.legend(loc='lower left', fontsize=14)
ax1.set_ylim(1.0, 1.18)

# --- RIGHT PANEL: f vs Wait Time ---
q_indices = sorted(df_fit['q_index'].unique())
q_colors = plt.cm.viridis(np.linspace(0, 0.9, len(q_indices)))

for i, q_idx in enumerate(q_indices):
    q_data = df_fit[df_fit['q_index'] == q_idx].sort_values('wait_time')
    q_val = q_data['q_val'].iloc[0]
    ax2.errorbar(q_data['wait_time'], q_data['f'], yerr=q_data['f_err'], fmt='-s', 
                 color=q_colors[i], label=f'$Q = {q_val:.4f} \ \AA^{{-1}}$', markersize=8, lw=2, capsize=4)

ax2.set_xlabel('Wait Time (min)', fontsize=20)
ax2.set_ylabel('Fast Fraction ($f$)', fontsize=20)
ax2.set_xticks([0, 10, 20])
ax2.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig('XPCS_Arrested_Dynamics_Split.pdf', dpi=300, bbox_inches='tight')
plt.show()