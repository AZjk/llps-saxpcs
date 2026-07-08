import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 1. READ GENERATED CSV FILES
data_dir = 'reduced_data'
df_ref10 = pd.read_csv(os.path.join(data_dir, 'Merged_Reference_10C.csv'))
df_ref30 = pd.read_csv(os.path.join(data_dir, 'Merged_Reference_30C.csv'))
df_meas10 = pd.read_csv(os.path.join(data_dir, 'Merged_Measurement_10C.csv'))
df_meas30 = pd.read_csv(os.path.join(data_dir, 'Merged_Measurement_30C.csv'))

# Define a local Gaussian peak with a linear background
def gauss_bg(x, a, x0, sigma, m, c):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + m * x + c

# 2. EXECUTE PLOT
plt.rcParams.update({
    'font.size': 20, 
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix'
})

fig, ax = plt.subplots(figsize=(10, 8))

# Main plot
ax.plot(df_ref10['Q(A^-1)'].values, df_ref10['I(Q)'].values, color='r', fillstyle='none', marker='o', markersize=2, linestyle='none', label='Reference, 10 C')
ax.plot(df_ref30['Q(A^-1)'].values, df_ref30['I(Q)'].values, color='k', fillstyle='none', marker='o', markersize=2, linestyle='none', label='Reference, 30 C')
ax.plot(df_meas10['Q(A^-1)'].values, df_meas10['I(Q)'].values, color='g', fillstyle='none', marker='o', markersize=2, linestyle='none', label='Measurement, 10 C')
ax.plot(df_meas30['Q(A^-1)'].values, df_meas30['I(Q)'].values, color='b', fillstyle='none', marker='o', markersize=2, linestyle='none', label='Measurement, 30 C')

# Formatting main plot
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Q ($\AA^{-1}$)')
ax.set_ylabel('I(Q) (arbs.)')
ax.legend(loc='lower left')

# --- ADD Q-SCALING FOR REFERENCE 30C ---
q_ref30_main = df_ref30['Q(A^-1)'].values
i_ref30_main = df_ref30['I(Q)'].values

# Define the low-Q fitting range for 3D domain scaling
q_min_scale, q_max_scale = 0.012, 0.04
mask_scale = (q_ref30_main > q_min_scale) & (q_ref30_main < q_max_scale)

# Fit the linear relationship in log-log space
log_q = np.log10(q_ref30_main[mask_scale])
log_i = np.log10(i_ref30_main[mask_scale])
slope, intercept = np.polyfit(log_q, log_i, 1)

# Generate points for the trendline directly through the data
q_line = np.linspace(q_min_scale, q_max_scale, 50)
i_line = (10**intercept) * (q_line**slope) 

# Plot the scaling line (thicker dashed line)
ax.plot(q_line, i_line, 'k--', lw=4.0)

# Add the scaling label at the top-left start of the fit line (offset vertically)
q_start = q_min_scale
i_start = (10**intercept) * (q_start**slope) * 1.5
ax.text(q_start, i_start, f'$\sim Q^{{{slope:.1f}}}$', fontsize=22, color='k', ha='left', va='bottom')
# ---------------------------------------

# 3. ADD INSET FIGURE (Top Right)
ax_ins = ax.inset_axes([0.52, 0.52, 0.45, 0.45]) 

q_ref30 = df_ref30['Q(A^-1)'].values
i_ref30 = df_ref30['I(Q)'].values

# Plot inset data
mask = q_ref30 > 0.22
ax_ins.plot(q_ref30[mask], i_ref30[mask], color='k', fillstyle='none', marker='o', markersize=2, linestyle='none')

print("\n--- Gaussian Fit Results for Reference 30C WAXS Peaks ---")

# --- FIT PEAK 1 (~0.74 A^-1) ---
m1 = (q_ref30 > 0.48) & (q_ref30 < 1.00)
q1, i1 = q_ref30[m1], i_ref30[m1]
p1_opt, _ = curve_fit(gauss_bg, q1, i1, p0=[np.ptp(i1), 0.74, 0.15, 0, np.min(i1)])

print(f"\nPeak 1 (Inter-sheet spacing):")
print(f"  Position (Q0): {p1_opt[1]:.4f} A^-1  (d = {2*np.pi/p1_opt[1]:.2f} A)")
print(f"  Width (Sigma): {p1_opt[2]:.4f} A^-1")
print(f"  Width (FWHM):  {2.355 * p1_opt[2]:.4f} A^-1")

ax_ins.plot(q1, gauss_bg(q1, *p1_opt), 'r-', lw=2.5)
ax_ins.axvline(p1_opt[1], color='r', linestyle='--', lw=1.5)
ax_ins.text(p1_opt[1] - 0.02, 0.95, f'{p1_opt[1]:.2f} $\\AA^{{-1}}$\n({2*np.pi/p1_opt[1]:.1f} $\\AA$)', 
            transform=ax_ins.get_xaxis_transform(), color='r', ha='right', va='top', fontsize=16)

# --- FIT PEAK 2 (~1.44 A^-1) ---
m2 = (q_ref30 > 1.18) & (q_ref30 < 1.70)
q2, i2 = q_ref30[m2], i_ref30[m2]
p2_opt, _ = curve_fit(gauss_bg, q2, i2, p0=[np.ptp(i2), 1.44, 0.15, 0, np.min(i2)])

print(f"\nPeak 2 (Inter-strand hydrogen bonding):")
print(f"  Position (Q0): {p2_opt[1]:.4f} A^-1  (d = {2*np.pi/p2_opt[1]:.2f} A)")
print(f"  Width (Sigma): {p2_opt[2]:.4f} A^-1")
print(f"  Width (FWHM):  {2.355 * p2_opt[2]:.4f} A^-1\n")

ax_ins.plot(q2, gauss_bg(q2, *p2_opt), 'r-', lw=2.5)
ax_ins.axvline(p2_opt[1], color='r', linestyle='--', lw=1.5)
ax_ins.text(p2_opt[1] - 0.02, 0.95, f'{p2_opt[1]:.2f} $\\AA^{{-1}}$\n({2*np.pi/p2_opt[1]:.1f} $\\AA$)', 
            transform=ax_ins.get_xaxis_transform(), color='r', ha='right', va='top', fontsize=16)

# Format inset
ax_ins.set_xlim(0.2, max(q_ref30[mask]))
ax_ins.set_xlabel(r'Q ($\AA^{-1}$)', fontsize=20)
ax_ins.set_ylabel('I(Q)', fontsize=20)
ax_ins.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.savefig('SAWAXS_2021-1_Fitted.pdf', dpi=300, bbox_inches='tight')
plt.show()