import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# 1. READ DATA
data_dir = 'reduced_data'
df = pd.read_csv(os.path.join(data_dir, 'Merged_Reference_10C.csv'))

# Q is strictly in A^-1
Q = df[df.columns[0]].values
I = df[df.columns[1]].values

# 2. DEFINE GUINIER REGION FOR UNASSEMBLED ELP (~23 A)
q_min = 0.020 
q_max = 0.055 

mask_fit = (Q >= q_min) & (Q <= q_max)
Q_fit = Q[mask_fit]
I_fit = I[mask_fit]

# 3. PERFORM GUINIER FIT
x = Q_fit**2
y = np.log(I_fit)

slope, intercept, r_value, p_value, std_err = linregress(x, y)

R_g = np.sqrt(-3 * slope)
R_g_err = 0.866 * (np.abs(slope)**-0.5) * std_err

print("--- Guinier Fit Results ---")
print(f"R_g: {R_g:.2f} ± {R_g_err:.2f} Å")
print(f"Max Q*R_g: {q_max * R_g:.2f} (Target < 1.3)")
print(f"R-squared: {r_value**2:.3f}")

# 4. EXECUTE PLOT
plt.rcParams.update({
    'font.size': 20, 
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix'
})

fig, ax = plt.subplots(figsize=(8, 6))

# Plot slightly wider raw data range for context
mask_plot = (Q >= 0.005) & (Q <= 0.08)
ax.plot(Q[mask_plot]**2, np.log(I[mask_plot]), 'ro', fillstyle='none', markersize=8, label='Reference 10 C')

# Plot fit line
fit_line = slope * x + intercept
ax.plot(x, fit_line, 'k-', linewidth=2.5, label=f'Fit ($R_g$ = {R_g:.1f} $\\AA$)')

# Formatting
ax.set_xlabel(r'$Q^2$ ($\AA^{-2}$)')
ax.set_ylabel(r'$\ln[I(Q)]$ (arbs.)')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Guinier_Fit_10C.pdf', dpi=300, bbox_inches='tight')
plt.show()