import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. READ DATA
data_dir = 'reduced_data'
df = pd.read_csv(os.path.join(data_dir, 'Merged_Reference_10C.csv'))

# Q is strictly in A^-1
Q = df['Q(A^-1)'].values
I = df['I(Q)'].values
# Intensity uncertainty (propagated in Read_12ID_SAWAXS.py); enables the same
# weighted Guinier fit the 12-ID beamline software performs.
I_err = df['I_err(Q)'].values if 'I_err(Q)' in df.columns else None

# 2. DEFINE GUINIER REGION FOR UNASSEMBLED ELP (~23 A)
q_min = 0.020
q_max = 0.055

mask_fit = (Q >= q_min) & (Q <= q_max) & (I > 0)
if I_err is not None:
    mask_fit &= (I_err > 0)
Q_fit = Q[mask_fit]
I_fit = I[mask_fit]

# 3. PERFORM WEIGHTED GUINIER FIT: ln(I) = intercept + slope * Q^2
#    slope = -Rg^2/3,  intercept = ln(I0).  Weight each point by 1/sigma_lnI^2
#    with sigma_lnI = sigma_I / I (error propagation of the log).  This matches
#    the beamline; an unweighted fit ignores the measured errors and reports an
#    artificially small uncertainty.
x = Q_fit**2
y = np.log(I_fit)
if I_err is not None:
    sigma_y = I_err[mask_fit] / I_fit
else:
    sigma_y = np.ones_like(y)          # fallback: unweighted (all equal weights)

w = 1.0 / sigma_y**2
X = np.vstack([x, np.ones_like(x)]).T
cov = np.linalg.inv(X.T @ (w[:, None] * X))   # covariance (absolute_sigma convention)
slope, intercept = cov @ (X.T @ (w * y))
slope_err = np.sqrt(cov[0, 0])
intercept_err = np.sqrt(cov[1, 1])

# reduced chi^2 as a fit-quality check
resid = (y - (slope * x + intercept)) / sigma_y
red_chi2 = np.sum(resid**2) / (len(x) - 2)

R_g = np.sqrt(-3 * slope)
R_g_err = (3.0 / (2.0 * R_g)) * slope_err     # error propagation of sqrt(-3 slope)
I_0 = np.exp(intercept)
I_0_err = I_0 * intercept_err                 # error propagation of exp(intercept)

print("--- Guinier Fit Results (weighted) ---")
print(f"I_0: {I_0:.4f} ± {I_0_err:.4f} cm^-1")
print(f"R_g: {R_g:.3f} ± {R_g_err:.3f} Å")
print(f"Max Q*R_g: {q_max * R_g:.4f} (Target < 1.3)")
print(f"reduced chi^2: {red_chi2:.3f}  (N = {len(x)} points)")

# 4. EXECUTE PLOT
plt.rcParams.update({
    'font.size': 20, 
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix'
})

fig, ax = plt.subplots(figsize=(8, 6))

# Plot slightly wider raw data range for context, with propagated error bars
mask_plot = (Q >= 0.005) & (Q <= 0.08) & (I > 0)
xp = Q[mask_plot]**2
yp = np.log(I[mask_plot])
if I_err is not None:
    yperr = I_err[mask_plot] / I[mask_plot]
    ax.errorbar(xp, yp, yerr=yperr, fmt='o', mfc='none', mec='r', ecolor='r',
                markersize=8, capsize=2, elinewidth=1, ls='none', label='Reference 10 C')
else:
    ax.plot(xp, yp, 'ro', fillstyle='none', markersize=8, label='Reference 10 C')

# Plot fit line
fit_line = slope * x + intercept
ax.plot(x, fit_line, 'k-', linewidth=2.5, label='Guinier fit')

# Fit-result annotation (mirrors the 12-ID beamline readout)
txt = (f'$I_0$ = {I_0:.3f} $\\pm$ {I_0_err:.3f}\n'
       f'$R_g$ = {R_g:.2f} $\\pm$ {R_g_err:.2f} $\\AA$\n'
       f'$Q_{{max}}\\cdot R_g$ = {q_max * R_g:.3f}')
ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=15, va='top', ha='right',
        bbox=dict(boxstyle='round', fc='white', ec='0.5'))

# Formatting
ax.minorticks_on()
ax.set_axisbelow(True)
ax.grid(which='major', ls='-', lw=0.5, color='0.80')
ax.grid(which='minor', ls='-', lw=0.3, color='0.90')
ax.set_xlabel(r'$Q^2$ ($\AA^{-2}$)')
ax.set_ylabel(r'$\ln[I(Q)]$ (arbs.)')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
ax.legend(loc='lower left', fontsize=15)

plt.tight_layout()
plt.savefig('Guinier_Fit_10C.pdf', dpi=300, bbox_inches='tight')
plt.show()