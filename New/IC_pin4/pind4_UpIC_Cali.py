
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.optimize import curve_fit

UpIC_dark = 83.6
DnIC_dark = 204.4
pind4_dark = 13141.9
ph_counts_dark = 3.04e8

UpIC = np.array([257609, 125204, 55586.5, 27036.8, 12296.8, 6028.8, 2714.4, 1361.1])
# pind4 = np.array([645687, 351204, 162281, 85326.9, 45776.8, 29016.8, 20163.9, 16553.5])-pind4_dark
DnIC = np.array([223888, 108873, 48413.4, 23613.7, 10807.5, 5367.4, 2485.5, 1310.9])
ph_counts = np.array([1.49e10, 8.12e9, 3.75e9, 1.97e9, 1.06e9, 6.71e8, 4.66e8, 3.83e8])-ph_counts_dark

def func(x, a, b):
    return a*x + b


crop = 1

popt, pcov = curve_fit(func, UpIC[crop:], ph_counts[crop:])


fs = 20
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(UpIC[crop:], ph_counts[crop:], 'ko')
ax.plot(UpIC[crop:], func(UpIC[crop:], *popt), 'b-')
ax.set_xlabel('Upstream Ion Chamber', fontsize=fs)
ax.set_ylabel('Number of Photons', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig('UpIC_pind4.pdf', dpi=100, format='pdf', facecolor='w', edgecolor='w', transparent=True)

## Air transmission is 0.868
fs = 20
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(np.arange(7), (DnIC[crop:]-DnIC_dark)/(UpIC[crop:]-UpIC_dark), 'ko:')
ax.set_xlabel('Number of measurements', fontsize=fs)
ax.set_ylabel('Transmission of Air', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig('UpIC_DnIC.pdf', dpi=100, format='pdf', facecolor='w', edgecolor='w', transparent=True)

print(f'photon counts={popt[0]:.2e}*Up_IC{popt[1]:.2e}')
print(f'Air absorption is read from the pdf figure to be 0.868')

