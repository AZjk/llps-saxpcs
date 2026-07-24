"""SAXS + XPCS for diluted H06 at four attenuation conditions, with the legend
labelled by the actual photon flux on the sample (not the nominal attenuation).

The nominal attenuator settings (a7, a9, a11, a14) do NOT give a monotonic flux
because the attenuator setup was wrong.  The true on-sample flux is recovered
from the upstream ion-chamber (IC) reading of each measurement:

    IC readout  --(Pind_calibration.csv)-->  calibrated PIN-diode current
                --(Flux_Cal.py formula)---->  photon flux on sample [ph/s]

The IC readout is stored in each results file at
``entry/instrument/incident_beam/incident_beam_intensity`` and must be scaled by
1e-9 (matches the numbers reported on Slide_Screenshot.png).
"""

import os
import glob

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# --- PATHS (all local, relative to this script) ---
here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(here, 'cluster_results')
calib_csv = os.path.join(here, 'Pind_calibration.csv')

# nominal attenuation label -> results file (diluted H06, c6c5-4)
FILES = {
    'a7':  'E0174_H06-c6c5-4_a0007_f100000_r00001_results.hdf',
    'a9':  'E0173_H06-c6c5-4_a0009_f100000_r00001_results.hdf',
    'a11': 'E0172_H06-c6c5-4_a0011_f100000_r00001_results.hdf',
    'a14': 'E0171_H06-c6c5-4_a0014_f100000_r00001_results.hdf',
}
ORDER = ['a7', 'a9', 'a11', 'a14']       # legend / colour order
COLORS = {'a7': 'C0', 'a9': 'C1', 'a11': 'C2', 'a14': 'C3'}

IC_PATH  = 'entry/instrument/incident_beam/incident_beam_intensity'
IC_SCALE = 1e-9                          # apply to the stored IC readout

# --- FLUX CONVERSION CONSTANTS (from Flux_Cal.py) ---
# calibrated PIN-diode current [A]  ->  photon flux on sample [ph/s].
# NOTE: we deliberately drop the attenuation-ratio factor used in Flux_Cal.py
# (there it recovers the *unattenuated* full-beam flux); here the IC/PIN reading
# already corresponds to the attenuated flux hitting the sample.
SPEC_RES      = 0.1882
PHOTON_ENERGY = 12.4e3                   # eV
EV2J          = 1.6e-19
LOSS_MIRROR   = 0.94 * 0.94             # mirror reflectivity
LOSS_DIAMOND  = 0.735                    # BPM + CVD diamond window at 8-ID-E
LOSS_FACTOR   = LOSS_MIRROR * LOSS_DIAMOND


def pind_to_flux(pind_current):
    """Calibrated PIN-diode current [A] -> corrected photon flux [ph/s]."""
    beam_power = pind_current / SPEC_RES
    raw_flux = beam_power / (PHOTON_ENERGY * EV2J)
    return raw_flux / LOSS_FACTOR


def ic_to_pind_slope(csv_path):
    """Fit calibrated-PIN vs up-IC (through the origin) from the calibration CSV.

    Only rows with a real positive IC reading and a numeric (non-saturated)
    calibrated-PIN value are used.  Both detectors respond linearly to flux and
    read ~0 with the shutter closed, so a single proportionality constant maps
    one to the other.
    """
    df = pd.read_csv(csv_path)
    ic = pd.to_numeric(df['Up IC (120 nA, 1e9 scale)'], errors='coerce')
    pind = pd.to_numeric(df['Calibrated pind'], errors='coerce')
    good = ic.notna() & pind.notna() & (ic > 0) & (pind > 0)
    x, y = ic[good].values, pind[good].values
    slope = np.sum(x * y) / np.sum(x * x)          # least squares through origin
    return slope


def fmt_flux(f):
    """Format a flux as '$m.m\\times10^{e}$ ph/s' for the legend."""
    exp = int(np.floor(np.log10(f)))
    mant = f / 10 ** exp
    return rf'${mant:.1f}\times10^{{{exp}}}$ ph/s'


# --- BUILD IC -> flux calibration ---
slope = ic_to_pind_slope(calib_csv)
print(f'IC -> calibrated-PIN slope (through origin): {slope:.4e}')

# --- READ DATA + COMPUTE FLUX ---
data = {}
print('\ncondition   IC [A]      cal-PIN [A]   flux [ph/s]')
for key in ORDER:
    with h5py.File(os.path.join(data_dir, FILES[key]), 'r') as hf:
        ic_raw = hf[IC_PATH][()]
        ic_raw = float(np.asarray(ic_raw).reshape(-1)[0])
        ic = ic_raw * IC_SCALE
        g2 = hf['xpcs/multitau/normalized_g2'][:]
        g2_err = hf['xpcs/multitau/normalized_g2_err'][:]
        saxs_1d = np.squeeze(hf['xpcs/temporal_mean/scattering_1d'][:])
        t_el = hf['xpcs/multitau/delay_list'][:]
        t0 = hf['entry/instrument/detector_1/frame_time'][()]
        ql_sta = hf['xpcs/qmap/static_v_list_dim0'][:]

    pind = slope * ic
    flux = pind_to_flux(pind)
    data[key] = dict(ic=ic, flux=flux, g2=g2, g2_err=g2_err, saxs_1d=saxs_1d,
                     q_saxs=ql_sta * 10.0, t_g2=t_el * t0)
    print(f'{key:>7}   {ic:.3e}   {pind:.3e}    {flux:.3e}')

# --- FIGURE ---
plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                     'mathtext.fontset': 'stix'})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

for key in ORDER:
    d = data[key]
    color = COLORS[key]
    label = fmt_flux(d['flux'])
    # left: SAXS I(q)
    axs[0].plot(d['q_saxs'], d['saxs_1d'], marker='o', mfc='none', mec=color,
                mew=0.6, ms=3, color=color, ls='none', label=label)
    # right: g2 at the first q bin
    axs[1].errorbar(d['t_g2'], d['g2'][:, 0], d['g2_err'][:, 0], marker='o',
                    mfc='none', mec=color, mew=0.6, ms=4, color=color,
                    ls='none', label=label)

axs[0].set_xlabel(r'$Q$ (nm$^{-1}$)')
axs[0].set_ylabel('Intensity (photon/pixel/frame)')
axs[0].set_title('SAXS')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].minorticks_on()
axs[0].grid(True, which='both', ls='--', alpha=0.5)
axs[0].legend(title='Flux on sample', fontsize=9)

axs[1].set_xlabel('Delay Time (s)')
axs[1].set_ylabel(r'g$_2$')
axs[1].set_title('XPCS')
axs[1].set_xscale('log')
axs[1].minorticks_on()
axs[1].grid(True, which='both', ls='--', alpha=0.5)
axs[1].legend(title='Flux on sample', fontsize=9)

out_pdf = os.path.join(here, 'g2_SAXPCS_Rad_Cali.pdf')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f'\nwrote {out_pdf}')
plt.show()
