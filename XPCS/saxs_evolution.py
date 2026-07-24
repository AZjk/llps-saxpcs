"""SI figure: full SAXS 1d evolution for all B0147 files (D0138 subtracted).

Every B0147 averaged file is plotted, coloured by elapsed time (time origin =
first B0147 file, frames 1-200).  B0146 (6 C reference, before the 30 C
isothermal) is shown as a grey dashed curve.  The legend is placed outside, on
the right-hand side of the axes.
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- PARAMETERS ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BACKGROUND_HEADER = 'D0138'
# Absolute scattering cross-section coefficients: convert the raw SAXS 1d to an
# absolute differential cross section (cm^-1) as
#     I_abs(q) = coef_sam * I_sample(q) - coef_buf * I_buffer(q)
coef_sam = 6.93e4                     # sample coefficient
coef_buf = 7.62e4                     # buffer coefficient
PHI_AVERAGE = True
CMAP = plt.cm.turbo

SAXS_PATH       = '/xpcs/temporal_mean/scattering_1d'
STATIC_MAP_PATH = '/xpcs/qmap/static_index_mapping'
STATIC_Q_PATH   = '/xpcs/qmap/static_v_list_dim0'
STATIC_PHI_PATH = '/xpcs/qmap/static_v_list_dim1'
START_TIME_PATH = '/entry/start_time'
TIME_FORMAT     = '%Y-%m-%d %H:%M:%S'

_name_re = re.compile(r'Average_([A-Za-z]\d+)_.*?_(\d+)_(\d+)_results')


def parse_name(fname):
    m = _name_re.search(os.path.basename(fname))
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, -1, -1)


def read_start_time(hf):
    raw = hf[START_TIME_PATH][()]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return datetime.strptime(str(raw).strip(), TIME_FORMAT)


def read_saxs_iq(hf, phi_average=True):
    intensity = np.asarray(hf[SAXS_PATH][()]).reshape(-1)
    idx_map = hf[STATIC_MAP_PATH][()]
    q_list = hf[STATIC_Q_PATH][()]
    n_phi = hf[STATIC_PHI_PATH].shape[0]
    q_idx = idx_map // n_phi
    uq = np.unique(q_idx)
    if phi_average and n_phi > 1:
        inten = np.array([np.nanmean(intensity[q_idx == qi]) for qi in uq])
    else:
        inten = np.array([intensity[q_idx == qi][0] for qi in uq])
    return q_list[uq], inten


# --- DISCOVER FILES ---
file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf')))
by_header, start_times = {}, {}
for fp in file_paths:
    header = parse_name(fp)[0]
    with h5py.File(fp, 'r') as hf:
        start_times[fp] = read_start_time(hf)
    by_header.setdefault(header, []).append(fp)
for h in by_header:
    by_header[h].sort(key=lambda p: parse_name(p)[1])

b0147 = by_header['B0147']
t_ref = start_times[b0147[0]]
elapsed = {fp: (start_times[fp] - t_ref).total_seconds() for fp in b0147}
norm = Normalize(vmin=0, vmax=max(elapsed.values()))

bg_I = None
if BACKGROUND_HEADER in by_header:
    with h5py.File(by_header[BACKGROUND_HEADER][0], 'r') as hf:
        _, bg_I = read_saxs_iq(hf, PHI_AVERAGE)

# --- FIGURE ---
plt.rcParams.update({'font.size': 14, 'font.family': 'serif', 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(9, 6))

for fp in by_header.get('B0146', []):
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = coef_sam * I - coef_buf * bg_I if bg_I is not None else coef_sam * I
    pos = I > 0
    ax.plot(q[pos], I[pos], color='0.5', marker='s', ls='none', ms=5,
            mfc='none', mew=1.2, label=r'6$^{\circ}$C Ref')

for fp in b0147:
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = coef_sam * I - coef_buf * bg_I if bg_I is not None else coef_sam * I
    pos = I > 0
    ax.plot(q[pos], I[pos], color=CMAP(norm(elapsed[fp])), marker='o', ls='none',
            ms=5, mfc='none', mew=1.2, label=f'{elapsed[fp]:.0f} s')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=18)
ax.set_ylabel(r'$d\Sigma/d\Omega$ (cm$^{-1}$)', fontsize=18)
ax.set_title(f'SAXS evolution ({BACKGROUND_HEADER} subtracted)', fontsize=15)
ax.minorticks_on()
ax.set_axisbelow(True)
ax.grid(which='major', ls='-', lw=0.5, color='0.80')
ax.grid(which='minor', ls='-', lw=0.3, color='0.90')
ax.legend(loc='upper right', fontsize=10, title='Elapsed time', frameon=True,
          ncol=2, columnspacing=1.0, handletextpad=0.4, labelspacing=0.3,
          framealpha=0.9)

plt.tight_layout()
plt.savefig('SAXS_Evolution_SI.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('wrote SAXS_Evolution_SI.pdf')
