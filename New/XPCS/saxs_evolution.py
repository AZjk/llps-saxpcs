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
data_dir = "/home/8-id-i/2022-1/babnigg202203_nexus/average/"
BACKGROUND_HEADER = 'D0138'
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
    I = I - bg_I if bg_I is not None else I
    pos = I > 0
    ax.plot(q[pos], I[pos], color='0.5', lw=2, ls='--', label='B0146 (6 °C ref)')

for fp in b0147:
    with h5py.File(fp, 'r') as hf:
        q, I = read_saxs_iq(hf, PHI_AVERAGE)
    I = I - bg_I if bg_I is not None else I
    pos = I > 0
    ax.plot(q[pos], I[pos], color=CMAP(norm(elapsed[fp])), lw=1.8,
            label=f'{elapsed[fp]:.0f} s')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$Q$ ($\AA^{-1}$)', fontsize=18)
ax.set_ylabel(r'$I(Q) - I_{\mathrm{buffer}}(Q)$ (a.u.)', fontsize=18)
ax.set_title(f'SAXS evolution ({BACKGROUND_HEADER} subtracted)', fontsize=15)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=11,
          title='Elapsed time', frameon=False)

plt.tight_layout()
plt.savefig('SAXS_Evolution_SI.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('wrote SAXS_Evolution_SI.pdf')
