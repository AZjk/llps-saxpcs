"""Read every HDF result file in a target directory and dump two CSVs.

For all ``*.hdf`` files found in ``data_dir`` this script extracts:
  * the multitau g2 curves           -> ``g2_data.csv``
  * the temporal-mean SAXS 1d curve  -> ``saxs_1d.csv``

Both CSVs carry a ``file_name`` column identifying the source file and an
``elapsed_s`` column giving each file's acquisition time (from
``/entry/start_time``) in seconds relative to the earliest file read.  Both are
constant within a file, so they are written only on the first row of each file's
block (blank thereafter); readers should forward-fill them.

g2_data.csv columns:   file_name, elapsed_s, q_index, q_val, tau, g2, g2_err
saxs_1d.csv columns:
  * with ``PHI_AVERAGE = False`` (per q/phi sector):
        file_name, elapsed_s, q_index, q, phi, intensity
  * with ``PHI_AVERAGE = True`` (azimuthally averaged I(q)):
        file_name, elapsed_s, q_index, q, intensity, n_phi
    where ``n_phi`` is how many phi sectors were averaged for that q bin.
"""

import glob
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

# --- PARAMETERS ---
data_dir = "/home/8-id-i/2022-1/babnigg202203_nexus/average/"

# If True and the data is partitioned in phi, average I(q, phi) over the phi
# sectors to produce a single azimuthally-averaged I(q) per q bin.  If there is
# only one phi sector this has no effect.
PHI_AVERAGE = True

# --- HDF field locations ---
START_TIME_PATH = '/entry/start_time'
TIME_FORMAT     = '%Y-%m-%d %H:%M:%S'
FRAME_TIME_PATH = '/entry/instrument/detector_1/frame_time'
DELAY_PATH      = '/xpcs/multitau/delay_list'
G2_PATH         = '/xpcs/multitau/normalized_g2'
G2_ERR_PATH     = '/xpcs/multitau/normalized_g2_err'
DYN_Q_PATH      = '/xpcs/qmap/dynamic_v_list_dim0'

SAXS_PATH       = '/xpcs/temporal_mean/scattering_1d'
STATIC_MAP_PATH = '/xpcs/qmap/static_index_mapping'
STATIC_Q_PATH   = '/xpcs/qmap/static_v_list_dim0'
STATIC_PHI_PATH = '/xpcs/qmap/static_v_list_dim1'


def read_start_time(hf):
    """Return the acquisition datetime stored in an open file."""
    raw = hf[START_TIME_PATH][()]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return datetime.strptime(str(raw).strip(), TIME_FORMAT)


def read_g2(hf, fname, elapsed_s):
    """Return a list of g2 records (one per tau/q point) for one open file."""
    t0 = hf[FRAME_TIME_PATH][()]
    t0 = t0.item() if isinstance(t0, np.ndarray) else t0
    tau = hf[DELAY_PATH][()] * t0
    tau = tau[:, 0] if tau.ndim > 1 else tau
    g2 = hf[G2_PATH][()]
    g2_err = hf[G2_ERR_PATH][()]
    q_vals = hf[DYN_Q_PATH][()]

    records = []
    n_q = g2.shape[1]
    for idx in range(n_q):
        q_val = q_vals[idx]
        for t, g, err in zip(tau, g2[:, idx], g2_err[:, idx]):
            if t > 0 and not np.isnan(g):
                records.append({
                    'file_name': fname, 'elapsed_s': elapsed_s,
                    'q_index': idx, 'q_val': q_val,
                    'tau': t, 'g2': g, 'g2_err': err,
                })
    return records


def read_saxs(hf, fname, elapsed_s, phi_average=False):
    """Return a list of SAXS 1d records for one open file.

    Each stored SAXS point maps to a flattened (q, phi) bin, row-major over
    (n_q, n_phi).  With ``phi_average`` the intensity is averaged over the phi
    sectors of each q bin, giving one azimuthally-averaged point per q.
    """
    intensity = np.asarray(hf[SAXS_PATH][()]).reshape(-1)
    idx_map = hf[STATIC_MAP_PATH][()]
    q_list = hf[STATIC_Q_PATH][()]
    phi_list = hf[STATIC_PHI_PATH][()]
    n_phi = phi_list.shape[0]

    q_idx = idx_map // n_phi
    phi_idx = idx_map % n_phi

    if phi_average and n_phi > 1:
        records = []
        for qi in np.unique(q_idx):
            sel = q_idx == qi
            records.append({
                'file_name': fname, 'elapsed_s': elapsed_s,
                'q_index': int(qi), 'q': q_list[qi],
                'intensity': np.nanmean(intensity[sel]), 'n_phi': int(np.sum(sel)),
            })
        return records

    records = []
    for I, qi, pi in zip(intensity, q_idx, phi_idx):
        records.append({
            'file_name': fname, 'elapsed_s': elapsed_s, 'q_index': int(qi),
            'q': q_list[qi], 'phi': phi_list[pi], 'intensity': I,
        })
    return records


# --- DISCOVER FILES ---
file_paths = sorted(glob.glob(os.path.join(data_dir, '*.hdf')))
assert file_paths, f'no HDF files found in {data_dir}'
print(f'found {len(file_paths)} HDF files in {data_dir}')

# Acquisition time of each file, and the earliest one as the elapsed-time origin.
start_times = {}
for file_path in file_paths:
    try:
        with h5py.File(file_path, 'r') as hf:
            start_times[file_path] = read_start_time(hf)
    except Exception as exc:
        print(f'  no start_time for {os.path.basename(file_path)}: {exc}')
t_ref = min(start_times.values())

# --- PROCESS DATA ---
g2_records = []
saxs_records = []

for file_path in file_paths:
    fname = os.path.basename(file_path)
    st = start_times.get(file_path)
    elapsed_s = (st - t_ref).total_seconds() if st is not None else np.nan
    try:
        with h5py.File(file_path, 'r') as hf:
            g2_records.extend(read_g2(hf, fname, elapsed_s))
            saxs_records.extend(read_saxs(hf, fname, elapsed_s, phi_average=PHI_AVERAGE))
    except Exception as exc:
        print(f'  skipped {fname}: {exc}')
        continue
    print(f'  read {fname}  (elapsed {elapsed_s:.0f} s)')

# --- SAVE CSVs ---
def blank_repeated(df, cols=('file_name', 'elapsed_s')):
    """Write the per-file columns only on the first row of each file's block.

    Rows are grouped contiguously by file, so blanking every ``file_name`` that
    repeats the one above leaves a single entry per file; ``elapsed_s`` (constant
    within a file) is blanked on the same rows.  Readers should forward-fill
    these columns (``df[c] = df[c].ffill()``).
    """
    df = df.copy()
    repeat = df['file_name'] == df['file_name'].shift()
    for c in cols:
        df.loc[repeat, c] = ''
    return df


blank_repeated(pd.DataFrame(g2_records)).to_csv('g2_data.csv', index=False)
blank_repeated(pd.DataFrame(saxs_records)).to_csv('saxs_1d.csv', index=False)
print(f'\nwrote g2_data.csv ({len(g2_records)} rows) and '
      f'saxs_1d.csv ({len(saxs_records)} rows)')
