#!/usr/bin/env python3
"""
average_sections.py

Split XPCS/SAXS HDF result files (selected by header and _r number range) into
equal sections, remove outliers within each section, average g2 / g2_err /
saxs_1d / saxs_2d, and write one output HDF file per section that preserves the
full Nexus structure of the input files.

Usage (CLI)
-----------
  python average_sections.py \\
      --header   E0110 \\
      --prefix   /path/to/cluster_results \\
      --range    1 1000 \\
      --sections 10 \\
      [--output_dir /path/to/output]   # default: <prefix>/averaged

Python API
----------
  from average_sections import average_sections
  average_sections(header='E0110', file_range=(1, 1000),
                   num_sections=10, prefix='/path/to/cluster_results')
"""

import argparse
import glob
import logging
import os
import re
import sys

import h5py
import numpy as np

# ── sibling utilities (split, outlier_removal) ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import split, outlier_removal

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# HDF paths for the four averaged datasets (no leading slash for h5py key access)
_G2      = 'xpcs/multitau/normalized_g2'
_G2_ERR  = 'xpcs/multitau/normalized_g2_err'
_SAXS1D  = 'xpcs/temporal_mean/scattering_1d'
_SAXS2D  = 'xpcs/temporal_mean/scattering_2d'

_AVERAGED_FIELDS = [
    ('g2',      _G2),
    ('g2_err',  _G2_ERR),
    ('saxs_1d', _SAXS1D),
    ('saxs_2d', _SAXS2D),
]


# ── private helpers ───────────────────────────────────────────────────────────

def _r_number(fpath):
    """Extract the integer _r##### repetition number from a filename."""
    m = re.search(r'_r(\d+)_results', os.path.basename(fpath))
    return int(m.group(1)) if m else -1


def _find_files(prefix, header, file_range):
    """Glob for header*.hdf and keep files whose _r number is in file_range."""
    flist = sorted(glob.glob(os.path.join(prefix, f'{header}*.hdf')))
    if not flist:
        raise FileNotFoundError(
            f'No HDF files match "{header}*.hdf" in {prefix}')

    r0, r1 = file_range
    flist = [f for f in flist if r0 <= _r_number(f) <= r1]
    if not flist:
        raise FileNotFoundError(
            f'No files for header={header!r} with _r in [{r0}, {r1}]')

    logger.info(f'Found {len(flist)} files for {header!r}, _r in [{r0}, {r1}]')
    return flist


def _read_section(flist):
    """Read the four averaged fields from every file in *flist*.

    Returns
    -------
    data_dict : dict[str, ndarray]
        Keys: 'g2', 'g2_err', 'saxs_1d', 'saxs_2d'.
        Each value is a float64 array with shape (N, ...) where N is the
        number of successfully read files.
    valid_files : list[str]
        Paths of files that were read without error.
    """
    stacks = {k: [] for k, _ in _AVERAGED_FIELDS}
    valid  = []

    for f in flist:
        try:
            with h5py.File(f, 'r') as hf:
                row = {k: hf[path][()].astype(np.float64)
                       for k, path in _AVERAGED_FIELDS}
            for k in stacks:
                stacks[k].append(row[k])
            valid.append(f)
        except Exception as exc:
            logger.warning(f'Skipping {os.path.basename(f)}: {exc}')

    if not valid:
        raise RuntimeError('No readable files in this section')

    return {k: np.stack(stacks[k], axis=0) for k in stacks}, valid


def _average_with_mask(data_dict, mask):
    """Average all fields, applying boolean *mask* along axis 0.

    g2_err uses quadrature-sum / count (error propagation for the mean of
    independent measurements).  All other fields use nanmean.
    """
    avg = {}
    for k in data_dict:
        arr = data_dict[k][mask]
        if k == 'g2_err':
            count  = np.sum(~np.isnan(arr), axis=0)
            avg[k] = (np.sqrt(np.nansum(arr ** 2, axis=0))
                      / np.maximum(count, 1))
        else:
            avg[k] = np.nanmean(arr, axis=0)
    return avg


def _out_filename(section_files, sec_idx, output_dir):
    """Build an output filename for a section by modifying the first file's name."""
    first = os.path.basename(section_files[0])
    r0    = _r_number(section_files[0])
    r1    = _r_number(section_files[-1])
    # Replace _r##### with _avg_secNN_r#####-r#####
    name  = re.sub(
        r'_r\d+_results',
        f'_avg_sec{sec_idx:02d}_r{r0:05d}-r{r1:05d}_results',
        first)
    return os.path.join(output_dir, name)


def _write_file(out_path, template_file, avg_dict, filelist):
    """Write a new HDF file with the Nexus structure of *template_file*.

    The entire tree is copied from the template; the four averaged datasets
    are then replaced with the computed averages, and the list of averaged
    filenames is stored at /xpcs/average/filelist.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with h5py.File(template_file, 'r') as src, h5py.File(out_path, 'w') as dst:
        # Copy the full Nexus tree from the first file in the section
        for name in src:
            src.copy(name, dst)

        # Replace each averaged dataset
        for k, hdf_path in _AVERAGED_FIELDS:
            del dst[hdf_path]
            dst.create_dataset(hdf_path,
                               data=avg_dict[k].astype(np.float32),
                               compression='gzip')

        # Store the list of averaged filenames
        grp = dst['xpcs'].require_group('average')
        grp.create_dataset(
            'filelist',
            data=np.array([os.path.basename(f).encode() for f in filelist]))

    logger.info(f'  -> {out_path}')


# ── public API ────────────────────────────────────────────────────────────────

def average_sections(header, file_range, num_sections=1,
                     prefix='.', output_dir=None):
    """Split, outlier-remove, and average XPCS/SAXS HDF files in sections.

    Parameters
    ----------
    header : str
        Filename header prefix, e.g. ``'E0110'``.
    file_range : (int, int)
        Inclusive _r number range, e.g. ``(1, 1000)``.
    num_sections : int
        Number of equal-sized sections to split the file list into. Default 1.
    prefix : str
        Directory containing the input HDF files.
    output_dir : str, optional
        Output directory for the averaged files.
        Defaults to ``<prefix>/averaged``.
    """
    if output_dir is None:
        output_dir = os.path.join(prefix, 'averaged')

    flist    = _find_files(prefix, header, file_range)
    sections = split(flist, num_sections)

    for sec_idx, sec_files in enumerate(sections):
        logger.info(
            f'--- Section {sec_idx:02d}/{num_sections - 1} '
            f'({len(sec_files)} files) ---')

        data_dict, valid_files = _read_section(sec_files)

        # Run outlier removal on g2, g2_err, saxs_1d.
        # saxs_2d is excluded here (large 2D arrays make cross-correlation slow);
        # the same mask is applied to it during averaging.
        outlier_input = {k: data_dict[k].copy()
                         for k in ('g2', 'g2_err', 'saxs_1d')}
        mask = outlier_removal(
            outlier_input,
            label=f'{header}_sec{sec_idx:02d}',
            percentile=5)

        n_kept    = int(np.sum(mask))
        n_removed = len(mask) - n_kept
        logger.info(
            f'  Outlier removal: {n_removed} removed, {n_kept} kept '
            f'out of {len(mask)}')

        avg_dict = _average_with_mask(data_dict, mask)

        out_path = _out_filename(valid_files, sec_idx, output_dir)
        _write_file(out_path, valid_files[0], avg_dict, valid_files)

    logger.info(
        f'Done — {num_sections} averaged file(s) written to: {output_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(
        description=(
            'Split XPCS HDF files into sections by _r number, '
            'remove outliers, and write one averaged Nexus file per section.'))
    ap.add_argument('--header',     required=True,
                    help='Filename header prefix (e.g. E0110)')
    ap.add_argument('--prefix',     required=True,
                    help='Directory containing the input HDF files')
    ap.add_argument('--range',      nargs=2, type=int, metavar=('START', 'END'),
                    required=True,
                    help='Inclusive _r number range (e.g. --range 1 1000)')
    ap.add_argument('--sections',   type=int, default=1,
                    help='Number of sections (default: 1)')
    ap.add_argument('--output_dir', default=None,
                    help='Output directory (default: <prefix>/averaged)')
    args = ap.parse_args()

    average_sections(
        header       = args.header,
        file_range   = tuple(args.range),
        num_sections = args.sections,
        prefix       = args.prefix,
        output_dir   = args.output_dir,
    )


if __name__ == '__main__':
    _cli()
