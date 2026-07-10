# Bug Report: Spurious single-q spikes in averaged SAXS-1D

**Date:** 2026-07-09
**Component:** `New/common/utils.py` — `apply_cross_corr_threshold()` (called via `outlier_removal()`)
**Affected output:** averaged `scattering_1d` written by `New/XPCS/Average_Ranges.py`
**Severity:** High — corrupts the averaged SAXS-1D curve (spikes ~10⁵× baseline). g2/g2_err inputs are also silently altered.
**Status:** Fixed (see below).

---

## Symptom

Averaged result files (e.g. `B0147_S3_7_300C10p_att00_Rq0_Average_00950_01050_results.hdf`)
show sharp, isolated single-q spikes in the SAXS-1D curve in pyXPCSViewer.

Notably:
- The **individual** raw result files (00950–00960) show **no** spikes in SAXS-1D.
- The **SAXS-2D** image of the averaged file shows **no** hot pixels.
- Re-averaging the pristine raw `scattering_1d` by hand (no outlier step) produces **no** spikes
  (max neighbor ratio 1.4×).

So the spikes are neither detector hot pixels nor an artifact of the mean itself.

## Root cause

`outlier_removal()` calls `apply_cross_corr_threshold()` once per field
(`g2`, `g2_err`, `saxs_1d`) on the arrays held in `data_dict`. The original code:

```python
def apply_cross_corr_threshold(x0, percentile=5, style='linear', ...):
    x = x0                    # reference, NOT a copy -> x and x0 alias the same array
    x[np.isnan(x)] = 0
    if style == 'log':        # saxs_1d is called with style='log'
        x[np.isnan(x)] = 1
        x[x <= 0] = 1         # writes literal 1.0 into data_dict['saxs_1d']
        x = np.log10(x)       # reassigns x, but the in-place damage above is already done
```

Because `x = x0` binds `x` to the **same** ndarray as `data_dict['saxs_1d']`, the in-place
assignments (`x[np.isnan(x)] = 0`, `x[x <= 0] = 1`) mutate the caller's data. The `x[x <= 0] = 1`
line is meant only to make the correlation metric well-defined, but it overwrites real data:

- Raw `scattering_1d` intensities are ~1e-5.
- At certain q-bins, some frames record exactly **0** (detector gaps / masked pixels that fall
  into that q-ring).
- Those zeros are replaced with the literal value **1.0** — about 5 orders of magnitude too large.

`average_datasets()` then averages this corrupted array. At a q-bin where `k` of `N` frames were
zeroed → set to 1.0, the mean is ≈ `k/N`, versus the true ~1e-5. That is the spike.

### Confirmation (range 950–1050, N=101 frames)

| q-bin | frames with 0 | buggy mean (≈k/N) | stored spike value | true (clean) mean |
|------:|--------------:|------------------:|-------------------:|------------------:|
| 950   | 11            | 0.109             | 0.112              | 2.3e-05           |
| 1115  | 25            | 0.247             | 0.225              | 1.4e-05           |
| 1363  | 7             | 0.069             | 0.045              | 6.9e-06           |
| 1468  | 37            | 0.366             | 0.337              | 3.2e-06           |

The stored spike values match `k/N` (small differences because outlier removal drops a few frames),
confirming the mechanism.

This also explains every observation: individual files are untouched on disk (and true zeros are
invisible on a log plot); SAXS-2D is untouched because the script never writes `scattering_2d`; and a
manual re-average without the outlier step is clean.

### Secondary effect

The same aliasing means the `g2` / `g2_err` branch (`style='linear'`) runs `x[np.isnan(x)] = 0` on
`data_dict['g2']` and `data_dict['g2_err']` in place, so NaNs in those arrays are silently converted
to 0 before averaging. Less visible, but still an unintended mutation of the caller's data.

## Fix

Operate on a copy so the routine never mutates the caller's arrays:

```python
def apply_cross_corr_threshold(x0, percentile=5, style='linear', ...):
    # Work on a copy: this routine rewrites NaNs and non-positive values (e.g.
    # x[x <= 0] = 1 below) purely to make the correlation metric well-defined.
    # If we mutated x0 in place it would corrupt data_dict, and those injected
    # 1.0's would later be averaged into saxs_1d as huge single-q spikes.
    x = np.array(x0, dtype=np.float64)
    x[np.isnan(x)] = 0
    ...
```

Single-line change (`x = x0` → `x = np.array(x0, dtype=np.float64)`). It is strictly safer: the
function's return value (the boolean mask) is unchanged; it only stops corrupting its input.

## Verification

After the fix, on range 950–1050:
- `data_dict` is unchanged by `outlier_removal()` (verified by array comparison before/after).
- Max neighbor spike ratio in the averaged SAXS-1D dropped from **~95,000×** to **1.10×**.
- The former spike bins now hold their correct ~1e-5 values.

All four averaged files were regenerated; each now reports a max spike ratio ≈ 1.1×:

```
Average_00950_01050  1.10
Average_01051_01150  1.07
Average_01151_01250  1.15
Average_01251_01313  1.10
```

## Recommendations

1. Keep the copy fix in `apply_cross_corr_threshold` (root cause).
2. Consider treating detector-gap zeros as missing (NaN) rather than valid 0 in `scattering_1d`, so
   `np.nanmean` ignores them in the average instead of counting them as real zero intensity.
3. As a defensive measure, helpers in `common/utils.py` that take arrays should avoid in-place
   mutation of inputs (copy-on-write), since `data_dict` is reused across fields and callers.
