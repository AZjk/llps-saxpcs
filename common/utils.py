import numpy as np
import sys
import os
from pyxpcsviewer import XpcsFile as XF
import skimage as skio
import h5py
import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging



logger = logging.getLogger(__name__)


def split(arr, n_segments: int):
    """split arr into n parts

    Parameters
    ----------
    arr : list
        input part
    n_segments : int 
        number of segments

    Returns
    -------
    list
        a list of n_segments lists
    """
    k, m = divmod(len(arr), n_segments)
    return list(arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] 
                for i in range(n_segments))


def read_keys_from_files_parallel(flist_list, num_cores=24):
    """
    Read keys from a list of files and process them in parallel
    """  
    with Pool(num_cores) as p:
        result = p.map(read_keys_from_files, flist_list)
        
    result = [x for x in result if x is not None]
    return result

    
def read_keys_from_files(flist, keys=('g2', 'g2_err', 'saxs_1d')):
    """
    This module provides functionalities to read, process, and analyze XPCS (X-ray Photon Correlation Spectroscopy) datasets. 
    It supports parallel reading of data files, splitting arrays into n-segments, averaging datasets, applying cross-correlation thresholds,
    and removing outliers based on correlation matrices.

    Functions
    ---------
    - split(arr, n_segments)
        Splits an array into specified number of segments.

    - read_keys_from_files_parallel(flist_list, num_cores)
        Reads keys from a list of files and processes them in parallel.

    - read_keys_from_files(flist, keys)
        Reads specific keys from a list of files.

    - average_datasets(flist, data_dict, mask)
        Averages datasets from a list of files or a dictionary with pre-read data.

    - apply_cross_corr_threshold(x0, percentile, style, debug_fig_ax, label)
        Applies a cross-correlation threshold to data using the specified percentile as the cutoff value.

    - outlier_removal(data_dict, label, percentile, plot_debug)
        Removes outliers from the data dictionary based on the correlation matrix.
    """

    def get_field(xf_obj, key):
        if key == 'saxs_1d':
            x = xf_obj.saxs_1d['Iq']
            assert x.ndim in [1, 2]
            if x.ndim == 2:
                try:
                    # print(np.sum(~np.isnan(x)))
                    x = np.nanmean(x, axis=0)
                except:
                    print(xf_obj.saxs_1d['Iq'])
                    raise
                return x
        else:
            return getattr(xf_obj, key)

    data_dict = {}
    for key in keys:
        data_dict[key] = []

    for f in flist:
        try:
            dset = XF(f, fields=keys)
            for k in keys:
                data_dict[k].append(get_field(dset, k))
        except Exception:
            logger.info(f'failed to read file {f}, skip this file')
            pass
    
    for k in keys:
        data_dict[k] = np.array(data_dict[k])
        
    return data_dict


def average_datasets(flist=None, data_dict=None, mask=None):
    if data_dict is None and flist is not None:
        keys=('g2', 'g2_err', 'saxs_1d')
        data_dict = read_keys_from_files(flist)
    elif data_dict is not None:
        keys = data_dict.keys()
    else:
        print('must provide flist or data_dict')
        raise

    average_dict = {}

    if mask is None:
        num_valid_files = data_dict['g2'].shape[0]
        mask = np.ones_like(num_valid_files, dtype=bool)
    else:
        num_valid_files = np.sum(mask)

    for k in keys:
        if k != 'g2_err':
            average_dict[k] = np.nanmean(data_dict[k][mask], axis=0)
        else:
            average_dict[k] = np.sqrt(np.nansum(data_dict[k][mask] ** 2, axis=0))
            average_dict[k] /= np.sum(~np.isnan(data_dict[k][mask]), axis=0)
        
    return average_dict


def apply_cross_corr_threshold(x0, percentile=5, style='linear', debug_fig_ax=None,
                               label='debug'):
    x = x0 
    x[np.isnan(x)] = 0
    if style == 'log':
        x[np.isnan(x)] = 1
        x[x <= 0] = 1
        x = np.log10(x)

    num = x.shape[0]
    result = np.zeros((num, num), dtype=np.float64)
    for n in range(num):
        for m in range(n + 1, num):
            val = np.sum(x[n] * x[m]) / np.sqrt(np.sum(x[n] ** 2) * np.sum(x[m] ** 2))
            result[n, m] = val
            result[m, n] = val

    result_1d = np.sum(result, axis=1)
    low_cutoff = np.percentile(result_1d, percentile)
    if debug_fig_ax is not None:
        title = f'{label}_{style=}_{percentile=}'
        debug_fig_ax.plot(result_1d, 'o')
        debug_fig_ax.hlines(low_cutoff, xmin=-1, xmax=len(result_1d))
        debug_fig_ax.set_title(title)
        # plt.savefig(f'debug_{label}.png', dpi=300)

    mask = result_1d >= low_cutoff
    return mask


def outlier_removal(data_dict, label='testrun', percentile=5, plot_debug=False):
    """
    Remove outliers from the data dictionary based on the correlation matrix.
    """
    mask_all = np.ones(len(data_dict['g2']), dtype=bool)
    num_features = len(data_dict.keys()) + 1
    """
    This module supports various data processing operations related to segmenting, reading, averaging, and filtering datasets.

    It supports parallel reading of data files, splitting arrays into n-segments, averaging datasets, applying cross-correlation thresholds,
    and removing outliers based on correlation matrices.

    Functions
    ---------
    - split(arr, n_segments)
        Splits an array into the specified number of segments.

    - read_keys_from_files_parallel(flist_list, num_cores)
        Reads keys from a list of files and processes them in parallel.

    - read_keys_from_files(flist, keys)
        Reads specific keys from a list of files.

    - average_datasets(flist, data_dict, mask)
        Averages datasets from a list of files or a dictionary with pre-read data.

    - apply_cross_corr_threshold(x0, percentile, style, debug_fig_ax, label)
        Applies a cross-correlation threshold to data using the specified percentile as the cutoff value.

    - outlier_removal(data_dict, label, percentile, plot_debug)
        Removes outliers from the data dictionary based on the correlation matrix.

    - average_datasets_without_outlier(args)
        Averages datasets after outlier removal.

    - average_datasets_without_outlier_parallel(args, num_cores)
        Averages datasets after outlier removal in parallel usage.

    - get_temperature(fname, zone_idx)
        Extracts temperature information from given file.

    - read_temperature_from_files(fnames, zone_idx)
        Reads temperature information from list of files.

    - process_group(group, prefix, num_sections, zone_idx, num_cores, skip_first_files, skip_last_files)
        Processes data group by reading file list, extracting temperatures, and optionally segmenting data and averaging datasets.
    """

    if plot_debug:
        fig, ax = plt.subplots(num_features, 1, figsize=(4, 2.4 * num_features),
                               sharex=True)
    else:
        ax = [None for _ in range(num_features)]

    for n, key in enumerate(data_dict.keys()):
        if key == 'saxs_1d':
            style = 'log'
        else:
            style = 'linear'
        mask = apply_cross_corr_threshold(data_dict[key], debug_fig_ax=ax[n],
                                          style=style,
                                          label=key, percentile=percentile)

        mask_all = np.logical_and(mask_all, mask)
    logger.info(f'{label=}: remove {np.sum(mask_all==False)} datasets out of {len(mask_all)}')

    if plot_debug:
        ax[-1].plot(mask_all, 'o')
        ax[-1].set_title('combined_axis')
        plt.tight_layout()
    
        if not os.path.isdir('debug'):
            os.mkdir('debug')
        plt.savefig(f'debug/debug_outlier_{label}.png', dpi=300)
        plt.close(fig)

    return mask_all


def average_datasets_without_outlier(args):
    """
    Average datasets without outlier.
    """
    mask = outlier_removal(args[0], label=args[1], percentile=5)
    avg_dict = average_datasets(data_dict=args[0], mask=mask)
    return avg_dict


def average_datasets_without_outlier_parallel(args, num_cores=24):
    with Pool(num_cores) as p:
        result = p.map(average_datasets_without_outlier, args)
    return result


def get_temperature(fname, zone_idx='auto'):
    if zone_idx == 'auto':
        zone_idx = (ord(os.path.basename(fname)[0]) - ord('A')) // 3 + 1
        
    assert 1 <= zone_idx <= 3, 'zone_idx must be in [1, 2, 3]'
    
    # key = f'/measurement/sample/QNW_Zone{zone_idx}_Temperature'
    key = f"/entry/sample/qnw{zone_idx}_temperature"
        
    try:
        with h5py.File(fname) as f:
            val = float(f[key][()][0])
    except Exception:
        val = np.nan
        logger.info(f'Failed to get temperature for file {fname}, return nan')
    return val


def read_temperature_from_files(fnames, zone_idx=1):
    return np.array([get_temperature(f, zone_idx) for f in fnames])


def process_group(group='B039',
                  prefix='/data/xpcs8/2022-1/babnigg202203/cluster_results_reanalysis',
                  num_sections=10,
                  zone_idx=1,
                  num_cores=24,
                  skip_first_files=0,
                  skip_last_files=0):
    
    
    # read file list in the folder
    flist = glob.glob(os.path.join(prefix, f'{group}*.hdf'))
    flist.sort()
    assert len(flist) > 0, f'no dataset is found in {prefix}'

    logger.info(f'total number of files in {group}  is {len(flist)}')

    logger.info(f'{skip_first_files=}, {skip_last_files=}')

    skip_end = len(flist) - skip_last_files
    flist = flist[skip_first_files: skip_end]
    assert len(flist) > 0, 'no dataset after the skip filter'
    
    # get the temperature
    temperature_list = read_temperature_from_files(flist, zone_idx=zone_idx)

    if num_sections <= 0:
        keys=('g2', 'g2_err', 'saxs_1d')
        data_dict = read_keys_from_files(flist, keys=keys)
        axf = XF(flist[0])
        t_el = axf.t_el
        ql_dyn = axf.dqlist
        ql_sta = axf.sqlist
        data_dict['temperature'] = temperature_list
        
        return data_dict, t_el, ql_dyn, ql_sta
        
    flist_sections = split(flist, num_sections)
    idx_sections = split(np.arange(len(flist)), num_sections)
    print('index\t T-min(C)\t T-max(C)\t T-mean(C)\t points')
    for n in range(num_sections):
        temp = temperature_list[idx_sections[n]]
        print(f'{n=:02d}\t {round(np.min(temp), 4):8}\t {round(np.max(temp), 4):8}\t {round(np.mean(temp), 4):8}\t {len(idx_sections[n])}')

    # read main field for averag
    data_dict_all = read_keys_from_files_parallel(flist_sections, num_cores=num_cores)
    
    # fig, ax = plt.subplots(1, 1)
    # for n, sub_list in enumerate(idx_sections):
    #     ax.plot(sub_list, temperature_list[sub_list], 'o', label=f'{n}')
    # plt.legend()
    
    # do outlier removal and average
    args = [(data_dict_all[n], f'{group}_section_{n:02d}') for n in range(num_sections)]
    avg_all = average_datasets_without_outlier_parallel(args, num_cores=num_cores)
    # mask = outlier_removal(data_dict, label=f'{group}_section_{n:02d}', percentile=5)
    # avg_dict = average_datasets(data_dict=data_dict, mask=mask)
    
    
    for n, avg_dict in enumerate(avg_all):
        avg_dict['temperature'] = temperature_list[idx_sections[n]]
        avg_dict['temperature_x'] = idx_sections[n]
    
    # get additonal field for plotting
    axf = XF(flist[0])
    t_el = axf.t_el
    ql_dyn = axf.dqlist
    ql_sta = axf.sqlist
    return avg_all, t_el, ql_dyn, ql_sta


if __name__ == '__main__':
    # test 01
    # flist = glob.glob('/home/8ididata/2021-2/babnigg202107_2/cluster_results_QZ/B039*')[0:100]
    # # print(read_keys_from_files(flist))
    # # print(read_temperature_from_files(flist))
    # data_dict = read_keys_from_files(flist)
    # print(data_dict.keys())
    # # print(data_dict['g2'].shape)
    # mask = outlier_removal(data_dict)
    # print(np.sum(mask))
    # res = average_datasets(data_dict=data_dict, mask=mask)

    # test 02
    a, b, c, d = process_group(group='B039', prefix='/home/8ididata/2021-2/babnigg202107_2/cluster_results_QZ',
                  skip_first_files=0, skip_last_files=30)
    print(a[0]['saxs_1d'])
    exit(0)