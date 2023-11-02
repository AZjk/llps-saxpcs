import numpy as np
from xpcs_viewer import XpcsFile as XF
import skimage as skio
import h5py
import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging


logger = logging.getLogger(__name__)


def split(arr, n: int):
    """split arr into n parts

    Parameters
    ----------
    arr : list
        input part
    n : int 
        number of parts

    Returns
    -------
    list
        a list of n lists
    """
    k, m = divmod(len(arr), n)
    return list(arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def read_keys_from_files_parallel(flist_list):
    with Pool(24) as p:
        result = p.map(read_keys_from_files, flist_list)
        
    result = [x for x in result if x is not None]
    return result

    
def read_keys_from_files(flist, keys=('g2', 'g2_err', 'saxs_1d')):
    def get_field(xf_obj, key):
        if key == 'saxs_1d':
            x = xf_obj.saxs_1d['Iq']
            assert x.ndim in [1, 2]
            if x.ndim == 2:
                x = np.nanmean(x, axis=0)
                return x
        else:
            return xf_obj.at(key)

    data_dict = {}
    for key in keys:
        data_dict[key] = []

    for f in flist:
        try:
            dset = XF(f, fields=keys)
            for k in keys:
                data_dict[k].append(get_field(dset, k))
        except Exception:
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

    norm_factor = {}
    for k in keys:
        average_dict[k] = np.nansum(data_dict[k][mask], axis=0)
        temp = np.sum(np.isnan(data_dict[k][mask]), axis=0)
        norm_factor[k] = np.ones_like(temp) * num_valid_files - temp

    for n, key in enumerate(keys):
        factor = norm_factor[key]
        if key == 'g2_err':
            factor = np.sqrt(factor)
        average_dict[key] /= factor
        
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


def outlier_removal(data_dict, label='testrun', percentile=5):
    mask_all = np.ones_like(len(data_dict['g2']), dtype=bool)
    num_features = len(data_dict.keys()) + 1
    fig, ax = plt.subplots(num_features, 1, figsize=(4, 2.4 * num_features),
                           sharex=True)

    for n, key in enumerate(data_dict.keys()):
        if key == 'saxs_1d':
            style = 'log'
        else:
            style = 'linear'
        mask = apply_cross_corr_threshold(data_dict[key], debug_fig_ax=ax[n],
                                          style=style,
                                          label=key, percentile=percentile)

        mask_all = np.logical_and(mask_all, mask)

    ax[-1].plot(mask_all, 'o')
    ax[-1].set_title('combined_axis')
    logger.info(f'{label=}: remove {np.sum(mask_all==False)} datasets out of {len(mask_all)}')
    plt.tight_layout() 
    plt.savefig(f'debug_{label}.png', dpi=300)
    plt.close(fig)
    return mask_all


def average_datasets_without_outlier(args):
    mask = outlier_removal(args[0], label=args[1], percentile=5)
    avg_dict = average_datasets(data_dict=args[0], mask=mask)
    return avg_dict


def average_datasets_without_outlier_parallel(args):
    with Pool(24) as p:
        result = p.map(average_datasets_without_outlier, args)
    return result


def get_temperature(fname, zone_idx=1):
    key = f'/measurement/sample/QNW_Zone{zone_idx}_Temperature'
    with h5py.File(fname) as f:
        val = f[key][()][0]
    return val


def read_temperature_from_files(fnames, zone_idx=1):
    return np.array([get_temperature(f, zone_idx) for f in fnames])


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == '__main__':
    flist = glob.glob('/home/8ididata/2021-2/babnigg202107_2/cluster_results_QZ/B039*')[0:100]
    # print(read_keys_from_files(flist))
    # print(read_temperature_from_files(flist))
    data_dict = read_keys_from_files(flist)
    mask = outlier_removal(data_dict)
    print(np.sum(mask))
    average_datasets(data_dict=data_dict, mask=mask)
