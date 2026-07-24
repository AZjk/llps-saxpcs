
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.optimize import curve_fit

import h5py
 

fn_dir = os.path.join('Y:/', '2022-1/babnigg202203/cluster_results/')
fn = 'AvgE0121_S3_6_Cycle_060C10p_att00_Rq0_00001_0001-100000.hdf'

print(fn_dir+fn)

with h5py.File(fn_dir+fn, 'r') as HDF_Result:
    det_dist = np.squeeze(HDF_Result.get('/measurement/instrument/detector/distance')[()])
    pix_dim_x = np.squeeze(HDF_Result.get('/measurement/instrument/detector/x_pixel_size')[()])
    pix_dim_y = np.squeeze(HDF_Result.get('/measurement/instrument/detector/x_pixel_size')[()])
    Up_IC = np.squeeze(HDF_Result.get('/measurement/instrument/source_end/I0Monitor')[()])
    Dn_IC = np.squeeze(HDF_Result.get('/measurement/instrument/source_end/TransmissionMonitor')[()])
    sample_thickness = np.squeeze(HDF_Result.get('/measurement/sample/thickness')[()])

    num_frames = np.squeeze(HDF_Result.get('/xpcs/data_end')[()])
    t0 = np.squeeze(HDF_Result.get('/measurement/instrument/detector/exposure_period')[()])

t_exp = t0*num_frames
F_ec = 6.25e4*(Up_IC/t_exp)-1.28e7
T_ec = Dn_IC/Up_IC/0.868
f = sample_thickness
Delta_Omega = (pix_dim_x/det_dist)*(pix_dim_y/det_dist)
