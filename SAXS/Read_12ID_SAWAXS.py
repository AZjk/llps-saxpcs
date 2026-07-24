import numpy as np
import os

def Read_12ID(fn_full):
    # .avg columns: q(A^-1)  I(q)  sqrt(I(q))  (col 2 is the intensity uncertainty)
    data = np.loadtxt(fn_full, comments='%')
    return data[:, 0], data[:, 1], data[:, 2]

def get_scaling_factor(q_sa, i_sa, q_wa, i_wa):
    q_min = max(q_sa.min(), q_wa.min())
    q_max = min(q_sa.max(), q_wa.max())
    mask_sa = (q_sa >= q_min) & (q_sa <= q_max)
    mask_wa = (q_wa >= q_min) & (q_wa <= q_max)
    if not any(mask_sa) or not any(mask_wa):
        return 1.0
    return np.mean(i_sa[mask_sa]) / np.mean(i_wa[mask_wa])

def merge_data(q_sa, i_sa, e_sa, q_wa, i_wa, e_wa):
    """Scale WAXS onto SAXS and concatenate, propagating the uncertainties."""
    scale = get_scaling_factor(q_sa, i_sa, q_wa, i_wa)
    i_wa_scaled = i_wa * scale
    e_wa_scaled = e_wa * scale               # scale factor treated as exact

    q_comb = np.concatenate([q_sa, q_wa])
    i_comb = np.concatenate([i_sa, i_wa_scaled])
    e_comb = np.concatenate([e_sa, e_wa_scaled])
    sort_idx = np.argsort(q_comb)

    return q_comb[sort_idx], i_comb[sort_idx], e_comb[sort_idx]

fn_path = '/home/8-id-i/2021-1/12-id-b/ZuoApr13/Processed/'

# 1. LOAD COORDINATES ONCE
SA_ql, _, _ = Read_12ID(fn_path + 'SPA1_10C_00070.avg')
WA_ql, _, _ = Read_12ID(fn_path + 'WPA1_10C_00070.avg')

# 2. LOAD INTENSITIES AND THEIR UNCERTAINTIES
# 10 C
_, SA_Iq_10C_1, SA_eq_10C_1 = Read_12ID(fn_path + 'SPA1_10C_00070.avg')
_, WA_Iq_10C_1, WA_eq_10C_1 = Read_12ID(fn_path + 'WPA1_10C_00070.avg')
_, SA_Iq_10C_2, SA_eq_10C_2 = Read_12ID(fn_path + 'SPA1_10C_00071.avg')
_, WA_Iq_10C_2, WA_eq_10C_2 = Read_12ID(fn_path + 'WPA1_10C_00071.avg')
_, SA_Iq_10C_Buf, SA_eq_10C_Buf = Read_12ID(fn_path + 'SBufferB_10C_00076.avg')
_, WA_Iq_10C_Buf, WA_eq_10C_Buf = Read_12ID(fn_path + 'WBufferB_10C_00076.avg')

# 30 C
_, SA_Iq_30C_1, SA_eq_30C_1 = Read_12ID(fn_path + 'SPA1B_30C_00078.avg')
_, WA_Iq_30C_1, WA_eq_30C_1 = Read_12ID(fn_path + 'WPA1B_30C_00078.avg')
_, SA_Iq_30C_2, SA_eq_30C_2 = Read_12ID(fn_path + 'SPA3B_30C_00081.avg')
_, WA_Iq_30C_2, WA_eq_30C_2 = Read_12ID(fn_path + 'WPA3B_30C_00081.avg')
_, SA_Iq_30C_Buf, SA_eq_30C_Buf = Read_12ID(fn_path + 'SBufferB_30C_00077.avg')
_, WA_Iq_30C_Buf, WA_eq_30C_Buf = Read_12ID(fn_path + 'WBufferB_30C_00077.avg')

# 3. SUBTRACT BACKGROUND (errors add in quadrature: sig_sub = sqrt(sig_s^2 + a^2 sig_b^2))
alpha_SA, alpha_WA = 1.0, 0.95
SA_10C_1_sub = SA_Iq_10C_1 - alpha_SA * SA_Iq_10C_Buf
WA_10C_1_sub = WA_Iq_10C_1 - alpha_WA * WA_Iq_10C_Buf
SA_30C_1_sub = SA_Iq_30C_1 - alpha_SA * SA_Iq_30C_Buf
WA_30C_1_sub = WA_Iq_30C_1 - alpha_WA * WA_Iq_30C_Buf

SA_10C_2_sub = SA_Iq_10C_2 - alpha_SA * SA_Iq_10C_Buf
WA_10C_2_sub = WA_Iq_10C_2 - alpha_WA * WA_Iq_10C_Buf
SA_30C_2_sub = SA_Iq_30C_2 - alpha_SA * SA_Iq_30C_Buf
WA_30C_2_sub = WA_Iq_30C_2 - alpha_WA * WA_Iq_30C_Buf

SA_10C_1_e = np.sqrt(SA_eq_10C_1**2 + (alpha_SA * SA_eq_10C_Buf)**2)
WA_10C_1_e = np.sqrt(WA_eq_10C_1**2 + (alpha_WA * WA_eq_10C_Buf)**2)
SA_30C_1_e = np.sqrt(SA_eq_30C_1**2 + (alpha_SA * SA_eq_30C_Buf)**2)
WA_30C_1_e = np.sqrt(WA_eq_30C_1**2 + (alpha_WA * WA_eq_30C_Buf)**2)

SA_10C_2_e = np.sqrt(SA_eq_10C_2**2 + (alpha_SA * SA_eq_10C_Buf)**2)
WA_10C_2_e = np.sqrt(WA_eq_10C_2**2 + (alpha_WA * WA_eq_10C_Buf)**2)
SA_30C_2_e = np.sqrt(SA_eq_30C_2**2 + (alpha_SA * SA_eq_30C_Buf)**2)
WA_30C_2_e = np.sqrt(WA_eq_30C_2**2 + (alpha_WA * WA_eq_30C_Buf)**2)

pl_range_10C_SA = np.arange(30,len(SA_Iq_10C_1)-10)
pl_range_30C_SA = np.arange(10,len(SA_Iq_30C_1)-10)
pl_range_10C_WA = np.arange(2,len(WA_Iq_10C_1)-240)
pl_range_30C_WA = np.arange(2,len(WA_Iq_30C_1)-200)

# 4. MERGE DATA AND EXPORT TO CSV
q_ref10, i_ref10, e_ref10 = merge_data(
    SA_ql[pl_range_10C_SA], SA_10C_1_sub[pl_range_10C_SA], SA_10C_1_e[pl_range_10C_SA],
    WA_ql[pl_range_10C_WA], WA_10C_1_sub[pl_range_10C_WA], WA_10C_1_e[pl_range_10C_WA])
q_ref30, i_ref30, e_ref30 = merge_data(
    SA_ql[pl_range_30C_SA], SA_30C_1_sub[pl_range_30C_SA], SA_30C_1_e[pl_range_30C_SA],
    WA_ql[pl_range_30C_WA], WA_30C_1_sub[pl_range_30C_WA], WA_30C_1_e[pl_range_30C_WA])
q_meas10, i_meas10, e_meas10 = merge_data(
    SA_ql[pl_range_10C_SA], SA_10C_2_sub[pl_range_10C_SA], SA_10C_2_e[pl_range_10C_SA],
    WA_ql[pl_range_10C_WA], WA_10C_2_sub[pl_range_10C_WA], WA_10C_2_e[pl_range_10C_WA])
q_meas30, i_meas30, e_meas30 = merge_data(
    SA_ql[pl_range_30C_SA], SA_30C_2_sub[pl_range_30C_SA], SA_30C_2_e[pl_range_30C_SA],
    WA_ql[pl_range_30C_WA], WA_30C_2_sub[pl_range_30C_WA], WA_30C_2_e[pl_range_30C_WA])

# Create the directory if it doesn't exist
out_dir = 'reduced_data'
os.makedirs(out_dir, exist_ok=True)

header_txt = 'Q(A^-1),I(Q),I_err(Q)'

# Save files into the 'reduced_data' folder (now with the propagated uncertainty)
np.savetxt(os.path.join(out_dir, 'Merged_Reference_10C.csv'), np.column_stack((q_ref10, i_ref10, e_ref10)), delimiter=',', header=header_txt, comments='')
np.savetxt(os.path.join(out_dir, 'Merged_Reference_30C.csv'), np.column_stack((q_ref30, i_ref30, e_ref30)), delimiter=',', header=header_txt, comments='')
np.savetxt(os.path.join(out_dir, 'Merged_Measurement_10C.csv'), np.column_stack((q_meas10, i_meas10, e_meas10)), delimiter=',', header=header_txt, comments='')
np.savetxt(os.path.join(out_dir, 'Merged_Measurement_30C.csv'), np.column_stack((q_meas30, i_meas30, e_meas30)), delimiter=',', header=header_txt, comments='')

print(f"Data successfully saved to the '{out_dir}' directory.")