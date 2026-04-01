import numpy as np
import matplotlib.pyplot as plt

def Read_12ID(fn_full, detector='SA'):
    data = np.loadtxt(fn_full, comments='%')
    q = data[:, 0]
    i = data[:, 1]
    
    if detector == 'SA':
        # Remove the last 52 lines for SAXS
        return q[15:-55], i[15:-55]
    elif detector == 'WA':
        # Remove the first line and the last 100 lines for WAXS
        return q[1:-270], i[1:-270]
    
    return q, i

fn_path = '/home/8-id-i/2021-3/12-id-b/qtZhang2/Processed/'

# 1. LOAD BACKGROUND BUFFERS (Sliced automatically on load)
_, i_sa_buf = Read_12ID(fn_path + 'SBuffer_6.0C_00016.avg', detector='SA')
_, i_wa_buf = Read_12ID(fn_path + 'WBuffer_6.0C_00016.avg', detector='WA')

alpha_SA = 1.0
alpha_WA = 0.95

# Dictionary for manual scaling factors
file_scales = {
    'A_H06_Stock_6.00C_00051': 1,
    'A_H06_Stock_27.85C_00196': 1,
    'A_H06_Stock_28.76C_00202': 1,
    'A_H06_Stock_29.68C_00208': 1,
    'A_H06_Stock_32.43C_00226': 1,
    'A_H06_Stock_33.35C_00232': 1,
    'A_H06_Stock_34.26C_00238': 1,
    'A_H06_Stock_35.18C_00244': 1,
    'A_H06_Stock_36.10C_00250': 1
}

plt.rcParams.update({'font.size': 16, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(10, 8))

for base, waxs_scale in file_scales.items():
    sa_fn = f"S{base}.avg"
    wa_fn = f"W{base}.avg"
    
    # Sliced automatically on load
    q_sa, i_sa = Read_12ID(fn_path + sa_fn, detector='SA')
    q_wa, i_wa = Read_12ID(fn_path + wa_fn, detector='WA')
    
    # 2. BACKGROUND SUBTRACTION (Arrays are guaranteed to match shapes)
    i_sa_sub = i_sa - (alpha_SA * i_sa_buf)
    i_wa_sub = i_wa - (alpha_WA * i_wa_buf)
    
    # 3. APPLY INDIVIDUAL SCALE
    i_wa_scaled = i_wa_sub * waxs_scale
    
    # 4. MERGE ARRAYS
    q_comb = np.concatenate([q_sa, q_wa])
    i_comb = np.concatenate([i_sa_sub, i_wa_scaled])
    
    # 5. STRIP ZEROS AND NEGATIVES
    valid_mask = i_comb > 0
    q_comb = q_comb[valid_mask]
    i_comb = i_comb[valid_mask]
    
    idx = np.argsort(q_comb)
    
    temp_label = base.split('_')[3] 
    
    ax.plot(q_comb[idx], i_comb[idx], label=temp_label, fillstyle='none', 
            marker='o', markersize=4, linestyle='none')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$q (\AA^{-1})$')
ax.set_ylabel('Intensity (arbs.)')
ax.legend(fontsize=12, ncol=2)

plt.tight_layout()
plt.savefig('Specific_TempSweep_Stitched_Subtracted.pdf', bbox_inches='tight')
plt.show()