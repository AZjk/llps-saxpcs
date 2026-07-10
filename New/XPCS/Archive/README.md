
It looks like Plot_Ave_Ranges.py does the averaging and automatic outlier removal by directly averaging from the original data at: /home/8-id-i/2022-1/babnigg202203_nexus/reprocess_results/

Plot_Ave.py, on the other hand, plots the manually averaged files from the same folder in /average.

What does the SAXS look like? Also, if showing SAXS, might as well try the background subtraction, absolute cross-section conversion, and including SAXS from lower temperatures as well


Prompt:

In /home/beams/8IDIUSER/Documents/llps-saxpcs/New/XPCS/Plot_Ave_Ranges.py, split the code into two: one reads the files with a given group, prefix and file_ranges, removes the outliers, average g2, g2_err and saxs_1d, and save them back into an hdf file with the same structure at the same directory of the original file. Add 'Average' and the range of the files to the file name to distinguish from the rest of the files.

In the averaged hdf file, modify field /entry/start_time to use the time of the first file for the range of the averaged files. Look for the time stamps in /home/beams/8IDIUSER/Documents/llps-saxpcs/timelist_2022-1.txt. Add one more field called /xpcs/average/file_list/ that records the name of all the files that are included in the average.

The background is D0138 in the same directory and was averaged with Miaoqi's code. Conversion to absolute cross-section is just multiplying some coefficient from the IC_pin4 calibration. 

If background subtraction doesn't bring the 6 C data to flat level, there's always the option to add correction coefficients.

-----------------
coef_sam = 6.93e4  # Absolute scattering cross-section coefficient for the samples
coef_buf = 7.62e4  # Absolute scattering cross-section coefficient for the buffer

for ii in range(len(avg_ramp_up)):
    avg_ramp_up[ii]["saxs_1d"] = coef_sam*avg_ramp_up[ii]["saxs_1d"] - coef_buf*avg_bg[0]["saxs_1d"]

for ii in range(len(avg_ramp_down)):
    avg_ramp_down[ii]["saxs_1d"] = coef_sam*avg_ramp_down[ii]["saxs_1d"] - coef_buf*avg_bg[0]["saxs_1d"]