### Synopsis

<!-- This folder scales the SAXS from H06 at different T, and plots:

1. Fitting of SAXS at different T using the Lorentzian function (Fig. a)
2. Overall scaled curve (inset)
3. Scaled coefficient as a function of temperature (Fig. b)

Suggestion: For each code, plot the intermediate data and save the processed result into some `.pickle` file. Use the final code only for assembling the figure and not data analysis.

### Code flow

`SAXS_Global_Scale.ipynb`:  -->

`8IDI_reduced_H06.ipynb`:
1. (Input) Collect the H06 SA-XPCS 0.1 C/min Temp-ramp-up results from the network drive.
2. (Process) Bin the results into 10 sets. 
3. (Output) Save the binned SA-XPCS result into a pickle file (`8IDI_reduced_H06.pickle`)

`SAXS_Global_Scale.ipynb`:
1. (Input) Load `8IDI_reduced_H06.pickle`
2. (Process) Attempt to overlap all 10 SAXS curves onto the same master curve by translating them linearly in the loglog plot. The overlap is determined by the $\chi^2$ value between the neighboring results.
3. (Output) The scaled results are saved into a pickle file (`SAXS_Global_Scale.pickle`). Notice that the last two temperatures were excluded from the scaled curve because they don't overlap with the rest.

`SAXS_Scale_Fit.ipynb`:
1. (Input) Load `SAXS_Global_Scale.pickle`.
2. (Process) Put all scaled results on one figure (the master curve). Perform linear regression using the generlized Lorentzian function to determine the eta value.
3. (Output) Eta value (read directly from the screen output)

`SAXS_Global_Fit.ipynb`:
1. (Input) Load `8IDI_reduced_H06.pickle`, `SAXS_Global_Scale.pickle`, Eta from `SAXS_Scale_Fit.ipynb`:
2. (Process) Using Eta, attempt to fit all SAXS results with the generalized Lorentzian. Note the highest temperature is excluded from the fit because the fit would crash.
3. (Process) Using the Q-scaling coefficient from `SAXS_Global_Scale.pickle`, plot the characteristic length $\xi(T)$ for the lower 8 temperatures. The characteristic length for the lowest temperature is taken as the Rg from Guinier analysis in 12-ID-B. Note this $\xi(T)$ is different from the fitting parameters in the generlized Lorentzian because the generlized Lorentzian does not describe the high Q data well.
4. (Process) Attempt to fit $\xi(T)$ vs. ($(T_0-T)^{\gamma}$). Here $T_0$ is the characteristic temperature and $\gamma$ is determined via linear regression. Note that $(T_0)$ could be lower than the highest $T$ measured, in which case the higher $T$ will not show up in the log-log plot.
5. (Output) Final figure that contains (a) SAXS results at different temperatures with the Lorentzian fit; (b) $\xi(T)$ vs. ($(T_0-T)^{\gamma}$).