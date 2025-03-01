"""
List of functions used to process the SAXS data for sample H06 during heating ramp.
"""

import numpy as np
from scipy.optimize import minimize


def preprocess_data(avg_data_list, ql_sta):
    """
    Preprocess the data to remove non-positive values from q and I.
    Returns cleaned q and avg_data_list.
    """
    cleaned_data_list = []
    cleaned_ql_sta = []

    for curve in avg_data_list[0:-1]:
        q = ql_sta
        i = curve['saxs_1d']
        
        # Filter to ensure both q and I are positive
        valid_indices = (q > 0) & (i > 0)
        cleaned_q = q[valid_indices]
        cleaned_i = i[valid_indices]

        if len(cleaned_q) > 0 and len(cleaned_i) > 0:
            cleaned_ql_sta.append(cleaned_q)
            cleaned_data_list.append({'saxs_1d': cleaned_i})

    return cleaned_data_list, cleaned_ql_sta


def chi_squared_direct(q_scale, i_scale, ref_x, ref_y, target_x, target_y):
    """
    Calculate chi-squared directly comparing the reference curve with the scaled target curve.
    Formula: chi^2 = sum((O - E)^2 / E)
    """
    target_x = np.log10(target_x)
    target_y = np.log10(target_y)
    ref_x = np.log10(ref_x)
    ref_y = np.log10(ref_y)

    scaled_x = target_x + np.log10(q_scale)
    scaled_y = target_y + np.log10(i_scale)

    range_x_min = np.max([np.min(scaled_x), np.min(ref_x)])
    range_x_max = np.min([np.max(scaled_x), np.max(ref_x)])
    range_x_overlap = np.where((scaled_x > range_x_min) & (scaled_x < range_x_max))

    ref_y_inter = np.interp(scaled_x[range_x_overlap], ref_x, ref_y)

    chi2 = np.mean((scaled_y[range_x_overlap] - ref_y_inter) ** 2)

    return chi2


def iterative_scaling(avg_data_list, ql_sta_cleaned):
    """
    Iteratively scale each curve in the dataset, using the previously scaled curve as the reference.
    Dynamically adjust the q_scale range based on the curve index.
    """
    ref_x = ql_sta_cleaned[0]
    ref_y = avg_data_list[0]['saxs_1d']

    scaling_results = []
    scaled_curves = []

    for i in range(1, len(avg_data_list)):  # Start from the second curve
        target_x = ql_sta_cleaned[i]
        target_y = avg_data_list[i]['saxs_1d']

        # Set dynamic q_scale range based on curve index
        if i <= 4:  # Curves 1–4
            q_scale_range = np.arange(0.5, 2.0, 0.01)
        elif i <= 6:  # Curves 5–6
            q_scale_range = np.arange(1.9, 4.2, 0.01)
        elif i == 7:  # Curve 7
            q_scale_range = np.arange(5.5, 7.0, 0.01)
        elif i == 8:  # Curve 8
            q_scale_range = np.arange(9.5, 11.0, 0.01)
        else:
            raise ValueError(f"Unexpected curve index: {i}")

        i_scale_range = np.arange(0.001, 1.5, 0.001)

        chi2_results = []
        for q_scale in q_scale_range:
            for i_scale in i_scale_range:
                chi2 = chi_squared_direct(q_scale, i_scale, ref_x, ref_y, target_x, target_y)
                chi2_results.append((q_scale, i_scale, chi2))

        best_q_scale, best_i_scale, min_chi2 = min(chi2_results, key=lambda x: x[2])
        scaling_results.append((i, best_q_scale, best_i_scale, min_chi2))

        scaled_x = target_x * best_q_scale
        scaled_y = target_y * best_i_scale
        scaled_curves.append((scaled_x, scaled_y))

        ref_x = scaled_x
        ref_y = scaled_y

    return scaling_results, scaled_curves


def normalize_scaled_curves(scaled_curves):
    """
    Normalize the intensity values based on the lowest q-value.
    """
    # Merge all scaled curves
    all_q = np.concatenate([curve[0] for curve in scaled_curves])
    all_i = np.concatenate([curve[1] for curve in scaled_curves])
    
    # Sort by q-value
    sorted_indices = np.argsort(all_q)
    all_q = all_q[sorted_indices]
    all_i = all_i[sorted_indices]
    
    # Find the lowest q-value and corresponding intensity
    min_q_index = np.argmin(all_q)
    reference_intensity = all_i[min_q_index]
    
    # Normalize intensities
    normalized_i = all_i / reference_intensity
    
    return all_q, normalized_i


def combine_scaled_data(scaled_curves):
    """
    Combine all scaled curves into a single dataset, ensuring only positive x and y values are included.
    """
    all_x, all_y = [], []
    for scaled_x, scaled_y in scaled_curves:
        valid_indices = (scaled_x > 0) & (scaled_y > 0)  # Ensure x > 0 and y > 0
        all_x.extend(scaled_x[valid_indices])
        all_y.extend(scaled_y[valid_indices])
    return np.array(all_x), np.array(all_y)


# --------------------------
# T0 Optimization using log–log fitting
# The model in log space is: log(x) = log(A) + n * log(T0 - T)
def optimize_T0_log(temperatures, x_values, x_errors, initial_delta=1.0, num_samples=1000):
    def residual(delta, x_vals, x_errs):
        T0 = np.max(temperatures) + delta
        T_diff = T0 - temperatures
        valid = T_diff > 0
        if not np.any(valid):
            return np.inf
        try:
            log_x = np.log(x_vals[valid])
            log_T_diff = np.log(T_diff[valid])
            # For log-space, the uncertainty: d(log(x)) ~ x_err / x.
            weights = x_vals[valid] / x_errs[valid]
            coeffs = np.polyfit(log_T_diff, log_x, 1, w=weights)
            fitted = np.polyval(coeffs, log_T_diff)
            chi2 = np.sum(((log_x - fitted) * weights)**2)
            return chi2
        except Exception:
            return np.inf

    result = minimize(lambda d: residual(d, x_values, x_errors), initial_delta, bounds=[(0.1, 3)])
    best_delta = result.x[0] if result.success else initial_delta
    best_T0 = np.max(temperatures) + best_delta

    # Monte Carlo error propagation
    T0_samples = []
    for _ in range(num_samples):
        perturbed_x = np.random.normal(x_values, x_errors)
        res = minimize(lambda d: residual(d, perturbed_x, x_errors), initial_delta, bounds=[(0.1, 3)])
        if res.success:
            T0_samples.append(np.max(temperatures) + res.x[0])
    T0_samples = np.array(T0_samples)
    T0_err = np.std(T0_samples)
    return best_T0, T0_err