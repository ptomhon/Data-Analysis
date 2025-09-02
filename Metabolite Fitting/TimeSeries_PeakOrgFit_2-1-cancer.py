import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# === SETTINGS ===
file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\integrated_data_250819-125332-Pyr70_2.csv"
save_folder = '125332_plot_integrals'  # Folder to save plots

target_peaks = [182.5, 178, 170, 160, 124]
target_peaks_labels = [f'P{peak}' for peak in target_peaks]
method = 'integrals'

# Define peak indices for substrate and products
substratepeak = 2
product1peak = 4  # CO2
product2peak = 3  # HCO3
product3peak = 1  # PYH
product4peak = 0  # LAC
smoothing = False
startPoint = 5

startTimep3 = -2.0  # Start time for product 3
startTimep4 = 0.0  # Start time for product 4

##Substrate settings
#Substrate corresponding to column 1 of .csv file
substrate = '1-13C-Pyruvate'      #Name of substrate

##Product settings
#product_1 corresponding to column 2 of .csv file
product_1 = 'CO2'       #Name of product_1

#product_2 corresponding to column 3 of .csv file
product_2 = 'HC03'       #Name of product_2

#product_3 corresponding to column 4 of .csv file
use_3 = 1       #0 if there is no metabolism and 1 if there is
product_3 = 'PYH'           #Name of product_3

#product_4 corresponding to column 5 of .csv file
use_4 = 1       #0 if there is no metabolism and 1 if there is
product_4 = 'LAC'       #Name of product_4

# === FUNCTIONS ===
def custom_savgol_filter(x, window_length, polyorder, isolation_range=1, ignore_last=0):
    """
    Apply a Savitzky-Golay filter while ignoring isolated zeros.
    Instead of applying smoothing to the entire signal and then replacing the last
    ignore_last indices, this function identifies the nonzero portion of the signal,
    and applies smoothing only up to the (M - ignore_last)-th nonzero element.
    
    Parameters:
      - x: 1D numpy array of data.
      - window_length: Window length for the filter (must be a positive odd integer).
      - polyorder: Polynomial order for the filter.
      - isolation_range: Number of indices on each side to check for isolated zeros.
      - ignore_last: Number of nonzero points at the tail to leave unsmoothed.
      
    Returns:
      - Smoothed numpy array.
    """
    x = np.array(x, dtype=float)
    x_mod = x.copy()
    n = len(x)
    
    # Replace isolated zeros as before.
    for i in range(n):
        if x[i] == 0:
            start = max(0, i - isolation_range)
            end = min(n, i + isolation_range + 1)
            neighbor_indices = [j for j in range(start, end) if j != i]
            if not any(x[j] == 0 for j in neighbor_indices):
                neighbors = [x[j] for j in neighbor_indices if x[j] != 0]
                if neighbors:
                    x_mod[i] = np.mean(neighbors)
    
    # Identify indices of nonzero points.
    nonzero_indices = np.where(x_mod != 0)[0]
    if len(nonzero_indices) == 0:
        # If no nonzero values, just return a full smoothing.
        return savgol_filter(x_mod, window_length=window_length, polyorder=polyorder, mode='mirror')
    
    M = len(nonzero_indices)
    if ignore_last >= M:
        # If ignore_last is too high, skip smoothing altogether.
        return x_mod
    
    # Determine the index up to which to smooth:
    # This is the (M - ignore_last)-th nonzero element.
    smooth_end = nonzero_indices[M - ignore_last] + 1  # +1 to include that index
    
    # Only smooth the portion up to smooth_end.
    if smooth_end < window_length:
        # Not enough points to smooth.
        smoothed = x_mod
    else:
        smoothed_main = savgol_filter(x_mod[:smooth_end], window_length=window_length, polyorder=polyorder, mode='mirror')
        smoothed = np.concatenate([smoothed_main, x_mod[smooth_end:]])
    
    # Ensure no negative values.
    smoothed[smoothed < 0] = 0
    return smoothed

def import_new_peak_data(csv_file, target_peaks, use_smoothing=False, window_length=5, polyorder=2, isolation_range=1, ignore_last=3):
    """
    Imports peak data from a CSV file with the new format:
      - First 3 rows are header rows.
      - Data rows: Column 0: Time; Columns 1-4: Peak Heights; Columns 5-8: Peak Integrals.
    Optionally applies smoothing to the numeric columns if use_smoothing is True.
    Returns a numpy array of shape (num_targets, num_rows, 4) where each target's array has columns:
      [Time, PeakPos, PeakHeight, PeakIntegral]
    """
    df = pd.read_csv(csv_file, skiprows=3, header=None)

    num_targets = len(target_peaks)

    # Dynamically determine column ranges
    height_cols = list(range(1, 1 + num_targets))
    integral_cols = list(range(1 + num_targets, 1 + 2 * num_targets))

    if use_smoothing:
        for col in height_cols + integral_cols:
            df.iloc[:, col] = custom_savgol_filter(
                df.iloc[:, col].values, window_length, polyorder, isolation_range, ignore_last
            )
    # Limit to first 150 rows (adjust if needed)
    df = df.iloc[:150, :]
    time_values = df.iloc[:, 0].values

    all_results = []
    for i in range(num_targets):
        height_col = df.iloc[:, i+1].values
        integral_col = df.iloc[:, i+1+num_targets].values
        peakpos_col = np.full_like(time_values, target_peaks[i], dtype=float)
        processed_array = np.column_stack((time_values, peakpos_col, height_col, integral_col))
        all_results.append(processed_array)
    return np.array(all_results)

def create_filtered_arrays(processed_data_array, target_peaks):
    filtered_data = {}
    for i, processed_array in enumerate(processed_data_array):
        peak_label = f"P{target_peaks[i]}"
        if method == 'heights':
            # Filter based on peak heights
            valid_indices = processed_array[:, 2] > 0
            filtered_data[peak_label] = processed_array[valid_indices]
        elif method == 'integrals':
            # Filter based on peak integrals
            valid_indices = processed_array[:, 3] > 0
            filtered_data[peak_label] = processed_array[valid_indices]
    return filtered_data

def extract_filtered_data(filtered_arrays, peak_label, start_index=0):
    """
    Extracts time and signal data from filtered_arrays for a given peak label.

    Parameters:
        filtered_arrays (dict): Dictionary containing processed NMR peak data.
        peak_label (str): Key corresponding to the target peak (e.g., 'P171').
        start_index (int): Number of initial points to exclude (default: 0).

    Returns:
        tuple: (time_array, signal_array) as numpy float64 arrays.
    """
    if peak_label not in filtered_arrays:
        raise ValueError(f"Peak label '{peak_label}' not found in filtered_arrays.")

    array = filtered_arrays[peak_label]
    time_array = np.array(array[start_index:, 0], dtype=np.float64)
    if method == 'heights':
        signal_array = np.array(array[start_index:, 2], dtype=np.float64)
    elif method == 'integrals':
        signal_array = np.array(array[start_index:, 3], dtype=np.float64)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'heights' or 'integrals'.")
    
    return time_array, signal_array

# === LOAD DATA ===
processed_data_array = import_new_peak_data(file_path, target_peaks, use_smoothing=smoothing)
filtered_arrays = create_filtered_arrays(processed_data_array, target_peaks)

print(processed_data_array.shape)

# Extract signals
tsub, sub = extract_filtered_data(filtered_arrays, target_peaks_labels[substratepeak], start_index=startPoint)
# t1, p1 = extract_filtered_data(filtered_arrays, target_peaks_labels[product1peak], start_index=startPoint)
# t2, p2 = extract_filtered_data(filtered_arrays, target_peaks_labels[product2peak], start_index=startPoint)
t3, p3 = extract_filtered_data(filtered_arrays, target_peaks_labels[product3peak], start_index=startPoint) if use_3 else (np.array([]), np.array([]))
t4, p4 = extract_filtered_data(filtered_arrays, target_peaks_labels[product4peak], start_index=startPoint) if use_4 else (np.array([]), np.array([]))

tsub_max = tsub[-1]
print(sub[1:10])

# # === SETUP MODELS FOR PRODUCT FIT ===
# # --- Define the two-compartment ODE system for CO₂ and bicarbonate ---
# def ode_system(t, y, params, sub_interp):
#     CO2, BIC = y
#     ksubCO, kBCCO, kCOBC, gam1, gam2 = params
#     sub_val = sub_interp(t)
#     dCO2 = ksubCO * sub_val - kBCCO * CO2 + kCOBC * BIC - gam1 * CO2
#     dBIC = kBCCO * CO2 - kCOBC * BIC - gam2 * BIC
#     return [dCO2, dBIC]

# def integrate_model(params, t_measured, sub_interp):
#     t_start = 0
#     t_end = t_measured[-1]
#     t_dense = np.linspace(t_start, t_end, 300)
#     sol = solve_ivp(lambda t, y: ode_system(t, y, params, sub_interp),
#                     t_span=(t_start, t_end),
#                     y0=[0, 0],
#                     t_eval=t_dense,
#                     method='RK45')
#     CO2_interp = interp1d(t_dense, sol.y[0], kind='linear', bounds_error=False, fill_value="extrapolate")
#     BIC_interp = interp1d(t_dense, sol.y[1], kind='linear', bounds_error=False, fill_value="extrapolate")

#     return CO2_interp(t_measured), BIC_interp(t_measured)

# # --- Define the residuals function for least squares fitting ---
# def residuals(params, t_CO, data_CO, t_BC, data_BC, sub_interp):
    
#     constrained_params = np.array([params[0], params[1], params[2], params[3], params[3]]) # Enforce gamBC == gamCO
    
#     model_CO2, _ = integrate_model(constrained_params, t_CO, sub_interp)
#     _, model_BIC = integrate_model(constrained_params, t_BC, sub_interp)
    
#     resid_CO = model_CO2 - data_CO
#     resid_BIC = model_BIC - data_BC
    
#     return np.concatenate([resid_CO, resid_BIC])

# --- Define one-compartment ODE system for additional products ---
def ode_single(t, y, k, gam):
    return k * sub_interp(t) - gam * y

def integrate_single(t_eval, k, gam):
    sol = solve_ivp(lambda t, y: ode_single(t, y, k, gam),
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=[0.0],
                    t_eval=t_eval,
                    method='RK45')
    return sol.y[0]

# --- Define the exponential decay function for substrate flux ---
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

# === FIT SUBSTRATE AND PRODUCTS ===
# Fit the substrate decay using an exponential decay model
init_guess_KL = [max(sub), 100, min(sub)]  # Initial guess for A, tau, C
popt, _ = curve_fit(exp_decay, tsub, sub, p0=init_guess_KL)
y_fit = exp_decay(tsub, *popt)
sub_interp = interp1d(tsub, y_fit, kind='linear', bounds_error=False, fill_value="extrapolate") # Create an interpolation function for substrate flux

# --- Fit CO₂ and HCO₃⁻ ---
# # init_guess = [ksubCO, kBCCO, kCOBC, gam]
# init_guess = [0.0005, 0.001, 0.001, 0.001]
# bounds_lower = [0.00001, 0.001, 0.001, 0.001]
# bounds_upper = [0.1, 0.1, 0.1, 0.1]

# result = least_squares(lambda p: residuals(p, t1, p1, t2, p2, sub_interp),
#                        x0=init_guess, bounds=(bounds_lower, bounds_upper))
# # best_params = result.x
# best_params = np.array([result.x[0], result.x[1], result.x[2], result.x[3], result.x[3]])  # gamBC = gamCO

# --- Fit additional products if they are used ---
if use_3:
    fixed_t3 = np.concatenate((np.array([startTimep3]), t3))
    fixed_p3 = np.concatenate((np.array([0.0]), p3))
    initial_guess_p3 = [0.001, 0.01]
    popt_p3, _ = curve_fit(integrate_single, fixed_t3, fixed_p3, p0=initial_guess_p3,
                       bounds=([0, 0], [np.inf, 0.2]))
    p3_fit, p3_gam = popt_p3

if use_4:
    fixed_t4 = np.concatenate((np.array([startTimep4]), t4))
    fixed_p4 = np.concatenate((np.array([0.0]), p4))
    initial_guess_p4 = [0.00001, 0.01]
    popt_p4, _ = curve_fit(integrate_single, fixed_t4, fixed_p4, p0=initial_guess_p4,
                       bounds=([0, 0], [np.inf, 0.2]))
    p4_fit, p4_gam = popt_p4


# === EXTRAPOLATE MODELS ===
# Extrapolate leucine model from t = -2 to t_max_KL

dense_time_extrap = np.linspace(0, tsub_max, 300)
# model_CO2_dense, _ = integrate_model(best_params, dense_time_extrap, sub_interp) # Extrapolate CO₂ and bicarbonate fits to t_max_KL
# _, model_BIC_dense = integrate_model(best_params, dense_time_extrap, sub_interp) # Extrapolate CO₂ and bicarbonate fits to t_max_KL
if use_3:
    p3_t_extrap = np.linspace(fixed_t3[0], tsub_max, 300)
    p3_model_extrap = integrate_single(p3_t_extrap, p3_fit, p3_gam)  # Extrapolate p3 fit
if use_4:   
    p4_t_extrap = np.linspace(fixed_t4[0], tsub_max, 300)
    p4_model_extrap = integrate_single(p4_t_extrap, p4_fit, p4_gam)  # Extrapolate p4 fit



# # =========================================
# # pH Model and Calculations (Using CO₂/BIC Fits)
# # =========================================
# # Extrapolate the common time grid from 0 to tsub_max for pH-related calculations
# gam1 = best_params[3]
# gam2 = best_params[4]
# epsilon = 1e-10  # Small value to avoid division by zero
# raw_ratio = (model_BIC_dense) / (model_CO2_dense + epsilon)
# adjusted_ratio = raw_ratio

# # Compute weighted average of adjusted_ratio over the time range 0 to 300 seconds
# time_min = 0
# time_max = 300
# mask = (dense_time_extrap >= time_min) & (dense_time_extrap <= time_max)



# ===========================================
# PRINTING RESULTS
# ===========================================
# Print fitted parameters
print("Fitted Parameters for Substrate (Exponential Decay):")
print(f"A: {popt[0]:.2f}, tau: {popt[1]:.2f}, C: {popt[2]:.2f}")
# print("Fitted Parameters for CO₂ and Bicarbonate:")
# print("CO₂ and Bicarbonate Model Parameters:")
# print(f"ksubCO: {best_params[0]:.5f}")
# print(f"kBCCO: {best_params[1]:.5f}")
# print(f"kCOBC: {best_params[2]:.5f}")
# print(f"gamCO: {best_params[3]:.5f}")
# print(f"gamBC: {best_params[4]:.5f}")
if use_3:
    print("Fitted Parameters for Product 3:")
    print(f"Product 3: {p3_fit:.6f}, {p3_gam:.6f}")
if use_4:
    print("Fitted Parameters for Product 4:")
    print(f"Product 4: {p4_fit:.6f}, {p4_gam:.6f}")
# # Calculate the weighted average of the adjusted ratio in the specified time range
# if np.any(mask):
#     weighted_avg = np.trapezoid(adjusted_ratio[mask], dense_time_extrap[mask]) / (dense_time_extrap[mask][-1] - dense_time_extrap[mask][0])
#     print(f"Weighted average of adjusted ratio from {time_min} to {time_max} sec: {weighted_avg:.2f}")
# else:
#     print("No data in the specified time range.")
# print(f"pH from the adjusted ratio weighted average: {np.log10(weighted_avg) + 6.1:.2f}")
# print(f"pH from the bicarbonate/CO2 conversion rates: {np.log10(best_params[1]/best_params[2]) + 6.1:.2f}")



# ============================
# PLOTTING (1x5)
# ============================
fig, axs = plt.subplots(1, 3, figsize=(10, 10))
axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

xmax = 400
xmin = -5
ymax= 50
ymin = -5

ymax2 = 33
ymin2 = ymax2 / 50

fitcolor = 'darkgrey'

# Subplot 1: Ketoleucine decay and exponential fit
axs[0].scatter(tsub, sub/100, color='red', alpha=0.7, label="Substrate data")
axs[0].plot(tsub, y_fit/100, fitcolor, lw=2, label="Model Substrate (Exponential Fit)")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Substrate Signal")
axs[0].set_title("Substrate Decay")
axs[0].set_xlim(xmin, xmax)
axs[0].set_ylim(ymin2, ymax2)
axs[0].legend()

# # Subplot 2: CO₂ accumulation
# axs[1].scatter(t1, p1, color='darkorange', alpha=0.7, label="CO₂ data")
# axs[1].plot(dense_time_extrap, model_CO2_dense, fitcolor, lw=2, label="Model CO₂ (Extrapolated)")
# axs[1].set_xlabel("Time")
# axs[1].set_ylabel("CO₂ Signal")
# axs[1].set_title("CO₂ Accumulation")
# axs[1].set_xlim(xmin, xmax)
# axs[1].set_ylim(ymin, ymax)
# axs[1].legend()

# # Subplot 3: Bicarbonate accumulation
# axs[2].scatter(t2, p2, color='purple', alpha=0.7, label="Bicarbonate data")
# axs[2].plot(dense_time_extrap, model_BIC_dense, fitcolor, lw=2, label="Model Bicarbonate (Extrapolated)")
# axs[2].set_xlabel("Time")
# axs[2].set_ylabel("Bicarbonate Signal")
# axs[2].set_title("Bicarbonate Accumulation")
# axs[2].set_xlim(xmin, xmax)
# axs[2].set_ylim(ymin, ymax)
# axs[2].legend()

# Subplot 4: Product accumulation (Extrapolated)
if use_3:
    axs[1].scatter(t3, p3, color='blue', alpha=0.7, label="Product 3 data")
    axs[1].plot(p3_t_extrap, p3_model_extrap, fitcolor, lw=2, label="Model Product 3 (Extrapolated)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Product 3 Signal")
    axs[1].set_title("Product 3 Accumulation")
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax*3)
    axs[1].legend()

# Subplot 5: Product accumulation (Extrapolated)
if use_4:
    axs[2].scatter(t4, p4, color='blue', alpha=0.7, label="Product 4 data")
    axs[2].plot(p4_t_extrap, p4_model_extrap, fitcolor, lw=2, label="Model Product 4 (Extrapolated)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Product 4 Signal")
    axs[2].set_title("Product 4 Accumulation")
    axs[2].set_ylim(-0.2, 10)
    axs[2].set_xlim(xmin, xmax)
    axs[2].legend()

# # Plot 6: Adjusted ratio (HCO₃⁻/CO₂)
# axs[5].plot(dense_time_extrap, adjusted_ratio, color='teal', lw=2)
# axs[5].set_xlabel("Time (s)")
# axs[5].set_ylabel("Adjusted Ratio")
# axs[5].set_title("Adjusted HCO₃⁻/CO₂ Ratio")
# axs[5].set_xlim(xmin, xmax)
# axs[5].set_ylim(0, 2.5)
# axs[5].grid(True)

# # Plot 7: Instantaneous pH over time
# pH_over_time = np.log10(adjusted_ratio) + 6.1
# axs[6].plot(dense_time_extrap, pH_over_time, color='darkgreen', lw=2)
# axs[6].set_xlabel("Time (s)")
# axs[6].set_ylabel("Estimated pH")
# axs[6].set_title("Estimated pH Over Time")
# axs[6].set_xlim(xmin, xmax)
# axs[6].set_ylim(4.5, 7.0)
# axs[6].grid(True)

# # Plot 8 (optional): Placeholder for additional plots
# axs[7].axis('off')  # Turn off the axis for the last subplot

plt.tight_layout()
plt.show()