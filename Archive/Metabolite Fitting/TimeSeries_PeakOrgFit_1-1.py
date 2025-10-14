import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# ===========================
# Smoothing Functionality
# ===========================
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



# ===========================
# Data Import with Optional Smoothing
# ===========================
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
    if use_smoothing:
        for col in list(range(1, 5)) + list(range(5, 9)):
            df.iloc[:, col] = custom_savgol_filter(df.iloc[:, col].values, window_length, polyorder, isolation_range, ignore_last)
    # Limit to first 150 rows (adjust if needed)
    df = df.iloc[:150, :]
    time_values = df.iloc[:, 0].values
    num_targets = len(target_peaks)
    all_results = []
    for i in range(num_targets):
        height_col = df.iloc[:, i+1].values
        integral_col = df.iloc[:, i+1+num_targets].values
        peakpos_col = np.full_like(time_values, target_peaks[i], dtype=float)
        processed_array = np.column_stack((time_values, peakpos_col, height_col, integral_col))
        all_results.append(processed_array)
    return np.array(all_results)

# ===========================
# Filtering and Plotting Functions
# ===========================
def plot_filtered_peaks(processed_data_array, xlim=None, ylim=None):
    plt.figure(figsize=(10, 6))
    for i, processed_array in enumerate(processed_data_array):
        time_values = processed_array[:, 0]
        height_values = processed_array[:, 2]
        valid_indices = height_values > 0
        filtered_data_array = processed_array[valid_indices]
        plt.plot(filtered_data_array[:, 0], filtered_data_array[:, 2], 'o-', 
                 label=f"Peak {i+1}", alpha=0.7, markerfacecolor='none')
    plt.xlabel("Time (s)")
    plt.ylabel("Peak Height")
    plt.title("Time vs. Peak Height (Filtered Non-Zero)")
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.show()

def create_filtered_arrays(processed_data_array, target_peaks):
    filtered_data = {}
    for i, processed_array in enumerate(processed_data_array):
        peak_label = f"P{target_peaks[i]}"
        valid_indices = processed_array[:, 2] > 0
        filtered_data[peak_label] = processed_array[valid_indices]
    return filtered_data

# ===========================
# User Settings and Data Import
# ===========================
use_smoothed_data = True  # Set to False to use unsmoothed data

# Define target peaks (by ppm) and mapping (order: 178 -> leucine, 171 -> ketoleucine, 160 -> bicarbonate, 124 -> CO₂)
target_peaks = [178, 171, 160, 124]
# file_path = r"G:\Shared drives\Vizma Life Sciences - Projects\Project Management\Wayne State University\KL Yeast Paper\Data\Data Analysis\integrated_run3_peak_data.csv"
file_path = r"D:\WSU\2025-02 Pyruvate Cell paper\Data Analysis\Integrated Data\integrated_data_pyr00-2.csv"
# file_path = r"D:\WSU\2025-02 Ketoleucine Cell paper\Data Analysis\integrated_data_kic002-2.csv"

# Import the data (processed_data_array shape: [num_targets, num_rows, 4])
processed_data_array = import_new_peak_data(file_path, target_peaks, use_smoothing=use_smoothed_data)
filtered_arrays = create_filtered_arrays(processed_data_array, target_peaks)
# Optionally: plot_filtered_peaks(processed_data_array, xlim=(0, 300), ylim=(0, 50))

# ===========================
# Kinetic Fittings and Modeling
# ===========================
START_EXCLUDE_BCCO = 2  # Exclude first N points (for CO₂/Bicarbonate)
START_EXCLUDE_L = 2     # Exclude first N points (for leucine)

# Mapping:
# P171 (target 171) → ketoleucine (for exponential decay fitting)
# P160 (target 160) → bicarbonate
# P124 (target 124) → CO₂
# P178 (target 178) → leucine

# Extract ketoleucine data (P171)
time_KL = np.array(filtered_arrays['P171'][:, 0], dtype=np.float64)[START_EXCLUDE_BCCO:]
data_KL = np.array(filtered_arrays['P171'][:, 2], dtype=np.float64)[START_EXCLUDE_BCCO:]

# Determine maximum time from ketoleucine data for extrapolation
t_max_KL = time_KL[-1]

# Extract CO₂ and bicarbonate data (from P124 and P160 respectively)
time_CO = np.array(filtered_arrays['P124'][:, 0], dtype=np.float64)[START_EXCLUDE_BCCO:]
data_CO = np.array(filtered_arrays['P124'][:, 2], dtype=np.float64)[START_EXCLUDE_BCCO:]

time_BC = np.array(filtered_arrays['P160'][:, 0], dtype=np.float64)[START_EXCLUDE_BCCO:]
data_BC = np.array(filtered_arrays['P160'][:, 2], dtype=np.float64)[START_EXCLUDE_BCCO:]

# --- Fit ketoleucine to exponential decay ---
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

init_guess_KL = [max(data_KL), 10, min(data_KL)]
popt_KL, _ = curve_fit(exp_decay, time_KL, data_KL, p0=init_guess_KL)
KL_fit = exp_decay(time_KL, *popt_KL)

# Create an interpolation function for ketoleucine flux
KL_interp = interp1d(time_KL, KL_fit, kind='linear', bounds_error=False, fill_value="extrapolate")

# --- Define the two-compartment ODE system for CO₂ and bicarbonate ---
def ode_system(t, y, params, KL_interp):
    CO2, BIC = y
    kkLCO, kBCCO, kCOBC, gam1, gam2 = params
    KL_val = KL_interp(t)
    dCO2 = kkLCO * KL_val - kBCCO * CO2 + kCOBC * BIC - gam1 * CO2
    dBIC = kBCCO * CO2 - kCOBC * BIC - gam2 * BIC
    return [dCO2, dBIC]

def integrate_model(params, t_measured, KL_interp):
    t_start = 0
    t_end = t_measured[-1]
    t_dense = np.linspace(t_start, t_end, 300)
    sol = solve_ivp(lambda t, y: ode_system(t, y, params, KL_interp),
                    t_span=(t_start, t_end),
                    y0=[0, 0],
                    t_eval=t_dense,
                    method='RK45')
    CO2_interp = interp1d(t_dense, sol.y[0], kind='linear', bounds_error=False, fill_value="extrapolate")
    BIC_interp = interp1d(t_dense, sol.y[1], kind='linear', bounds_error=False, fill_value="extrapolate")
    return CO2_interp(t_measured), BIC_interp(t_measured)

def residuals(params, t_CO, data_CO, t_BC, data_BC, KL_interp):
    model_CO2, _ = integrate_model(params, t_CO, KL_interp)
    _, model_BIC = integrate_model(params, t_BC, KL_interp)
    resid_CO = model_CO2 - data_CO
    resid_BIC = model_BIC - data_BC
    return np.concatenate([resid_CO, resid_BIC])

# Fit CO₂/Bicarbonate model
init_guess = [0.001, 0.01, 0.01, 0.01, 0.01]
bounds_lower = [0, 0, 0, 0.0001, 0.0001]
bounds_upper = [np.inf, np.inf, np.inf, np.inf, np.inf]
result = least_squares(lambda p: residuals(p, time_CO, data_CO, time_BC, data_BC, KL_interp),
                       x0=init_guess, bounds=(bounds_lower, bounds_upper))
best_params = result.x
print("Fitted Parameters:")
print(f"  KL_decay = {popt_KL[1]:.2f}")
print(f"  kkLCO = {best_params[0]:.4g}")
print(f"  kBCCO = {best_params[1]:.4g}")
print(f"  kCOBC = {best_params[2]:.4g}")
print(f"  gam1  = {best_params[3]:.4g}")
print(f"  gam2  = {best_params[4]:.4g}")

# Extrapolate CO₂ and bicarbonate fits to t_max_KL
dense_time_extrap = np.linspace(0, t_max_KL, 300)
model_CO2_dense, _ = integrate_model(best_params, dense_time_extrap, KL_interp)
_, model_BIC_dense = integrate_model(best_params, dense_time_extrap, KL_interp)

# ============================
# Leucine Accumulation Model
# ============================
# Extract leucine data (P178)
time_L = np.array(filtered_arrays['P178'][:, 0], dtype=np.float64)[START_EXCLUDE_L:]
data_L = np.array(filtered_arrays['P178'][:, 2], dtype=np.float64)[START_EXCLUDE_L:]
# Build fixed time vector starting at -2 and then using leucine data time points
fixed_time = np.concatenate((np.array([-2.0]), time_L))
fixed_data = np.concatenate((np.array([0.0]), data_L))

def ode_leucine(t, y, kLL, gam):
    return kLL * KL_interp(t) - gam * y

def integrate_leucine(t_eval, kLL, gam):
    sol = solve_ivp(lambda t, y: ode_leucine(t, y, kLL, gam),
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=[0.0],
                    t_eval=t_eval,
                    method='RK45')
    return sol.y[0]

# Fit leucine accumulation model using the fixed time vector
initial_guess_L = [0.001, 0.00001]
popt_L, _ = curve_fit(integrate_leucine, fixed_time, fixed_data, p0=initial_guess_L,
                       bounds=([0, 0], [np.inf, 0.2]))
kLL_fit, gam_fit = popt_L
print(f"  kLL = {kLL_fit:.6f}")
print(f"  gam3  = {gam_fit:.6f}")

# Extrapolate leucine model from t = -2 to t_max_KL
t_fit_extrap = np.linspace(fixed_time[0], t_max_KL, 300)
L_model_extrap = integrate_leucine(t_fit_extrap, kLL_fit, gam_fit)

# =========================================
# pH Model and Calculations (Using CO₂/BIC Fits)
# =========================================
# Extrapolate the common time grid from 0 to t_max_KL for pH-related calculations
t_common = np.linspace(0, t_max_KL, 500)
model_CO2_common, model_BIC_common = integrate_model(best_params, t_common, KL_interp)
gam1 = best_params[3]
gam2 = best_params[4]
epsilon = 1e-8
raw_ratio = model_BIC_common / (model_CO2_common + epsilon)
adjusted_ratio = raw_ratio * (gam1 / gam2)

# Compute weighted average of adjusted_ratio over the time range 0 to 300 seconds
time_min = 0
time_max = 300
mask = (t_common >= time_min) & (t_common <= time_max)
if np.any(mask):
    weighted_avg = np.trapz(adjusted_ratio[mask], t_common[mask]) / (t_common[mask][-1] - t_common[mask][0])
    print(f"Weighted average of adjusted ratio from {time_min} to {time_max} sec: {weighted_avg:.2f}")
else:
    print("No data in the specified time range.")

print(f"pH from the adjusted ratio weighted average: {np.log10(weighted_avg) + 6.1:.2f}")
print(f"pH from the bicarbonate/CO2 conversion rates: {np.log10(best_params[1]/best_params[2]) + 6.1:.2f}")

# ============================
# PLOTTING (1x5)
# ============================
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

xmax = 400
xmin = -5
ymax= 180
ymin = -5

# Subplot 1: Ketoleucine decay and exponential fit
axs[0].scatter(time_KL, data_KL/100, color='red', alpha=0.7, label="Pyruvate data")
axs[0].plot(time_KL, KL_fit/100, color='blue', lw=2, label="Exp. Decay Fit")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Pyruvate Signal")
axs[0].set_title("Pyruvate Decay")
axs[0].set_xlim(xmin, xmax)
axs[0].legend()

# Subplot 2: CO₂ accumulation
axs[1].scatter(time_CO, data_CO, color='blue', alpha=0.7, label="CO₂ data")
axs[1].plot(dense_time_extrap, model_CO2_dense, color='green', lw=2, label="Model CO₂ (Extrapolated)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("CO₂ Signal")
axs[1].set_title("CO₂ Accumulation")
axs[1].set_xlim(xmin, xmax)
axs[1].set_ylim(ymin, ymax)
axs[1].legend()

# Subplot 3: Bicarbonate accumulation
axs[2].scatter(time_BC, data_BC, color='purple', alpha=0.7, label="Bicarbonate data")
axs[2].plot(dense_time_extrap, model_BIC_dense, color='orange', lw=2, label="Model Bicarbonate (Extrapolated)")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Bicarbonate Signal")
axs[2].set_title("Bicarbonate Accumulation")
axs[2].set_xlim(xmin, xmax)
axs[2].set_ylim(ymin, ymax)
axs[2].legend()

# Subplot 4: Adjusted ratio (Bicarbonate/CO₂)
axs[3].plot(t_common, adjusted_ratio, 'k-', lw=2)
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Adjusted Ratio\n(BIC/CO₂ * gam1/gam2)")
axs[3].set_title("Adjusted Bicarbonate/CO₂ Ratio")
axs[3].set_xlim(xmin, xmax)

# Subplot 5: Leucine accumulation (Extrapolated)
axs[4].scatter(time_L, data_L, color='blue', alpha=0.7, label="Pyruvate hydrate data")
axs[4].plot(t_fit_extrap, L_model_extrap, color='green', lw=2, label="Model Hydrate (Extrapolated)")
axs[4].set_xlabel("Time")
axs[4].set_ylabel("Hydrate Signal")
axs[4].set_title("Hydrate Accumulation")
axs[4].set_xlim(xmin, xmax)
axs[4].set_ylim(ymin, ymax)
axs[4].legend()

plt.tight_layout()
plt.show()

# ============================
# SAVING SUBPLOTS AS PDFs
# ============================

import os
import re

# Prompt the user for the folder name to save subplots
save_folder = "run6_plot_peakheights"

# Determine the directory where the CSV file is located and create the new folder there
csv_directory = os.path.dirname(file_path)
output_dir = os.path.join(csv_directory, save_folder)
os.makedirs(output_dir, exist_ok=True)

# Helper function to sanitize subplot titles for safe filenames
def sanitize_title(title):
    title = title.replace(" ", "_")
    return re.sub(r'[^\w_]', '', title)

# Force the figure to render so that we can extract the tight bounding boxes
plt.draw()
renderer = fig.canvas.get_renderer()

# Save each subplot (axis) in the figure as an individual PDF file
for ax in axs:
    # Use the subplot's title or default to "subplot" if title is empty
    subplot_title = ax.get_title() or "subplot"
    safe_title = sanitize_title(subplot_title)
    filename = os.path.join(output_dir, f"{save_folder}_{safe_title}.pdf")
    
    # Get the tight bounding box for this axis in figure coordinates
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    
    # Save the subplot area as a PDF file
    fig.savefig(filename, bbox_inches=bbox)
    print(f"Saved subplot '{subplot_title}' to: {filename}")