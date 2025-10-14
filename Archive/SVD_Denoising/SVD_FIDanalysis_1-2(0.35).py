import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from scipy.signal import find_peaks
import time

import os
import matplotlib.pyplot as plt
import numpy as np
import time

ppmadjust = 100  # ppm adjustment for frequency axis
b1freq = 3.82882  # frequency in Hz for the 0.35T MRI scanner

# --------------------------
# Load FID data from CSV
# --------------------------
def load_fid_data(file_path):
    """
    Load FID data from a CSV file where:
       - Column 1: Time in milliseconds
       - Column 2: Real component
       - Column 3: Imaginary component
    """
    data = np.loadtxt(file_path, delimiter=',')
    time_arr = data[:, 0] / 1000000  # convert ms to seconds
    real_part = data[:, 1]
    imag_part = data[:, 2]
    fid = real_part + 1j * imag_part
    return time_arr, fid

# ----------------------------------------------------
# SVD-based denoising using Hankel matrix and anti-diagonal averaging
# ----------------------------------------------------
def average_anti_diagonals(M):
    """
    Average the elements along the anti-diagonals of a square matrix M.
    """
    N, _ = M.shape
    M_new = np.copy(M)
    for d in range(0, 2 * N - 1):
        indices = [(i, d - i) for i in range(max(0, d - (N - 1)), min(d + 1, N))]
        if indices:
            avg = np.mean([M[i, j] for (i, j) in indices])
            for (i, j) in indices:
                M_new[i, j] = avg
    return M_new

def denoise_fid(fid_segment, L=500, k=4, n_iter=5):
    """
    Denoise a complex FID segment using iterative SVD-based Hankel matrix denoising.
    
    Parameters:
        fid_segment (np.ndarray): 1D complex array (length n_denoise).
        L (int): Number of rows for the Hankel matrix (default 500).
                 For a square Hankel matrix, n_denoise should equal 2*L - 1.
        k (int): Number of singular values to retain (default 4).
        n_iter (int): Number of iterations for refinement (default 5).
    
    Returns:
        np.ndarray: Denoised FID segment (length = 2*L - 1).
    """
    N = len(fid_segment)
    M = N - L + 1  
    H = hankel(fid_segment[:L], fid_segment[L-1:])
    
    H_denoise = np.copy(H)
    for _ in range(n_iter):
        U, s, Vh = svd(H_denoise, full_matrices=False, lapack_driver='gesdd')
        U_trunc = U[:, :k]
        s_trunc = s[:k]
        Vh_trunc = Vh[:k, :]
        H_trunc = U_trunc @ np.diag(s_trunc) @ Vh_trunc
        H_denoise = average_anti_diagonals(H_trunc)
    
    first_col = H_denoise[:, 0]      
    last_row = H_denoise[1:, -1]       
    denoised_segment = np.concatenate([first_col, last_row])
    
    return denoised_segment

# ----------------------------------------------------
# Zero-filling and exponential apodization function
# ----------------------------------------------------
def zero_fill_and_apodize(signal, target_length, dt, T2_apod=0.1):
    """
    Zero-fill a 1D signal to a target length and apply exponential apodization.
    
    Parameters:
        signal (np.ndarray): 1D input signal.
        target_length (int): Desired length after zero-filling.
        dt (float): Sampling interval (seconds).
        T2_apod (float): Decay constant for exponential apodization.
    
    Returns:
        tuple: (processed_signal, time_vector)
    """
    current_length = len(signal)
    if target_length < current_length:
        raise ValueError("target_length must be >= length of the input signal.")
    padded_signal = np.pad(signal, (0, target_length - current_length), mode='constant', constant_values=0)
    t = np.arange(target_length) * dt
    apodization_filter = np.exp(-T2_apod * t)
    processed_signal = padded_signal * apodization_filter
    return processed_signal, t

# ----------------------------------------------------
# Phase correction function
# ----------------------------------------------------
def apply_phase_correction(fid, phase_angle):
    """Apply a phase correction to the FID.
       - phase_angle: Phase shift in degrees
    """
    phase_rad = np.deg2rad(phase_angle)  # Convert degrees to radians
    return fid * np.exp(-1j * phase_rad)  # Apply phase shift

# ----------------------------------------------------
# Baseline correction function
# ----------------------------------------------------
def correct_baseline(fft_data, n):
    """
    Corrects the FFT baseline by subtracting the average of the first n and last n data points.

    Parameters:
        fft_data (np.ndarray): Array of FFT data (real or complex values).
        n (int): Number of points from the beginning and end to average.

    Returns:
        np.ndarray: Baseline-corrected FFT data.
    """
    baseline = np.mean(np.concatenate((fft_data[:n], fft_data[-n:])))
    return fft_data - baseline

# # ----------------------------------------------------
# # FFT quality check function
# # ----------------------------------------------------
def check_fft_validity(fid_segment, dt, T2_apod, phase_corr_angle, allowed_ppms, ppm_threshold, target_length=65536, integration_threshold=0.01, edge_epsilon_ratio=0.001):
    """
    Process the denoised FID segment and check the FFT for three types of artifacts:
      1. Negative peaks in the FFT.
      2. Peaks (above a relative threshold) that are not within ±ppm_threshold of any allowed_ppm.
      3. Peaks whose integrated area (from edge to edge, where the edge is defined as when the signal falls to a small fraction 
         of the peak height) is less than integration_threshold.
    
    Parameters:
        fid_segment (np.ndarray): 1D FID segment (time domain).
        dt (float): Sampling interval.
        T2_apod (float): Exponential apodization decay constant.
        phase_corr_angle (float): Phase correction angle in degrees.
        allowed_ppms (list): List of allowed chemical shifts (e.g. [170, 178, 160, 124]).
        ppm_threshold (float): Allowed deviation in ppm from each allowed value.
        target_length (int): Length for zero-filling.
        integration_threshold (float): Minimum required integrated area for each peak.
        edge_epsilon_ratio (float): Fraction of the peak height used to define the peak edge.
    
    Returns:
        bool: True if FFT passes all checks, False otherwise.
    """
    # Process the signal (zero-fill, apodize, phase-correct)
    processed_signal, t = zero_fill_and_apodize(fid_segment, target_length, dt, T2_apod)
    proc_signal = apply_phase_correction(processed_signal, phase_corr_angle)
    
    # Compute FFT and frequency axis
    n = len(proc_signal)
    fft_signal = np.fft.fft(proc_signal)
    freqs = np.fft.fftfreq(n, d=dt)
    fft_shifted = np.fft.fftshift(fft_signal)
    freqs_shifted = np.fft.fftshift(freqs)
    # Reverse arrays so that the highest positive frequency appears on the left
    fft_plot = fft_shifted[::-1]
    freqs_plot = freqs_shifted[::-1]
    
    # Baseline correction on the real part
    fft_plot_real = correct_baseline(fft_plot.real, 100)
    
    # Convert frequency axis (Hz) to chemical shift in ppm using your formula
    ppm_axis = -((freqs_plot) / b1freq) + ppmadjust
    
    # Condition 1: Check for negative peaks.
    if np.min(fft_plot_real) < -5:
        print("Quality check failed: Negative peaks detected (min value = {:.4f}).".format(np.min(fft_plot_real)))
        return False
    
    # Condition 2: Check that all significant peaks lie within allowed ppm windows.
    height_threshold = 1 #0.01 * np.max(fft_plot_real)
    prominence_threshold = 4 #0.02 * np.max(fft_plot_real)
    peaks, properties = find_peaks(fft_plot_real, height=height_threshold, prominence=prominence_threshold)
    if len(peaks) == 0:
        print("Quality check failed: No peaks found with height threshold {:.4f} and prominence threshold {:.4f}.".format(height_threshold, prominence_threshold))
        return False
    for peak_idx in peaks:
        peak_ppm = ppm_axis[peak_idx]
        if not any(abs(peak_ppm - allowed) <= ppm_threshold for allowed in allowed_ppms):
            print("Quality check failed: Peak at {:.2f} ppm is outside allowed windows (±{} ppm).".format(peak_ppm, ppm_threshold))
            return False
        
    # Condition 3: For each detected peak, integrate from edge to edge.
    # The edge is defined as the index where the signal falls below (edge_epsilon_ratio * peak height).
    for peak_idx in peaks:
        peak_val = fft_plot_real[peak_idx]
        epsilon = peak_val * edge_epsilon_ratio
        
        # Find left edge: move left until the signal falls below epsilon
        left_index = peak_idx
        while left_index > 0 and fft_plot_real[left_index] > epsilon:
            left_index -= 1
        
        # Find right edge: move right until the signal falls below epsilon
        right_index = peak_idx
        while right_index < len(fft_plot_real) - 1 and fft_plot_real[right_index] > epsilon:
            right_index += 1
        
        # Compute the integral over the region between the detected edges.
        peak_integral = np.trapezoid(fft_plot_real[left_index:right_index+1], ppm_axis[left_index:right_index+1])
        print("Peak at {:.2f} ppm: height = {:.4f}, integral = {:.4f}".format(ppm_axis[peak_idx], peak_val, peak_integral))
        if peak_integral < integration_threshold:
            print("Quality check failed: Integrated area for peak at {:.2f} ppm is too low ({:.4f} < threshold {:.4f}).".format(ppm_axis[peak_idx], peak_integral, integration_threshold))
            return False
    
    return True

# ----------------------------------------------------
# Plotting function for FIDs and FFTs
# ----------------------------------------------------
def plot_denoised_results(time, noisy_signal, denoised_signal, mode="fid"):
    """
    Plot either the time-domain FID (real and imaginary parts)
    or the FFT (real part only) for both the noisy and denoised signals.
    
    For FFT plots, the frequency axis is arranged from +2500 Hz (left) to -2500 Hz (right).
    
    Parameters:
        time (np.ndarray): 1D time axis.
        noisy_signal (np.ndarray): Complex noisy signal.
        denoised_signal (np.ndarray): Complex denoised signal.
        mode (str): 'fid' for time-domain plots, 'fft' for frequency-domain plots.
    """
    if mode.lower() == "fid":
        plt.figure(figsize=(10, 5))
        plt.plot(time, noisy_signal.real, label="Noisy FID - Real", color="blue")
        plt.plot(time, noisy_signal.imag, label="Noisy FID - Imag", color="cyan")
        plt.plot(time, denoised_signal.real, label="Denoised FID - Real", color="red")
        plt.plot(time, denoised_signal.imag, label="Denoised FID - Imag", color="magenta")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Time-Domain FID")
        plt.legend()
        plt.show()
    elif mode.lower() == "fft":
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        n = len(noisy_signal)
        # Compute FFTs for both signals.
        fft_noisy = np.fft.fft(noisy_signal)
        fft_denoised = np.fft.fft(denoised_signal)
        # Frequency axis (in Hz)
        freqs = np.fft.fftfreq(n, d=dt)
        # Shift the FFT and frequency axis so that zero frequency is centered.
        fft_noisy_shifted = np.fft.fftshift(fft_noisy)
        fft_denoised_shifted = np.fft.fftshift(fft_denoised)
        freqs_shifted = np.fft.fftshift(freqs)
        # Reverse arrays so that the highest positive frequency appears on the left.
        fft_noisy_plot = fft_noisy_shifted[::-1]
        fft_denoised_plot = fft_denoised_shifted[::-1]
        freqs_plot = freqs_shifted[::-1]
        
        # Baseline correction on the real parts
        fft_noisy_plot = correct_baseline(fft_noisy_plot.real, 100)
        fft_denoised_plot = correct_baseline(fft_denoised_plot.real, 100)
        
        # Convert frequency axis (Hz) to chemical shift in ppm using your formula:
        ppm_axis = -((freqs_plot) / b1freq) + ppmadjust
        
        plt.figure(figsize=(10, 5))
        plt.plot(ppm_axis, fft_noisy_plot, label="Noisy FFT (Real)", color="blue")
        plt.plot(ppm_axis, fft_denoised_plot, label="Denoised FFT (Real)", color="red")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Amplitude")
        plt.title("FFT of FID (Real Part)")
        plt.legend()
        plt.gca().invert_xaxis()  # Reverse x-axis direction
        plt.show()
    else:
        raise ValueError("mode must be either 'fid' or 'fft'")

# ----------------------------------------------------
# Main processing functions (single and multiple folder options)
# ----------------------------------------------------
# Single folder processing
def process_single_folder(folder, base_path,
                          L=2000, T2_apod=5, phase_corr_angle=10,
                          initial_k=4, allowed_ppms=None, ppm_threshold=1,
                          target_length=8192, n_iter=2):
    """
    Process a single folder:
      - Loads the data,
      - Denoises using iterative SVD (reducing k until the FFT quality check passes),
      - Zero-fills, apodizes, and phase-corrects,
      - Plots the time-domain and frequency-domain results.
      
    Parameters:
        folder (int): The folder number to process.
        base_path (str): Base directory path where folder subdirectories reside.
        L (int): Number of rows for Hankel matrix construction.
        T2_apod (float): Apodization decay constant.
        phase_corr_angle (float): Phase correction angle (degrees).
        initial_k (int): Starting k value for denoising.
        allowed_ppms (list): Allowed chemical shifts (default: [178, 170, 160, 124]).
        ppm_threshold (float): Allowed deviation in ppm.
        target_length (int): Zero-filled length for processing.
        n_iter (int): Number of iterations for the SVD denoising loop.
    """
    start_time = time.time()
    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    
    # Construct file path.
    file_path = base_path + f"/{folder}/fid.csv"
    print(f"Processing folder {folder} at {file_path}")
    
    # Load full FID (expected length = 8192)
    time_full, fid_full = load_fid_data(file_path)
    dt = time_full[1] - time_full[0]
    
    # --- Denoising on a subset ---
    n_denoise = 2 * L - 1  # For a square Hankel matrix
    fid_segment = fid_full[:n_denoise]
    
    k = initial_k
    # Loop: decrease k until the FFT of the denoised segment passes the quality check.
    while k >= 1:
        print(f"Trying denoising with k = {k} ...")
        denoised_segment = denoise_fid(fid_segment, L=L, k=k, n_iter=n_iter)
        if check_fft_validity(denoised_segment, dt, T2_apod, phase_corr_angle, allowed_ppms, ppm_threshold):
            print(f"FFT quality check passed with k = {k}.")
            break
        else:
            print(f"FFT quality check failed with k = {k}. Reducing k...")
            k -= 1
            if k < 1:
                print("Warning: k reached below 1. Proceeding with the current result.")
                break
    
    # --- Zero-filling and exponential apodization ---
    noisy_processed, new_time = zero_fill_and_apodize(fid_full, target_length, dt, T2_apod)
    denoised_processed, _ = zero_fill_and_apodize(denoised_segment, target_length, dt, T2_apod)
    
    # Apply phase correction.
    noisy_processed = apply_phase_correction(noisy_processed, phase_corr_angle)
    denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)

    end_time = time.time()
    
    # --- Plot the results ---
    plot_denoised_results(new_time, noisy_processed, denoised_processed, mode="fid")
    plot_denoised_results(new_time, noisy_processed, denoised_processed, mode="fft")
    
    elapsed_time = end_time - start_time

    print("Single folder processing complete.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

# ====================================================
# Multiple folder processing with learning feature
# ====================================================
def process_multiple_folders(start_folder, end_folder, base_path,
                             n_threshold, initial_k=4, L=2000,
                             T2_apod=5, phase_corr_angle=10,
                             allowed_ppms=None, ppm_threshold=1,
                             target_length=65536, n_iter=2,
                             target_peaks=None, metabolite_names=None,
                             tolerance=1, output_csv=None, time_interval=3.5):
    """
    Process a range of folders and for each folder:
      - Loads and processes the FID (SVD denoising, zero-filling, apodization, and phase correction),
      - Computes the FFT from the already processed denoised data,
      - Dynamically defines integration windows for each detected peak by scanning outwards from the peak 
        until the signal reaches near the baseline (defined as the median of the signal). The window stops 
        when the signal difference from baseline is less than 5% of the peak amplitude above baseline.
        For adjacent peaks, if the computed windows would overlap, the common boundary is set to the index 
        of the minimum signal between the peaks.
      - Computes the peak height and integral over these windows,
      - Matches these to user-defined target peaks (in ppm) within a given tolerance,
      - Saves the integrated data for each folder.
    
    Parameters:
        start_folder (int): Starting folder number.
        end_folder (int): Ending folder number.
        base_path (str): Base directory where folder subdirectories reside.
        n_threshold (int): Number of consecutive folders with a lower optimal k required to update the initial k.
        initial_k (int): Starting k value.
        L (int): Number of rows for Hankel matrix construction.
        T2_apod (float): Apodization decay constant.
        phase_corr_angle (float): Phase correction angle in degrees.
        allowed_ppms (list): Allowed chemical shifts for quality check (default: [178, 170, 160, 124]).
        ppm_threshold (float): Allowed deviation in ppm.
        target_length (int): Zero-filled length for processing.
        n_iter (int): Number of iterations for SVD denoising.
        target_peaks (list): Target peak positions (in ppm). Default: [178, 171, 160, 124].
        metabolite_names (list): Metabolite names corresponding to target peaks.
                                 Default: ["alanine", "pyruvate", "bicarbonate", "CO2"].
        tolerance (float): Tolerance (in ppm) for matching integrated peaks.
        output_csv (str): Full file path to save the CSV data.
        time_interval (float): Time increment between folders.
    """
    import os, csv
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.integrate import simpson

    loop_start_time = time.time()

    # Set default values.
    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    if target_peaks is None:
        target_peaks = [178, 171, 160, 124]
    if metabolite_names is None:
        metabolite_names = ["alanine", "pyruvate", "bicarbonate", "CO2"]

    # New helper functions to determine integration boundaries based on baseline.
    def baseline_left_edge(signal, peak_idx, fraction=0.05):
        baseline = np.median(signal)
        i = peak_idx
        while i > 0 and (signal[i] - baseline) > fraction * (signal[peak_idx] - baseline):
            i -= 1
        return i

    def baseline_right_edge(signal, peak_idx, fraction=0.05):
        baseline = np.median(signal)
        i = peak_idx
        while i < len(signal) - 1 and (signal[i] - baseline) > fraction * (signal[peak_idx] - baseline):
            i += 1
        return i

    current_initial_k = initial_k
    consecutive_decreased_count = 0
    results = []  # One row per folder.

    for folder in range(start_folder, end_folder + 1):
        
        start_time = time.time()

        print(f"\nProcessing folder {folder} with initial k = {current_initial_k}")
        file_path = os.path.join(base_path, str(folder), "fid.csv")
        try:
            time_full, fid_full = load_fid_data(file_path)
        except Exception as e:
            print(f"Failed to load data for folder {folder}: {e}")
            continue

        dt = time_full[1] - time_full[0]
        n_denoise = 2 * L - 1
        fid_segment = fid_full[:n_denoise]

        k = current_initial_k
        optimal_k = k  # store optimal k for this folder
        while k >= 1:
            print(f"Folder {folder}: Trying denoising with k = {k} ...")
            denoised_segment = denoise_fid(fid_segment, L=L, k=k, n_iter=n_iter)
            if check_fft_validity(denoised_segment, dt, T2_apod, phase_corr_angle,
                                  allowed_ppms, ppm_threshold):
                print(f"Folder {folder}: FFT quality check passed with k = {k}.")
                optimal_k = k
                break
            else:
                print(f"Folder {folder}: FFT quality check failed with k = {k}. Reducing k...")
                k -= 1
                if k < 1:
                    print(f"Folder {folder}: Warning: k reached below 1. Proceeding with the current result.")
                    optimal_k = k
                    break
        print(f"Folder {folder}: Optimal k determined as {optimal_k}.")

        # Process FID: zero-fill, apodize, and phase-correct.
        noisy_processed, new_time = zero_fill_and_apodize(fid_full, target_length, dt, T2_apod)
        denoised_processed, _ = zero_fill_and_apodize(denoised_segment, target_length, dt, T2_apod)
        noisy_processed = apply_phase_correction(noisy_processed, phase_corr_angle)
        denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)
        print(f"Folder {folder}: Processing complete.")

        # ----- Integration Section (inside the loop) -----
        n = len(denoised_processed)
        fft_data = np.fft.fft(denoised_processed)
        freqs = np.fft.fftfreq(n, d=dt)
        fft_data_shifted = np.fft.fftshift(fft_data)
        freqs_shifted = np.fft.fftshift(freqs)
        fft_data_plot = fft_data_shifted[::-1]
        freqs_plot = freqs_shifted[::-1]
        # Convert frequency (Hz) to chemical shift (ppm).
        ppm_axis = -((freqs_plot) / b1freq) + ppmadjust
        signal = fft_data_plot.real

        # Identify peaks with a low threshold.
        peak_threshold = 0.005 * np.max(signal)
        peak_prominence = 0.01 * np.max(signal)
        peaks, _ = find_peaks(signal, height=peak_threshold, prominence=peak_prominence, distance=20)
        print(f"Folder {folder}: Detected {len(peaks)} peaks via find_peaks.")

        if len(peaks) == 0:
            print(f"Folder {folder}: No peaks detected. Skipping integration.")
            continue

        # Compute integration windows for each peak using the baseline method.
        peak_windows = []
        for peak in peaks:
            left_edge = baseline_left_edge(signal, peak, fraction=0.05)
            right_edge = baseline_right_edge(signal, peak, fraction=0.05)
            peak_windows.append((left_edge, right_edge))

        # For adjacent peaks, if the windows overlap, define the common boundary as the index of the minimum between them.
        for i in range(len(peak_windows) - 1):
            if peak_windows[i][1] > peak_windows[i+1][0]:
                overlapping_region = signal[peaks[i]:peaks[i+1]+1]
                boundary = peaks[i] + np.argmin(overlapping_region)
                peak_windows[i] = (peak_windows[i][0], boundary)
                peak_windows[i+1] = (boundary, peak_windows[i+1][1])

        # Compute integration for each peak.
        peak_data = []
        for idx, peak in enumerate(peaks):
            left_edge, right_edge = peak_windows[idx]
            position = ppm_axis[peak]
            height = signal[peak]
            integral = simpson(signal[left_edge:right_edge+1], ppm_axis[left_edge:right_edge+1])
            peak_data.append({"position_converted": position,
                              "height": height,
                              "integral": integral,
                              "left_edge": ppm_axis[left_edge],
                              "right_edge": ppm_axis[right_edge]})
            print(f"Folder {folder}:")
            print(f"  Peak at {position:.2f} ppm")
            print(f"  Height = {height:.4f}")
            print(f"  Integral = {integral:.4f}")
            print(f"  Integration range (ppm): [{ppm_axis[left_edge]:.2f}, {ppm_axis[right_edge]:.2f}]")

        # Match each target peak (metabolite) with the detected peaks.
        target_heights = []
        target_integrals = []
        for target in target_peaks:
            match_found = False
            for peak in peak_data:
                if abs(peak["position_converted"] - target) <= tolerance:
                    target_heights.append(peak["height"])
                    target_integrals.append(peak["integral"])
                    match_found = True
                    break
            if not match_found:
                target_heights.append(0)
                target_integrals.append(0)

        time_point = (folder - 1) * time_interval
        results.append([time_point] + target_heights + target_integrals)
        print(f"Folder {folder}: Integrated data - Heights {target_heights}, Integrals {target_integrals}")
        # ----- End Integration Section -----

        # Learning feature: update initial k if consecutive folders show a lower optimal k.
        if optimal_k < current_initial_k:
            consecutive_decreased_count += 1
        else:
            consecutive_decreased_count = 0
        if consecutive_decreased_count >= n_threshold:
            print(f"Learning update: Updating initial k from {current_initial_k} to {optimal_k} after "
                  f"{consecutive_decreased_count} consecutive folders with lower k.")
            current_initial_k = current_initial_k - 1 if current_initial_k > 1 else 1
            consecutive_decreased_count = 0

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing complete in {elapsed_time:.2f} seconds.")

    print("Multiple folder processing complete.")

    # ----- Saving Section (after loop) -----
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        header1 = [""] + ["Height"] * len(target_peaks) + ["Integral"] * len(target_peaks)
        header2 = ["Time"] + [str(tp) for tp in target_peaks] + [str(tp) for tp in target_peaks]
        header3 = [""] + metabolite_names + metabolite_names
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header1)
            writer.writerow(header2)
            writer.writerow(header3)
            for row in results:
                writer.writerow(row)
        print(f"Integrated peak data saved to '{output_csv}' successfully!")
    
    loop_end_time = time.time()
    loop_elapsed_time = loop_end_time - loop_start_time
    print(f"Total elapsed time: {loop_elapsed_time:.2f} seconds")

# ====================================================
# Code for plotting multiple folders in a single window
#

# ====================================================
#  --- Helper function to plot on a given axis ---
# ====================================================
def plot_denoised_on_ax(ax, new_time, noisy_processed, denoised_processed, mode, folder, xlim=None, ylim=None, plot_noisy_fft=True):
    """
    Plot the processed data on the provided axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        new_time (np.ndarray): Time axis (after zero-filling).
        noisy_processed (np.ndarray): Processed noisy FID.
        denoised_processed (np.ndarray): Processed denoised FID.
        mode (str): 'fid' or 'fft' to choose time- or frequency-domain plot.
        folder (int): Folder number (for title labeling).
        xlim (tuple, optional): x-axis limits as (xmin, xmax). Default is None.
        ylim (tuple, optional): y-axis limits as (ymin, ymax). Default is None.
        plot_noisy_fft (bool): In FFT mode, if True plot both noisy and denoised curves,
                               otherwise plot only the denoised FFT.
    """
    if mode.lower() == "fid":
        ax.plot(new_time, noisy_processed.real, label="Noisy FID - Real", color="blue")
        ax.plot(new_time, noisy_processed.imag, label="Noisy FID - Imag", color="cyan")
        ax.plot(new_time, denoised_processed.real, label="Denoised FID - Real", color="red")
        ax.plot(new_time, denoised_processed.imag, label="Denoised FID - Imag", color="magenta")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Folder {folder} - FID")
        ax.legend()
    elif mode.lower() == "fft":
        dt = new_time[1] - new_time[0] if len(new_time) > 1 else 1.0
        n = len(noisy_processed)
        # Compute FFTs for both signals.
        fft_noisy = np.fft.fft(noisy_processed)
        fft_denoised = np.fft.fft(denoised_processed)
        freqs = np.fft.fftfreq(n, d=dt)
        # Shift FFT and frequency axis.
        fft_noisy_shifted = np.fft.fftshift(fft_noisy)
        fft_denoised_shifted = np.fft.fftshift(fft_denoised)
        freqs_shifted = np.fft.fftshift(freqs)
        # Reverse arrays so that the highest positive frequency appears on the left.
        fft_noisy_plot = fft_noisy_shifted[::-1]
        fft_denoised_plot = fft_denoised_shifted[::-1]
        freqs_plot = freqs_shifted[::-1]
        # Baseline correction on the real parts (assuming correct_baseline is defined)
        fft_noisy_plot = correct_baseline(fft_noisy_plot.real, 100)
        fft_denoised_plot = correct_baseline(fft_denoised_plot.real, 100)
        # Convert frequency axis (Hz) to chemical shift in ppm.
        ppm_axis = -((freqs_plot) / b1freq) + ppmadjust
        if plot_noisy_fft:
            ax.plot(ppm_axis, fft_noisy_plot, label="Noisy FFT (Real)", color="blue")
        ax.plot(ppm_axis, fft_denoised_plot, label="Denoised FFT (Real)", color="red")
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Folder {folder} - FFT")
        ax.legend()
        ax.invert_xaxis()  # Reverse x-axis direction
    else:
        raise ValueError("mode must be either 'fid' or 'fft'")
    
    # Apply uniform axis limits if provided.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

# ====================================================
# --- Helper function to save an individual plot as PDF ---
# ====================================================
def save_individual_plot(new_time, noisy_processed, denoised_processed, mode, folder, output_pdf_dir, xlim=None, ylim=None, plot_noisy_fft=True):
    """
    Plot the processed data on a single-plot figure and save as PDF.
    
    Parameters:
        new_time (np.ndarray): Time axis.
        noisy_processed (np.ndarray): Processed noisy FID.
        denoised_processed (np.ndarray): Processed denoised FID.
        mode (str): 'fid' or 'fft'.
        folder (int): Folder number.
        output_pdf_dir (str): Directory in which to save the PDF.
        xlim (tuple, optional): x-axis limits.
        ylim (tuple, optional): y-axis limits.
        plot_noisy_fft (bool): For FFT mode, if True plot both curves; else only denoised.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_denoised_on_ax(ax, new_time, noisy_processed, denoised_processed, mode, folder,
                        xlim=xlim, ylim=ylim, plot_noisy_fft=plot_noisy_fft)
    fig.tight_layout()
    output_file = os.path.join(output_pdf_dir, f"folder_{folder}_{mode}.pdf")
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Saved plot for folder {folder} as {output_file}")

# ====================================================
# --- Main function to process four folders and plot in a single window ---
# ====================================================
def process_four_folders_plots(folder_list, base_path, mode="fid", save_plots=False, output_pdf_dir=None,
                               xlim=None, ylim=None, plot_noisy_fft=True,
                               L=2000, T2_apod=5, phase_corr_angle=10, initial_k=4,
                               allowed_ppms=None, ppm_threshold=1, target_length=65536, n_iter=2):
    """
    Process FID data from four specified folders, plot each result as a subplot in a 2×2 grid,
    and optionally save each subplot as a PDF.
    
    Parameters:
        folder_list (list): List of 4 folder numbers to process.
        base_path (str): Base directory where the folders (each with a fid.csv) reside.
        mode (str): 'fid' or 'fft' for time- or frequency-domain plots.
        save_plots (bool): If True, save each subplot as a separate PDF.
        output_pdf_dir (str): Directory where PDFs will be saved (required if save_plots is True).
        xlim (tuple, optional): Uniform x-axis limits to apply to all plots.
        ylim (tuple, optional): Uniform y-axis limits to apply to all plots.
        plot_noisy_fft (bool): In FFT mode, if True plot both noisy and denoised curves; otherwise only plot denoised FFT.
        L, T2_apod, phase_corr_angle, initial_k, allowed_ppms, ppm_threshold,
        target_length, n_iter: Parameters passed to the processing functions.
    """
    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    if len(folder_list) != 4:
        raise ValueError("folder_list must contain exactly 4 folder numbers.")
    if save_plots and output_pdf_dir is None:
        raise ValueError("When save_plots is True, output_pdf_dir must be provided.")
    
    # Store results for each folder.
    processed_results = []
    
    # Process each folder similar to process_single_folder.
    for folder in folder_list:
        file_path = os.path.join(base_path, str(folder), "fid.csv")
        print(f"\nProcessing folder {folder} at {file_path}")
        start_time = time.time()
        # Load full FID data.
        time_full, fid_full = load_fid_data(file_path)
        dt = time_full[1] - time_full[0]
        n_denoise = 2 * L - 1  # for square Hankel matrix
        fid_segment = fid_full[:n_denoise]
        
        k = initial_k
        while k >= 1:
            print(f"Folder {folder}: Trying denoising with k = {k} ...")
            denoised_segment = denoise_fid(fid_segment, L=L, k=k, n_iter=n_iter)
            if check_fft_validity(denoised_segment, dt, T2_apod, phase_corr_angle, allowed_ppms, ppm_threshold):
                print(f"Folder {folder}: FFT quality check passed with k = {k}.")
                break
            else:
                print(f"Folder {folder}: FFT quality check failed with k = {k}. Reducing k...")
                k -= 1
                if k < 1:
                    print("Warning: k reached below 1. Proceeding with the current result.")
                    break
        
        # Zero-fill, apodize, and phase-correct.
        noisy_processed, new_time = zero_fill_and_apodize(fid_full, target_length, dt, T2_apod)
        denoised_processed, _ = zero_fill_and_apodize(denoised_segment, target_length, dt, T2_apod)
        noisy_processed = apply_phase_correction(noisy_processed, phase_corr_angle)
        denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)
        elapsed_time = time.time() - start_time
        print(f"Folder {folder}: Processing complete in {elapsed_time:.2f} seconds.")
        
        processed_results.append((folder, new_time, noisy_processed, denoised_processed))
        
        # Optionally save individual plot as PDF.
        if save_plots:
            save_individual_plot(new_time, noisy_processed, denoised_processed, mode, folder, output_pdf_dir,
                                   xlim=xlim, ylim=ylim, plot_noisy_fft=plot_noisy_fft)
    
    # Create a single figure with a 2x2 grid for the four folders.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # simplify indexing
    for idx, (folder, new_time, noisy_processed, denoised_processed) in enumerate(processed_results):
        plot_denoised_on_ax(axes[idx], new_time, noisy_processed, denoised_processed, mode, folder,
                            xlim=xlim, ylim=ylim, plot_noisy_fft=plot_noisy_fft)
    fig.tight_layout()
    plt.show()

# =============================================================================
# Example usage: Single folder processing
# =============================================================================
# For single folder processing (plots the results):

process_single_folder(folder=2, base_path=r"D:\WSU\Animal data may 2025 visit\copy of the 13C data\2025-05-07-animal injection coil #9\animal05", 
                      allowed_ppms=[98], n_iter=2, L=1000, initial_k=3, T2_apod=1, phase_corr_angle=75)
# process_single_folder(folder=10, base_path=r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-04-03\250403-123410 7degCarbon-Cells (Pyr70_1)", 
#                       allowed_ppms=[178, 176, 170, 160, 124], n_iter=2, L=4000, initial_k=7, T2_apod=1.5, phase_corr_angle=10)
# process_single_folder(folder=15, base_path=r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-04-03\250403-164228 7degCarbon-Cells (KL70_1)", 
#                       allowed_ppms=[178, 176, 171, 160, 124], n_iter=2, L=4000, initial_k=7, T2_apod=1.5, phase_corr_angle=10)
# process_single_folder(folder=20, base_path=r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-03-12\250312-121333 7degCarbon-Cells (Pyr_KIC_CoPol_Cells_001)", 
#                       allowed_ppms=[178, 176, 171, 170, 160, 124], n_iter=2, L=4000, initial_k=8, T2_apod=1, phase_corr_angle=10)

# =============================================================================
# Example usage: Multiple folder processing
# =============================================================================
# Process multiple folders and save integrated data to a CSV file.

# # Pyruvate data processing
# if __name__ == '__main__':
#     # Base path where folder subdirectories (each containing fid.csv) reside.
#     base_path = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-04-03\250403-182557 7degCarbon-Cells (Pyr70_buffer)"
#     # Full path (including CSV filename) where the integrated results should be saved.
#     output_csv = r"D:\WSU\2025-02 Pyruvate Cell paper\Data Analysis\integrated_data_250403-Pyr70_buffer.csv"

#     process_multiple_folders(start_folder=1, end_folder=150,
#                              base_path=base_path,
#                              n_threshold=2, initial_k=2, L=4000, T2_apod=1.5, phase_corr_angle=10,
#                              allowed_ppms=[178, 176, 170, 160, 124],
#                              ppm_threshold=1, 
#                              target_length=65536, n_iter=2,
#                              target_peaks=[178, 176, 171, 160, 124],
#                              metabolite_names=["hydrate", "pyr-form", "pyruvate", "bicarbonate", "CO2"],
#                              tolerance=1,
#                              output_csv=output_csv,
#                              time_interval=3.5)

# # Ketoleucine data processing
# if __name__ == '__main__':
#     # Base path where folder subdirectories (each containing fid.csv) reside.
#     base_path = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-04-03\250403-164228 7degCarbon-Cells (KL70_1)"
#     # Full path (including CSV filename) where the integrated results should be saved.
#     output_csv = r"D:\WSU\2025-02 Ketoleucine Cell paper\Data Analysis\integrated_data_250403-KL70_1.csv"

#     process_multiple_folders(start_folder=1, end_folder=150,
#                              base_path=base_path,
#                              n_threshold=3, initial_k=7, L=4000, T2_apod=1.5, phase_corr_angle=10,
#                              allowed_ppms=[178, 176, 171, 160, 124],
#                              ppm_threshold=1, 
#                              target_length=65536, n_iter=2,
#                              target_peaks=[178, 176, 171, 160, 124],
#                              metabolite_names=["hydrate", "kl-form", "ketoleucine", "bicarbonate", "CO2"],
#                              tolerance=1,
#                              output_csv=output_csv,
#                              time_interval=3.5)

# # Co-polarization data processing
# if __name__ == '__main__':
#     # Base path where folder subdirectories (each containing fid.csv) reside.
#     base_path = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-03-12\250312-121333 7degCarbon-Cells (Pyr_KIC_CoPol_Cells_001)"
#     # Full path (including CSV filename) where the integrated results should be saved.
#     output_csv = r"D:\WSU\2025-02 Ketoleucine Cell paper\Data Analysis\integrated_data_copol001-2.csv"

#     process_multiple_folders(start_folder=1, end_folder=150,
#                              base_path=base_path,
#                              n_threshold=3, initial_k=8, L=4000, T2_apod=1.5, phase_corr_angle=10,
#                              allowed_ppms=[178, 171, 170, 160, 124],
#                              ppm_threshold=1, 
#                              target_length=65536, n_iter=2,
#                              target_peaks=[178.8, 178.2, 171.2, 170.2, 160.3, 124.8],
#                              metabolite_names=["pyv_hydrate", "kl_hydrate", "ketoleucine", "pyruvate", "bicarbonate", "CO2"],
#                              tolerance=0.3,
#                              output_csv=output_csv,
#                              time_interval=3.5)

# =============================================================================
# Four folder processing and plotting
# =============================================================================
# Process four folders and plot results in a single window.
# The processed data for each folder can be saved as individual PDFs.

# folder_numbers = [10, 30, 60, 90]  # example folder numbers chosen by the user
# # base_path = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-02-06\250206-131739 7degCarbon-Cells (KIC_SLIC_002)"  # adjust base path as needed
# base_path=r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-03-12\250312-121333 7degCarbon-Cells (Pyr_KIC_CoPol_Cells_001)"
# output_dir = r"D:\WSU\2025-03 Co-polarization Cell paper\Data Analysis\Plotted Data\Run 1 Plots"       # directory where individual PDFs will be saved

# # Set uniform axis limits; for instance, for FFT plots:
# x_axis_limits = (220, 0)  # chemical shift ppm (note: in FFT plots, x-axis is often inverted)
# y_axis_limits = (-7.5, 200)   # amplitude limits

# process_four_folders_plots(folder_numbers, base_path, mode="fft", save_plots=True, output_pdf_dir=output_dir,
#                            xlim=x_axis_limits, ylim=y_axis_limits, plot_noisy_fft=False,
#                            L=4000, T2_apod=1.5, phase_corr_angle=10, initial_k=8,
#                            allowed_ppms=[178, 171, 170, 160, 124], ppm_threshold=1, target_length=65536, n_iter=2)