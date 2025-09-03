import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from scipy.signal import find_peaks
from scipy.integrate import simpson

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
    time_arr = data[:, 0] / 1000.0  # convert ms to seconds
    real_part = data[:, 1]
    imag_part = data[:, 2]
    fid = real_part + 1j * imag_part
    return time_arr, fid

# ----------------------------------------------------
# Fast in-place anti-diagonal averaging (square M)
# ----------------------------------------------------
def average_anti_diagonals_inplace(M):
    """
    Average elements along anti-diagonals of square matrix M, in place.
    Identical values to a naive allocate+average implementation.
    """
    N, _ = M.shape
    for d in range(0, 2*N - 1):
        i0 = max(0, d - (N - 1))
        i1 = min(d, N - 1)
        cnt = i1 - i0 + 1
        # accumulate once
        s = 0.0 + 0.0j
        for idx in range(cnt):
            i = i0 + idx
            j = d - i
            s += M[i, j]
        avg = s / cnt
        # write back
        for idx in range(cnt):
            i = i0 + idx
            j = d - i
            M[i, j] = avg
    return M

# ----------------------------------------------------
# SVD-based denoising using Hankel matrix
# ----------------------------------------------------
def denoise_fid(fid_segment, L=500, k=4, n_iter=5):
    """
    Denoise a complex FID segment using iterative SVD Hankel denoising.
    Returns a denoised segment of length 2*L - 1.
    """
    N = len(fid_segment)
    H = hankel(fid_segment[:L], fid_segment[L-1:])  # shape (L, L)
    H_d = H.copy()

    for _ in range(n_iter):
        U, s, Vh = svd(H_d, full_matrices=False, lapack_driver='gesdd')
        U_k = U[:, :k]
        Vh_k = Vh[:k, :]
        # Broadcast s across columns instead of constructing np.diag
        H_trunc = (U_k * s[:k]) @ Vh_k
        average_anti_diagonals_inplace(H_trunc)
        H_d = H_trunc  # reuse buffer

    # reconstruct 1D from Hankel
    first_col = H_d[:, 0]
    last_row  = H_d[1:, -1]
    return np.concatenate([first_col, last_row])

# ----------------------------------------------------
# Zero-filling + apodization (legacy) and new cached helpers
# ----------------------------------------------------
def zero_fill_and_apodize(signal, target_length, dt, T2_apod=0.1):
    """(Kept for backward-compat)"""
    current_length = len(signal)
    if target_length < current_length:
        raise ValueError("target_length must be >= length of the input signal.")
    padded_signal = np.pad(signal, (0, target_length - current_length), mode='constant')
    t = np.arange(target_length) * dt
    apod = np.exp(-T2_apod * t)
    return padded_signal * apod, t

def make_apodization(target_length, dt, T2_apod):
    t = np.arange(target_length) * dt
    apod = np.exp(-T2_apod * t)
    return t, apod

# Cache for frequency & ppm axes keyed by (target_length, dt)
_FREQ_CACHE = {}
def get_freq_axes(target_length, dt):
    key = (target_length, dt)
    if key not in _FREQ_CACHE:
        freqs = np.fft.fftfreq(target_length, d=dt)
        freqs_shifted = np.fft.fftshift(freqs)
        freqs_plot = freqs_shifted[::-1]
        ppm_axis = -((freqs_plot - (2500 - 639.08016399999997)) / 15.507665)
        _FREQ_CACHE[key] = (freqs_plot, ppm_axis)
    return _FREQ_CACHE[key]

# ----------------------------------------------------
# Phase and baseline helpers
# ----------------------------------------------------
def apply_phase_correction(fid, phase_angle_deg):
    return fid * np.exp(-1j * np.deg2rad(phase_angle_deg))

def correct_baseline(fft_real, n=100):
    baseline = np.mean(np.concatenate((fft_real[:n], fft_real[-n:])))
    return fft_real - baseline

# ----------------------------------------------------
# Distance-aware hybrid matcher (order = order of target_peaks)
# ----------------------------------------------------
def match_targets_to_peaks(
    peak_data, target_peaks, tolerance=1.0, prefer="height",
    unique=True, alpha=0.25
):
    """
    Hybrid matcher (distance-aware):
      • Consider only candidates within ±tolerance of each target.
      • Score = (metric / max_metric_in_window) - alpha * (dist / tolerance)
      • Highest score wins (tie-break by larger metric, then closer).
      • If unique=True, each detected peak can be used only once.
    """
    if prefer not in ("height", "integral"):
        raise ValueError("prefer must be 'height' or 'integral'")
    metric_key = "height" if prefer == "height" else "integral"

    available = set(range(len(peak_data)))
    target_heights, target_integrals = [], []

    for tp in target_peaks:
        cands = []
        for i, pk in enumerate(peak_data):
            if unique and i not in available:
                continue
            dppm = abs(pk["position_converted"] - tp)
            if dppm <= tolerance:
                cands.append((i, pk, dppm))

        if not cands:
            target_heights.append(0.0)
            target_integrals.append(0.0)
            continue

        mmax = max(pk[metric_key] for _, pk, _ in cands) or 1.0
        best = None
        best_tuple = None  # (score, metric, -dppm)
        for i, pk, dppm in cands:
            metric = pk[metric_key]
            score = (metric / mmax) - alpha * (dppm / tolerance)
            tup = (score, metric, -dppm)
            if (best is None) or (tup > best_tuple):
                best, best_tuple = (i, pk), tup

        best_i, best_pk = best
        target_heights.append(best_pk["height"])
        target_integrals.append(best_pk["integral"])
        if unique:
            available.remove(best_i)

    return target_heights, target_integrals

# ----------------------------------------------------
# Plotting function for FIDs and FFTs
# ----------------------------------------------------
def plot_denoised_on_ax(ax, new_time, noisy_processed, denoised_processed,
                        mode, folder, xlim=None, ylim=None, plot_noisy_fft=True):
    if mode.lower() == "fid":
        ax.plot(new_time, noisy_processed.real, label="Noisy FID - Real")
        ax.plot(new_time, noisy_processed.imag, label="Noisy FID - Imag")
        ax.plot(new_time, denoised_processed.real, label="Denoised FID - Real")
        ax.plot(new_time, denoised_processed.imag, label="Denoised FID - Imag")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Folder {folder} - FID")
        ax.legend()
    elif mode.lower() == "fft":
        dt = new_time[1] - new_time[0] if len(new_time) > 1 else 1.0
        n = len(denoised_processed)
        # FFT for denoised (always)
        fft_denoised = np.fft.fft(denoised_processed)
        # freq/ppm axes from cache
        _, ppm_axis = get_freq_axes(n, dt)
        fft_den_plot = np.fft.fftshift(fft_denoised)[::-1].real
        fft_den_plot = correct_baseline(fft_den_plot, 100)

        if plot_noisy_fft:
            fft_noisy = np.fft.fft(noisy_processed)
            fft_noi_plot = np.fft.fftshift(fft_noisy)[::-1].real
            fft_noi_plot = correct_baseline(fft_noi_plot, 100)
            ax.plot(ppm_axis, fft_noi_plot, label="Noisy FFT (Real)")

        ax.plot(ppm_axis, fft_den_plot, label="Denoised FFT (Real)")
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Folder {folder} - FFT")
        ax.legend()
        ax.invert_xaxis()
    else:
        raise ValueError("mode must be either 'fid' or 'fft'")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

# ----------------------------------------------------
# FFT quality check (kept functionally the same)
# ----------------------------------------------------
def check_fft_validity(fid_segment, dt, T2_apod, phase_corr_angle,
                       allowed_ppms, ppm_threshold,
                       target_length=65536, integration_threshold=0.01,
                       edge_epsilon_ratio=0.001):
    processed_signal, _ = zero_fill_and_apodize(fid_segment, target_length, dt, T2_apod)
    proc_signal = apply_phase_correction(processed_signal, phase_corr_angle)

    n = len(proc_signal)
    fft_signal = np.fft.fft(proc_signal)
    _, ppm_axis = get_freq_axes(n, dt)
    fft_plot_real = correct_baseline(np.fft.fftshift(fft_signal)[::-1].real, 100)

    if np.min(fft_plot_real) < -30:
        print(f"Quality check failed: Negative peaks (min={np.min(fft_plot_real):.4f}).")
        return False

    height_threshold = 0.25
    prominence_threshold = 0.02 * np.max(fft_plot_real)
    peaks, _ = find_peaks(fft_plot_real, height=height_threshold, prominence=prominence_threshold)
    if len(peaks) == 0:
        print(f"Quality check failed: No peaks found (h_thr={height_threshold:.4f}, prom_thr={prominence_threshold:.4f}).")
        return False

    for p in peaks:
        ppm = ppm_axis[p]
        if not any(abs(ppm - a) <= ppm_threshold for a in allowed_ppms):
            print(f"Quality check failed: Peak at {ppm:.2f} ppm outside allowed windows (±{ppm_threshold}).")
            return False

    for p in peaks:
        peak_val = fft_plot_real[p]
        epsilon = peak_val * edge_epsilon_ratio
        # left edge
        li = p
        while li > 0 and fft_plot_real[li] > epsilon:
            li -= 1
        # right edge
        ri = p
        N = len(fft_plot_real) - 1
        while ri < N and fft_plot_real[ri] > epsilon:
            ri += 1

        integral = np.trapz(fft_plot_real[li:ri+1], ppm_axis[li:ri+1])
        print(f"Peak at {ppm_axis[p]:.2f} ppm: height={peak_val:.4f}, integral={integral:.4f}")
        if integral < integration_threshold:
            print(f"Quality check failed: Area too low at {ppm_axis[p]:.2f} ppm ({integral:.4f} < {integration_threshold}).")
            return False

    return True

# ----------------------------------------------------
# Baseline-edge helpers that reuse a precomputed baseline
# ----------------------------------------------------
def baseline_left_edge(signal, peak_idx, baseline, fraction=0.05):
    thr = fraction * (signal[peak_idx] - baseline)
    i = peak_idx
    while i > 0 and (signal[i] - baseline) > thr:
        i -= 1
    return i

def baseline_right_edge(signal, peak_idx, baseline, fraction=0.05):
    thr = fraction * (signal[peak_idx] - baseline)
    i = peak_idx
    N = len(signal) - 1
    while i < N and (signal[i] - baseline) > thr:
        i += 1
    return i
# ----------------------------------------------------

# ----------------------------------------------------
# Single folder processing
# ----------------------------------------------------
def process_single_folder(
    folder, base_path,
    L=2000, T2_apod=5, phase_corr_angle=10,
    initial_k=4, allowed_ppms=None, ppm_threshold=1,
    target_length=65536, n_iter=2,
    target_peaks=None, metabolite_names=None,
    tolerance=1.0, time_interval=None,
    plot_time_domain=True, plot_fft=True, plot_noisy_fft=True,
    xlim=None, ylim=None
):
    start_time = time.time()

    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    if target_peaks is None:
        target_peaks = [178, 171, 160, 124]
    if metabolite_names is None:
        metabolite_names = ["alanine", "pyruvate", "bicarbonate", "CO2"]

    file_path = os.path.join(base_path, str(folder), "fid.csv")
    print(f"Processing folder {folder} at {file_path}")
    time_full, fid_full = load_fid_data(file_path)
    dt = time_full[1] - time_full[0]

    # 1) Denoise with k loop + FFT quality check
    n_denoise = 2 * L - 1
    fid_segment = fid_full[:n_denoise]
    k = initial_k
    optimal_k = k
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

    # 2) Zero-fill, apodize, phase-correct (cached apodization)
    new_time, apod = make_apodization(target_length, dt, T2_apod)
    noisy_processed   = np.pad(fid_full,         (0, target_length - len(fid_full)        ), mode='constant') * apod
    denoised_processed= np.pad(denoised_segment, (0, target_length - len(denoised_segment)), mode='constant') * apod
    noisy_processed   = apply_phase_correction(noisy_processed,   phase_corr_angle)
    denoised_processed= apply_phase_correction(denoised_processed,phase_corr_angle)
    print(f"Folder {folder}: Processing complete.")

    # 3) FFT & integration windows
    n = len(denoised_processed)
    fft_data = np.fft.fft(denoised_processed)
    _, ppm_axis = get_freq_axes(n, dt)
    signal_real = correct_baseline(np.fft.fftshift(fft_data)[::-1].real, 100)

    peak_threshold = 2
    peak_prominence = 5
    peaks, _ = find_peaks(signal_real, height=peak_threshold, prominence=peak_prominence, distance=20)
    print(f"Folder {folder}: Detected {len(peaks)} peaks via find_peaks.")

    peak_windows = []
    peak_data = []
    if len(peaks) > 0:
        baseline = np.median(signal_real)
        # initial windows
        for p in peaks:
            L_edge = baseline_left_edge(signal_real, p, baseline, fraction=0.05)
            R_edge = baseline_right_edge(signal_real, p, baseline, fraction=0.05)
            peak_windows.append((L_edge, R_edge))
        # resolve overlaps by splitting at min
        for i in range(len(peak_windows) - 1):
            L_i, R_i = peak_windows[i]
            L_j, R_j = peak_windows[i+1]
            if R_i > L_j:
                local = signal_real[peaks[i]:peaks[i+1]+1]
                boundary = peaks[i] + np.argmin(local)
                peak_windows[i]   = (L_i, boundary)
                peak_windows[i+1] = (boundary, R_j)
        # compute integrals
        for idx, p in enumerate(peaks):
            L_edge, R_edge = peak_windows[idx]
            pos_ppm = ppm_axis[p]
            height = signal_real[p]
            integral = simpson(signal_real[L_edge:R_edge+1], ppm_axis[L_edge:R_edge+1])
            peak_data.append({
                "position_converted": pos_ppm,
                "height": height,
                "integral": integral,
                "left_edge": ppm_axis[L_edge],
                "right_edge": ppm_axis[R_edge],
            })
            print(f"  Peak at {pos_ppm:.2f} ppm | Height={height:.4f} | Integral={integral:.4f} "
                  f"| Range [{ppm_axis[L_edge]:.2f}, {ppm_axis[R_edge]:.2f}]")
    else:
        print(f"Folder {folder}: No peaks detected. Skipping integration.")

    # 4) Match to targets (distance-aware hybrid)
    print(f"Target list (in-order): {target_peaks} | tolerance={tolerance}")
    target_heights, target_integrals = match_targets_to_peaks(
        peak_data,
        target_peaks=target_peaks,
        tolerance=tolerance,
        prefer="height",
        unique=True,
        alpha=0.25
    )
    print(f"Folder {folder}: Matched data - Heights {target_heights}, Integrals {target_integrals}")

    end_time = time.time()
    print(f"Single folder processing complete in {end_time - start_time:.2f} s.")

    # 5) Optional plotting
    if plot_time_domain:
        fig_fid, ax_fid = plt.subplots(figsize=(10, 4))
        plot_denoised_on_ax(ax_fid, new_time, noisy_processed, denoised_processed,
                            mode="fid", folder=folder, xlim=xlim, ylim=ylim)
        fig_fid.tight_layout()
        plt.show()

    if plot_fft:
        fig_fft, ax_fft = plt.subplots(figsize=(10, 4))
        plot_denoised_on_ax(ax_fft, new_time, noisy_processed, denoised_processed,
                            mode="fft", folder=folder, xlim=xlim, ylim=ylim,
                            plot_noisy_fft=plot_noisy_fft)
        fig_fft.tight_layout()
        plt.show()

    time_point = None
    if time_interval is not None:
        time_point = (folder - 1) * float(time_interval)

    return {
        "folder": folder,
        "file_path": file_path,
        "optimal_k": optimal_k,
        "new_time": new_time,
        "noisy_processed": noisy_processed,
        "denoised_processed": denoised_processed,
        "ppm_axis": ppm_axis,
        "signal_real": signal_real,
        "peaks": peaks,
        "peak_windows": peak_windows,
        "peak_data": peak_data,
        "target_heights": target_heights,
        "target_integrals": target_integrals,
        "time_point": time_point,
        "metabolite_names": metabolite_names,
        "target_peaks": target_peaks,
    }

# ====================================================
# Multiple folder processing with learning feature
# ====================================================
def process_multiple_folders(start_folder, end_folder, base_path,
                             n_threshold, initial_k=4, L=2000,
                             T2_apod=5, phase_corr_angle=10,
                             allowed_ppms=None, ppm_threshold=1,
                             target_length=65536, n_iter=2,
                             target_peaks=None, metabolite_names=None,
                             tolerance=1.0, output_csv=None, time_interval=3.5):
    loop_start_time = time.time()

    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    if target_peaks is None:
        target_peaks = [178, 171, 160, 124]
    if metabolite_names is None:
        metabolite_names = ["alanine", "pyruvate", "bicarbonate", "CO2"]

    current_initial_k = initial_k
    consecutive_decreased_count = 0
    results = []  # rows: [time_point] + heights + integrals

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

        # Denoise + quality check
        k = current_initial_k
        optimal_k = k
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

        # Process: zero-fill, apodize, phase-correct (cached)
        new_time, apod = make_apodization(target_length, dt, T2_apod)
        noisy_processed    = np.pad(fid_full,         (0, target_length - len(fid_full)        ), mode='constant') * apod
        denoised_processed = np.pad(denoised_segment, (0, target_length - len(denoised_segment)), mode='constant') * apod
        noisy_processed    = apply_phase_correction(noisy_processed,    phase_corr_angle)
        denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)
        print(f"Folder {folder}: Processing complete.")

        # Integration path (same as single)
        n = len(denoised_processed)
        fft_data = np.fft.fft(denoised_processed)
        _, ppm_axis = get_freq_axes(n, dt)
        signal = correct_baseline(np.fft.fftshift(fft_data)[::-1].real, 100)

        peak_threshold = 3
        peak_prominence = 5
        peaks, _ = find_peaks(signal, height=peak_threshold, prominence=peak_prominence, distance=20)
        print(f"Folder {folder}: Detected {len(peaks)} peaks via find_peaks.")

        if len(peaks) == 0:
            print(f"Folder {folder}: No peaks detected. Skipping integration.")
            continue

        baseline = np.median(signal)
        peak_windows = []
        for p in peaks:
            left_edge  = baseline_left_edge(signal, p, baseline, fraction=0.05)
            right_edge = baseline_right_edge(signal, p, baseline, fraction=0.05)
            peak_windows.append((left_edge, right_edge))

        for i in range(len(peak_windows) - 1):
            if peak_windows[i][1] > peak_windows[i+1][0]:
                local = signal[peaks[i]:peaks[i+1]+1]
                boundary = peaks[i] + np.argmin(local)
                peak_windows[i]   = (peak_windows[i][0], boundary)
                peak_windows[i+1] = (boundary, peak_windows[i+1][1])

        peak_data = []
        for idx, p in enumerate(peaks):
            L_edge, R_edge = peak_windows[idx]
            position = ppm_axis[p]
            height   = signal[p]
            integral = simpson(signal[L_edge:R_edge+1], ppm_axis[L_edge:R_edge+1])
            peak_data.append({
                "position_converted": position,
                "height": height,
                "integral": integral,
                "left_edge": ppm_axis[L_edge],
                "right_edge": ppm_axis[R_edge]
            })
            print(f"Folder {folder}: Peak {idx+1} @ {position:.2f} ppm | H={height:.4f} | Int={integral:.4f} "
                  f"| Range [{ppm_axis[L_edge]:.2f}, {ppm_axis[R_edge]:.2f}]")

        # Match to targets (distance-aware hybrid, no duplicates)
        target_heights, target_integrals = match_targets_to_peaks(
            peak_data,
            target_peaks=target_peaks,
            tolerance=tolerance,
            prefer="height",
            unique=True,
            alpha=0.25
        )
        time_point = (folder - 1) * time_interval
        results.append([time_point] + target_heights + target_integrals)
        print(f"Folder {folder}: Integrated data - Heights {target_heights}, Integrals {target_integrals}")

        # Adaptive k "learning"
        if optimal_k < current_initial_k:
            consecutive_decreased_count += 1
        else:
            consecutive_decreased_count = 0
        if consecutive_decreased_count >= n_threshold:
            print(f"Learning update: updating initial k from {current_initial_k} to {optimal_k}")
            current_initial_k = max(1, current_initial_k - 1)
            consecutive_decreased_count = 0

        print(f"Processing complete in {time.time() - start_time:.2f} s.")

    print("Multiple folder processing complete.")

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        header1 = [""] + ["Height"] * len(target_peaks) + ["Integral"] * len(target_peaks)
        header2 = ["Time"] + [str(tp) for tp in target_peaks] + [str(tp) for tp in target_peaks]
        header3 = [""] + metabolite_names + metabolite_names
        with open(output_csv, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header1)
            w.writerow(header2)
            w.writerow(header3)
            for row in results:
                w.writerow(row)
        print(f"Integrated peak data saved to '{output_csv}' successfully!")

    print(f"Total elapsed time: {time.time() - loop_start_time:.2f} s")

# =============================================================================
# Example usage: Single folder processing
# =============================================================================
# For single folder processing (plots the results):


# Cancer cell data processing
# data_a = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-120551 7degCarbon-Cells (PYR70_1)"
# data_b = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-125332 7degCarbon-Cells (PYR07_2)"
# data_c =r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-134130 7degCarbon-Cells (PYR07_3)"

# data_a = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-120551 7degCarbon-Cells (PYR70_1)"
# data_b = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-125332 7degCarbon-Cells (PYR07_2)"
# data_c = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-134130 7degCarbon-Cells (PYR07_3)"

# data_a = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-120551 7degCarbon-Cells (PYR70_1)"
# data_b = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-125332 7degCarbon-Cells (PYR07_2)"
# data_c = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-134130 7degCarbon-Cells (PYR07_3)"

data_a = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-26\250826-112726 7degCarbon-Cells (PYR70_1)"
data_b = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-26\250826-122114 7degCarbon-Cells (PYR70_2)"
data_c = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-26\250826-130032 7degCarbon-Cells (PYR70_3)"
data_d = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-26\250826-151918 7degCarbon-Cells (PYR70_5)"
data_e = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-26\250826-160303 7degCarbon-Cells (PYR70_6)"
data_f = r"D:\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-26\250826-164823 7degCarbon-Cells (PYR70_7)"

# process_single_folder(folder=10, base_path=data_c,
#                       initial_k=8, L=4000, T2_apod=1.5, phase_corr_angle=10,
#                       allowed_ppms=[182.5, 178, 170, 160, 124], 
#                       ppm_threshold=20,
#                       target_length=65536, n_iter=2,
#                       target_peaks=[182.5, 178, 170, 160, 124],
#                       metabolite_names=["lactate", "hydrate", "pyruvate", "bicarbonate", "CO2"],
#                       tolerance=1, time_interval=3.5,
#                       plot_time_domain=True, plot_fft=True, plot_noisy_fft=True) 

# =============================================================================
# Example usage: Multiple folder processing
# =============================================================================
# Process multiple folders and save integrated data to a CSV file.

# Pyruvate data processing
if __name__ == '__main__':
    # Base path where folder subdirectories (each containing fid.csv) reside.
    base_path = data_d
    # Full path (including CSV filename) where the integrated results should be saved.
    output_csv = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\integrated_data_250826-151918-Pyr70_5.csv"

    process_multiple_folders(start_folder=4, end_folder=200,
                             base_path=base_path,
                             n_threshold=3, initial_k=10, L=4000, T2_apod=1.5, phase_corr_angle=10,
                             allowed_ppms=[182.5, 178, 170, 160, 124],
                             ppm_threshold=20, 
                             target_length=65536, n_iter=2,
                             target_peaks=[182.5, 178, 170, 160, 124],
                             metabolite_names=["lactate", "hydrate", "pyruvate", "bicarbonate", "CO2"],
                             tolerance=1,
                             output_csv=output_csv,
                             time_interval=3.5)