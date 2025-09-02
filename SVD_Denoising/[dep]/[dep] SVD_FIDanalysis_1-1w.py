# SVD_FIDanalysis_1-1 (windowed SVD enabled)
# -----------------------------------------------------------------------------
# This file is your original SVD_FIDanalysis_1-1.py with a minimal, surgical
# addition of a windowed SVD path that divides a 1D FID into user-defined
# segments with an overlap ratio, denoises each segment using the *existing*
# denoise_fid(...) routine (unchanged), and overlap-adds the results with
# Hann cross-fades. All other behavior and downstream functionality remains
# identical. The new behavior is OFF by default (n_segments=1).
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from scipy.signal import find_peaks
import time
import os

# --------------------------
# Load FID data from CSV
# --------------------------
def load_fid_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    time_arr = data[:, 0] / 1000  # ms -> s
    real_part = data[:, 1]
    imag_part = data[:, 2]
    fid = real_part + 1j * imag_part
    return time_arr, fid

# ----------------------------------------------------
# SVD-based denoising using Hankel matrix and anti-diagonal averaging
# ----------------------------------------------------
def average_anti_diagonals(M):
    """Average elements along the anti-diagonals of a matrix M.
    Works for both square and rectangular matrices (complex-safe)."""
    L, C = M.shape  # rows, cols
    M_new = np.copy(M)
    # anti-diagonal index d runs from 0 .. L+C-2, with i+j=d
    for d in range(L + C - 1):
        i_start = max(0, d - (C - 1))
        i_end = min(L - 1, d)
        if i_start > i_end:
            continue
        # collect values on this anti-diagonal
        vals = [M[i, d - i] for i in range(i_start, i_end + 1)]
        if not vals:
            continue
        avg = np.mean(vals)
        for i in range(i_start, i_end + 1):
            j = d - i
            M_new[i, j] = avg
    return M_new


def denoise_fid(fid_segment, L=500, k=4, n_iter=5):
    """Unchanged core SVD denoiser (Cadzow-like with anti-diagonal averaging).
    Returns a denoised segment with the *same* length as fid_segment.
    """
    N = len(fid_segment)
    M = N - L + 1
    if M <= 0:
        # Fallback: if segment too short for requested L, reduce L automatically
        L = max(2, min(N, int(np.ceil(0.5 * N))))
        M = N - L + 1
        if M <= 0:
            return fid_segment.copy()

    H = hankel(fid_segment[:L], fid_segment[L-1:])

    H_denoise = np.copy(H)
    for _ in range(int(max(0, n_iter))):
        U, s, Vh = svd(H_denoise, full_matrices=False, lapack_driver='gesdd')
        kk = int(np.clip(k, 1, min(U.shape[1], Vh.shape[0])))
        H_trunc = (U[:, :kk] * s[:kk]) @ Vh[:kk, :]
        H_denoise = average_anti_diagonals(H_trunc)

    first_col = H_denoise[:, 0]
    last_row  = H_denoise[1:, -1]
    return np.concatenate([first_col, last_row])  # length N

# ----------------------------------------------------
# NEW: Window planner + overlap-add synthesis using the existing denoiser
# ----------------------------------------------------

def _hann_vec(n: int):
    if n <= 1:
        return np.ones(n, float)
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))

def _halfcos_ramp(n: int):
    """0→1 half-cosine ramp that hits EXACTLY 0 at start and 1 at end.
    This avoids a gain jump at the end of the overlap.
    """
    if n <= 1:
        return np.ones(n, float)
    t = np.arange(n, dtype=float)
    return 0.5 * (1.0 - np.cos(np.pi * t / (n - 1)))


def _plan_by_segments(N: int, n_segments: int, overlap_ratio: float):
    """
    Return list of (start, length) windows covering ~N samples with
    uniform segment length and an overlap of overlap_ratio in (0..0.95).
    The last window is truncated if it would run past N.
    """
    n_segments = max(1, int(n_segments))
    overlap_ratio = float(np.clip(overlap_ratio, 0.0, 0.95))
    if n_segments == 1:
        return [(0, N)]

    # Choose step and window_len to achieve requested overlap fraction
    # Let step = (1 - overlap_ratio) * window_len  => window_len = step/(1 - r)
    # We want n_segments windows to roughly span N with overlap, so
    # coverage ≈ window_len + (n_segments-1)*step ≈ N
    # Choose step ≈ N / ( (n_segments-1) + 1/(1-r) )
    denom = (n_segments - 1) + 1.0 / max(1e-6, (1.0 - overlap_ratio))
    step = max(1, int(round(N / denom)))
    win_len = max(8, int(round(step / max(1e-6, (1.0 - overlap_ratio)))))

    starts = [i * step for i in range(n_segments)]
    windows = []
    for s in starts:
        e = min(N, s + win_len)
        if e > s:
            windows.append((s, e - s))
    # Ensure coverage of the tail
    if windows and (windows[-1][0] + windows[-1][1] < N):
        last_s = windows[-1][0]
        windows[-1] = (last_s, N - last_s)
    return windows


def denoise_fid_windowed(x: np.ndarray,
                         *,
                         L: int,
                         k: int,
                         n_iter: int,
                         n_segments: int = 1,
                         overlap_ratio: float = 0.25,
                         pad_fraction: float = 0.1,
                         return_extended: bool = True) -> np.ndarray:
    """
    Windowed SVD using the existing denoise_fid on each segment.
    Mirrors hankel_consensus-2 with extra stability:
      • Pre-window zero-fill extension so all windows/weights are exact
      • Absolute Hann cross-fades (unity partition) — no final normalization
      • Optional context padding per window (pad_fraction of window length):
        denoise a slightly larger slice and crop back to the core window
      • Per-window complex LS alignment to assembled signal over the overlap
      • No tail fade — rely on global exponential apodization later
      • Optionally return the extended-length result (recommended)
    If n_segments=1, falls back to the single-window path.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.size
    if n_segments <= 1:
        return denoise_fid(x, L=L, k=k, n_iter=n_iter)

    # ---- Plan windows and extend length so windows fit exactly ----
    nW = int(max(1, n_segments))
    r = float(np.clip(overlap_ratio, 0.0, 0.95))

    denom = (nW - (nW - 1) * r)
    denom = max(1e-6, denom)
    win_len = int(max(8, round(N / denom)))
    xf = int(max(0, round(r * win_len)))
    step = max(1, win_len - xf)

    coverage = win_len + (nW - 1) * step
    N_ext = max(N, coverage)

    # zero-fill to N_ext BEFORE windowing
    xp = np.zeros(N_ext, dtype=np.complex128)
    xp[:N] = x

    # Build exact windows on the extended length
    windows: list[tuple[int,int]] = []  # (s, e)
    for i in range(nW):
        s = i * step
        e = s + win_len
        e = min(e, N_ext)
        if e > s:
            windows.append((int(s), int(e)))
    if not windows:
        return x.copy()

    # Ensure last ends at N_ext exactly
    s_last, e_last = windows[-1]
    if e_last < N_ext:
        windows[-1] = (s_last, N_ext)

    # ---- Absolute weights per window (unity partition) ----
    weights = [np.zeros(N_ext, float) for _ in windows]
    for i, (s, e) in enumerate(windows):
        weights[i][s:e] = 1.0
    for i in range(len(windows) - 1):
        s1, e1 = windows[i]
        s2, e2 = windows[i + 1]
        ovl_a = max(s1, s2)
        ovl_b = min(e1, e2)
        ovl = max(0, ovl_b - ovl_a)
        if ovl <= 0:
            continue
        xf_loc = min(xf, ovl)
        if xf_loc <= 0:
            continue
        h = _hann_vec(2 * xf_loc)
        ramp = h[:xf_loc]  # 0..1 increasing
        sl = slice(ovl_b - xf_loc, ovl_b)
        weights[i][sl]     = 1.0 - ramp
        weights[i + 1][sl] = ramp

    # ---- Denoise per window with optional context padding and synthesize ----
    y_ext = np.zeros(N_ext, dtype=np.complex128)
    wsum_sofar = np.zeros(N_ext, dtype=float)  # for alignment only

    for i, (s, e) in enumerate(windows):
        ln = e - s
        pad = int(max(0, round(pad_fraction * ln)))
        s_pad = max(0, s - pad)
        e_pad = min(N_ext, e + pad)

        segp = xp[s_pad:e_pad]
        ln_pad = e_pad - s_pad
        # Balanced Hankel on the padded segment
        L_eff = min(L, max(2, int(round(0.5 * ln_pad))))
        segp_dn = denoise_fid(segp, L=L_eff, k=k, n_iter=n_iter)

        # Crop back to the core window
        start_in_dn = s - s_pad
        seg_dn = segp_dn[start_in_dn : start_in_dn + ln]
        if seg_dn.size > ln:
            seg_dn = seg_dn[:ln]
        elif seg_dn.size < ln:
            seg_dn = np.pad(seg_dn, (0, ln - seg_dn.size), mode='constant')

        # Complex LS align to the *current assembled* estimate over overlap
        if i > 0:
            s_prev, e_prev = windows[i-1]
            ovl_a = max(s_prev, s)
            ovl_b = min(e_prev, e)
            ovl = max(0, ovl_b - ovl_a)
            if ovl > 0:
                sl = slice(ovl_a, ovl_b)
                denom = np.maximum(wsum_sofar[sl], 1e-12)
                y_ref = y_ext[sl] / denom
                a = seg_dn[ovl_a - s : ovl_b - s]
                aa = np.vdot(a, a) + 1e-12
                ay = np.vdot(a, y_ref)
                alpha = ay / aa
                seg_dn = seg_dn * alpha

        w = weights[i]
        y_ext[s:e] += seg_dn * w[s:e]
        wsum_sofar[s:e] += w[s:e]

    # Return extended array (recommended for smoother FFT after global apodization)
    return y_ext if return_extended else y_ext[:N]

# ----------------------------------------------------
# Zero-filling and exponential apodization
# ----------------------------------------------------
def zero_fill_and_apodize(signal, target_length, dt, T2_apod=0.1):
    current_length = len(signal)
    if target_length < current_length:
        raise ValueError("target_length must be >= length of the input signal.")
    padded_signal = np.pad(signal, (0, target_length - current_length), mode='constant', constant_values=0)
    t = np.arange(target_length) * dt
    apodization_filter = np.exp(-T2_apod * t)
    processed_signal = padded_signal * apodization_filter
    return processed_signal, t

# ----------------------------------------------------
# Phase and baseline utilities (unchanged)
# ----------------------------------------------------

def apply_phase_correction(fid, phase_angle):
    phase_rad = np.deg2rad(phase_angle)
    return fid * np.exp(-1j * phase_rad)


def correct_baseline(fft_data, n):
    baseline = np.mean(np.concatenate((fft_data[:n], fft_data[-n:])))
    return fft_data - baseline

# ----------------------------------------------------
# FFT quality check (unchanged)
# ----------------------------------------------------

def check_fft_validity(fid_segment, dt, T2_apod, phase_corr_angle, allowed_ppms, ppm_threshold, target_length=65536, integration_threshold=0.01, edge_epsilon_ratio=0.001):
    processed_signal, t = zero_fill_and_apodize(fid_segment, target_length, dt, T2_apod)
    proc_signal = apply_phase_correction(processed_signal, phase_corr_angle)
    n = len(proc_signal)
    fft_signal = np.fft.fft(proc_signal)
    freqs = np.fft.fftfreq(n, d=dt)
    fft_shifted = np.fft.fftshift(fft_signal)
    freqs_shifted = np.fft.fftshift(freqs)
    fft_plot = fft_shifted[::-1]
    freqs_plot = freqs_shifted[::-1]
    fft_plot_real = correct_baseline(fft_plot.real, 100)
    ppm_axis = -((freqs_plot - (2500 - 639.08016399999997)) / 15.507665)

    if np.min(fft_plot_real) < -8:
        return False

    height_threshold = 0.25
    prominence_threshold = 0.02 * np.max(fft_plot_real)
    peaks, properties = find_peaks(fft_plot_real, height=height_threshold, prominence=prominence_threshold)
    if len(peaks) == 0:
        return False
    for peak_idx in peaks:
        peak_ppm = ppm_axis[peak_idx]
        if not any(abs(peak_ppm - allowed) <= ppm_threshold for allowed in allowed_ppms):
            return False

    for peak_idx in peaks:
        peak_val = fft_plot_real[peak_idx]
        epsilon = peak_val * edge_epsilon_ratio
        left_index = peak_idx
        while left_index > 0 and fft_plot_real[left_index] > epsilon:
            left_index -= 1
        right_index = peak_idx
        while right_index < len(fft_plot_real) - 1 and fft_plot_real[right_index] > epsilon:
            right_index += 1
        peak_integral = np.trapz(fft_plot_real[left_index:right_index+1], ppm_axis[left_index:right_index+1])
        if peak_integral < integration_threshold:
            return False
    return True

# ----------------------------------------------------
# Plotting (unchanged)
# ----------------------------------------------------

def plot_denoised_results(time, noisy_signal, denoised_signal, mode="fid"):
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
        fft_noisy = np.fft.fft(noisy_signal)
        fft_denoised = np.fft.fft(denoised_signal)
        freqs = np.fft.fftfreq(n, d=dt)
        fft_noisy_shifted = np.fft.fftshift(fft_noisy)
        fft_denoised_shifted = np.fft.fftshift(fft_denoised)
        freqs_shifted = np.fft.fftshift(freqs)
        fft_noisy_plot = fft_noisy_shifted[::-1]
        fft_denoised_plot = fft_denoised_shifted[::-1]
        freqs_plot = freqs_shifted[::-1]
        fft_noisy_plot = correct_baseline(fft_noisy_plot.real, 100)
        fft_denoised_plot = correct_baseline(fft_denoised_plot.real, 100)
        ppm_axis = -((freqs_plot - (2500 - 639.08016399999997)) / 15.507665)
        plt.figure(figsize=(10, 5))
        plt.plot(ppm_axis, fft_noisy_plot, label="Noisy FFT (Real)", color="blue")
        plt.plot(ppm_axis, fft_denoised_plot, label="Denoised FFT (Real)", color="red")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Amplitude")
        plt.title("FFT of FID (Real Part)")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.show()
    else:
        raise ValueError("mode must be either 'fid' or 'fft'")

# ----------------------------------------------------
# Main processing functions — UPDATED to optionally use windowed SVD
# ----------------------------------------------------

def process_single_folder(folder, base_path,
                          L=2000, T2_apod=5, phase_corr_angle=10,
                          initial_k=4, allowed_ppms=None, ppm_threshold=1,
                          target_length=65536, n_iter=2,
                          # NEW window controls
                          n_segments=1, overlap_ratio=0.25):
    start_time = time.time()
    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]

    file_path = base_path + f"/{folder}/fid.csv"
    print(f"Processing folder {folder} at {file_path}")

    time_full, fid_full = load_fid_data(file_path)
    dt = time_full[1] - time_full[0]

    n_denoise = min(len(fid_full), 2 * L - 1)
    fid_segment = fid_full[:n_denoise]

    k = initial_k
    while k >= 1:
        print(f"Trying denoising with k = {k} ...")
        if n_segments <= 1:
            denoised_segment = denoise_fid(fid_segment, L=L, k=k, n_iter=n_iter)
        else:
            denoised_segment = denoise_fid_windowed(fid_segment, L=L, k=k, n_iter=n_iter,
                                                    n_segments=n_segments, overlap_ratio=overlap_ratio)
        if check_fft_validity(denoised_segment, dt, T2_apod, phase_corr_angle, allowed_ppms, ppm_threshold):
            print(f"FFT quality check passed with k = {k}.")
            break
        else:
            print(f"FFT quality check failed with k = {k}. Reducing k...")
            k -= 1
            if k < 1:
                print("Warning: k reached below 1. Proceeding with the current result.")
                break

    noisy_processed, new_time = zero_fill_and_apodize(fid_full, target_length, dt, T2_apod)
    denoised_processed, _ = zero_fill_and_apodize(denoised_segment, target_length, dt, T2_apod)
    noisy_processed = apply_phase_correction(noisy_processed, phase_corr_angle)
    denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)
    end_time = time.time()

    plot_denoised_results(new_time, noisy_processed, denoised_processed, mode="fid")
    plot_denoised_results(new_time, noisy_processed, denoised_processed, mode="fft")

    elapsed_time = end_time - start_time
    print("Single folder processing complete.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


def process_multiple_folders(start_folder, end_folder, base_path,
                             n_threshold, initial_k=4, L=2000,
                             T2_apod=5, phase_corr_angle=10,
                             allowed_ppms=None, ppm_threshold=1,
                             target_length=65536, n_iter=2,
                             target_peaks=None, metabolite_names=None,
                             tolerance=1, output_csv=None, time_interval=3.5,
                             # NEW window controls
                             n_segments=1, overlap_ratio=0.25):
    import csv
    from scipy.integrate import simpson

    loop_start_time = time.time()

    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    if target_peaks is None:
        target_peaks = [178, 171, 160, 124]
    if metabolite_names is None:
        metabolite_names = ["alanine", "pyruvate", "bicarbonate", "CO2"]

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
    results = []

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
        n_denoise = min(len(fid_full), 2 * L - 1)
        fid_segment = fid_full[:n_denoise]

        k = current_initial_k
        optimal_k = k
        while k >= 1:
            print(f"Folder {folder}: Trying denoising with k = {k} ...")
            if n_segments <= 1:
                denoised_segment = denoise_fid(fid_segment, L=L, k=k, n_iter=n_iter)
            else:
                denoised_segment = denoise_fid_windowed(fid_segment, L=L, k=k, n_iter=n_iter,
                                                        n_segments=n_segments, overlap_ratio=overlap_ratio)
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

        noisy_processed, new_time = zero_fill_and_apodize(fid_full, target_length, dt, T2_apod)
        denoised_processed, _ = zero_fill_and_apodize(denoised_segment, target_length, dt, T2_apod)
        noisy_processed = apply_phase_correction(noisy_processed, phase_corr_angle)
        denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)
        print(f"Folder {folder}: Processing complete.")

        n = len(denoised_processed)
        fft_data = np.fft.fft(denoised_processed)
        freqs = np.fft.fftfreq(n, d=dt)
        fft_data_shifted = np.fft.fftshift(fft_data)
        freqs_shifted = np.fft.fftshift(freqs)
        fft_data_plot = fft_data_shifted[::-1]
        freqs_plot = freqs_shifted[::-1]
        ppm_axis = -((freqs_plot - (2500 - 639.08016399999997)) / 15.507665)
        signal = fft_data_plot.real

        peak_threshold = 3
        peak_prominence = 5
        peaks, _ = find_peaks(signal, height=peak_threshold, prominence=peak_prominence, distance=20)
        print(f"Folder {folder}: Detected {len(peaks)} peaks via find_peaks.")
        if len(peaks) == 0:
            print(f"Folder {folder}: No peaks detected. Skipping integration.")
            continue

        peak_windows = []
        for peak in peaks:
            left_edge = baseline_left_edge(signal, peak, fraction=0.05)
            right_edge = baseline_right_edge(signal, peak, fraction=0.05)
            peak_windows.append((left_edge, right_edge))

        for i in range(len(peak_windows) - 1):
            if peak_windows[i][1] > peak_windows[i+1][0]:
                overlapping_region = signal[peaks[i]:peaks[i+1]+1]
                boundary = peaks[i] + np.argmin(overlapping_region)
                peak_windows[i] = (peak_windows[i][0], boundary)
                peak_windows[i+1] = (boundary, peak_windows[i+1][1])

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
            print(f"Folder {folder}: Peak {idx+1} at {position:.2f} ppm — height {height:.4f}, integral {integral:.4f}")

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
        print(f"Processing complete in {end_time - start_time:.2f} seconds.")

    print("Multiple folder processing complete.")

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
    print(f"Total elapsed time: {loop_end_time - loop_start_time:.2f} seconds")

# -----------------------------------------------------------------------------
# Example: to enable windowing without changing your pipelines, pass e.g.
#   n_segments=6, overlap_ratio=0.25
# to process_single_folder(...) or process_multiple_folders(...).
# Defaults keep the original single-window behavior.
# -----------------------------------------------------------------------------




process_single_folder(folder=10,
                      base_path=r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-08-19\250819-134130 7degCarbon-Cells (PYR07_3)",
                        L=4000, T2_apod=1.0, phase_corr_angle=10,
                        initial_k=2, allowed_ppms=[182.5, 178, 170, 160, 124], ppm_threshold=1,
                        target_length=65536, n_iter=2,
                        # NEW window controls
                        n_segments=6, overlap_ratio=0.5)
