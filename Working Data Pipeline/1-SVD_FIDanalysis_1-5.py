import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from scipy.signal import find_peaks
from scipy.integrate import simpson

import os
np.__config__.show()
print("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS", os.getenv("OPENBLAS_NUM_THREADS"))
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from time import perf_counter

# Prefer PROPACK (faster & more robust for dense low-rank), fallback to ARPACK
from scipy.sparse.linalg import svds

from collections import deque
import math


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
# Fast in-place anti-diagonal averaging (vectorized)
# ----------------------------------------------------
def average_anti_diagonals_inplace(M: np.ndarray) -> np.ndarray:
    """
    Replace each anti-diagonal (i+j = const) by its mean, in place.
    Works for real or complex arrays. Same numerical result as the loop.
    """
    N0, N1 = M.shape
    assert N0 == N1, "Expected square Hankel matrix"
    N = N0

    # Build index grid and corresponding anti-diagonal bins
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    bins = (i + j).ravel()                        # shape (N*N,)

    # Accumulate real part
    Mr = M.real.ravel()
    count = np.bincount(bins, minlength=2*N - 1)
    sum_r = np.bincount(bins, weights=Mr, minlength=2*N - 1)

    if np.iscomplexobj(M):
        # Accumulate imaginary part if needed
        Mi = M.imag.ravel()
        sum_i = np.bincount(bins, weights=Mi, minlength=2*N - 1)

        # Compute means (avoid divide-by-zero; empty bins don't occur here)
        mean_r = sum_r[count > 0] / count[count > 0]
        mean_i = sum_i[count > 0] / count[count > 0]

        means = np.empty(2*N - 1, dtype=np.complex128)
        means[count > 0] = mean_r + 1j * mean_i
        # Scatter back to M
        M[:] = means[(i + j)]
    else:
        mean_r = sum_r[count > 0] / count[count > 0]
        means = np.empty(2*N - 1, dtype=M.dtype)
        means[count > 0] = mean_r
        M[:] = means[(i + j)]

    return M

# ----------------------------------------------------
# Hankel view creation for SVD denoising
# ----------------------------------------------------

def hankel_view(x: np.ndarray, L: int) -> np.ndarray:
    """
    Return an (L, L) Hankel view H where H[i, j] = x[i + j].
    Requires len(x) >= 2*L - 1. Returns a read-only strided view.
    """
    x = np.ascontiguousarray(x)
    if x.ndim != 1:
        x = x.ravel()
    if x.size < 2*L - 1:
        raise ValueError("Input too short for Hankel of size L.")
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(L, L), strides=(stride, stride))


# ----------------------------------------------------
# Truncated SVD helper function (using svds)
# ----------------------------------------------------
def truncated_svd_rank_k(H: np.ndarray, k: int):
    """
    Compute ONLY the top-k singular triplets (U_k, s_k, Vh_k) of dense H.
    Uses scipy.sparse.linalg.svds under the hood (PROPACK if available).
    Returns U_k [L x k], s_k [k], Vh_k [k x L] sorted by descending s_k.
    """
    # svds returns ascending singular values; we reverse-order them
    # Use tight tol to match full SVD numerically; increase if you want faster.
    uk, sk, vhk = svds(H, k=k, which="LM", tol=1e-7, maxiter=None)  # complex supported
    order = np.argsort(sk)[::-1]
    sk = sk[order]
    uk = uk[:, order]
    vhk = vhk[order, :]
    return uk, sk, vhk

# ----------------------------------------------------
# SVD-based denoising using Hankel matrix
# ----------------------------------------------------
def denoise_fid(fid_segment, L=500, k=4, n_iter=5):
    """
    Denoise a complex FID segment using iterative SVD Hankel denoising.
    Returns a denoised segment of length 2*L - 1.
    """
    # Build Hankel as a stride-based view and keep Fortran order for LAPACK
    H_d = np.asfortranarray(hankel_view(fid_segment, L))

    for _ in range(n_iter):
        # Fastest LAPACK path; avoid extra checks/copies
        t_s = perf_counter()
        # SVD call
        # === rank-k SVD only (huge speedup when k << L) ===
        try:
            Uk, sk, Vhk = truncated_svd_rank_k(H_d, k)
        except Exception:
            # robust fallback (rare): use full SVD if svds doesn’t converge
            U, s, Vh = svd(H_d, full_matrices=False, lapack_driver='gesdd',
                        overwrite_a=True, check_finite=False)
            Uk, sk, Vhk = U[:, :k], s[:k], Vh[:k, :]

        # Scale rows of Vh by singular values in-place, then one GEMM
        Vhk *= sk[:, None]
        H_trunc = Uk @ Vhk

        t_e = perf_counter()
        print(f"[TIMING] SVD: {(t_e - t_s)*1000:.1f} ms")

        # Vectorized anti-diagonal averaging (in-place)
        t_a = perf_counter()
        average_anti_diagonals_inplace(H_trunc)
        t_b = perf_counter()
        print(f"[TIMING] AntiDiag: {(t_b - t_a)*1000:.1f} ms")

        # Keep F-order for next iteration to help BLAS
        H_d = np.asfortranarray(H_trunc)


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

    if np.min(fft_plot_real) < -50:
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
# Saving Helpers
# ----------------------------------------------------
import os
import csv
import numpy as np

def ensure_dir(path: str):
    """Create directory if it doesn't exist (no-op if None/empty)."""
    if path:
        os.makedirs(path, exist_ok=True)

def init_summary_csv(path: str, target_peaks, metabolite_names):
    """
    Create the summary CSV with the 3 header lines if it doesn't exist yet.
    Safe to call multiple times; headers are only written once.
    """
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        header1 = [""] + ["Height"] * len(target_peaks) + ["Integral"] * len(target_peaks)
        header2 = ["Time"] + [str(tp) for tp in target_peaks] + [str(tp) for tp in target_peaks]
        header3 = [""] + list(metabolite_names) + list(metabolite_names)
        with open(path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header1)
            w.writerow(header2)
            w.writerow(header3)

def append_summary_row(path: str, row: list):
    """Append a single row to the summary CSV (already initialized)."""
    with open(path, mode="a", newline="") as f:
        csv.writer(f).writerow(row)

def save_fid_csv(path: str, time_s: np.ndarray, denoised: np.ndarray):
    """
    Save a denoised FID to CSV with columns: time_s, denoised_real, denoised_imag
    """
    ensure_dir(os.path.dirname(path))
    arr = np.column_stack([time_s, denoised.real, denoised.imag])
    np.savetxt(path, arr, delimiter=",", header="time_s,denoised_real,denoised_imag", comments="")

def save_fft_csv(path: str, freq_hz: np.ndarray, ppm_axis: np.ndarray, real_fft: np.ndarray):
    """
    Save a denoised spectrum to CSV with columns: freq_hz, ppm, denoised_fft_real
    """
    ensure_dir(os.path.dirname(path))
    arr = np.column_stack([freq_hz, ppm_axis, real_fft])
    np.savetxt(path, arr, delimiter=",", header="freq_hz,ppm,denoised_fft_real", comments="")

def get_per_folder_output_dir(output_dir: str | None, output_name: str | None) -> str | None:
    """
    Returns <output_dir>/<output_name> and ensures it exists.
    If output_dir or output_name is missing, returns None.
    """
    if not output_dir or not output_name:
        return None
    subdir = os.path.join(output_dir, output_name)
    ensure_dir(subdir)
    return subdir


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
def process_multiple_folders(base_path, start_folder=1, end_folder=200,
                             n_threshold=3, initial_k=4, L=2000,
                             T2_apod=5, phase_corr_angle=10,
                             allowed_ppms=None, ppm_threshold=1,
                             target_length=65536, n_iter=2,
                             target_peaks=None, metabolite_names=None,
                             tolerance=1.0,
                             output_dir=None,        # e.g., r"D:\...\Data Analysis"
                             output_name=None,       # e.g., "integrated_data_250610-Run5"
                             save_per_folder=True,
                             time_interval=3.5,
                             k_learning_skip=5,  # skip first N folders for learning
                             k_learning_window=9,
                             k_learning_trim=5,
                             per_inner_folder_cb=None,   # callable(done_count:int, total_count:int, folder_num:int)
                             should_stop_cb=None         # callable() -> bool, if True: stop after current inner folder
                            ):
    """
    Live-updating summary CSV at <output_dir>/<output_name>.csv.
    Per-folder FID/FFT CSVs saved under <output_dir>/<output_name>/ as:
      folderNNN_fid.csv  and  folderNNN_fft.csv
    """
    import time
    from scipy.signal import find_peaks
    from scipy.integrate import simpson

    loop_start_time = time.time()

    if allowed_ppms is None:
        allowed_ppms = [178, 170, 160, 124]
    if target_peaks is None:
        target_peaks = [178, 171, 160, 124]
    if metabolite_names is None:
        metabolite_names = ["alanine", "pyruvate", "bicarbonate", "CO2"]

    # 1) Initialize summary CSV at the ROOT of output_dir
    summary_csv_path = None
    if output_dir and output_name:
        summary_csv_path = os.path.join(output_dir, f"{output_name}.csv")
        init_summary_csv(summary_csv_path, target_peaks, metabolite_names)

    # 2) Prepare subdirectory for per-folder files: <output_dir>/<output_name>/
    per_folder_dir = get_per_folder_output_dir(output_dir, output_name)

    current_initial_k = initial_k
    # Rolling window of the last up to 5 finishing k’s
    recent_finishing_k = deque(maxlen=int(k_learning_window))

    # BEFORE the loop over numbered folders:
    # set up counters, etc.
    inner_start = int(start_folder)
    inner_end   = int(end_folder)
    inner_total = max(0, inner_end - inner_start + 1)
    done_count = 0

    for folder in range(start_folder, end_folder + 1):
        start_time = time.time()
        print(f"\nProcessing folder {folder} with initial k = {current_initial_k}")

        # --- Load FID ---
        file_path = os.path.join(base_path, str(folder), "fid.csv")
        try:
            time_full, fid_full = load_fid_data(file_path)
        except Exception as e:
            print(f"Failed to load data for folder {folder}: {e}")
            # still append a zero row for continuity if desired
            if summary_csv_path:
                time_point = (folder - 1) * time_interval
                zeros = [0.0] * (2 * len(target_peaks))
                append_summary_row(summary_csv_path, [time_point] + zeros)
            continue

        dt = time_full[1] - time_full[0]
        n_denoise = 2 * L - 1
        fid_segment = fid_full[:n_denoise]

        # --- Denoise with quality check ---
        k = current_initial_k
        optimal_k = k
        while k >= 1:
            print(f"Folder {folder}: Trying denoising with k = {k} ...")
            denoised_segment = denoise_fid(fid_segment, L=L, k=k, n_iter=n_iter)
            if check_fft_validity(denoised_segment, dt, T2_apod, phase_corr_angle,
                                  allowed_ppms, ppm_threshold, target_length=target_length):
                print(f"Folder {folder}: FFT quality check passed with k = {k}.")
                optimal_k = k
                break
            else:
                print(f"Folder {folder}: FFT quality check failed with k = {k}. Reducing k...")
                k -= 1
                if k < 1:
                    print(f"Folder {folder}: Warning: k below 1. Proceeding with current result.")
                    optimal_k = k
                    break
        print(f"Folder {folder}: Optimal k determined as {optimal_k}.")

        # --- Zero-fill, apodize, phase-correct ---
        noisy_processed, new_time = zero_fill_and_apodize(fid_full, target_length, dt, T2_apod)
        denoised_processed, _     = zero_fill_and_apodize(denoised_segment, target_length, dt, T2_apod)
        noisy_processed    = apply_phase_correction(noisy_processed,    phase_corr_angle)
        denoised_processed = apply_phase_correction(denoised_processed, phase_corr_angle)
        print(f"Folder {folder}: Processing complete.")

        # --- Spectrum for integration ---
        n = len(denoised_processed)
        fft_data = np.fft.fft(denoised_processed)
        freqs = np.fft.fftfreq(n, d=dt)
        fft_data_shifted = np.fft.fftshift(fft_data)
        freqs_shifted    = np.fft.fftshift(freqs)
        fft_data_plot = fft_data_shifted[::-1]
        freqs_plot    = freqs_shifted[::-1]
        ppm_axis = -((freqs_plot - (2500 - 639.08016399999997)) / 15.507665)
        signal = fft_data_plot.real

        # --- Peaks & windows ---
        peak_threshold = 3
        peak_prominence = 5
        peaks, _ = find_peaks(signal, height=peak_threshold, prominence=peak_prominence, distance=20)
        print(f"Folder {folder}: Detected {len(peaks)} peaks via find_peaks.")

        if len(peaks) == 0:
            print(f"Folder {folder}: No peaks detected. Skipping integration.")
            # Save per-folder files even if no peaks
            if save_per_folder and per_folder_dir:
                fid_path = os.path.join(per_folder_dir, f"folder{folder:03d}_fid.csv")
                fft_path = os.path.join(per_folder_dir, f"folder{folder:03d}_fft.csv")
                save_fid_csv(fid_path, new_time, denoised_processed)
                save_fft_csv(fft_path, freqs_plot, ppm_axis, signal)
            # Append zeros to summary (time continuity)
            if summary_csv_path:
                time_point = (folder - 1) * time_interval
                zeros = [0.0] * (2 * len(target_peaks))
                append_summary_row(summary_csv_path, [time_point] + zeros)
            continue

        baseline = np.median(signal)
        peak_windows = []
        for p in peaks:
            thr = 0.05 * (signal[p] - baseline)
            li = p
            while li > 0 and (signal[li] - baseline) > thr:
                li -= 1
            ri = p
            Nsig = len(signal) - 1
            while ri < Nsig and (signal[ri] - baseline) > thr:
                ri += 1
            peak_windows.append((li, ri))

        for i in range(len(peak_windows) - 1):
            L_i, R_i = peak_windows[i]
            L_j, R_j = peak_windows[i+1]
            if R_i > L_j:
                local = signal[peaks[i]:peaks[i+1]+1]
                boundary = peaks[i] + np.argmin(local)
                peak_windows[i]   = (L_i, boundary)
                peak_windows[i+1] = (boundary, R_j)

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

        target_heights, target_integrals = match_targets_to_peaks(
            peak_data,
            target_peaks=target_peaks,
            tolerance=tolerance,
            prefer="height",
            unique=True,
            alpha=0.25
        )
        time_point = (folder - 1) * time_interval
        row = [time_point] + target_heights + target_integrals
        print(f"Folder {folder}: Integrated data - Heights {target_heights}, Integrals {target_integrals}")

        # --- Save per-folder files to <output_dir>/<output_name>/ ---
        if save_per_folder and per_folder_dir:
            fid_path = os.path.join(per_folder_dir, f"folder{folder:03d}_fid.csv")
            fft_path = os.path.join(per_folder_dir, f"folder{folder:03d}_fft.csv")
            save_fid_csv(fid_path, new_time, denoised_processed)
            save_fft_csv(fft_path, freqs_plot, ppm_axis, signal)

        # --- Append summary row immediately (live CSV at root) ---
        if summary_csv_path:
            append_summary_row(summary_csv_path, row)

        # --- Dynamic k adaptation: skip first k_learning_skip; then wait until window is FULL before adapting ---
        if done_count < int(k_learning_skip):
            # still skipping; don't include this folder's k in the average
            to_go = int(k_learning_skip) - done_count
            print(
                "Learning warm-up (skip):"
                f" processed={done_count}, skipping this folder’s finishing_k={optimal_k};"
                f" {to_go} more to skip before collecting"
            )
        else:
            # start collecting after the skip window
            recent_finishing_k.append(optimal_k)
            W_current = len(recent_finishing_k)
            W_target  = int(k_learning_window)
            M         = int(k_learning_trim)

            # Do NOT adapt until window is FULL
            if W_current < W_target:
                print(
                    "Learning warm-up (collecting):"
                    f" collected={W_current}/{W_target} finishing_k after skip;"
                    " not adapting yet"
                )
            else:
                # Window is full: compute trimmed-mean of middle M over the last W_target values
                s = sorted(recent_finishing_k)
                if M > W_target:
                    M = W_target
                start = (W_target - M) // 2
                mid = s[start:start + M]
                avg_k = float(sum(mid) / len(mid))
                next_k = int(math.ceil(avg_k))
                next_k = max(1, min(L, next_k))  # clamp to [1, L]
                print(
                    "Learning update (trimmed, full window):"
                    f" window={W_target}, trim_size={M}, middle={mid}, avg={avg_k:.3f}"
                    f" -> next starting k = {next_k} (was {current_initial_k})"
                )
                current_initial_k = next_k


        done_count += 1
        if callable(per_inner_folder_cb):
            try:
                per_inner_folder_cb(done_count, inner_total, folder)
            except Exception:
                pass  # never let UI issues break processing

        # allow graceful stop BETWEEN numbered folders
        if callable(should_stop_cb):
            try:
                if should_stop_cb():
                    print(f"Stop requested — halting after folder {folder}.")
                    break
            except Exception:
                pass

        print(f"Processing complete in {time.time() - start_time:.2f} s.")

    print("Multiple folder processing complete.")
    print(f"Total elapsed time: {time.time() - loop_start_time:.2f} s")



# =============================================================================
# Example usage: Single folder processing
# =============================================================================
# For single folder processing (plots the results):


# Cancer cell data processing

data_a = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-095127 7degCarbon-Cells (PYR70_1)"
# data_b = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-103408 7degCarbon-Cells (PYR70_2)"
# data_c = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-111237 7degCarbon-Cells (PYR70_3)"
# data_d = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-114635 7degCarbon-Cells (PYR70_4)"
# data_e = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-121617 7degCarbon-Cells (PYR70_5)"
# data_f = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-124410 7degCarbon-Cells (PYR70_6)"
# data_g = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-132819 7degCarbon-Cells (PYR70_7)"
# data_h = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-141545 7degCarbon-Cells (PYR70_8)"
# data_i = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-150424 7degCarbon-Cells (PYR70_9)"
# data_j = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-154101 7degCarbon-Cells (PYR70_10)"
# data_k = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-162418 7degCarbon-Cells (PYR70_11)"
# data_m = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-08\251008-170058 7degCarbon-Cells (PYR70_12)"

# data_a = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-10\251010-104503 7degCarbon-Cells (PYR70_1)"
# data_b = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-10\251010-113315 7degCarbon-Cells (PYR70_2)"
# data_c = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-10\251010-121921 7degCarbon-Cells (PYR70_3)"
# data_d = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-10\251010-141120 7degCarbon-Cells (PYR70_4)"
data_e = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-10\251010-150009 7degCarbon-Cells (PYR70_5)"
data_f = r"C:\Data\[temp] WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-10\251010-154517 7degCarbon-Cells (PYR70_6)"

# process_single_folder(folder=10, base_path=data_d,
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
    base_path = data_a
    # Full path (including CSV filename) where the integrated results should be saved.
    # output_dir = r"C:\Data\[temp] WSU\2025-08 Leukemia Cancer Cells\Data Analysis\2025-10-10 SVD Proc-Int"
    output_dir = r"D:\WSU\Projects\Eukaryote Experiments\2025-10 Pilot Breast Cancer Cell Experiments\Data Analysis\2025-10-08 SVD Proc-Int"
    output_name="integrated_data_251008-095127 (PYR70_1)-fast_test3"

    # PYR RUN
    process_multiple_folders(
        start_folder=75, end_folder=110, 
        base_path=base_path,
        n_threshold=3, initial_k=10, L=4000, T2_apod=1.5, phase_corr_angle=10,
        allowed_ppms=[182.5, 178, 170, 160, 124], 
        ppm_threshold=20, target_length=65536, n_iter=2,
        target_peaks=[182.5, 178, 170, 160, 124],
        metabolite_names=["lactate","hydrate","pyruvate","bicarbonate","CO2"],
        tolerance=1.0,
        output_dir=output_dir,
        output_name=output_name,
        save_per_folder=True,
        time_interval=2.6391
    )

    # # KL RUN
    # process_multiple_folders(
    #     start_folder=4, end_folder=200, 
    #     base_path=base_path,
    #     n_threshold=3, initial_k=10, L=4000, T2_apod=1.5, phase_corr_angle=10,
    #     allowed_ppms=[182.5, 178.3, 173.2, 171.3, 170.2, 160, 124], 
    #     ppm_threshold=20, target_length=65536, n_iter=2,
    #     target_peaks=[182.5, 178.3, 173.2, 171.3, 170.2, 160, 124],
    #     metabolite_names=["lactate","hydrate","unk1", "ketoleucine", "unk2", "bicarbonate","CO2"],
    #     tolerance=1.0,
    #     output_dir=output_dir,
    #     output_name=output_name,
    #     save_per_folder=True,
    #     time_interval=2.4710
    # )

    # # COPOL RUN
    # process_multiple_folders(
    #     start_folder=4, end_folder=200, 
    #     base_path=base_path,
    #     n_threshold=3, initial_k=12, L=4000, T2_apod=1.5, phase_corr_angle=10,
    #     allowed_ppms=[182.5, 178.6, 178.3, 173, 171, 170], 
    #     ppm_threshold=20, target_length=65536, n_iter=2,
    #     target_peaks=[182.5, 178.6, 178.3, 173, 171.4, 170.3],
    #     metabolite_names=["lactate", "pyrhydrate", "klhydrate", "unk1", "ketoleucine", "pyruvate"],
    #     tolerance=1.0,
    #     output_dir=output_dir,
    #     output_name=output_name,
    #     save_per_folder=True,
    #     time_interval=2.3854
    # )