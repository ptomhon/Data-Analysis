# ===== Standard Library =====
import sys, os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ===== Third-Party Libraries =====
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QListWidget, QTextEdit, QMessageBox, QTabWidget, QCheckBox, QLabel, QDoubleSpinBox
)

import re


# ========= USER SETTINGS (paste/adjust your values here) =========
# These mirror your original script and are read by the functions below.

# 1) Data + analysis toggles
target_peaks = [182.5, 178, 170, 160, 124]     # ppm list (order matters)
target_peaks_labels = [f'P{p}' for p in target_peaks]
method = 'integrals'                            # 'integrals' or 'heights'
substratepeak = 2                               # index into target_peaks_labels for Pyruvate
product4peak = 0                                # index into target_peaks_labels for Lactate
smoothing = False                               # enable Savitzky-Golay denoising
startPoint = 1                                  # drop first N points before fitting

# 2) Plot window (used later by GUI plotting)
x_display_min = -5
x_display_max = 400

# 3) Model/fit limits (TwoSiteConfig below):
#    - Set kinetic bounds, amplitude-scale bounds, and tinj fit/fixed behavior there.


# ========= Dataclasses =========
@dataclass
class TwoSiteConfig:
    allow_curve_scales: bool = True  # allow separate sP, sL amplitude scales in the fit
    # Bounds (physiology/sequence-guided)
    kpl_bounds: Tuple[float, float] = (0.00001, 0.02)         # s^-1
    Rp_eff_bounds: Tuple[float, float] = (0.015, 0.07)    # s^-1  (Rp_eff = 1/T1p + kpl)
    Rl_bounds: Tuple[float, float] = (0.01, 0.2)        # s^-1  (~25-100 s T1-like)
    tinj_extra_hi: float = 30.0                          # allow +30 s above raw first time
    # Control whether t_inj is fixed or fitted
    fit_tinj: bool = True                  # <— set False to hard-code
    fixed_tinj_value: float = 16.0          # <— your chosen global t_inj (seconds)
    # Amplitude scale bounds (if used)
    sP_bounds: Tuple[float, float] = (1.8, 2.2)   # <- EDIT as you like (must be >0)
    sL_bounds: Tuple[float, float] = (0.6, 0.65)
    # Flip-angle correction (off by default; can add later)
    flip_angle_deg: Optional[float] = None
    TR_s: Optional[float] = None

CFG = TwoSiteConfig()

@dataclass
class FitResult:
    """Return container for the joint fit."""
    P0: float
    kpl: float
    Rp_eff: float
    Rl: float
    tinj: float
    sP: float
    sL: float
    success: bool
    message: str


# ========= Function signatures (paste your bodies from the original file) =========

# ===== Your smoothing helper (unchanged behavior) =====
def custom_savgol_filter(x, window_length, polyorder, isolation_range=1, ignore_last=3):
    x = np.array(x, dtype=float)
    x_mod = x.copy()
    n = len(x)
    for i in range(n):
        if x[i] == 0:
            start = max(0, i - isolation_range)
            end = min(n, i + isolation_range + 1)
            neighbor_indices = [j for j in range(start, end) if j != i]
            if not any(x[j] == 0 for j in neighbor_indices):
                neighbors = [x[j] for j in neighbor_indices if x[j] != 0]
                if neighbors:
                    x_mod[i] = np.mean(neighbors)

    nonzero_indices = np.where(x_mod != 0)[0]
    if len(nonzero_indices) == 0:
        return savgol_filter(x_mod, window_length=window_length, polyorder=polyorder, mode='mirror')

    M = len(nonzero_indices)
    if ignore_last >= M:
        return x_mod

    smooth_end = nonzero_indices[M - ignore_last] + 1
    if smooth_end < window_length:
        smoothed = x_mod
    else:
        smoothed_main = savgol_filter(x_mod[:smooth_end], window_length=window_length, polyorder=polyorder, mode='mirror')
        smoothed = np.concatenate([smoothed_main, x_mod[smooth_end:]])
    smoothed[smoothed < 0] = 0
    return smoothed

# ===== Import, filter, and extract (kept structure) =====
def import_new_peak_data(csv_file, target_peaks, use_smoothing=False, window_length=5, polyorder=2, isolation_range=1, ignore_last=3):
    df = pd.read_csv(csv_file, skiprows=3, header=None)
    num_targets = len(target_peaks)
    height_cols = list(range(1, 1 + num_targets))
    integral_cols = list(range(1 + num_targets, 1 + 2 * num_targets))

    if use_smoothing:
        for col in height_cols + integral_cols:
            df.iloc[:, col] = custom_savgol_filter(
                df.iloc[:, col].values, window_length, polyorder, isolation_range, ignore_last
            )
    df = df.iloc[:150, :]  # keep your original 150-row cap
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
    filtered_data: Dict[str, np.ndarray] = {}
    for i, processed_array in enumerate(processed_data_array):
        peak_label = f"P{target_peaks[i]}"
        if method == 'heights':
            valid = processed_array[:, 2] > 0
            filtered_data[peak_label] = processed_array[valid]
        elif method == 'integrals':
            valid = processed_array[:, 3] > 0
            filtered_data[peak_label] = processed_array[valid]
        else:
            raise ValueError("method must be 'integrals' or 'heights'")
    return filtered_data

def extract_filtered_data(filtered_arrays, peak_label, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
    if peak_label not in filtered_arrays:
        raise ValueError(f"Peak '{peak_label}' not found.")
    arr = filtered_arrays[peak_label]
    t = np.array(arr[start_index:, 0], dtype=np.float64)
    if method == 'heights':
        y = np.array(arr[start_index:, 2], dtype=np.float64)
    else:
        y = np.array(arr[start_index:, 3], dtype=np.float64)
    return t, y

# ===== Utilities =====
def _small_flip_increment(alpha_deg: float, TR_s: float) -> float:
    """Return additional loss rate from repeated small flip angle excitations."""
    alpha = np.deg2rad(alpha_deg)
    return -np.log(np.cos(alpha)) / TR_s

def _safe_exp(x):
    # prevent overflow in exp if someone passes extreme guesses
    x = np.clip(x, -700, 700)
    return np.exp(x)

# ===== Two-site closed-form model =====
def two_site_closed_form(t: np.ndarray,
                         P0: float, kpl: float, Rp_eff: float, Rl: float, tinj: float,
                         alpha_deg: Optional[float] = None, TR_s: Optional[float] = None):
    """
    Evaluate P(t), L(t) at display time t with injection delay tinj.
    Model uses Rp_eff = (1/T1p) + kpl and Rl = (1/T1l) [no flip-angle correction by default].
    If alpha_deg and TR_s are given, an extra decay increment is added to both rates.
    """
    teff = np.asarray(t) + tinj
    teff = np.clip(teff, 0.0, None)  # enforce nonnegative 'physical' time inside the model

    Rp = Rp_eff
    Rl_eff = Rl
    if (alpha_deg is not None) and (TR_s is not None):
        inc = _small_flip_increment(alpha_deg, TR_s)
        Rp = Rp_eff + inc
        Rl_eff = Rl + inc

    # Pyruvate
    P = P0 * _safe_exp(-Rp * teff)

    # Lactate
    denom = (Rl_eff - Rp)
    close = np.isclose(denom, 0.0, atol=1e-9)
    L = np.empty_like(P)

    if np.any(close):
        # Limit when Rl ≈ Rp: L = P0 * kpl * teff * exp(-Rl*teff)
        L[close] = P0 * kpl * teff[close] * _safe_exp(-Rl_eff * teff[close])
    if np.any(~close):
        L[~close] = (P0 * kpl / denom) * (_safe_exp(-Rp * teff[~close]) - _safe_exp(-Rl_eff * teff[~close]))

    return P, L

# ===== Joint residual builder =====
def _build_residuals(theta, tP, Pobs, tL, Lobs, cfg: TwoSiteConfig, tinj: float):
    if cfg.allow_curve_scales:
        P0, kpl, Rp_eff, Rl, sP, sL = theta
    else:
        P0, kpl, Rp_eff, Rl = theta
        sP, sL = 1.0, 1.0

    Pmod, _ = two_site_closed_form(
        t=tP, P0=P0, kpl=kpl, Rp_eff=Rp_eff, Rl=Rl, tinj=tinj,
        alpha_deg=cfg.flip_angle_deg, TR_s=cfg.TR_s
    )
    _, Lmod_atL = two_site_closed_form(
        t=tL, P0=P0, kpl=kpl, Rp_eff=Rp_eff, Rl=Rl, tinj=tinj,
        alpha_deg=cfg.flip_angle_deg, TR_s=cfg.TR_s
    )

    Pmod *= sP
    Lmod_atL *= sL

    wp = 1.0 / max(np.max(Pobs), 1e-9)
    wl = 1.0 / max(np.max(Lobs), 1e-9)

    rP = wp * (Pmod - Pobs)
    rL = wl * (Lmod_atL - Lobs)
    return np.concatenate([rP, rL], axis=0)

# ===== Fitter function =====
def fit_two_site_joint(tP_raw, Pobs_raw, tL_raw, Lobs_raw, cfg: TwoSiteConfig) -> FitResult:
    # time alignment (display t=0 at first kept P point)
    t0 = float(tP_raw[0])
    tP = tP_raw - t0
    tL = tL_raw - t0

    # ---------- t_inj handling ----------
    # initial heuristic if we were to fit:
    tinj0_guess = max(0.0, t0 + 5.0)
    tinj_lo = max(0.0, t0)
    tinj_hi = t0 + cfg.tinj_extra_hi

    # choose tinj for this run
    if cfg.fit_tinj:
        tinj_for_residuals = tinj0_guess
    else:
        tinj_for_residuals = float(cfg.fixed_tinj_value)

    # ---------- initial guesses for kinetic params ----------
    P0_0 = max(float(np.max(Pobs_raw)), 1e-6)
    early = min(len(tP), 8)
    if early >= 3 and np.all(Pobs_raw[:early] > 0):
        coeffs = np.polyfit(tP[:early], np.log(Pobs_raw[:early]), 1)
        Rp0 = float(np.clip(-coeffs[0], cfg.Rp_eff_bounds[0], cfg.Rp_eff_bounds[1]))
    else:
        Rp0 = 0.02
    kpl0 = 0.01
    Rl0 = 0.02
    sP0, sL0 = 1.0, 1.0

    # ---------- build θ, bounds depending on whether tinj is fitted ----------
    if cfg.allow_curve_scales:
        if cfg.fit_tinj:
            # Ensure initial guesses are inside bounds (optional but nice)
            sP0 = float(np.clip(sP0, cfg.sP_bounds[0], cfg.sP_bounds[1]))
            sL0 = float(np.clip(sL0, cfg.sL_bounds[0], cfg.sL_bounds[1]))

            x0 = np.array([P0_0, kpl0, Rp0, Rl0, sP0, sL0, tinj0_guess], dtype=float)
            lb = np.array([
                0.0,
                cfg.kpl_bounds[0],
                cfg.Rp_eff_bounds[0],
                cfg.Rl_bounds[0],
                cfg.sP_bounds[0],   # <-- sP lower from config
                cfg.sL_bounds[0],   # <-- sL lower from config
                tinj_lo
            ], dtype=float)
            ub = np.array([
                np.inf,
                cfg.kpl_bounds[1],
                cfg.Rp_eff_bounds[1],
                cfg.Rl_bounds[1],
                cfg.sP_bounds[1],   # <-- sP upper from config
                cfg.sL_bounds[1],   # <-- sL upper from config
                tinj_hi
            ], dtype=float)
        else:
            # Ensure initial guesses are inside bounds (optional but nice)
            sP0 = float(np.clip(sP0, cfg.sP_bounds[0], cfg.sP_bounds[1]))
            sL0 = float(np.clip(sL0, cfg.sL_bounds[0], cfg.sL_bounds[1]))

            x0 = np.array([P0_0, kpl0, Rp0, Rl0, sP0, sL0], dtype=float)
            lb = np.array([
                0.0,
                cfg.kpl_bounds[0],
                cfg.Rp_eff_bounds[0],
                cfg.Rl_bounds[0],
                cfg.sP_bounds[0],   # <-- sP lower from config
                cfg.sL_bounds[0]    # <-- sL lower from config
            ], dtype=float)
            ub = np.array([
                np.inf,
                cfg.kpl_bounds[1],
                cfg.Rp_eff_bounds[1],
                cfg.Rl_bounds[1],
                cfg.sP_bounds[1],   # <-- sP upper from config
                cfg.sL_bounds[1]    # <-- sL upper from config
            ], dtype=float)
    else:
        if cfg.fit_tinj:
            x0 = np.array([P0_0, kpl0, Rp0, Rl0, tinj0_guess], dtype=float)
            lb = np.array([0.0,  cfg.kpl_bounds[0],  cfg.Rp_eff_bounds[0], cfg.Rl_bounds[0], tinj_lo])
            ub = np.array([np.inf, cfg.kpl_bounds[1], cfg.Rp_eff_bounds[1], cfg.Rl_bounds[1], tinj_hi])
        else:
            x0 = np.array([P0_0, kpl0, Rp0, Rl0], dtype=float)
            lb = np.array([0.0,  cfg.kpl_bounds[0],  cfg.Rp_eff_bounds[0], cfg.Rl_bounds[0]])
            ub = np.array([np.inf, cfg.kpl_bounds[1], cfg.Rp_eff_bounds[1], cfg.Rl_bounds[1]])

    # ---------- least-squares ----------
    if cfg.fit_tinj:
        # Wrapper to unpack tinj from θ
        def fun_fit(theta):
            if cfg.allow_curve_scales:
                *kin, sP, sL, tinj = theta
                theta_kin = (*kin, sP, sL)
            else:
                *kin, tinj = theta
                theta_kin = tuple(kin)
            # reuse the same residual builder; pass current tinj
            return _build_residuals(theta_kin, tP, Pobs_raw, tL, Lobs_raw, cfg, tinj)

        res = least_squares(fun_fit, x0=x0, bounds=(lb, ub), method='trf', max_nfev=5000, verbose=0)

        if cfg.allow_curve_scales:
            P0, kpl, Rp_eff, Rl, sP, sL, tinj = res.x
        else:
            P0, kpl, Rp_eff, Rl, tinj = res.x
            sP, sL = 1.0, 1.0

    else:
        # tinj is fixed; keep it out of θ and pass it separately to residuals
        def fun_fixed(theta):
            return _build_residuals(theta, tP, Pobs_raw, tL, Lobs_raw, cfg, tinj_for_residuals)

        res = least_squares(fun_fixed, x0=x0, bounds=(lb, ub), method='trf', max_nfev=5000, verbose=0)

        if cfg.allow_curve_scales:
            P0, kpl, Rp_eff, Rl, sP, sL = res.x
        else:
            P0, kpl, Rp_eff, Rl = res.x
            sP, sL = 1.0, 1.0
        tinj = tinj_for_residuals

    return FitResult(P0=P0, kpl=kpl, Rp_eff=Rp_eff, Rl=Rl, tinj=tinj,
                     sP=sP, sL=sL, success=res.success, message=res.message)

# ===== Orchestrator function =====
def run_fit_on_file(
    csv_path: str,
    *,
    target_peaks: list,
    target_peaks_labels: list,
    substratepeak: int,
    product4peak: int,
    startPoint: int,
    smoothing: bool,
    CFG: TwoSiteConfig,
    x_display_min: float,
    x_display_max: float,
    method_for_display: str = None,
):
    """
    Orchestrates one complete analysis without plotting.

    Parameters
    ----------
    csv_path : str
        Path to the integrated CSV.
    target_peaks, target_peaks_labels : list
        As defined in USER SETTINGS (Step 1).
    substratepeak, product4peak : int
        Indexes into target_peaks_labels for Pyruvate and Lactate.
    startPoint : int
        Number of initial points to drop before fitting.
    smoothing : bool
        Whether to apply Savitzky–Golay smoothing during import.
    CFG : TwoSiteConfig
        Model bounds/behavior (fit_tinj, bounds, scale limits, etc.).
    x_display_min, x_display_max : float
        Plot window limits (used to precompute dense model arrays for GUI plotting).
    method_for_display : str, optional
        If provided, overrides global `method` for y-selection during extraction
        (useful if GUI wants to temporarily display heights vs integrals).

    Returns
    -------
    payload : dict
        {
          # file info
          'file': <filename>,

          # raw series (after filtering + startPoint)
          'tP_raw': np.ndarray,
          'Pobs_raw': np.ndarray,
          'tL_raw': np.ndarray,
          'Lobs_raw': np.ndarray,

          # fit result (FitResult fields)
          'fit': FitResult,

          # derived/diagnostic values
          'T1p_like': float,    # ≈ 1/(Rp_eff - kpl) if positive
          'T1l_like': float,    # ≈ 1/Rl
          't0': float,          # time reference shift used in display alignment

          # dense display domain + model curves (precomputed for GUI plots)
          't_dense': np.ndarray,
          'Pmod_dense': np.ndarray,
          'Lmod_dense': np.ndarray,
          'ratio_dense': np.ndarray,         # L/P (scaled) on dense grid
          'ratio_asymptote': float,          # (sL/sP) * kpl / (Rl - Rp_eff) or np.inf

          # residuals evaluated on data points (normalized)
          'rP_norm': np.ndarray,
          'rL_norm': np.ndarray,

          # meta/context
          'method_used': 'integrals' | 'heights',
          'x_display_min': float,
          'x_display_max': float,
        }
    """
    # --- capture/override method if needed for display y-selection
    method_used = method_for_display if method_for_display is not None else globals().get("method", "integrals")

    # --- load and preprocess
    processed = import_new_peak_data(csv_path, target_peaks, use_smoothing=smoothing)
    filtered = create_filtered_arrays(processed, target_peaks)

    # --- extract series: P (pyruvate) and L (lactate)
    P_key = target_peaks_labels[substratepeak]
    L_key = target_peaks_labels[product4peak]
    tP_raw, Pobs_raw = extract_filtered_data(filtered, P_key, start_index=startPoint)
    tL_raw, Lobs_raw = extract_filtered_data(filtered, L_key, start_index=startPoint)

    # --- run fit
    fit = fit_two_site_joint(tP_raw, Pobs_raw, tL_raw, Lobs_raw, CFG)

    # --- derived T1-like values (no FA correction here)
    T1p_like = np.inf
    if fit.Rp_eff > fit.kpl:
        T1p_like = 1.0 / max(fit.Rp_eff - fit.kpl, 1e-12)
    T1l_like = 1.0 / max(fit.Rl, 1e-12)

    # --- display alignment (t=0 at first pyruvate point)
    t0 = float(tP_raw[0])
    tP = tP_raw - t0
    tL = tL_raw - t0

    # --- dense time grid for smooth curves in GUI plots
    t_dense_max = max(float(np.max(tP)) if tP.size else 0.0,
                      float(np.max(tL)) if tL.size else 0.0,
                      x_display_max)
    t_dense = np.linspace(x_display_min, t_dense_max, 600)

    # --- evaluate model curves on dense grid (scaled)
    Pmod_dense, Lmod_dense = two_site_closed_form(
        t=t_dense, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj,
        alpha_deg=CFG.flip_angle_deg, TR_s=CFG.TR_s
    )
    Pmod_dense *= fit.sP
    Lmod_dense *= fit.sL

    # L/P ratio (scaled); guard against division by tiny P
    eps = 1e-12
    ratio_dense = Lmod_dense / np.maximum(Pmod_dense, eps)

    # asymptotes (unscaled and scaled)
    Delta = fit.Rl - fit.Rp_eff
    if Delta > 0:
        ratio_asymptote_unscaled = fit.kpl / Delta
        ratio_asymptote = (fit.sL / fit.sP) * ratio_asymptote_unscaled
    else:
        ratio_asymptote_unscaled = np.inf
        ratio_asymptote = np.inf

    # --- compute normalized residuals on data points for diagnostics
    Pmod_data, _ = two_site_closed_form(
        t=tP, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj,
        alpha_deg=CFG.flip_angle_deg, TR_s=CFG.TR_s
    )
    Pmod_data *= fit.sP
    _, Lmod_data = two_site_closed_form(
        t=tL, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj,
        alpha_deg=CFG.flip_angle_deg, TR_s=CFG.TR_s
    )
    Lmod_data *= fit.sL

    wp = 1.0 / max(float(np.max(Pobs_raw)), 1e-9)
    wl = 1.0 / max(float(np.max(Lobs_raw)), 1e-9)
    rP_norm = wp * (Pmod_data - Pobs_raw)
    rL_norm = wl * (Lmod_data - Lobs_raw)

    # --- assemble payload for GUI
    payload = dict(
        file=Path(csv_path).name,
        # raw series
        tP_raw=tP, Pobs_raw=Pobs_raw,
        tL_raw=tL, Lobs_raw=Lobs_raw,
        # fit container
        fit=fit,
        # derived
        T1p_like=T1p_like,
        T1l_like=T1l_like,
        t0=t0,
        # dense display products
        t_dense=t_dense,
        Pmod_dense=Pmod_dense,
        Lmod_dense=Lmod_dense,
        ratio_dense=ratio_dense,
        ratio_asymptote=ratio_asymptote,
        ratio_asymptote_unscaled=ratio_asymptote_unscaled,
        # residuals on data points
        rP_norm=rP_norm,
        rL_norm=rL_norm,
        # meta
        method_used=method_used,
        x_display_min=float(x_display_min),
        x_display_max=float(x_display_max),
    )
    return payload

# =========================
# Step 3 — GUI (with export checkboxes)
# =========================
def _fig_2x2_from_payload(payload, y_max_p=None, y_max_l=None):
    """Build the full 2×2 figure (Pyruvate, Lactate, Residuals, L/P ratio) from run_fit_on_file payload."""
    tP = payload["tP_raw"]; Pobs = payload["Pobs_raw"]
    tL = payload["tL_raw"]; Lobs = payload["Lobs_raw"]
    t_dense = payload["t_dense"]
    Pmod_dense = payload["Pmod_dense"]; Lmod_dense = payload["Lmod_dense"]
    ratio_dense = payload["ratio_dense"]; ratio_asymptote = payload["ratio_asymptote"]
    rP = payload["rP_norm"]; rL = payload["rL_norm"]
    xmin = payload["x_display_min"]; xmax = payload["x_display_max"]

    fig, axs = plt.subplots(2, 2, figsize=(13, 8.8))
    axP, axL, axR, axQ = axs.ravel()

    # Pyruvate
    axP.scatter(tP, Pobs, s=18, alpha=0.8, label='Pyruvate (data)')
    axP.plot(t_dense, Pmod_dense, lw=2.0, label='Pyruvate (model)')
    axP.set_xlim(xmin, xmax); axP.set_xlabel('Time (s)'); axP.set_ylabel('Signal (a.u.)')
    if y_max_p is not None:
        axP.set_ylim(-float(y_max_p)/10, float(y_max_p))
    axP.set_title('Pyruvate'); axP.legend()

    # Lactate
    axL.scatter(tL, Lobs, s=18, alpha=0.8, label='Lactate (data)', color='tab:orange')
    axL.plot(t_dense, Lmod_dense, lw=2.0, label='Lactate (model)', color='tab:red')
    axL.set_xlim(xmin, xmax); axL.set_xlabel('Time (s)'); axL.set_ylabel('Signal (a.u.)')
    if y_max_l is not None:
        axL.set_ylim(-float(y_max_l)/10, float(y_max_l))
    axL.set_title('Lactate'); axL.legend()

    # Residuals (normalized)
    axR.axhline(0, color='k', lw=1)
    axR.plot(tP, rP, '.', ms=6, label='Pyruvate residuals (norm)')
    axR.plot(tL, rL, '.', ms=6, label='Lactate residuals (norm)', color='tab:red')
    axR.set_xlim(xmin, xmax); axR.set_xlabel('Time (s)')
    axR.set_ylabel('Norm. residual'); axR.set_title('Residuals'); axR.legend()

    # Ratio (L/P, scaled)
    axQ.plot(t_dense, ratio_dense, lw=2.0)
    axQ.set_xlim(xmin, xmax); axQ.set_xlabel('Time (s)'); axQ.set_ylabel('Lactate / Pyruvate (model)')
    axQ.set_title('L/P Ratio (model fit)')
    if np.isfinite(ratio_asymptote):
        axQ.axhline(ratio_asymptote, ls=':', lw=1)
        axQ.annotate(f"asymptote = {ratio_asymptote:.3g}",
                     xy=(axQ.get_xlim()[0], ratio_asymptote),
                     xytext=(5, 5), textcoords='offset points')
    fig.tight_layout()
    return fig

def _result_row_from_payload(payload):
    """Flatten key metrics into a dict row for CSV export and on-screen printout."""
    fit = payload["fit"]
    row = {
        "file": payload["file"],
        "P0": fit.P0,
        "kpl": fit.kpl,
        "Rp_eff": fit.Rp_eff,
        "Rl": fit.Rl,
        "tinj": fit.tinj,
        "sP": fit.sP,
        "sL": fit.sL,
        "T1p_like": payload["T1p_like"],
        "T1l_like": payload["T1l_like"],
        "L_over_P_asymptote_unscaled": payload["ratio_asymptote_unscaled"],  # kpl/(Rl-Rp_eff)
        "L_over_P_asymptote_scaled": payload["ratio_asymptote"],             # (sL/sP)*unscaled
        "success": fit.success,
        "message": fit.message,
    }
    return row

def _derive_outstem_from_name(stem: str, tag: str):
    """
    From a filename stem, produce the naming per spec:
      take substring from first '(' onward if present; else use '_' + stem
      then append _{tag}_atYYYYMMDD-HHMM
    """
    if "(" in stem:
        name_part = stem[stem.index("("):]
    else:
        name_part = "_" + stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{name_part}_{tag}_at{timestamp}"

def extract_exptdate(filename: str):
    """Extract the 6-digit experiment date from filename after 'integrated_data_'."""
    m = re.search(r"integrated_data_(\d{6})", filename)
    return m.group(1) if m else None

class FitGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Site Fit GUI")
        self.resize(1100, 800)
        self.output_folder = None
        self.canvases = []  # keep strong references to canvases

        root = QVBoxLayout(self)

        # --- Top controls (input/output folders)
        top = QHBoxLayout()
        self.btn_in = QPushButton("Select Input Folder")
        self.btn_in.clicked.connect(self.on_select_input)
        self.btn_out = QPushButton("Select Output Folder")
        self.btn_out.clicked.connect(self.on_select_output)
        top.addWidget(self.btn_in)
        top.addWidget(self.btn_out)
        root.addLayout(top)

        # --- Export checkboxes
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Exports:"))
        self.cb_png = QCheckBox("Plot PNG")
        self.cb_pdf = QCheckBox("Plot PDF")
        self.cb_csv_each = QCheckBox("Per-file CSV")
        self.cb_csv_summary = QCheckBox("Summary CSV")
        # defaults (enable everything)
        self.cb_png.setChecked(True)
        self.cb_pdf.setChecked(True)
        self.cb_csv_each.setChecked(True)
        self.cb_csv_summary.setChecked(True)
        for cb in (self.cb_png, self.cb_pdf, self.cb_csv_each, self.cb_csv_summary):
            opts.addWidget(cb)
        # --- Y-axis max limits for top plots
        ylims = QHBoxLayout()
        ylims.addWidget(QLabel("Y max (Pyruvate, Lactate):"))
        self.sb_ymax_p = QDoubleSpinBox()
        self.sb_ymax_p.setRange(0, 1e12)
        self.sb_ymax_p.setDecimals(1)
        self.sb_ymax_p.setValue(3500.0)  # default for Pyruvate

        self.sb_ymax_l = QDoubleSpinBox()
        self.sb_ymax_l.setRange(0, 1e12)
        self.sb_ymax_l.setDecimals(1)
        self.sb_ymax_l.setValue(50.0)    # default for Lactate

        ylims.addWidget(self.sb_ymax_p)
        ylims.addWidget(self.sb_ymax_l)
        root.addLayout(ylims)
        root.addLayout(opts)

        # --- File list + Run
        middle = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(self.file_list.MultiSelection)
        middle.addWidget(self.file_list, 2)

        right = QVBoxLayout()
        self.btn_run = QPushButton("Run Fits")
        self.btn_run.clicked.connect(self.on_run)
        right.addWidget(self.btn_run)
        right.addStretch(1)
        middle.addLayout(right, 1)
        root.addLayout(middle)

        # --- Tabs for results and plots
        self.tabs = QTabWidget()
        self.output_tab = QTextEdit()
        self.output_tab.setReadOnly(True)
        self.tabs.addTab(self.output_tab, "Results")
        root.addWidget(self.tabs)

    def on_select_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not folder:
            return
        self.file_list.clear()
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                self.file_list.addItem(str(Path(folder) / f))
        self.output_tab.append(f"Loaded input folder: {folder}")

    def on_select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        self.output_folder = Path(folder)
        self.output_tab.append(f"Output folder set: {self.output_folder}")

    def on_run(self):
        selected = [it.text() for it in self.file_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "No files", "Please select at least one CSV.")
            return
        if self.output_folder is None and any([self.cb_png.isChecked(), self.cb_pdf.isChecked(),
                                               self.cb_csv_each.isChecked(), self.cb_csv_summary.isChecked()]):
            QMessageBox.warning(self, "No output folder", "Select an output folder for exports.")
            return

        # Clear previous tabs (except results)
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)
        self.canvases.clear()

        do_png = self.cb_png.isChecked()
        do_pdf = self.cb_pdf.isChecked()
        do_csv_each = self.cb_csv_each.isChecked()
        do_csv_summary = self.cb_csv_summary.isChecked()

        all_rows = []
        self.output_tab.append("\nRunning fits...\n")

        for file in selected:
            try:
                payload = run_fit_on_file(
                    file,
                    target_peaks=target_peaks,
                    target_peaks_labels=target_peaks_labels,
                    substratepeak=substratepeak,
                    product4peak=product4peak,
                    startPoint=startPoint,
                    smoothing=smoothing,
                    CFG=CFG,
                    x_display_min=x_display_min,
                    x_display_max=x_display_max,
                )
            except Exception as e:
                self.output_tab.append(f"ERROR processing {Path(file).name}: {e}")
                continue

            # Print results to GUI
            row = _result_row_from_payload(payload)
            all_rows.append(row)
            self.output_tab.append(f"=== {row['file']} ===")
            for k, v in row.items():
                if k != "file":
                    self.output_tab.append(f"{k}: {v}")
            self.output_tab.append("----")

            # Create figure and embed as a tab
            fig = _fig_2x2_from_payload(
                payload,
                y_max_p=self.sb_ymax_p.value(),
                y_max_l=self.sb_ymax_l.value(),
            )
            canvas = FigureCanvas(fig)
            canvas.draw()                         # <— make sure it renders
            self.tabs.addTab(canvas, Path(file).name)
            self.canvases.append(canvas)          # <— keep a reference

            # Exports
            if self.output_folder is not None:
                stem = Path(file).stem
                exptdate = extract_exptdate(stem) or ""
                prefix = f"{exptdate}_" if exptdate else ""
                outstem = f"{prefix}{_derive_outstem_from_name(stem, 'fit')}"

                if do_csv_each:
                    out_csv = self.output_folder / f"{outstem}.csv"
                    pd.DataFrame([row]).to_csv(out_csv, index=False)

                if do_png:
                    out_png = self.output_folder / f"{outstem}.png"
                    fig.savefig(out_png, dpi=300)

                if do_pdf:
                    out_pdf = self.output_folder / f"{outstem}.pdf"
                    fig.savefig(out_pdf)

            plt.close(fig)  # free memory; canvas keeps a reference for display

        # Summary CSV (optional)
        if do_csv_summary and all_rows and self.output_folder is not None:
            first_file = selected[0] if selected else ""
            exptdate = extract_exptdate(os.path.basename(first_file)) or ""
            datepart = f"{exptdate}_" if exptdate else ""
            summary_path = self.output_folder / f"summary_{datepart}fit_at{datetime.now().strftime('%Y%m%d-%H%M')}.csv"

            pd.DataFrame(all_rows).to_csv(summary_path, index=False)
            self.output_tab.append(f"\nSummary saved to {summary_path}")

        self.output_tab.append("\nDone.\n")

def main():
    app = QApplication(sys.argv)
    gui = FitGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
