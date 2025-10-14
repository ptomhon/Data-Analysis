# TimeSeries_TwoSiteFit.py  (replaces substrate-prefit + product ODE with joint two-site fit)
# Requires: numpy, pandas, scipy, matplotlib
# Uses your original I/O structures (import_new_peak_data, create_filtered_arrays, extract_filtered_data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

# =========================
# USER SETTINGS (same style as your script)
# =========================
file_path = r"D:\WSU\Projects\Eukaryote Experiments\2025-08 Pilot Leukemia Cell Experiments\Data Analysis\2025-09-26 SVD Proc-Int\integrated_data_250926-114829 (PYR70_1).csv"
# file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\SVD Proc-Int 2025 09 05\integrated_data_250826-112726 (PYR07_1).csv"
# file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\SVD Proc-Int 2025 09 05\integrated_data_250826-122114 (PYR07_2).csv"
# file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\SVD Proc-Int 2025 09 05\integrated_data_250826-130032 (PYR07_3).csv"
# file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\SVD Proc-Int 2025 09 05\integrated_data_250826-151918 (PYR07_5).csv"
# file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\SVD Proc-Int 2025 09 05\integrated_data_250826-160303 (PYR70_6).csv"
# file_path = r"D:\WSU\Projects\2025-08 Leukemia Cell Experiments\Data Analysis\SVD Proc-Int 2025 09 05\integrated_data_250826-164823 (PYR70_7).csv"


save_folder = "two_site_plots"     # folder name for any saved plots (optional, not created here)
target_peaks = [182.5, 178, 170, 160, 124]
target_peaks_labels = [f'P{p}' for p in target_peaks]
method = 'integrals'               # 'integrals' or 'heights' (unchanged from your code)

# Peaks (use your original indexing: substrate at index 2; lactate at index 0 in your example config)
substratepeak = 2                  # pyruvate
product4peak = 0                   # lactate
smoothing = False
startPoint = 1                     # drop first N points per your workflow

# Optional plotting window
x_display_min = -5
x_display_max = 400

# =========================
# Two-site model config
# =========================
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

# =========================
# Utilities
# =========================
def _small_flip_increment(alpha_deg: float, TR_s: float) -> float:
    """Return additional loss rate from repeated small flip angle excitations."""
    alpha = np.deg2rad(alpha_deg)
    return -np.log(np.cos(alpha)) / TR_s

def _safe_exp(x):
    # prevent overflow in exp if someone passes extreme guesses
    x = np.clip(x, -700, 700)
    return np.exp(x)

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

# =========================
# Two-site closed-form model
# =========================
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

# =========================
# Joint residual builder
# =========================
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


# =========================
# Fitter
# =========================
@dataclass
class FitResult:
    P0: float
    kpl: float
    Rp_eff: float
    Rl: float
    tinj: float
    sP: float
    sL: float
    success: bool
    message: str

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


def main():
    # ---- Load data (same structures as your original code)
    processed = import_new_peak_data(file_path, target_peaks, use_smoothing=smoothing)
    filtered = create_filtered_arrays(processed, target_peaks)

    # ---- Extract P (substrate) and L (lactate)
    tP_raw, Pobs_raw = extract_filtered_data(filtered, target_peaks_labels[substratepeak], start_index=startPoint)
    tL_raw, Lobs_raw = extract_filtered_data(filtered, target_peaks_labels[product4peak], start_index=startPoint)

    # ---- Fit joint two-site model
    fit = fit_two_site_joint(tP_raw, Pobs_raw, tL_raw, Lobs_raw, CFG)

    # ---- Report
    print("\n=== Joint Two-Site Fit (Pyruvate↦Lactate) ===")
    print(f"Success: {fit.success}  ({fit.message})")
    print(f"P0                 : {fit.P0:.6g}")
    print(f"kpl (s^-1)         : {fit.kpl:.6g}")
    print(f"Rp_eff (s^-1)      : {fit.Rp_eff:.6g}")
    print(f"Rl (s^-1)          : {fit.Rl:.6g}")
    mode = "fitted" if CFG.fit_tinj else f"fixed @ {CFG.fixed_tinj_value:.3f} s"
    print(f"tinj (s)           : {fit.tinj:.6g}  [{mode}]")
    print(f"sP, sL (scales)    : {fit.sP:.4g}, {fit.sL:.4g}")

    # Derived T1 estimates (if you want to interpret as T1-like without flip-angle correction)
    T1p_like = np.inf
    if fit.Rp_eff > fit.kpl:
        T1p_like = 1.0 / max(fit.Rp_eff - fit.kpl, 1e-12)
    T1l_like = 1.0 / max(fit.Rl, 1e-12)
    print(f"Derived T1p_like(s): {T1p_like:.3f}   (≈ 1/(Rp_eff - kpl))")
    print(f"Derived T1l_like(s): {T1l_like:.3f}   (≈ 1/Rl)")


    # ---- Build aligned display times (t=0 at first kept P point) and model curves
    t0 = float(tP_raw[0])
    tP = tP_raw - t0
    tL = tL_raw - t0
    t_dense = np.linspace(x_display_min, max(np.max(tP), np.max(tL), x_display_max), 600)

    Pmod_dense, Lmod_dense = two_site_closed_form(
        t=t_dense, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj,
        alpha_deg=CFG.flip_angle_deg, TR_s=CFG.TR_s
    )
    Pmod_dense *= fit.sP
    Lmod_dense *= fit.sL


    # ---- L/P ratio on the model curves (safe division)
    eps = 1e-12
    ratio_dense = Lmod_dense / np.maximum(Pmod_dense, eps)

    # (Optional) ignore times where P is essentially zero to avoid meaningless blowups
    valid = Pmod_dense > (np.max(Pmod_dense) * 1e-4)
    ratio_valid = np.where(valid, ratio_dense, np.nan)


    # ---- Asymptote of the *scaled* ratio
    Delta = fit.Rl - fit.Rp_eff  # R_l - R_p,eff
    if Delta > 0:
        ratio_asymptote_unscaled = fit.kpl / Delta
        ratio_asymptote = (fit.sL/fit.sP) * ratio_asymptote_unscaled
    else:
        ratio_asymptote_unscaled = np.inf
        ratio_asymptote = np.inf


    # ---- Plateau criterion: first time within X% of scaled asymptote
    PLATEAU_FRACTION = 0.99  # or 0.95
    if np.isfinite(ratio_asymptote):
        target = PLATEAU_FRACTION * ratio_asymptote

        # ignore divide-by-small artifacts where P ~ 0
        validP = Pmod_dense > (np.max(Pmod_dense) * 1e-4)
        ratio_valid = np.where(validP, ratio_dense, np.nan)

        reached = np.where(ratio_valid >= target)[0]
        if reached.size > 0:
            t_plateau = t_dense[reached[0]]
            ratio_at_plateau = ratio_valid[reached[0]]
        else:
            t_plateau = np.nan
            ratio_at_plateau = np.nan
    else:
        t_plateau = np.nan
        ratio_at_plateau = np.nan

    # Report asymptotes and plateau for clarity
    if np.isfinite(ratio_asymptote_unscaled):
        print(f"Asymptotic L/P (unscaled model): {ratio_asymptote_unscaled:.6g}  [= kpl / (Rl - Rp_eff)]")
    else:
        print("Asymptotic L/P (unscaled model): diverges (Rl <= Rp_eff)")

    if np.isfinite(ratio_asymptote):
        print(f"Asymptotic L/P (scaled, plotted): {ratio_asymptote:.6g}  [= (sL/sP) * kpl / (Rl - Rp_eff)]")
        if np.isfinite(t_plateau):
            print(f"Plateau reached (≥{PLATEAU_FRACTION*100:.0f}% of scaled asymptote) at t = {t_plateau:.3f} s")
        else:
            print(f"Plateau not reached within plotted window (≥{PLATEAU_FRACTION*100:.0f}% criterion).")
    else:
        print("Asymptotic L/P (scaled): diverges (Rl <= Rp_eff); plateau time undefined.")


    # ---- Plots
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.8))
    axP, axL, axR, axQ = axs.ravel()   # axQ will be the Ratio panel

    # Pyruvate
    axP.scatter(tP, Pobs_raw, s=18, alpha=0.8, label='Pyruvate (data)')
    axP.plot(t_dense, Pmod_dense, lw=2.0, label='Pyruvate (model)')
    axP.set_xlim(x_display_min, x_display_max); axP.set_xlabel('Time (s)'); axP.set_ylabel('Signal (a.u.)')
    axP.set_title('Pyruvate'); axP.legend()

    # Lactate
    axL.scatter(tL, Lobs_raw, s=18, alpha=0.8, label='Lactate (data)', color='tab:orange')
    # Evaluate L at the same dense display times
    _, L_dense = two_site_closed_form(t=t_dense, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj)
    L_dense *= fit.sL
    axL.plot(t_dense, L_dense, lw=2.0, label='Lactate (model)', color='tab:red')
    axL.set_xlim(x_display_min, x_display_max); axL.set_xlabel('Time (s)'); axL.set_ylabel('Signal (a.u.)')
    axL.set_title('Lactate'); axL.legend()

    # Residuals (normalized)
    from numpy import max as npmax
    wp = 1.0 / max(npmax(Pobs_raw), 1e-9)
    wl = 1.0 / max(npmax(Lobs_raw), 1e-9)

    # model evaluated on data points
    Pmod_data, _ = two_site_closed_form(t=tP, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj)
    Pmod_data *= fit.sP
    _, Lmod_data = two_site_closed_form(t=tL, P0=fit.P0, kpl=fit.kpl, Rp_eff=fit.Rp_eff, Rl=fit.Rl, tinj=fit.tinj)
    Lmod_data *= fit.sL

    rP = wp * (Pmod_data - Pobs_raw)
    rL = wl * (Lmod_data - Lobs_raw)
    axR.axhline(0, color='k', lw=1)
    axR.plot(tP, rP, '.', ms=6, label='Pyruvate residuals (norm)')
    axR.plot(tL, rL, '.', ms=6, label='Lactate residuals (norm)', color='tab:red')
    axR.set_xlim(x_display_min, x_display_max); axR.set_xlabel('Time (s)')
    axR.set_ylabel('Norm. residual'); axR.set_title('Residuals'); axR.legend()

    # ---- Ratio panel (L/P, model)
    axQ.plot(t_dense, ratio_dense, lw=2.0)
    axQ.set_xlim(x_display_min, x_display_max)
    axQ.set_xlabel('Time (s)')
    axQ.set_ylabel('Lactate / Pyruvate (model)')
    axQ.set_title('L/P Ratio (model fit)')
    # Mark the maximum if it exists
    # Draw the scaled asymptote at the correct height
    if np.isfinite(ratio_asymptote):
        axQ.axhline(ratio_asymptote, ls=':', lw=1)
        axQ.annotate(f"asymptote = {ratio_asymptote:.3g}",
                    xy=(axQ.get_xlim()[0], ratio_asymptote),
                    xytext=(5, 5), textcoords='offset points')

    # Mark first time reaching the plateau threshold
    if np.isfinite(t_plateau):
        axQ.axvline(t_plateau, ls='--', lw=1)
        axQ.annotate(f"{PLATEAU_FRACTION*100:.0f}% at {t_plateau:.1f}s",
                    xy=(t_plateau, ratio_at_plateau),
                    xytext=(5, -15), textcoords='offset points')



    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
