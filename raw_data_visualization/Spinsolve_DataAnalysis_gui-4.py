# Spinsolve_DataAnalysis_gui-4.py — Multi-Peak Deconvolution enhancement
# Updates from gui-3:
# 1) Mode toggle: "Basic Single Integration" vs "Multi-Peak Deconvolution"
# 2) Deconvolution mode: auto peak detection, Lorentzian/pseudo-Voigt fitting,
#    hybrid integration with fractional overlap attribution, color-coded regions
# 3) Lineshape toggle: Pure Lorentzian vs Pseudo-Voigt
# 4) Individual Lorentzian/Voigt traces overlaid as dashed lines
# 5) Integral Table updated with per-peak columns in deconvolution mode
# 6) Peaks labelled by ppm position
# 7) Mode switch resets figures but preserves loaded data & phase offsets

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize_scalar, curve_fit
from scipy.signal import find_peaks

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QDial, QTextEdit,
    QListWidget, QListWidgetItem, QSizePolicy, QDialog, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

# Compatibility: numpy >= 2.0 renamed trapezoid → trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or np.trapezoid

# ---------------------------
# Processing utilities
# ---------------------------

def exponential_apodization(fid, time, lb):
    return fid * np.exp(-lb * time)

def zero_fill(fid, target_length=65536):
    if len(fid) >= target_length:
        return fid
    return np.pad(fid, (0, target_length - len(fid)), mode='constant')

def apply_phase_correction(fid, angle_deg):
    return fid * np.exp(-1j * np.deg2rad(angle_deg))

def compute_fft(fid, dt):
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fid), d=dt))
    return freqs, spectrum

def optimize_phase(fid, time_arr):
    dt = float(np.mean(np.diff(time_arr)))

    def objective(angle):
        corr = apply_phase_correction(fid, angle)
        _, spectrum = compute_fft(corr, dt)
        return -np.sum(np.real(spectrum) ** 2)

    res = minimize_scalar(objective, bounds=(0.0, 10.0), method='bounded')
    return float(res.x)

def read_fid_csv(path):
    df = pd.read_csv(path, header=None)
    time = df.iloc[:, 0].astype(float).to_numpy() / 1000.0  # ms -> s
    real = df.iloc[:, 1].astype(float).to_numpy()
    imag = df.iloc[:, 2].astype(float).to_numpy()
    fid = real + 1j * imag
    return fid, time

def ppm_conversion(freqs):
    return -((freqs - (2500 - 639.08016399999997)) / 15.507665)

def get_processed_spectrum(fid, time, lb, user_phase_offset):
    """Common processing pipeline: apodize, phase, FFT, baseline correct.
    Returns (ppm, real, opt_phase, total_phase, freqs)."""
    fid_apod = exponential_apodization(fid, time, lb)
    opt_phase = optimize_phase(fid_apod, time)
    total_phase = opt_phase + user_phase_offset
    fid_corr = apply_phase_correction(fid_apod, total_phase)
    fid_zf = zero_fill(fid_corr)

    dt = float(time[1] - time[0])
    freqs, spectrum = compute_fft(fid_zf, dt)
    ppm = ppm_conversion(freqs)

    real = np.real(spectrum)
    baseline = np.mean(real[np.abs(freqs) > 200]) if np.any(np.abs(freqs) > 200) else 0.0
    real = real - baseline

    return ppm, real, opt_phase, total_phase, freqs


# ---------------------------
# Basic single integration
# ---------------------------

def compute_integral(fid, time, lb, user_phase_offset, ppm_start, ppm_stop):
    """Compute just the integral value without generating a plot."""
    ppm, real, _, _, _ = get_processed_spectrum(fid, time, lb, user_phase_offset)

    mask = (ppm > ppm_stop) & (ppm < ppm_start)
    ppm_vals = ppm[mask]
    real_vals = real[mask]

    order = np.argsort(ppm_vals)
    ppm_vals = ppm_vals[order]
    real_vals = real_vals[order]

    integral = float(_trapezoid(real_vals, ppm_vals)) if len(ppm_vals) > 1 else 0.0
    return integral

def process_and_plot(fid, time, lb, user_phase_offset, ppm_start, ppm_stop):
    ppm, real, opt_phase, total_phase, freqs = get_processed_spectrum(
        fid, time, lb, user_phase_offset)

    mask = (ppm > ppm_stop) & (ppm < ppm_start)
    ppm_vals = ppm[mask]
    real_vals = real[mask]

    order = np.argsort(ppm_vals)
    ppm_vals = ppm_vals[order]
    real_vals = real_vals[order]

    integral = float(_trapezoid(real_vals, ppm_vals)) if len(ppm_vals) > 1 else 0.0

    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(ppm, real, lw=1)
    ax.set_xlabel("ppm")
    ax.set_ylabel("Real FFT (a.u.)")
    ax.invert_xaxis()

    if len(ppm_vals) > 1:
        ax.fill_between(ppm_vals, real_vals, alpha=0.2)

    ax.set_title(
        f"Integral: {integral:,.3f} | Phase offset: {user_phase_offset:.2f}° (opt {opt_phase:.2f}°)")
    fig.tight_layout()
    return fig


# ---------------------------
# Lineshape models
# ---------------------------

def lorentzian(x, amplitude, center, gamma):
    """Pure Lorentzian lineshape."""
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)

def pseudo_voigt(x, amplitude, center, gamma, eta):
    """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian, eta in [0,1].
    gamma is the HWHM for both components."""
    L = amplitude / (1.0 + ((x - center) / gamma) ** 2)
    G = amplitude * np.exp(-np.log(2) * ((x - center) / gamma) ** 2)
    return eta * L + (1.0 - eta) * G

def _build_multi_peak_model(n_peaks, lineshape):
    """Return a callable model(x, *params) for n_peaks of the given lineshape."""
    if lineshape == "lorentzian":
        def model(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                A, x0, g = params[3*i], params[3*i+1], params[3*i+2]
                y += lorentzian(x, A, x0, g)
            return y
        params_per_peak = 3
    else:  # pseudo-voigt
        def model(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                A, x0, g, eta = params[4*i], params[4*i+1], params[4*i+2], params[4*i+3]
                y += pseudo_voigt(x, A, x0, g, eta)
            return y
        params_per_peak = 4
    return model, params_per_peak


# ---------------------------
# Multi-peak deconvolution engine
# ---------------------------

def deconvolve_spectrum(ppm_vals, real_vals, n_peaks_requested, lineshape="lorentzian"):
    """
    Detect peaks, fit multi-peak lineshape model, compute hybrid integrals.

    Returns:
        peak_results: list of dicts with keys:
            'center_ppm', 'integral', 'fit_params', 'ppm_region', 'signal_region',
            'attributed_signal'
        fit_curve: array of fitted sum over ppm_vals
        individual_curves: list of arrays, one per peak
        n_found: number of peaks actually found
        warning: str or None
    """
    warning = None

    # --- Peak detection ---
    # find_peaks works on the data; we want the tallest n_peaks_requested
    # Use prominence to rank peaks
    peaks_idx, properties = find_peaks(real_vals, prominence=0)
    if len(peaks_idx) == 0:
        # Fallback: just take the maximum
        peaks_idx = np.array([np.argmax(real_vals)])
        properties = {'prominences': np.array([real_vals[peaks_idx[0]]])}

    prominences = properties.get('prominences', real_vals[peaks_idx])
    # Sort by prominence descending, take top n
    rank_order = np.argsort(prominences)[::-1]
    n_found = min(n_peaks_requested, len(peaks_idx))
    if n_found < n_peaks_requested:
        warning = (f"Requested {n_peaks_requested} peaks but only {n_found} "
                   f"found with sufficient prominence. Proceeding with {n_found}.")
    top_idx = peaks_idx[rank_order[:n_found]]
    # Sort selected peaks by ppm position (ascending ppm)
    top_idx = top_idx[np.argsort(ppm_vals[top_idx])]

    # --- Initial guesses ---
    model_func, params_per_peak = _build_multi_peak_model(n_found, lineshape)

    p0 = []
    lower = []
    upper = []
    ppm_range = ppm_vals[-1] - ppm_vals[0] if len(ppm_vals) > 1 else 1.0

    for idx in top_idx:
        A_guess = real_vals[idx]
        x0_guess = ppm_vals[idx]
        # Estimate gamma from half-height width (rough)
        half_h = A_guess / 2.0
        above_half = np.where(real_vals > half_h)[0]
        if len(above_half) > 1:
            # Find the width segment containing this peak
            # Split above_half into contiguous groups, find the one containing idx
            diffs = np.diff(above_half)
            splits = np.where(diffs > 1)[0] + 1
            groups = np.split(above_half, splits)
            gamma_guess = 0.1  # default
            for grp in groups:
                if idx in grp or (grp[0] <= idx <= grp[-1]):
                    gamma_guess = max(0.02, (ppm_vals[grp[-1]] - ppm_vals[grp[0]]) / 2.0)
                    break
        else:
            gamma_guess = 0.1

        if lineshape == "lorentzian":
            p0.extend([A_guess, x0_guess, gamma_guess])
            lower.extend([0, x0_guess - 2.0, 0.005])
            upper.extend([max(A_guess * 5, A_guess + 1.0), x0_guess + 2.0, max(ppm_range / 2, 0.2)])
        else:
            p0.extend([A_guess, x0_guess, gamma_guess, 0.5])
            lower.extend([0, x0_guess - 2.0, 0.005, 0.0])
            upper.extend([max(A_guess * 5, A_guess + 1.0), x0_guess + 2.0, max(ppm_range / 2, 0.2), 1.0])

    # --- Curve fit ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(model_func, ppm_vals, real_vals, p0=p0,
                                bounds=(lower, upper), maxfev=20000)
    except (RuntimeError, ValueError):
        # If fit fails (convergence or bounds issue), use initial guesses
        popt = np.array(p0)
        if warning:
            warning += " Fit did not converge; using initial estimates."
        else:
            warning = "Fit did not converge; using initial estimates."

    # --- Evaluate individual curves ---
    individual_curves = []
    fitted_centers = []
    for i in range(n_found):
        if lineshape == "lorentzian":
            A, x0, g = popt[3*i], popt[3*i+1], popt[3*i+2]
            curve_i = lorentzian(ppm_vals, A, x0, g)
            fitted_centers.append(x0)
        else:
            A, x0, g, eta = popt[4*i], popt[4*i+1], popt[4*i+2], popt[4*i+3]
            curve_i = pseudo_voigt(ppm_vals, A, x0, g, eta)
            fitted_centers.append(x0)
        individual_curves.append(curve_i)

    fit_curve = model_func(ppm_vals, *popt)

    # --- Fractional attribution & hybrid integration ---
    # For each point, compute the fraction each peak contributes to the total fit
    total_fit = np.zeros_like(ppm_vals)
    for c in individual_curves:
        total_fit += c
    # Avoid division by zero
    total_fit_safe = np.where(total_fit > 1e-12, total_fit, 1e-12)

    fractions = [c / total_fit_safe for c in individual_curves]

    # Hybrid signal strategy:
    #   For each peak, compute a "raw" hybrid that uses the larger of:
    #     - fractional_actual = fraction_i * actual_signal  (captures real shape)
    #     - individual_fit_i                                (fills in overlap zones)
    #   Then normalize all hybrids so their sum equals the actual signal at every
    #   point, preventing over-counting while preserving the relative distribution.

    raw_hybrids = []
    for i in range(n_found):
        attributed_actual = fractions[i] * np.maximum(real_vals, 0.0)
        attributed_fit = individual_curves[i]
        hybrid = np.maximum(attributed_actual, attributed_fit)
        hybrid = np.maximum(hybrid, 0.0)
        raw_hybrids.append(hybrid)

    # Normalize: scale so sum of hybrids = actual signal (clamped non-negative)
    raw_sum = np.zeros_like(ppm_vals)
    for h in raw_hybrids:
        raw_sum += h
    raw_sum_safe = np.where(raw_sum > 1e-12, raw_sum, 1e-12)
    actual_nonneg = np.maximum(real_vals, 0.0)

    peak_results = []
    for i in range(n_found):
        # Scale each hybrid proportionally so sum matches actual data
        hybrid_normalized = raw_hybrids[i] * (actual_nonneg / raw_sum_safe)

        integral_i = float(_trapezoid(hybrid_normalized, ppm_vals)) if len(ppm_vals) > 1 else 0.0

        peak_results.append({
            'center_ppm': fitted_centers[i],
            'integral': integral_i,
            'fit_params': (popt[params_per_peak*i:params_per_peak*(i+1)]).tolist(),
            'attributed_signal': hybrid_normalized,
        })

    return peak_results, fit_curve, individual_curves, n_found, warning


# ---------------------------
# Deconvolution plot
# ---------------------------

# Color cycle for peak regions
PEAK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def compute_deconv_integrals(fid, time, lb, user_phase_offset, ppm_start, ppm_stop,
                              n_peaks, lineshape):
    """Compute deconvolution integrals without plotting. Returns (peak_results, warning)."""
    ppm, real, opt_phase, total_phase, freqs = get_processed_spectrum(
        fid, time, lb, user_phase_offset)

    mask = (ppm > ppm_stop) & (ppm < ppm_start)
    ppm_w = ppm[mask]
    real_w = real[mask]
    order = np.argsort(ppm_w)
    ppm_w = ppm_w[order]
    real_w = real_w[order]

    if len(ppm_w) < 5:
        return [], "Integration window too narrow."

    peak_results, _, _, _, warning = deconvolve_spectrum(
        ppm_w, real_w, n_peaks, lineshape)
    return peak_results, warning


def process_and_plot_deconv(fid, time, lb, user_phase_offset, ppm_start, ppm_stop,
                            n_peaks, lineshape):
    """Full deconvolution processing + plot generation."""
    ppm, real, opt_phase, total_phase, freqs = get_processed_spectrum(
        fid, time, lb, user_phase_offset)

    mask = (ppm > ppm_stop) & (ppm < ppm_start)
    ppm_w = ppm[mask]
    real_w = real[mask]
    order = np.argsort(ppm_w)
    ppm_w = ppm_w[order]
    real_w = real_w[order]

    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(ppm, real, lw=1, color='black', zorder=5)
    ax.set_xlabel("ppm")
    ax.set_ylabel("Real FFT (a.u.)")
    ax.invert_xaxis()

    # Zoom to integration region with margin so deconvolution detail is visible
    margin = (ppm_start - ppm_stop) * 0.1
    ax.set_xlim(ppm_start + margin, ppm_stop - margin)

    warning = None
    if len(ppm_w) < 5:
        ax.set_title("Integration window too narrow")
        fig.tight_layout()
        return fig, warning

    peak_results, fit_curve, individual_curves, n_found, warning = deconvolve_spectrum(
        ppm_w, real_w, n_peaks, lineshape)

    # Plot color-coded attributed regions
    for i, pr in enumerate(peak_results):
        color = PEAK_COLORS[i % len(PEAK_COLORS)]
        ax.fill_between(ppm_w, pr['attributed_signal'], alpha=0.25, color=color,
                        label=f"{pr['center_ppm']:.1f} ppm: {pr['integral']:,.1f}",
                        zorder=2)

    # Plot individual fit curves as dashed lines
    for i, curve in enumerate(individual_curves):
        color = PEAK_COLORS[i % len(PEAK_COLORS)]
        ax.plot(ppm_w, curve, '--', lw=1.2, color=color, alpha=0.8, zorder=4)

    # Plot total fit as thin dotted line
    ax.plot(ppm_w, fit_curve, ':', lw=1, color='gray', alpha=0.6, zorder=3,
            label='Total fit')

    # Build title with per-peak integrals
    total_integral = sum(pr['integral'] for pr in peak_results)
    parts = [f"{pr['center_ppm']:.1f}: {pr['integral']:,.1f}" for pr in peak_results]
    integrals_str = " | ".join(parts)
    title = (f"{integrals_str} | Total: {total_integral:,.1f}\n"
             f"Phase offset: {user_phase_offset:.2f}° (opt {opt_phase:.2f}°)")
    ax.set_title(title, fontsize=9)

    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    return fig, warning


# ---------------------------
# Copyable Table Widget
# ---------------------------

class CopyableTableWidget(QTableWidget):
    """QTableWidget with Ctrl+C → tab-delimited copy for Excel."""
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        selection = self.selectedRanges()
        if not selection:
            return
        min_row = min(r.topRow() for r in selection)
        max_row = max(r.bottomRow() for r in selection)
        min_col = min(r.leftColumn() for r in selection)
        max_col = max(r.rightColumn() for r in selection)

        selected_cells = set()
        for r in selection:
            for row in range(r.topRow(), r.bottomRow() + 1):
                for col in range(r.leftColumn(), r.rightColumn() + 1):
                    selected_cells.add((row, col))

        lines = []
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                if (row, col) in selected_cells:
                    item = self.item(row, col)
                    row_data.append(item.text() if item else "")
                else:
                    row_data.append("")
            lines.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(lines))


# ---------------------------
# Integral Table Dialog
# ---------------------------

class IntegralTableDialog(QDialog):
    """
    Shows integrals in a copyable table.
    Basic mode:  File Name | Transient | Integral
    Deconv mode: File Name | Transient | Peak1 (ppm) | Peak2 (ppm) | ... | Total
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Integral Table")
        self.setMinimumSize(500, 400)
        self.setModal(False)

        layout = QVBoxLayout(self)

        self.table = CopyableTableWidget()
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout.addWidget(self.table)

        hint_label = QLabel("Tip: Select cells and press Ctrl+C to copy (tab-delimited for Excel)")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint_label)

    def populate_basic(self, data, phase_offsets, lb, ppm_start, ppm_stop):
        """Populate for Basic Single Integration mode."""
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File Name", "Transient", "Integral"])
        self.table.setRowCount(len(data))

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        for i, (fid, time, folder_name, transient_num) in enumerate(data):
            offset = phase_offsets[i]
            integral = compute_integral(fid, time, lb, offset, ppm_start, ppm_stop)

            self.table.setItem(i, 0, QTableWidgetItem(folder_name))
            self.table.setItem(i, 1, QTableWidgetItem(str(transient_num)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{integral:.3f}"))

    def populate_deconv(self, data, phase_offsets, lb, ppm_start, ppm_stop,
                        n_peaks, lineshape):
        """Populate for Multi-Peak Deconvolution mode."""
        # First pass on first dataset to determine actual number of peaks found
        # (used for column headers)
        if not data:
            return

        # Process first dataset to get representative peak positions
        fid0, time0, _, _ = data[0]
        results0, _ = compute_deconv_integrals(
            fid0, time0, lb, phase_offsets[0], ppm_start, ppm_stop, n_peaks, lineshape)
        n_actual = len(results0)

        # Build column headers
        base_cols = ["File Name", "Transient"]
        peak_cols = []
        for pr in results0:
            peak_cols.append(f"{pr['center_ppm']:.1f} ppm")
        # If other rows find different peak positions, we'll still use these headers
        # (the peak count should be consistent within a run)
        total_col = ["Total"]
        all_cols = base_cols + peak_cols + total_col
        n_cols = len(all_cols)

        self.table.setColumnCount(n_cols)
        self.table.setHorizontalHeaderLabels(all_cols)
        self.table.setRowCount(len(data))

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, n_cols):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)

        for i, (fid, time, folder_name, transient_num) in enumerate(data):
            offset = phase_offsets[i]
            peak_results, _ = compute_deconv_integrals(
                fid, time, lb, offset, ppm_start, ppm_stop, n_peaks, lineshape)

            self.table.setItem(i, 0, QTableWidgetItem(folder_name))
            self.table.setItem(i, 1, QTableWidgetItem(str(transient_num)))

            total = 0.0
            for j, pr in enumerate(peak_results):
                col_idx = 2 + j
                if col_idx < n_cols - 1:  # guard
                    self.table.setItem(i, col_idx, QTableWidgetItem(f"{pr['integral']:.3f}"))
                    total += pr['integral']

            self.table.setItem(i, n_cols - 1, QTableWidgetItem(f"{total:.3f}"))


# ---------------------------
# Main Window
# ---------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spinsolve Data Analysis — Date/Folder Picker")

        # State holders
        self.selected_folders = []
        self.data = []
        self.phase_offsets = []
        self.figs = []
        self.current = 0
        self.integral_dialog = None

        # ---------------- Layout ----------------
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top form
        form = QFormLayout()
        self.path_edit = QLineEdit(r"D:\WSU\Raw Data\Spinsolve-1.4T_13C")
        self.date_edit = QLineEdit(datetime.now().strftime("%Y-%m-%d"))

        self.set_date_btn = QPushButton("Set Date")
        self.set_date_btn.clicked.connect(self.on_set_date)

        top_row = QHBoxLayout()
        top_row.addWidget(self.path_edit)
        top_row.addWidget(self.date_edit)
        top_row.addWidget(self.set_date_btn)
        form.addRow(QLabel("Main Path | Date (YYYY-MM-DD) | Set Date:"), top_row)

        # Folder selection list + session log
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.folder_list.setDisabled(True)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        folders_and_log = QWidget()
        folders_and_log_layout = QHBoxLayout(folders_and_log)
        folders_and_log_layout.setContentsMargins(0, 0, 0, 0)
        folders_and_log_layout.addWidget(self.folder_list, 3)
        folders_and_log_layout.addWidget(self.log_view, 2)

        self.folder_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_view.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        form.addRow(QLabel("Folders in selected date:"), folders_and_log)

        self.select_btn = QPushButton("Select folders")
        self.select_btn.setDisabled(True)
        self.select_btn.clicked.connect(self.on_select_folders)
        form.addRow(self.select_btn)

        # Processing controls
        self.lb_spin = QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 10.0)
        self.lb_spin.setSingleStep(0.1)
        self.lb_spin.setValue(1.0)

        self.ppm_start = QDoubleSpinBox()
        self.ppm_start.setRange(0.0, 300.0)
        self.ppm_start.setSingleStep(0.1)
        self.ppm_start.setValue(175.0)

        self.ppm_stop = QDoubleSpinBox()
        self.ppm_stop.setRange(0.0, 300.0)
        self.ppm_stop.setSingleStep(0.1)
        self.ppm_stop.setValue(160.0)

        form.addRow("Apodization (lb):", self.lb_spin)
        form.addRow("PPM Start:", self.ppm_start)
        form.addRow("PPM Stop:", self.ppm_stop)

        # ---- Integration mode selector ----
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Basic Single Integration", "Multi-Peak Deconvolution"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        form.addRow("Integration Mode:", self.mode_combo)

        # ---- Deconvolution-specific controls (shown/hidden) ----
        self.deconv_widget = QWidget()
        deconv_layout = QHBoxLayout(self.deconv_widget)
        deconv_layout.setContentsMargins(0, 0, 0, 0)

        deconv_layout.addWidget(QLabel("Number of Peaks:"))
        self.n_peaks_spin = QSpinBox()
        self.n_peaks_spin.setRange(1, 10)
        self.n_peaks_spin.setValue(3)
        deconv_layout.addWidget(self.n_peaks_spin)

        deconv_layout.addWidget(QLabel("Lineshape:"))
        self.lineshape_combo = QComboBox()
        self.lineshape_combo.addItems(["Lorentzian", "Pseudo-Voigt"])
        deconv_layout.addWidget(self.lineshape_combo)

        deconv_layout.addStretch(1)
        self.deconv_widget.setVisible(False)
        form.addRow("", self.deconv_widget)

        # Transient range inputs
        self.transient_start = QLineEdit("1")
        self.transient_stop = QLineEdit("20")
        form.addRow("Transient Start:", self.transient_start)
        form.addRow("Transient Stop:", self.transient_stop)

        root.addLayout(form)

        # Run controls
        btns = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.on_run)
        self.reint_btn = QPushButton("Re-integrate")
        self.reint_btn.clicked.connect(self.on_reintegrate)
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.on_next)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.reint_btn)
        btns.addStretch(1)
        btns.addWidget(self.prev_btn)
        btns.addWidget(self.next_btn)
        root.addLayout(btns)

        # Phase controls
        phase_row = QHBoxLayout()
        self.decr_phase_btn = QPushButton("Phase -")
        self.incr_phase_btn = QPushButton("Phase +")
        self.phase_dial = QDial()
        self.phase_dial.setRange(1, 10)
        self.phase_dial.setValue(1)
        self.phase_dial.setNotchesVisible(True)
        self.phase_label = QLabel("1")
        self.phase_dial.valueChanged.connect(lambda v: self.phase_label.setText(str(v)))
        self.decr_phase_btn.clicked.connect(self.decrease_phase)
        self.incr_phase_btn.clicked.connect(self.increase_phase)
        phase_row.addWidget(self.decr_phase_btn)
        phase_row.addWidget(self.incr_phase_btn)
        phase_row.addWidget(QLabel("Phase Step:"))
        phase_row.addWidget(self.phase_dial)
        phase_row.addWidget(self.phase_label)
        root.addLayout(phase_row)

        # Integral Table button
        integral_btn_row = QHBoxLayout()
        self.integral_table_btn = QPushButton("Integral Table")
        self.integral_table_btn.clicked.connect(self.on_show_integral_table)
        integral_btn_row.addWidget(self.integral_table_btn)
        integral_btn_row.addStretch(1)
        root.addLayout(integral_btn_row)

        # Plot section
        self.canvas = None
        self.toolbar = None
        self.plot_container = QWidget()
        self.plot_container.setMinimumHeight(420)
        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.plot_container)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        self.layout = root

    # ---------------- Helpers ----------------

    def log(self, text: str):
        self.log_view.append(text)

    def _resolved_date_path(self):
        base = self.path_edit.text().strip()
        date = self.date_edit.text().strip()
        return os.path.join(base, date)

    def _is_deconv_mode(self):
        return self.mode_combo.currentIndex() == 1

    def _get_lineshape(self):
        return "lorentzian" if self.lineshape_combo.currentIndex() == 0 else "pseudo-voigt"

    # ---------------- Mode switch ----------------

    def on_mode_changed(self, index):
        is_deconv = (index == 1)
        self.deconv_widget.setVisible(is_deconv)

        # Reset figures but keep data & phase offsets
        if self.data:
            self.figs = [None] * len(self.data)
            self._clear_plot()
            self._close_integral_dialog()
            self.log(f"[Mode] Switched to {'Multi-Peak Deconvolution' if is_deconv else 'Basic Single Integration'}. "
                     f"Click Re-integrate to update the current trace, or Run to reprocess all.")

    # ---------------- Actions ----------------

    def on_set_date(self):
        date_path = self._resolved_date_path()
        if not os.path.isdir(date_path):
            self.folder_list.clear()
            self.folder_list.setDisabled(True)
            self.select_btn.setDisabled(True)
            self.log(f"[Set Date] Date folder not found:\n{date_path}")
            return

        subs = [d for d in sorted(os.listdir(date_path)) if os.path.isdir(os.path.join(date_path, d))]
        self.folder_list.clear()
        for name in subs:
            QListWidgetItem(name, self.folder_list)

        self.folder_list.setDisabled(False)
        self.select_btn.setDisabled(False)
        self.log(f"[Set Date] Resolved date folder:\n{date_path}\nFound {len(subs)} subfolder(s). "
                 f"Select one or more and click 'Select folders'.")

    def on_select_folders(self):
        items = self.folder_list.selectedItems()
        self.selected_folders = [it.text() for it in items]
        date_path = self._resolved_date_path()
        if not self.selected_folders:
            self.log("[Select folders] No folders selected. Please choose at least one.")
            return
        msg = "\n".join(self.selected_folders)
        self.log(f"[Select folders] Date path:\n{date_path}\nSelected folders:\n{msg}")

    def _clear_plot(self):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas = None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
            self.toolbar = None

    def _close_integral_dialog(self):
        if self.integral_dialog is not None and self.integral_dialog.isVisible():
            self.integral_dialog.close()
        self.integral_dialog = None

    def _refresh_integral_dialog(self):
        if self.integral_dialog is not None and self.integral_dialog.isVisible():
            lb = float(self.lb_spin.value())
            ppm_start = float(self.ppm_start.value())
            ppm_stop = float(self.ppm_stop.value())
            if self._is_deconv_mode():
                n_peaks = self.n_peaks_spin.value()
                lineshape = self._get_lineshape()
                self.integral_dialog.populate_deconv(
                    self.data, self.phase_offsets, lb, ppm_start, ppm_stop, n_peaks, lineshape)
            else:
                self.integral_dialog.populate_basic(
                    self.data, self.phase_offsets, lb, ppm_start, ppm_stop)

    def on_show_integral_table(self):
        if not self.data:
            self.log("[Integral Table] No data loaded. Run analysis first.")
            return

        if self.integral_dialog is not None and self.integral_dialog.isVisible():
            self.integral_dialog.raise_()
            self.integral_dialog.activateWindow()
            return

        self.integral_dialog = IntegralTableDialog(self)

        lb = float(self.lb_spin.value())
        ppm_start = float(self.ppm_start.value())
        ppm_stop = float(self.ppm_stop.value())

        if self._is_deconv_mode():
            n_peaks = self.n_peaks_spin.value()
            lineshape = self._get_lineshape()
            self.integral_dialog.populate_deconv(
                self.data, self.phase_offsets, lb, ppm_start, ppm_stop, n_peaks, lineshape)
        else:
            self.integral_dialog.populate_basic(
                self.data, self.phase_offsets, lb, ppm_start, ppm_stop)

        self.integral_dialog.show()

    def _generate_fig(self, idx):
        """Generate figure for dataset at idx using current mode settings."""
        fid, time, _, _ = self.data[idx]
        offset = self.phase_offsets[idx]
        lb = float(self.lb_spin.value())
        ppm_start = float(self.ppm_start.value())
        ppm_stop = float(self.ppm_stop.value())

        if self._is_deconv_mode():
            n_peaks = self.n_peaks_spin.value()
            lineshape = self._get_lineshape()
            fig, warning = process_and_plot_deconv(
                fid, time, lb, offset, ppm_start, ppm_stop, n_peaks, lineshape)
            if warning:
                self.log(f"[Deconv] {warning}")
            return fig
        else:
            return process_and_plot(fid, time, lb, offset, ppm_start, ppm_stop)

    def on_run(self):
        self._close_integral_dialog()

        date_path = self._resolved_date_path()
        if not os.path.isdir(date_path):
            self.log(f"[Run] Date folder not found:\n{date_path}")
            return
        if not self.selected_folders:
            self.log("[Run] Please select folder(s) (and click 'Select folders') before running.")
            return

        folder_mode = "Multi-Transient" if len(self.selected_folders) == 1 else "Multi-Experiment"
        integration_mode = "Multi-Peak Deconvolution" if self._is_deconv_mode() else "Basic Single Integration"
        self.log(f"[Run] Folder mode: {folder_mode} | Integration: {integration_mode}")

        self.data = []
        self.phase_offsets = []
        self.figs = []
        self.current = 0

        try:
            if folder_mode == "Multi-Experiment":
                for folder_name in sorted(self.selected_folders):
                    fid_path = os.path.join(date_path, folder_name, "1", "fid.csv")
                    if not os.path.isfile(fid_path):
                        raise FileNotFoundError(f"Missing fid.csv: {fid_path}")
                    fid, time = read_fid_csv(fid_path)
                    self.data.append((fid, time, folder_name, 1))
                    self.phase_offsets.append(0.0)
                    self.log(f"[Run] Loaded {fid_path}")
            else:
                folder_name = self.selected_folders[0]
                target_path = os.path.join(date_path, folder_name)

                try:
                    t_start = int(self.transient_start.text())
                    t_stop = int(self.transient_stop.text())
                except ValueError:
                    raise ValueError("Transient Start/Stop must be integers.")
                if t_stop < t_start:
                    raise ValueError("Transient Stop must be >= Transient Start.")

                numeric_subs = sorted(
                    int(s) for s in os.listdir(target_path)
                    if s.isdigit() and os.path.isdir(os.path.join(target_path, s))
                )
                if not numeric_subs:
                    raise FileNotFoundError(f"No numeric subfolders found in {target_path}")

                for n in numeric_subs:
                    if t_start <= n <= t_stop:
                        fid_path = os.path.join(target_path, str(n), "fid.csv")
                        if not os.path.isfile(fid_path):
                            self.log(f"[Run] Skipping missing {fid_path}")
                            continue
                        fid, time = read_fid_csv(fid_path)
                        self.data.append((fid, time, folder_name, n))
                        self.phase_offsets.append(0.0)
                        self.log(f"[Run] Loaded {fid_path}")

            if not self.data:
                raise RuntimeError("No datasets loaded.")

            # Generate all figures
            for idx in range(len(self.data)):
                fig = self._generate_fig(idx)
                self.figs.append(fig)

            self.update_canvas()
            self.log("[Run] Plot updated.")

        except Exception as e:
            self.log(f"[Run Error] {str(e)}")
            self._clear_plot()
            self.status_label.setText("Ready")

    def on_reintegrate(self):
        if not self.data:
            self.log("[Re-integrate] Nothing to re-integrate. Load data first.")
            return

        fig = self._generate_fig(self.current)
        self.figs[self.current] = fig
        self._redraw_current()
        self.log("[Re-integrate] Updated current trace.")

        self._refresh_integral_dialog()

    def _redraw_current(self):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas = None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
            self.toolbar = None

        if self.figs[self.current] is None:
            # Generate on demand if not yet created (e.g. after mode switch)
            self.figs[self.current] = self._generate_fig(self.current)

        self.canvas = FigureCanvas(self.figs[self.current])
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()

        _, _, ts_name, trans_num = self.data[self.current]
        self.status_label.setText(f"Timestamp: {ts_name} | Transient: {trans_num}")

    def update_canvas(self):
        self._redraw_current()

    def on_prev(self):
        if not self.data:
            return
        self.current = (self.current - 1) % len(self.data)
        self._redraw_current()

    def on_next(self):
        if not self.data:
            return
        self.current = (self.current + 1) % len(self.data)
        self._redraw_current()

    def increase_phase(self):
        if not self.data:
            return
        step = float(self.phase_dial.value())
        self.phase_offsets[self.current] += step
        self.on_reintegrate()

    def decrease_phase(self):
        if not self.data:
            return
        step = float(self.phase_dial.value())
        self.phase_offsets[self.current] -= step
        self.on_reintegrate()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    sys.exit(app.exec_())
