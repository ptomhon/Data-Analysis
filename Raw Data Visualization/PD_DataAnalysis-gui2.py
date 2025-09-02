import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QCheckBox, QDial, QMessageBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

# ---------------------------
# Small helpers
# ---------------------------

def as_np(x):
    """Safe conversion to numpy array for pandas or numpy inputs."""
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)

# ---------------------------
# Basic Processing Functions
# ---------------------------

def exponential_apodization(fid, time, lb=1.0):
    return fid * np.exp(-lb * time)


def zero_fill(fid, target_length=65536):
    original_length = len(fid)
    if target_length is None:
        target_length = 2 ** int(np.ceil(np.log2(original_length)))
    pad = max(0, target_length - original_length)
    return np.pad(fid, (0, pad), 'constant')


def apply_phase_correction(fid, phase_angle_deg):
    return fid * np.exp(-1j * np.deg2rad(phase_angle_deg))


def compute_fft(fid, dt):
    spectrum = np.fft.fft(fid)
    spectrum = np.fft.fftshift(spectrum)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(len(fid), d=dt))
    return freq_axis, spectrum


def optimize_phase(fid, time_arr, maxphase=150):
    dt = float(np.mean(np.diff(time_arr))) if len(time_arr) > 1 else 1.0
    def objective(angle):
        corr = apply_phase_correction(fid, angle)
        _, spec = compute_fft(corr, dt)
        real_part = np.real(spec)
        return -np.sum(real_part**2)
    coarse_angles = np.arange(0, maxphase + 1, 2)
    coarse_scores = [objective(a) for a in coarse_angles]
    best_coarse = coarse_angles[int(np.argmin(coarse_scores))]
    lo = max(0, best_coarse - 15)
    hi = min(maxphase, best_coarse + 15)
    res = minimize_scalar(objective, bounds=(lo, hi), method='bounded')
    return float(res.x)

# ---------------------------
# CSV Reading (PD strict format: col0 complex string, col1 time)
# ---------------------------

def read_pd_fid_strict(path):
    df = pd.read_csv(path, header=None)
    data_strings = as_np(df.iloc[:, 0].astype(str))
    time_col = as_np(df.iloc[:, 1])

    # complex like '3+4i' or '3+4j'
    fid = np.array([complex(str(s).replace('i', 'j')) for s in data_strings], dtype=np.complex128)

    # time is already in seconds (may be string/complex-like) — coerce safely to float seconds
    try:
        tconv = np.array([complex(str(val).replace('i', 'j')).real for val in time_col], dtype=float)
    except Exception:
        tconv = as_np(pd.to_numeric(time_col, errors='coerce'))

    return fid, tconv

# ---------------------------
# PPM Conversion (carrier mode: 0 Hz → 0 ppm)
# ---------------------------

def hz_to_ppm(freq_hz: np.ndarray, larmor_hz: float) -> np.ndarray:
    """
    Spinsolve-style ppm with ref_hz = 0:
      ppm = - freq_hz / (larmor_hz / 1e6)
    """
    hz_per_ppm = float(larmor_hz) / 1e6
    return -(freq_hz / hz_per_ppm)


# ---------------------------
# Processing Class
# ---------------------------

class Processor:
    def __init__(self, lb=1.0, phase_offset_deg=0.0, ppm_start=175.0, ppm_stop=160.0,
                 use_ppm=True, larmor_hz=6.124e6):
        self.lb = lb
        self.phase_offset = phase_offset_deg
        self.ppm_start = ppm_start
        self.ppm_stop = ppm_stop
        self.use_ppm = use_ppm
        self.larmor_hz = larmor_hz

    def process(self, fid, time, extra_phase_offset=0.0):
        # --- Apodize ---
        fid_apod = exponential_apodization(fid, time, self.lb)
        # --- Optimize phase (simple bounded search) ---
        opt_phase = optimize_phase(fid_apod, time)
        total_phase = float(opt_phase + self.phase_offset + extra_phase_offset)
        fid_corr = apply_phase_correction(fid_apod, total_phase)
        # --- Zero-fill & FFT ---
        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
        fid_zf = zero_fill(fid_corr, target_length=65536)
        freq_hz, spec = compute_fft(fid_zf, dt)
        real = np.real(spec)*100000
        # --- Baseline from |f|>200 Hz ---
        noise_mask = (np.abs(freq_hz) > 200.0)
        baseline = np.mean(real[noise_mask]) if np.any(noise_mask) else 0.0
        real_bc = real - baseline
        # --- Axis in ppm (Spinsolve-style: ref_hz = 0) ---
        ppm = hz_to_ppm(freq_hz, self.larmor_hz) if self.use_ppm else freq_hz

        # --- Integration window (Spinsolve-style) ---
        # Convention: ppm_start > ppm_stop
        mask = (ppm > self.ppm_stop) & (ppm < self.ppm_start)

        if np.any(mask):
            x = ppm[mask]
            y = real_bc[mask]
            order = np.argsort(x)
            integral = float(np.trapezoid(y[order], x[order]))*100000
        else:
            integral = 0.0
        # --- Plot full spectrum in ppm/Hz (simple & reliable) ---
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(ppm, real_bc, lw=1.0)
        ax.set_title(f"FFT in {'PPM' if self.use_ppm else 'Hz'} (Phase: {total_phase:.2f}°, Integral: {integral:.3f})")
        ax.set_xlabel("PPM" if self.use_ppm else "Frequency (Hz)")
        ax.set_ylabel("Real Intensity")
        if self.use_ppm:
            ax.invert_xaxis()
        # Inset box with integration readout (updates on Re-integrate)
        overlay = f"Int: {integral:.3f}\nRange: {self.ppm_start:.2f}–{self.ppm_stop:.2f} {'ppm' if self.use_ppm else 'Hz'}"
        ax.text(0.98, 0.02, overlay, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        return fig

# ---------------------------
# Path Helpers for New Folder Logic
# ---------------------------

def parse_date_token_from_main_path(main_path: str) -> str:
    tail = os.path.basename(os.path.normpath(main_path))
    return tail.split('_', 1)[0] if '_' in tail else tail


def build_experiment_folder(parent_dir: str, date_token: str, run_number: int, sample_name: str) -> str:
    return os.path.join(parent_dir, f"{date_token}_SACQ_{run_number}_{sample_name}")


def find_case_insensitive_file(directory: str, candidate_names=None):
    if candidate_names is None:
        candidate_names = ["FID.csv", "fid.csv"]
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        return None
    lower_map = {f.lower(): f for f in files}
    for name in candidate_names:
        if name.lower() in lower_map:
            return os.path.join(directory, lower_map[name.lower()])
    return None

# ---------------------------
# Main Window
# ---------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PD Data Analysis — Modified (Spinsolve-style GUI)")
        self.current_index = 0
        self.data_list = []      # list of (fid, time)
        self.phase_offsets = []  # per-run extra phase offsets (deg)
        self.figures = []
        self.proc = Processor()
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)

        # ---- Inputs ----
        form = QFormLayout()
        self.main_path_edit = QLineEdit(r"D:\\WSU\\Raw Data\\PureDevices-0.55T\\2025-08-25\\25-Aug-2025_Aceticacid_SACQ")
        form.addRow("Main Path:", self.main_path_edit)
        self.sample_name_edit = QLineEdit("13C_Aceticacid")
        form.addRow("Sample Name:", self.sample_name_edit)
        self.start_number_edit = QLineEdit("1")
        form.addRow("Experiment Start #:", self.start_number_edit)
        self.end_number_edit = QLineEdit("")
        form.addRow("Experiment End #:", self.end_number_edit)

        self.plot_fft_check = QCheckBox("Plot FFT")
        self.plot_fft_check.setChecked(True)
        form.addRow(self.plot_fft_check)

        self.lb_spin = QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 2.0)
        self.lb_spin.setSingleStep(0.1)
        self.lb_spin.setValue(1.0)
        form.addRow("Apodization lb (1/s):", self.lb_spin)

        self.ppm_start_spin = QDoubleSpinBox()
        self.ppm_start_spin.setRange(-1000.0, 1000.0)
        self.ppm_start_spin.setSingleStep(0.1)
        self.ppm_start_spin.setValue(175.0)
        form.addRow("Integration Start (ppm):", self.ppm_start_spin)
        self.ppm_stop_spin = QDoubleSpinBox()
        self.ppm_stop_spin.setRange(-1000.0, 1000.0)
        self.ppm_stop_spin.setSingleStep(0.1)
        self.ppm_stop_spin.setValue(160.0)
        form.addRow("Integration Stop (ppm):", self.ppm_stop_spin)

        self.layout.addLayout(form)

        # ---- Run / Re-integrate ----
        btn_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_experiment)
        btn_row.addWidget(self.run_button)
        self.reintegrate_button = QPushButton("Re-integrate / Update")
        self.reintegrate_button.clicked.connect(self.update_canvas)
        btn_row.addWidget(self.reintegrate_button)
        self.layout.addLayout(btn_row)

        # ---- Navigation ----
        nav = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        nav.addWidget(self.prev_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        nav.addWidget(self.next_button)
        self.layout.addLayout(nav)

        # ---- Phase Adjustment Controls ----
        phase_layout = QHBoxLayout()
        self.decr_phase_button = QPushButton("Phase -")
        self.decr_phase_button.clicked.connect(self.decrease_phase)
        phase_layout.addWidget(self.decr_phase_button)
        self.incr_phase_button = QPushButton("Phase +")
        self.incr_phase_button.clicked.connect(self.increase_phase)
        phase_layout.addWidget(self.incr_phase_button)
        self.phase_dial = QDial()
        self.phase_dial.setMinimum(1)
        self.phase_dial.setMaximum(10)
        self.phase_dial.setValue(1)
        self.phase_dial.setNotchesVisible(True)
        phase_layout.addWidget(QLabel("Phase Step:"))
        phase_layout.addWidget(self.phase_dial)
        self.phase_value_label = QLabel(str(self.phase_dial.value()))
        self.phase_dial.valueChanged.connect(lambda v: self.phase_value_label.setText(str(v)))
        phase_layout.addWidget(self.phase_value_label)
        self.layout.addLayout(phase_layout)

        # ---- Canvas & Toolbar ----
        self.canvas = None
        self.toolbar = None

        # ---- Status label ----
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

    # ---------------------------
    # Core actions
    # ---------------------------
    def run_experiment(self):
        try:
            self.data_list.clear()
            self.phase_offsets.clear()
            self.figures.clear()
            self.current_index = 0

            self.proc.lb = float(self.lb_spin.value())
            self.proc.ppm_start = float(self.ppm_start_spin.value())
            self.proc.ppm_stop = float(self.ppm_stop_spin.value())

            main_path = self.main_path_edit.text().strip()
            sample_name = self.sample_name_edit.text().strip()
            if not main_path or not os.path.isdir(main_path):
                raise ValueError("Main Path looks invalid.")

            start_str = self.start_number_edit.text().strip()
            end_str = self.end_number_edit.text().strip()
            if not start_str:
                raise ValueError("Provide an Experiment Start #")
            start_num = int(start_str)
            end_num = int(end_str) if end_str else start_num

            date_token = parse_date_token_from_main_path(main_path)
            parent_dir = main_path

            for run in range(start_num, end_num + 1):
                exp_dir = build_experiment_folder(parent_dir, date_token, run, sample_name)
                fid_path_csv = find_case_insensitive_file(exp_dir, ["FID.csv", "fid.csv"]) 
                if not fid_path_csv or not os.path.isfile(fid_path_csv):
                    raise FileNotFoundError(f"Missing FID.csv for run {run}: {os.path.join(exp_dir, 'FID.csv')} (case-insensitive)")
                fid, time = read_pd_fid_strict(fid_path_csv)
                self.data_list.append((fid, time))
                self.phase_offsets.append(0.0)
                fig = self.proc.process(fid, time, extra_phase_offset=0.0)
                self.figures.append(fig)

            self.status_label.setText(f"Loaded runs {start_num}–{end_num} for {date_token} :: {sample_name}")
            self.update_canvas()
        except Exception as e:
            self._error_box(str(e))

    def update_canvas(self):
        if not self.data_list:
            return
        self.proc.lb = float(self.lb_spin.value())
        self.proc.ppm_start = float(self.ppm_start_spin.value())
        self.proc.ppm_stop = float(self.ppm_stop_spin.value())
        fid, time = self.data_list[self.current_index]
        offset = float(self.phase_offsets[self.current_index])
        fig = self.proc.process(fid, time, extra_phase_offset=offset)
        self.figures[self.current_index] = fig
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        if self.toolbar is not None:
            self.layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.canvas.draw_idle()
        run_num = int(self.start_number_edit.text() or 1) + self.current_index
        self.status_label.setText(f"Run {run_num} | Phase offset {offset:+.1f}° | lb {self.proc.lb:.2f} | Window {self.proc.ppm_stop:.2f}–{self.proc.ppm_start:.2f} ppm")

    # ---------------------------
    # Navigation & Phase tweaks
    # ---------------------------
    def show_next(self):
        if self.data_list and self.current_index < len(self.data_list) - 1:
            self.current_index += 1
            self.update_canvas()

    def show_previous(self):
        if self.data_list and self.current_index > 0:
            self.current_index -= 1
            self.update_canvas()

    def decrease_phase(self):
        if self.data_list:
            step = float(self.phase_dial.value())
            self.phase_offsets[self.current_index] -= step
            self.update_canvas()

    def increase_phase(self):
        if self.data_list:
            step = float(self.phase_dial.value())
            self.phase_offsets[self.current_index] += step
            self.update_canvas()

    # ---------------------------
    # Utilities
    # ---------------------------
    def _error_box(self, text: str):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(text)
        msg.exec_()


# ---------------------------
# Main Entrypoint
# ---------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
