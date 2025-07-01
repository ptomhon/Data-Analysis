import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox, QDial, QMessageBox
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

# ---------------------------
# Basic Processing Functions
# ---------------------------
def exponential_apodization(fid, time, lb=1):
    """Apply exponential apodization (line broadening) to the FID."""
    return fid * np.exp(-lb * time)

def zero_fill(fid, target_length=65536):
    """Zero-fill the FID to a target length."""
    original_length = len(fid)
    if target_length is None:
        target_length = 2 ** int(np.ceil(np.log2(original_length)))
    return np.pad(fid, (0, target_length - original_length), 'constant')

def apply_phase_correction(fid, phase_angle):
    """Apply phase correction (in degrees) to the FID."""
    phase_rad = np.deg2rad(phase_angle)
    return fid * np.exp(-1j * phase_rad)

def compute_fft(fid, dt):
    """Compute the FFT, shift zero-frequency to center, and return frequency axis and spectrum."""
    spectrum = np.fft.fft(fid)
    spectrum = np.fft.fftshift(spectrum)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(len(fid), d=dt))
    return freq_axis, spectrum

# ---------------------------
# Optimize Phase Function (with coarse and fine search)
# ---------------------------
def optimize_phase(fid, time_arr, maxphase=150):
    """Optimize phase correction by maximizing peak alignment and reducing imaginary leakage."""
    dt = np.mean(np.diff(time_arr))
    def objective(phase_angle):
        corrected_fid = apply_phase_correction(fid, phase_angle)
        _, spectrum = compute_fft(corrected_fid, dt)
        real_part = np.real(spectrum)
        imag_part = np.imag(spectrum)
        peaks, _ = find_peaks(real_part, height=0)
        peak_heights = real_part[peaks]
        peak_alignment_score = np.sum(np.abs(peak_heights))
        peak_variance = np.std(peak_heights)
        imaginary_leakage = np.sum(np.abs(imag_part))
        return peak_alignment_score + 0.1 * peak_variance
    # Coarse search over angles from 80° to maxphase in 2° steps
    coarse_search_angles = np.arange(80, maxphase + 1, 2)
    best_coarse_phase = min(coarse_search_angles, key=objective)
    print(f"Best coarse phase: {best_coarse_phase}°")
    # Fine-tune: search within ±15° around best_coarse_phase, clamped to [0, maxphase]
    lower_bound = max(0, best_coarse_phase - 15)
    upper_bound = min(maxphase, best_coarse_phase + 15)
    result = minimize_scalar(objective, bounds=(lower_bound, upper_bound), method="bounded")
    return result.x

# ---------------------------
# File Reading Function
# ---------------------------
def read_data(run_number, main_branch, exp_date, exp_date_2, sample_name, sample_name2, filetype):
    """
    Construct the filepath for a given run number and read the CSV file.
    Assumes:
      Column 0: FID as a complex string (e.g., "3+4j")
      Column 1: time
    """
    if filetype == 'HP':
        filename = f'{exp_date_2}SACQ_{run_number}_13C_{sample_name}\\FID.csv'
    else:
        filename = f'{exp_date_2}SACQ_{run_number}_{sample_name}\\FID.csv'
    file_path = os.path.join(main_branch, exp_date, exp_date_2 + sample_name2, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    data = pd.read_csv(file_path, header=None)
    return data

# ---------------------------
# Processing and Figure Creation Function
# ---------------------------
def process_and_create_figure(data, run_number, filetype, plot_fft, phase_offset=0):
    """
    Process the loaded CSV data and return a matplotlib Figure.
    
    For FFT analysis:
      1. Exponential apodization.
      2. Phase optimization using optimize_phase.
      3. Add a user-specified phase offset (in degrees).
      4. Zero-fill.
      5. Compute the FFT.
      6. Baseline correction on the real part (using noise region |freq| > 200 Hz).
      7. Integrate the corrected real part over a selected frequency window.
    
    For FID analysis, the raw FID magnitude is plotted.
    """
    data_strings = data[0].to_numpy()
    time_arr = data[1].to_numpy()
    try:
        fid = np.array([complex(num.replace('i', 'j')) for num in data_strings])
    except Exception as e:
        raise ValueError("Error converting FID complex numbers: " + str(e))
    try:
        time_conv = np.array([complex(num.replace('i', 'j')) for num in time_arr])
        if np.allclose(time_conv.imag, 0):
            time_conv = time_conv.real
        time_arr = time_conv
    except Exception:
        time_arr = time_arr.astype(float)
    
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    overlay_text = ""
    
    if plot_fft:
        # 1. Apodization
        fid_apodized = exponential_apodization(fid, time_arr, lb=1)
        # 2. Optimize phase
        optimal_phase = optimize_phase(fid_apodized, time_arr, maxphase=150)
        # 3. Apply user phase offset
        used_phase = optimal_phase + phase_offset
        fid_corrected = apply_phase_correction(fid_apodized, used_phase)
        print(f"Run {run_number}: Optimal phase = {optimal_phase:.2f}°, offset = {phase_offset:+.1f}°, used phase = {used_phase:.2f}°")
        # 4. Zero-fill
        fid_padded = zero_fill(fid_corrected, target_length=65536)
        dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 1
        freq, fft_data = compute_fft(fid_padded, dt)
        # 5. Baseline correction on the real part (noise region: |freq| > 200 Hz)
        noise_mask = (np.abs(freq) > 200)
        noise_values = np.real(fft_data)[noise_mask]
        baseline_average = np.mean(noise_values)
        print("Baseline (real) average:", baseline_average)
        corrected_real = np.real(fft_data) - baseline_average
        # 6. Integration on the real part
        if filetype == 'HP':
            integration_mask = (freq >= -200) & (freq <= 200)
        else:
            integration_mask = (freq >= -70) & (freq <= -35)
        selected_freq = freq[integration_mask]
        selected_magnitude = corrected_real[integration_mask]
        sort_idx = np.argsort(selected_freq)
        selected_freq = selected_freq[sort_idx]
        selected_magnitude = selected_magnitude[sort_idx]
        integral_value = np.trapz(selected_magnitude, selected_freq) * 10e6
        print(f"Integrated real FFT for run {run_number}: {integral_value:.3f}")
        overlay_text = f"Integral: {integral_value:.3f}\nPhase: {used_phase:.2f}°"
        # Plot the baseline-corrected real part of the FFT
        ax.plot(freq, corrected_real, label="FFT Real Part (Baseline Corrected)", color='blue')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Real Intensity")
        ax.set_title(f"FFT (Run {run_number})")
    else:
        n_highest = 3
        peakFID = np.average(np.partition(np.abs(fid), -n_highest)[-n_highest:])
        print(f"Peak FID for run {run_number}: {peakFID*10e9:.2f}")
        overlay_text = f"Peak FID: {peakFID*10e9:.2f}"
        ax.plot(time_arr, np.abs(fid), label="FID Magnitude", color="blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("FID Magnitude")
        ax.set_title(f"FID (Run {run_number})")
    
    ax.text(0.95, 0.95, overlay_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))
    ax.legend(loc="upper left")
    return fig

# ---------------------------
# PyQt GUI Main Window with Navigation, Dynamic Phase Adjustment, and a Knob
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PD Data Analysis")
        self.current_index = 0
        self.data_list = []      # To store raw data for each run
        self.phase_offsets = []  # To store phase adjustments (in degrees) for each run (initially 0)
        self.figures = []        # Cached figures for each run
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        # Experiment Input Form
        form_layout = QFormLayout()
        self.main_branch_edit = QLineEdit(r'G:\Shared drives\Vizma Life Sciences - Projects\Product Development\Experiments\Data\Vizma\PureDevices Benchtop PD-V-01\\')
        form_layout.addRow("Main Branch Path:", self.main_branch_edit)
        self.exp_date_edit = QLineEdit("2025-02-20")
        form_layout.addRow("Experiment Date (YYYY-MM-DD):", self.exp_date_edit)
        self.sample_name_edit = QLineEdit("35mM_5.2mM_depolarization")
        form_layout.addRow("Sample Name:", self.sample_name_edit)
        self.filetype_combo = QComboBox()
        self.filetype_combo.addItems(["HP", "Standard"])
        form_layout.addRow("File Type:", self.filetype_combo)
        self.start_number_edit = QLineEdit("6")
        form_layout.addRow("Start Run Number:", self.start_number_edit)
        self.end_number_edit = QLineEdit("13")
        form_layout.addRow("End Run Number:", self.end_number_edit)
        self.plot_fft_check = QCheckBox("Plot FFT")
        self.plot_fft_check.setChecked(True)
        form_layout.addRow("Plot FFT:", self.plot_fft_check)
        self.main_layout.addLayout(form_layout)
        
        # Run Experiment Button
        self.run_button = QPushButton("Run Experiment")
        self.run_button.clicked.connect(self.run_experiment)
        self.main_layout.addWidget(self.run_button)
        
        # Navigation Buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)
        self.main_layout.addLayout(nav_layout)
        
        # Phase Adjustment Controls
        phase_layout = QHBoxLayout()
        self.decr_phase_button = QPushButton("Phase -")
        self.decr_phase_button.clicked.connect(self.decrease_phase)
        phase_layout.addWidget(self.decr_phase_button)
        self.incr_phase_button = QPushButton("Phase +")
        self.incr_phase_button.clicked.connect(self.increase_phase)
        phase_layout.addWidget(self.incr_phase_button)
        self.phase_dial = QDial()
        self.phase_dial.setMinimum(1)
        self.phase_dial.setMaximum(5)
        self.phase_dial.setValue(1)
        self.phase_dial.setNotchesVisible(True)
        phase_layout.addWidget(QLabel("Phase Step:"))
        phase_layout.addWidget(self.phase_dial)
        self.phase_value_label = QLabel(str(self.phase_dial.value()))
        phase_layout.addWidget(self.phase_value_label)
        self.phase_dial.valueChanged.connect(lambda val: self.phase_value_label.setText(str(val)))
        self.main_layout.addLayout(phase_layout)
        
        # Placeholders for FigureCanvas and Navigation Toolbar
        self.canvas = None
        self.toolbar = None
    
    def run_experiment(self):
        try:
            self.main_branch = self.main_branch_edit.text()
            self.exp_date = self.exp_date_edit.text()
            self.sample_name = self.sample_name_edit.text()
            self.filetype = self.filetype_combo.currentText()
            self.start_number = int(self.start_number_edit.text())
            self.end_number = int(self.end_number_edit.text())
            self.plot_fft = self.plot_fft_check.isChecked()
            self.exp_date_2 = datetime.strptime(self.exp_date, '%Y-%m-%d').strftime('%d-%b-%Y_')
            if self.filetype == 'HP':
                self.sample_name2 = self.sample_name + '_SACQ'
            else:
                self.sample_name2 = self.sample_name
            
            self.data_list = []
            self.phase_offsets = []
            self.figures = []
            
            for run in range(self.start_number, self.end_number + 1):
                data = read_data(run, self.main_branch, self.exp_date, self.exp_date_2, self.sample_name, self.sample_name2, self.filetype)
                self.data_list.append(data)
                self.phase_offsets.append(0)
                fig = process_and_create_figure(data, run, self.filetype, self.plot_fft, phase_offset=0)
                self.figures.append(fig)
            self.current_index = 0
            self.update_canvas()
        except FileNotFoundError as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(str(e))
            msg.setWindowTitle("File Not Found")
            msg.exec_()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("An error occurred: " + str(e))
            msg.setWindowTitle("Error")
            msg.exec_()
    
    def update_canvas(self):
        current_run = self.start_number + self.current_index
        try:
            fig = process_and_create_figure(self.data_list[self.current_index],
                                            current_run,
                                            self.filetype,
                                            self.plot_fft,
                                            phase_offset=self.phase_offsets[self.current_index])
            self.figures[self.current_index] = fig
        except Exception as e:
            print(f"Error updating figure for run {current_run}: {e}")
            return
        if self.canvas is not None:
            self.main_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        if self.toolbar is not None:
            self.main_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
        self.canvas = FigureCanvas(self.figures[self.current_index])
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)
        self.canvas.draw()
    
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
            step = self.phase_dial.value()
            self.phase_offsets[self.current_index] -= step
            print(f"Run {self.start_number + self.current_index}: New phase offset = {self.phase_offsets[self.current_index]}°")
            self.update_canvas()
    
    def increase_phase(self):
        if self.data_list:
            step = self.phase_dial.value()
            self.phase_offsets[self.current_index] += step
            print(f"Run {self.start_number + self.current_index}: New phase offset = {self.phase_offsets[self.current_index]}°")
            self.update_canvas()

# ---------------------------
# Main: Start the Qt Application
# ---------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
