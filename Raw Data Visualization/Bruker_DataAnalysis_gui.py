# PD_DataAnalysis_Rewritten.py
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import nmrglue as ng

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QDial, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

def exponential_apodization(fid, time, lb):
    return fid * np.exp(-lb * time)

def zero_fill(fid, target_length=262144):
    return np.pad(fid, (0, target_length - len(fid)), 'constant')

def apply_phase_correction(fid, angle):
    return fid * np.exp(-1j * np.deg2rad(angle))

def compute_fft(fid, dt):
    fft = np.fft.fftshift(np.fft.fft(fid))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fid), d=dt))
    return freqs, fft

def optimize_phase(fid, time_arr):
    dt = np.mean(np.diff(time_arr))
    def objective(angle):
        corr = apply_phase_correction(fid, angle)
        _, spectrum = compute_fft(corr, dt)
        return -np.sum(np.real(spectrum)**2)
    result = minimize_scalar(objective, bounds=(0, 10), method='bounded')
    return result.x

def read_bruker_fid(path):
    dic, data = ng.bruker.read(path)
    if data.ndim == 2 and data.shape[0] == 2:
        fid = data[0] + 1j * data[1]
    else:
        fid = data.astype(np.complex128)
    # fid = np.real(fid)
    dt = 1.0 / float(dic["acqus"]["SW_h"]) # sample spacing from SW_h
    time = np.arange(data.size) * dt
    return fid, time, dic

def ppm_conversion(freqs, dic):
    sfo1 = float(dic["acqus"]["SFO1"])     # spectrometer frequency in MHz
    o1 = float(dic["acqus"]["O1"])         # carrier frequency in Hz
    return (freqs * -1e6 / sfo1) + (o1 / sfo1)

def process_and_plot(fid, time, lb, phase_offset, ppm_start, ppm_stop, dic):
    fid_apod = exponential_apodization(fid, time, lb)
    opt_phase = optimize_phase(fid_apod, time)
    total_phase = opt_phase + phase_offset
    fid_corr = apply_phase_correction(fid_apod, total_phase)
    fid_zf = zero_fill(fid_corr)
    dt = time[1] - time[0]
    freqs, spectrum = compute_fft(fid_zf, dt)
    ppm = ppm_conversion(freqs, dic)
    real = np.real(spectrum)
    baseline = np.mean(real[np.abs(freqs) > 200])
    real -= baseline
    mask = (ppm > ppm_stop) & (ppm < ppm_start)
    ppm_vals = ppm[mask]
    real_vals = real[mask]
    sort_idx = np.argsort(ppm_vals)
    ppm_vals = ppm_vals[sort_idx]
    real_vals = real_vals[sort_idx]
    integral = np.trapz(real_vals, ppm_vals)

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ppm, real)
    ax.set_title(f"FFT in PPM (Phase: {total_phase:.2f}°, Integral: {integral:.2f})")
    ax.set_xlabel("PPM")
    ax.set_ylabel("Real Intensity")
    ax.invert_xaxis()
    return fig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FID Analyzer")
        self.data = []
        self.phase_offsets = []
        self.figs = []
        self.current = 0
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form = QFormLayout()
        self.path_edit = QLineEdit("D:/WSU/Raw Data/Spinsolve-1.4T_13C")
        self.lb_spin = QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 2.0)
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
        self.fft_check = QCheckBox("Plot FFT")
        self.fft_check.setChecked(True)
        self.folder_select = QComboBox()
        self.path_edit.editingFinished.connect(self.populate_folder_list)

        form.addRow("Main Path:", self.path_edit)
        form.addRow("Apodization (lb):", self.lb_spin)
        form.addRow("PPM Start:", self.ppm_start)
        form.addRow("PPM Stop:", self.ppm_stop)
        form.addRow(self.fft_check)
        form.addRow("Select Data Folder:", self.folder_select)

        layout.addLayout(form)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)

        self.integrate_btn = QPushButton("Re-integrate")
        self.integrate_btn.clicked.connect(self.update_canvas)
        layout.addWidget(self.integrate_btn)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

        # Phase adjustment controls
        phase_layout = QHBoxLayout()
        self.decr_phase_btn = QPushButton("Phase -")
        self.incr_phase_btn = QPushButton("Phase +")
        self.phase_dial = QDial()
        self.phase_dial.setRange(1, 10)
        self.phase_dial.setValue(1)
        self.phase_dial.setNotchesVisible(True)
        self.phase_label = QLabel("1")
        self.phase_dial.valueChanged.connect(lambda val: self.phase_label.setText(str(val)))
        self.decr_phase_btn.clicked.connect(self.decrease_phase)
        self.incr_phase_btn.clicked.connect(self.increase_phase)

        phase_layout.addWidget(self.decr_phase_btn)
        phase_layout.addWidget(self.incr_phase_btn)
        phase_layout.addWidget(QLabel("Phase Step:"))
        phase_layout.addWidget(self.phase_dial)
        phase_layout.addWidget(self.phase_label)
        layout.addLayout(phase_layout)

        self.canvas = None
        self.toolbar = None
                # Inset label for timestamp or transient number
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.layout = layout

    def populate_folder_list(self):
        path = self.path_edit.text()
        if os.path.isdir(path):
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            self.folder_select.clear()
            self.folder_select.addItems(sorted(folders))

    def run(self):
        self.data.clear()
        self.phase_offsets.clear()
        self.figs.clear()
        self.current = 0
        try:
            base = self.path_edit.text()
            selected = self.folder_select.currentText()
            lb = self.lb_spin.value()
            ppm_start = self.ppm_start.value()
            ppm_stop = self.ppm_stop.value()

            fid_path = os.path.join(base, selected)
            if not os.path.isdir(fid_path):
                raise ValueError(f"No FID folder found in: {fid_path}")

            fid, time, dic = read_bruker_fid(fid_path)
            self.data.append((fid, time, dic))
            self.phase_offsets.append(0)
            fig = process_and_plot(fid, time, lb, 0, ppm_start, ppm_stop, dic)
            self.figs.append(fig)
            self.status_label.setText(f"Loaded: {selected}")
            self.update_canvas()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def update_canvas(self):
        if not self.data:
            return
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        if self.toolbar:
            self.layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)

        fid, time, dic = self.data[self.current]
        lb = self.lb_spin.value()
        offset = self.phase_offsets[self.current]
        ppm_start = self.ppm_start.value()
        ppm_stop = self.ppm_stop.value()
        fig = process_and_plot(fid, time, lb, offset, ppm_start, ppm_stop, dic)
        self.figs[self.current] = fig

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()

        selected = self.folder_select.currentText()
        self.status_label.setText(f"Folder: {selected}")

    def show_next(self):
        if self.current < len(self.data) - 1:
            self.current += 1
            self.update_canvas()

    def show_prev(self):
        if self.current > 0:
            self.current -= 1
            self.update_canvas()

    def increase_phase(self):
        if self.data:
            step = self.phase_dial.value()
            self.phase_offsets[self.current] += step
            self.update_canvas()

    def decrease_phase(self):
        if self.data:
            step = self.phase_dial.value()
            self.phase_offsets[self.current] -= step
            self.update_canvas()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
