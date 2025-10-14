# ===== Standard Library =====
import sys, os
from pathlib import Path
from datetime import datetime

# ===== Third-Party =====
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")  # ensure Qt backend
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QTextEdit, QHBoxLayout, QMessageBox,
    QCheckBox, QLabel, QLineEdit
)


# ----------------------------
# Core analysis function
# ----------------------------
def compute_metrics(csv_path,
                    time_cutoff=None,    # e.g., 60.0 seconds
                    xlim=None,           # e.g., 400
                    ylim=None,           # e.g., 50
                    export_png=False,
                    export_pdf=False,
                    export_csv=False,    # per-file CSV
                    output_dir: Path = None):
    """
    Loads a CSV in the 'stacked' format:
      - row 0: 'Height'/'Integral' flags
      - row 1: ppm / labels (ignored)
      - row 2: substrate names (e.g., 'lactate', 'pyruvate')
      - row 3+: numeric values
      - col 0: Time
    Builds time + integral columns for known substrates and computes metrics.
    Also plots Lactate vs Time (optionally saved).

    Returns:
        dict with metrics for printing/export.
    """
    raw = pd.read_csv(csv_path, header=None)

    # Build column names using row0 (type) + row2 (substrate)
    colnames = []
    for i, (ctype, substrate) in enumerate(zip(raw.iloc[0], raw.iloc[2])):
        if i == 0:
            colnames.append("Time")
        else:
            if ctype == "Integral":
                colnames.append(f"Integral_{substrate}")
            else:
                colnames.append(None)  # ignore heights

    # Data rows start at row index 3
    df = raw.iloc[3:].copy()
    df.columns = colnames
    df = df.loc[:, [c for c in df.columns if c is not None]]

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Required series
    if "Integral_lactate" not in df.columns or "Integral_pyruvate" not in df.columns:
        raise ValueError("Could not find required columns: Integral_lactate and Integral_pyruvate")

    time = df["Time"]
    lactate = df["Integral_lactate"]
    pyruvate = df["Integral_pyruvate"]

    # Safe ratio
    pyruvate_safe = pyruvate.mask(pyruvate == 0)
    ratio = lactate / pyruvate_safe

    # Apply time cutoff (for peak search only)
    if time_cutoff is not None:
        mask = time <= time_cutoff
        sub_time = time[mask]
        sub_lac = lactate[mask]
        sub_ratio = ratio[mask]
    else:
        sub_time, sub_lac, sub_ratio = time, lactate, ratio

    # Peak indices relative to original df
    idx_ratio = sub_ratio.idxmax()
    idx_lac = sub_lac.idxmax()

    # Results
    results = {
        "file": Path(csv_path).name,
        "peak_ratio_lac/pyr": float(ratio.loc[idx_ratio]),
        "peak_lactate_integral": float(lactate.loc[idx_lac]),
        "time_at_peak_ratio": float(time.loc[idx_ratio]),
        "time_at_peak_lactate": float(time.loc[idx_lac]),
        "idx_peak_ratio": int(idx_ratio),
        "idx_peak_lactate": int(idx_lac),
        "time_cutoff_used": float(time_cutoff) if time_cutoff is not None else None,
    }

    # Plot lactate vs time
    plt.figure(figsize=(6, 4))
    plt.plot(time, lactate, label="Lactate integral")
    plt.xlabel("Time (s)")
    plt.ylabel("Lactate Integral")
    if xlim:
        plt.xlim(0, xlim)
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.title(Path(csv_path).name)
    plt.legend()
    plt.tight_layout()

    # --- Optional exports (per-file) ---
    csv_path = Path(csv_path)
    outdir = Path(output_dir) if output_dir is not None else csv_path.parent
    name = csv_path.stem
    # pull from first "(" inclusive; else fallback to "_name"
    if "(" in name:
        name_part = name[name.index("("):]
    else:
        name_part = "_" + name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    outstem = f"{name_part}_proc_at{timestamp}"

    if export_csv:
        out_csv = outdir / f"{outstem}.csv"
        pd.DataFrame([results]).to_csv(out_csv, index=False)

    if export_png:
        out_png = outdir / f"{outstem}.png"
        plt.savefig(out_png, dpi=300)

    if export_pdf:
        out_pdf = outdir / f"{outstem}.pdf"
        plt.savefig(out_pdf)

    # Close the figure to avoid popping windows / memory growth (GUI prints text only)
    plt.close()

    return results


# ----------------------------
# GUI
# ----------------------------
class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Analysis GUI")
        self.resize(950, 700)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- Top row: input/output folder buttons
        top = QHBoxLayout()
        self.folder_btn = QPushButton("Select Input Folder")
        self.folder_btn.clicked.connect(self.load_folder)
        top.addWidget(self.folder_btn)

        self.out_btn = QPushButton("Select Output Folder")
        self.out_btn.clicked.connect(self.select_output_folder)
        top.addWidget(self.out_btn)

        self.layout.addLayout(top)

        # --- Export options
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Exports:"))
        self.cb_png = QCheckBox("Plot PNG")
        self.cb_pdf = QCheckBox("Plot PDF")
        self.cb_csv_each = QCheckBox("Per-file CSV")
        self.cb_csv_summary = QCheckBox("Summary CSV")

        # Defaults per your request
        self.cb_png.setChecked(True)     # PNG default ON
        self.cb_pdf.setChecked(False)    # PDF default OFF
        self.cb_csv_each.setChecked(False)   # per-file CSV default OFF
        self.cb_csv_summary.setChecked(True) # summary CSV default ON

        for cb in (self.cb_png, self.cb_pdf, self.cb_csv_each, self.cb_csv_summary):
            opts.addWidget(cb)

        self.layout.addLayout(opts)

        # --- Time cutoff input
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Time cutoff (s):"))
        self.time_cutoff_input = QLineEdit("60.0")
        cutoff_layout.addWidget(self.time_cutoff_input)
        self.layout.addLayout(cutoff_layout)

        # --- File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(self.file_list.MultiSelection)
        self.layout.addWidget(self.file_list)

        # --- Run button
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.run_btn)

        # --- Output log
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        # remember selected output folder
        self.output_folder = None

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not folder:
            return
        self.file_list.clear()
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                self.file_list.addItem(str(Path(folder) / f))
        self.output.append(f"Loaded input folder: {folder}")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = Path(folder)
            self.output.append(f"Output folder set: {self.output_folder}")

    def run_analysis(self):
        selected = [item.text() for item in self.file_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "No file selected", "Please select at least one CSV.")
            return

        # Export options
        export_png = self.cb_png.isChecked()
        export_pdf = self.cb_pdf.isChecked()
        export_csv_each = self.cb_csv_each.isChecked()
        export_csv_summary = self.cb_csv_summary.isChecked()

        # If any export is enabled, require an output folder
        if (export_png or export_pdf or export_csv_each or export_csv_summary) and not self.output_folder:
            QMessageBox.warning(self, "No output folder", "Please select an output folder for exports.")
            return

        # Time cutoff from input
        try:
            TIME_CUTOFF = float(self.time_cutoff_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Time cutoff must be a number.")
            return

        self.output.append("Running analysis...\n")
        rows = []

        # Fixed plot limits
        XLIM = 400
        YLIM = 50

        for file in selected:
            try:
                res = compute_metrics(
                    file,
                    time_cutoff=TIME_CUTOFF,
                    xlim=XLIM,
                    ylim=YLIM,
                    export_png=export_png,
                    export_pdf=export_pdf,
                    export_csv=export_csv_each,
                    output_dir=self.output_folder
                )
                rows.append(res)

                # Print results for this file
                self.output.append(f"=== {res['file']} ===")
                for k, v in res.items():
                    self.output.append(f"{k}: {v}")
                self.output.append("----")

            except Exception as e:
                self.output.append(f"ERROR: {Path(file).name}: {e}")

        # Summary CSV
        if export_csv_summary and rows:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            summary_path = self.output_folder / f"summary_proc_at{timestamp}.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            self.output.append(f"Summary saved: {summary_path}")

        self.output.append("\nDone.\n")


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    app = QApplication(sys.argv)
    gui = AnalysisGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
