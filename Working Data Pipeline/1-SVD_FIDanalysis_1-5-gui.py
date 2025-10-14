# gui_multi_folders.py  (Block 1/4)

import os
import sys
import re
import io
import math
import json
import time
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QTextEdit, QSplitter, QHBoxLayout, QVBoxLayout,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar
)

# --- Import your processing code from the attached file ---
# It must be in the same directory or adapt the import path accordingly.
# If you prefer, you can replace this with:  from SVD_FIDanalysis import ...
sys.path.append(str(Path(__file__).parent))
from _io import StringIO

# Adjust this name/path if needed to match your file name
import importlib.util
PROC_MODULE_PATH = Path(__file__).with_name("1-SVD_FIDanalysis_1-5.py")
spec = importlib.util.spec_from_file_location("fidproc", str(PROC_MODULE_PATH))
fidproc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fidproc)

# We will use these exactly as in your code:
#   - process_multiple_folders(...)
#   - get_per_folder_output_dir(...)
process_multiple_folders = fidproc.process_multiple_folders
get_per_folder_output_dir = getattr(fidproc, "get_per_folder_output_dir", None)
if get_per_folder_output_dir is None:
    # Fallback: per-folder dir is <output_dir>/<output_name>/
    def get_per_folder_output_dir(output_dir, output_name):
        if not output_dir or not output_name:
            return None
        out = os.path.join(output_dir, output_name)
        os.makedirs(out, exist_ok=True)
        return out

# ---------- Presets ----------
# These mirror/extend the parameters your process_multiple_folders already supports.
# PYR_RUN is the default. KL_RUN and COPOL_RUN are provided as editable presets.
PRESETS = {
    "PYR_RUN": {
        "initial_k": 10,
        "L": 4000,
        "T2_apod": 1.5,
        "phase_corr_angle": 10,
        "allowed_ppms": [182.5, 178, 170, 160, 124],
        "ppm_threshold": 20,
        "target_length": 65536,
        "n_iter": 2,
        "target_peaks": [182.5, 178, 170, 160, 124],
        "metabolite_names": ["lactate", "hydrate", "pyruvate", "bicarbonate", "CO2"],
        "tolerance": 1.0,
        "save_per_folder": True,
        # learning parameter from your function signature:
        "n_threshold": 3,  # used only if your in-file learning logic refers to it
        # GUI default time_interval (per folder override-able):
        "default_time_interval": 2.5,
        "k_learning_skip": 3,
        "k_learning_window": 9,
        "k_learning_trim": 5,
    },
    "KL_RUN": {
        # Start with PYR defaults; user can edit
        "initial_k": 10,
        "L": 4000,
        "T2_apod": 1.5,
        "phase_corr_angle": 10,
        "allowed_ppms": [182.5, 178.3, 173.2, 171.3, 170.2, 160, 124],
        "ppm_threshold": 20,
        "target_length": 65536,
        "n_iter": 2,
        "target_peaks": [182.5, 178.3, 173.2, 171.3, 170.2, 160, 124],
        "metabolite_names": ["lactate","hydrate","unk1", "ketoleucine", "unk2", "bicarbonate","CO2"],
        "tolerance": 1.0,
        "save_per_folder": True,
        "n_threshold": 2,
        "default_time_interval": 2.5,
        "k_learning_skip": 3,
        "k_learning_window": 9,
        "k_learning_trim": 5,
    },
    "COPOL_RUN": {
        # Start with PYR defaults; user can edit
        "initial_k": 12,
        "L": 4000,
        "T2_apod": 1.5,
        "phase_corr_angle": 10,
        "allowed_ppms": [182.5, 178.6, 178.3, 173, 171, 170],
        "ppm_threshold": 20,
        "target_length": 65536,
        "n_iter": 2,
        "target_peaks": [182.5, 178.6, 178.3, 173, 171, 170],
        "metabolite_names": ["lactate", "pyrhydrate", "klhydrate", "unk1", "ketoleucine", "pyruvate"],
        "tolerance": 1.0,
        "save_per_folder": True,
        "n_threshold": 2,
        "default_time_interval": 2.5,
        "k_learning_skip": 3,
        "k_learning_window": 9,
        "k_learning_trim": 5,
    },
}

# ---------- Helpers ----------

class EmittingStream(io.TextIOBase):
    """Capture stdout and emit lines via a Qt signal (provided by the owner)."""
    def __init__(self, write_callback):
        super().__init__()
        self._write_cb = write_callback
        self._buffer = ""

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_cb(line + "\n")
        return len(s)

    def flush(self):
        if self._buffer:
            self._write_cb(self._buffer)
            self._buffer = ""

@dataclass
class FolderRunSpec:
    base_path: str                   # selected base_path (experiment root)
    start_folder: Optional[int]      # e.g., 1
    end_folder: Optional[int]        # e.g., 200
    time_interval: float             # per-folder time interval
    output_name: str                 # subfolder name under output_dir

# gui_multi_folders.py  (Block 2/4)

class ParametersDialog(QDialog):
    """Shows/edits the processing parameters and lets the user apply presets."""
    def __init__(self, current_params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Parameters")
        self.resize(680, 520)
        self.params = dict(current_params)  # working copy

        # --- Widgets ---
        layout = QVBoxLayout(self)

        # Preset buttons
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        self.btn_pyr = QPushButton("PYR_RUN (default)")
        self.btn_kl = QPushButton("KL_RUN")
        self.btn_copol = QPushButton("COPOL_RUN")
        preset_row.addWidget(self.btn_pyr)
        preset_row.addWidget(self.btn_kl)
        preset_row.addWidget(self.btn_copol)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        # Form for all parameters
        form = QFormLayout()

        # A small helper to create typed inputs
        self.widgets = {}

        def spin_i(key, rng=(1, 1_000_000)):
            w = QSpinBox()
            w.setRange(*rng)
            w.setValue(int(self.params.get(key, 0)))
            self.widgets[key] = w
            return w

        def spin_f(key, rng=(0.0, 1e9), step=0.1, decimals=6):
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
            w.setSingleStep(step)
            w.setRange(*rng)
            w.setValue(float(self.params.get(key, 0.0)))
            self.widgets[key] = w
            return w

        def text_list(key):
            w = QLineEdit()
            # show as comma-separated
            v = self.params.get(key, [])
            w.setText(", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
            self.widgets[key] = w
            return w

        def text_str(key):
            w = QLineEdit(str(self.params.get(key, "")))
            self.widgets[key] = w
            return w

        # Numeric & list params
        form.addRow("initial_k", spin_i("initial_k", (1, 9999)))
        form.addRow("L", spin_i("L", (2, 1_000_000)))
        form.addRow("T2_apod", spin_f("T2_apod", (0, 1e6), step=0.5, decimals=3))
        form.addRow("phase_corr_angle", spin_f("phase_corr_angle", (-360.0, 360.0), step=1.0, decimals=3))
        form.addRow("allowed_ppms (comma)", text_list("allowed_ppms"))
        form.addRow("ppm_threshold", spin_f("ppm_threshold", (0.0, 1e6), step=0.1, decimals=3))
        form.addRow("target_length", spin_i("target_length", (1, 10_000_000)))
        form.addRow("n_iter", spin_i("n_iter", (1, 64)))
        form.addRow("target_peaks (comma)", text_list("target_peaks"))
        form.addRow("metabolite_names (comma)", text_list("metabolite_names"))
        form.addRow("tolerance", spin_f("tolerance", (0.0, 1e6), step=0.1, decimals=6))
        cb = QCheckBox()
        cb.setChecked(bool(self.params.get("save_per_folder", True)))
        self.widgets["save_per_folder"] = cb
        form.addRow("save_per_folder", cb)
        form.addRow("n_threshold", spin_i("n_threshold", (1, 64)))
        form.addRow("default_time_interval", spin_f("default_time_interval", (0.0, 1e6), step=0.1, decimals=3))
        form.addRow("k_learning_skip (omit first n folders)", spin_i("k_learning_skip", (0, 1000)))
        form.addRow("k_learning_window (rolling W)", spin_i("k_learning_window", (1, 1000)))
        form.addRow("k_learning_trim (middle M of W)", spin_i("k_learning_trim", (1, 1000)))

        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply = QPushButton("Apply")
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_apply)
        layout.addLayout(btn_row)

        # Signals
        self.btn_pyr.clicked.connect(lambda: self._apply_preset("PYR_RUN"))
        self.btn_kl.clicked.connect(lambda: self._apply_preset("KL_RUN"))
        self.btn_copol.clicked.connect(lambda: self._apply_preset("COPOL_RUN"))
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._on_apply)

        # Default preset visual cue (no change to values yet)
        self._apply_preset("PYR_RUN", update_fields=True, silent=True)

    def _apply_preset(self, name: str, update_fields=True, silent=False):
        preset = PRESETS.get(name, PRESETS["PYR_RUN"])
        # Keep existing manual edits unless we're explicitly updating fields
        for k, v in preset.items():
            if update_fields:
                self.params[k] = v
        if update_fields:
            # push to widgets
            for k, w in self.widgets.items():
                v = self.params.get(k)
                if isinstance(w, QSpinBox):
                    w.setValue(int(v))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(v))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QLineEdit):
                    if isinstance(v, (list, tuple)):
                        w.setText(", ".join(map(str, v)))
                    else:
                        w.setText(str(v))
        if not silent:
            QMessageBox.information(self, "Preset Applied", f"Applied preset: {name}")

    def _on_apply(self):
        # Pull from widgets into self.params
        out = {}
        for k, w in self.widgets.items():
            if isinstance(w, QSpinBox):
                out[k] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                out[k] = float(w.value())
            elif isinstance(w, QCheckBox):
                out[k] = bool(w.isChecked())
            elif isinstance(w, QLineEdit):
                text = w.text().strip()
                # try list parsing (comma-separated)
                if k in ("allowed_ppms", "target_peaks", "metabolite_names"):
                    if k == "metabolite_names":
                        # keep as strings
                        items = [s.strip() for s in text.split(",")] if text else []
                    else:
                        # keep as floats
                        items = [float(s.strip()) for s in text.split(",")] if text else []
                    out[k] = items
                else:
                    out[k] = text
        # cast a few known types back
        out["initial_k"] = int(out.get("initial_k", 4))
        out["L"] = int(out.get("L", 2000))
        out["target_length"] = int(out.get("target_length", 65536))
        out["n_iter"] = int(out.get("n_iter", 2))
        out["n_threshold"] = int(out.get("n_threshold", 2))
        out["default_time_interval"] = float(out.get("default_time_interval", 3.5))
        out["T2_apod"] = float(out.get("T2_apod", 5))
        out["phase_corr_angle"] = float(out.get("phase_corr_angle", 10))
        out["ppm_threshold"] = float(out.get("ppm_threshold", 1))
        out["tolerance"] = float(out.get("tolerance", 1.0))
        out["save_per_folder"] = bool(out.get("save_per_folder", True))
        out["k_learning_skip"] = int(out.get("k_learning_skip", 3))
        out["k_learning_window"] = int(out.get("k_learning_window", 9))
        out["k_learning_trim"] = int(out.get("k_learning_trim", 5))

        self.params.update(out)
        self.accept()

    def get_params(self) -> Dict[str, Any]:
        return dict(self.params)


class FolderOverridesDialog(QDialog):
    """Per-folder overrides (time_interval, output_name) for the selected base_paths."""
    def __init__(self, base_paths: List[str], default_time_interval: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Per-Folder Overrides")
        self.resize(760, 420)
        self.base_paths = list(base_paths)
        self.default_time_interval = float(default_time_interval)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Set optional folder range (start/end), plus time_interval and output_name per base_path.\n"
            "Leave start/end blank to analyze ALL available numbered folders (default)."
        )
        layout.addWidget(info)

        self.table = QTableWidget(len(self.base_paths), 5)
        self.table.setHorizontalHeaderLabels(
            ["Folder", "start_folder", "end_folder", "time_interval", "output_name"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)

        for row, p in enumerate(self.base_paths):
            name = os.path.basename(os.path.normpath(p))
            # Folder name (read-only)
            item_folder = QTableWidgetItem(name)
            item_folder.setFlags(item_folder.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_folder)

            # start_folder / end_folder default blank = analyze all
            self.table.setItem(row, 1, QTableWidgetItem(""))  # start_folder
            self.table.setItem(row, 2, QTableWidgetItem(""))  # end_folder

            # time_interval default from preset
            self.table.setItem(row, 3, QTableWidgetItem(f"{self.default_time_interval:.3f}"))

            # output_name in downstream-compatible format: integrated_data_<YYYYMMDD-HHMM> (<RUN_ID>)
            m_ts = re.search(r"(\d{6}-\d{6})", name)
            timestamp = m_ts.group(1) if m_ts else "unknown"
            m_run = re.search(r"\(([^)]+)\)", name)
            run_id = m_run.group(1) if m_run else "RUN"
            output_default = f"integrated_data_{timestamp} ({run_id})"
            self.table.setItem(row, 4, QTableWidgetItem(output_default))

        layout.addWidget(self.table)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok = QPushButton("OK")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        layout.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._on_ok)

        self.overrides: List[Dict[str, Any]] = []

    def _on_ok(self):
        self.overrides.clear()
        for r, p in enumerate(self.base_paths):
            # Columns: 0=Folder, 1=start, 2=end, 3=time_interval, 4=output_name
            start_text = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            end_text   = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip()
            ti_text    = (self.table.item(r, 3).text() if self.table.item(r, 3) else "").strip()
            on_text    = (self.table.item(r, 4).text() if self.table.item(r, 4) else "").strip()

            # Optional ints
            start_folder = None
            end_folder = None
            if start_text:
                if not start_text.isdigit():
                    QMessageBox.warning(self, "Invalid Value",
                                        f"Row {r+1}: start_folder must be a positive integer or blank")
                    return
                start_folder = int(start_text)
            if end_text:
                if not end_text.isdigit():
                    QMessageBox.warning(self, "Invalid Value",
                                        f"Row {r+1}: end_folder must be a positive integer or blank")
                    return
                end_folder = int(end_text)
            if start_folder is not None and end_folder is not None and end_folder < start_folder:
                QMessageBox.warning(self, "Invalid Value",
                                    f"Row {r+1}: end_folder must be >= start_folder")
                return

            # Required fields
            try:
                ti = float(ti_text)
            except Exception:
                QMessageBox.warning(self, "Invalid Value",
                                    f"Row {r+1}: time_interval must be a number")
                return
            if not on_text:
                QMessageBox.warning(self, "Invalid Value",
                                    f"Row {r+1}: output_name cannot be empty")
                return

            self.overrides.append({
                "base_path": p,
                "start_folder": start_folder,
                "end_folder": end_folder,
                "time_interval": ti,
                "output_name": on_text
            })
        self.accept()

    def get_overrides(self) -> List[Dict[str, Any]]:
        return list(self.overrides)

# gui_multi_folders.py  (Block 3/4)

class RunnerThread(QThread):
    prog = pyqtSignal(int, str)           # progress 0..100, message
    line = pyqtSignal(str)                # result log line
    folder_done = pyqtSignal(str, str)    # (base_path, per_folder_output_dir)
    all_done = pyqtSignal(bool, str)      # success, message

    def __init__(self,
                 specs: List[FolderRunSpec],
                 params: Dict[str, Any],
                 output_dir: str,
                 session_output_name: str,
                 parent=None):
        super().__init__(parent)
        self.specs = specs
        self.params = dict(params)
        self.output_dir = output_dir
        self.session_output_name = session_output_name
        self._stop = False

    def stop(self):
        self._stop = True

    def _prepare_kwargs(self, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Only pass kwargs that the target function accepts, to avoid signature mismatch."""
        import inspect
        sig = inspect.signature(process_multiple_folders)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in base_kwargs.items() if k in allowed}

    def _write_parameters_txt(self):
        # Write once per session at OUTPUT_DIR/PARAMETERS.txt
        os.makedirs(self.output_dir, exist_ok=True)
        ts_compact = time.strftime("%Y%m%d-%H%M") # for filename
        path = os.path.join(self.output_dir, f"PARAMETERS_{ts_compact}.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("PARAMETERS (session)\n")
                f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"output_dir: {self.output_dir}\n")
                # f.write(f"session_output_name: {self.session_output_name}\n")
                f.write("\n-- Effective Parameters --\n")
                json.dump(self.params, f, indent=2, default=str)
                f.write("\n\n-- Folder Specs --\n")
                for s in self.specs:
                    f.write(json.dumps(asdict(s), indent=2))
                    f.write("\n")
        except Exception as e:
            self.line.emit(f"[WARN] Could not write PARAMETERS.txt: {e}")


    def _write_proc_txt(self, per_folder_dir: str, base_path: str, buffered_stdout: str):
        # Save a per-folder PROC summary file alongside the CSVs
        try:
            os.makedirs(per_folder_dir, exist_ok=True)
            bn = os.path.basename(os.path.normpath(base_path))
            proc_path = os.path.join(per_folder_dir, f"{bn}_PROC.txt")
            with open(proc_path, "w", encoding="utf-8") as f:
                f.write(f"PROC for {bn}\n")
                f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("-- Parameters --\n")
                json.dump(self.params, f, indent=2, default=str)
                f.write("\n\n-- Stdout / Results --\n")
                f.write(buffered_stdout or "(no output captured)")
        except Exception as e:
            self.line.emit(f"[WARN] Could not write PROC file: {e}")

    def run(self):
        ok = True
        msg = "Done."
        try:
            # One-time session PARAMETERS.txt
            self._write_parameters_txt()

            total = len(self.specs)
            for idx, spec in enumerate(self.specs, 1):
                if self._stop:
                    ok = False
                    msg = "Stopped by user."
                    break

                self.prog.emit(int((idx - 1) / max(1, total) * 100), f"Starting {spec.base_path}")

                # Capture stdout from the processing for the Results Log and PROC file
                buf = StringIO()

                def _emit(line):
                    # real-time emission; also buffer for PROC
                    self.line.emit(line.rstrip("\n"))
                    buf.write(line)

                # Temporarily redirect stdout
                old_stdout = sys.stdout
                sys.stdout = EmittingStream(_emit)
                try:
                    # Build kwargs per folder
                    # Resolve start/end with safe defaults if user left them blank
                    sf = spec.start_folder if spec.start_folder is not None else int(self.params.get("start_folder", 1))
                    ef = spec.end_folder   if spec.end_folder   is not None else int(self.params.get("end_folder", 200))

                    base_kwargs = {
                        "base_path": spec.base_path,
                        "start_folder": sf,
                        "end_folder": ef,
                        "time_interval": spec.time_interval,
                        "output_dir": self.output_dir,
                        "output_name": spec.output_name,
                        # the rest come from self.params (processing parameters)
                        **self.params,
                    }

                    # ---- Callbacks for per-numbered-folder progress & stop ----
                    spec_total = len(self.specs)
                    spec_index = idx  # 1-based from the for-loop

                    def per_inner_folder_cb(done_count: int, inner_total: int, folder_num: int):
                        try:
                            if inner_total <= 0:
                                frac = (spec_index) / max(1, spec_total)
                            else:
                                outer = (spec_index - 1) / max(1, spec_total)
                                inner = (done_count / inner_total) / max(1, spec_total)
                                frac = outer + inner
                            pct = int(max(0.0, min(1.0, frac)) * 100.0)
                            base_name = os.path.basename(os.path.normpath(spec.base_path))
                            self.prog.emit(pct, f"{base_name} — folder {done_count}/{max(1, inner_total)} (#{folder_num})")
                        except Exception:
                            pass

                    def should_stop_cb() -> bool:
                        return bool(self._stop)

                    base_kwargs["per_inner_folder_cb"] = per_inner_folder_cb
                    base_kwargs["should_stop_cb"] = should_stop_cb

                    kwargs = self._prepare_kwargs(base_kwargs)
                    process_multiple_folders(**kwargs)

                
                finally:
                    # Restore stdout
                    try:
                        sys.stdout.flush()
                    except Exception:
                        pass
                    sys.stdout = old_stdout

                # Determine per-folder output dir (matches where CSVs go)
                try:
                    per_folder_dir = get_per_folder_output_dir(self.output_dir, spec.output_name)
                except Exception:
                    per_folder_dir = os.path.join(self.output_dir, spec.output_name)

                # Write PROC file from buffered stdout
                self._write_proc_txt(per_folder_dir, spec.base_path, buf.getvalue())

                self.folder_done.emit(spec.base_path, per_folder_dir)
                self.prog.emit(int(idx / max(1, total) * 100), f"Finished {spec.base_path}")

            self.prog.emit(100, "All folders processed.")
        except Exception as e:
            ok = False
            msg = f"Error during run: {e}\n{traceback.format_exc()}"
            self.line.emit(msg)
        finally:
            self.all_done.emit(ok, msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hankel SVD Denoising — Multi-Folder GUI")
        self.resize(1100, 720)

        # State
        self.input_root: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.session_output_name: str = "OUTPUT"
        self.selected_paths: List[str] = []
        self.specs: List[FolderRunSpec] = []
        self.current_params: Dict[str, Any] = dict(PRESETS["PYR_RUN"])  # default preset

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Top controls row
        top = QHBoxLayout()
        self.btn_choose_input = QPushButton("1) Choose Input Root…")
        self.lbl_input = QLabel("No input root selected")
        self.btn_choose_output = QPushButton("2) Choose Output Folder…")
        self.lbl_output = QLabel("No output folder selected")
        self.btn_params = QPushButton("Parameters…")
        top.addWidget(self.btn_choose_input)
        top.addWidget(self.lbl_input, 1)
        top.addSpacing(12)
        top.addWidget(self.btn_choose_output)
        top.addWidget(self.lbl_output, 1)
        top.addSpacing(12)
        top.addWidget(self.btn_params)
        outer.addLayout(top)

        # Middle: folder list + controls + progress
        mid = QHBoxLayout()
        self.list_subfolders = QListWidget()
        self.list_subfolders.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_subfolders.setMinimumWidth(360)
        self.list_subfolders.setAlternatingRowColors(True)

        right_col = QVBoxLayout()
        self.btn_refresh = QPushButton("Refresh Folders")
        self.btn_select = QPushButton("3) Select Highlighted Folders")
        self.btn_overrides = QPushButton("4) Set Per-Folder Overrides")
        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        right_col.addWidget(self.btn_refresh)
        right_col.addWidget(self.btn_select)
        right_col.addWidget(self.btn_overrides)
        right_col.addStretch(1)
        right_col.addWidget(self.progress)
        right_col.addWidget(self.btn_run)
        right_col.addWidget(self.btn_stop)

        mid.addWidget(self.list_subfolders, 2)
        mid.addLayout(right_col, 1)

        outer.addLayout(mid)

        # Logs split: Selection Log (left) and Results Log (right)
        split = QSplitter(Qt.Horizontal)
        self.log_selection = QTextEdit()
        self.log_selection.setReadOnly(True)
        self.log_selection.setPlaceholderText("Selection Log…")
        self.log_results = QTextEdit()
        self.log_results.setReadOnly(True)
        self.log_results.setPlaceholderText("Results Log (keeps last ~10 folders)…")
        split.addWidget(self.log_selection)
        split.addWidget(self.log_results)
        split.setSizes([500, 600])
        outer.addWidget(split, 1)

        # Wiring
        self.btn_choose_input.clicked.connect(self.on_choose_input)
        self.btn_choose_output.clicked.connect(self.on_choose_output)
        self.btn_refresh.clicked.connect(self.populate_subfolders)
        self.btn_select.clicked.connect(self.on_select_folders)
        self.btn_overrides.clicked.connect(self.on_set_overrides)
        self.btn_params.clicked.connect(self.on_open_params)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_stop.clicked.connect(self.on_stop)

        # Thread
        self.runner: Optional[RunnerThread] = None

    # ---------- UI helpers ----------
    def append_selection_log(self, text: str):
        self.log_selection.append(text)
        self.log_selection.ensureCursorVisible()

    def append_results_log(self, text: str):
        self.log_results.append(text)
        self.log_results.ensureCursorVisible()

    # ---------- Actions ----------
    def on_choose_input(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Input Root")
        if d:
            self.input_root = d
            self.lbl_input.setText(d)
            self.append_selection_log(f"[Input root] {d}")
            self.populate_subfolders()

    def on_choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if d:
            self.output_dir = d
            self.lbl_output.setText(d)
            self.append_selection_log(f"[Output folder] {d}")

    def populate_subfolders(self):
        self.list_subfolders.clear()
        if not self.input_root:
            return
        try:
            # List immediate subfolders
            for name in sorted(os.listdir(self.input_root)):
                p = os.path.join(self.input_root, name)
                if os.path.isdir(p):
                    item = QListWidgetItem(name)
                    item.setData(Qt.UserRole, p)
                    self.list_subfolders.addItem(item)
            self.append_selection_log(f"Found {self.list_subfolders.count()} subfolders.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to list subfolders:\n{e}")

    def on_select_folders(self):
        # Preserve the order in the list widget
        indexes = sorted(self.list_subfolders.selectedIndexes(), key=lambda x: x.row())
        paths = [self.list_subfolders.item(i.row()).data(Qt.UserRole) for i in indexes]
        if not paths:
            QMessageBox.information(self, "No Selection", "Select one or more subfolders first.")
            return
        self.selected_paths = paths
        self.append_selection_log("Selected folders (in order):")
        for p in self.selected_paths:
            self.append_selection_log(f"  • {p}")
        self.append_selection_log(f"Total selected: {len(self.selected_paths)}")

    def on_set_overrides(self):
        if not self.selected_paths:
            QMessageBox.information(self, "No Selection", "Select folders first.")
            return
        default_ti = float(self.current_params.get("default_time_interval", 3.5))
        dlg = FolderOverridesDialog(self.selected_paths, default_ti, self)
        if dlg.exec_() == QDialog.Accepted:
            ov = dlg.get_overrides()
            # Build specs with defaults for start/end folders
            self.specs = []
            for o in ov:
                self.specs.append(
                    FolderRunSpec(
                        base_path=o["base_path"],
                        start_folder=o.get("start_folder"),   # None => analyze all
                        end_folder=o.get("end_folder"),       # None => analyze all
                        time_interval=float(o["time_interval"]),
                        output_name=str(o["output_name"])
                    )
                )
            self.append_selection_log("Per-folder overrides set:")
            for s in self.specs:
                self.append_selection_log(
                    f"  • {s.base_path} | time_interval={s.time_interval} | output_name={s.output_name}"
                )
        else:
            self.append_selection_log("Per-folder overrides canceled.")

    def on_open_params(self):
        dlg = ParametersDialog(self.current_params, self)
        if dlg.exec_() == QDialog.Accepted:
            self.current_params = dlg.get_params()
            self.append_selection_log("[Parameters] Updated.")
        else:
            self.append_selection_log("[Parameters] No changes.")

    def on_run(self):
        if not self.selected_paths:
            QMessageBox.information(self, "Missing Selection", "Select folders first.")
            return
        if not self.specs:
            # If user skipped overrides, generate simple defaults
            default_ti = float(self.current_params.get("default_time_interval", 3.5))
            self.specs = []
            for p in self.selected_paths:
                base = os.path.basename(os.path.normpath(p))
                m_ts = re.search(r"(\d{6}-\d{6})", base)
                timestamp = m_ts.group(1) if m_ts else "unknown"
                m_run = re.search(r"\(([^)]+)\)", base)
                run_id = m_run.group(1) if m_run else "RUN"
                output_name = f"integrated_data_{timestamp} ({run_id})"
                self.specs.append(
                    FolderRunSpec(
                        base_path=p, start_folder=1, end_folder=200,
                        time_interval=default_ti,
                        output_name=output_name
                    )
                )
            self.append_selection_log("[Info] Using default per-folder overrides.")

        # # Ask for a session-level output_name used for PARAMETERS.txt root
        # name, ok = QInputDialog_getText(
        #     self, "Session Output Name",
        #     "Enter a session-level output_name for PARAMETERS.txt:",
        #     self.session_output_name
        # )
        # if not ok:
        #     return
        # self.session_output_name = name or "OUTPUT"

        if not self.output_dir:
            QMessageBox.information(self, "Missing Output Folder", "Choose an output folder first.")
            return

        # Start worker
        self.progress.setValue(0)
        self.append_results_log("=== Run started ===")
        self.runner = RunnerThread(self.specs, self.current_params, self.output_dir, self.session_output_name)
        self.runner.prog.connect(self.on_prog)
        self.runner.line.connect(self.on_line)
        self.runner.folder_done.connect(self.on_folder_done)
        self.runner.all_done.connect(self.on_all_done)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.runner.start()

    def on_stop(self):
        if self.runner and self.runner.isRunning():
            self.runner.stop()
            self.append_results_log("Stop requested — will halt after current folder.")

    def on_prog(self, pct: int, msg: str):
        self.progress.setValue(pct)
        if msg:
            self.statusBar().showMessage(msg)

    def on_line(self, text: str):
        # Append to results log but keep memory in check
        self.log_results.append(text)
        MAX_LINES = 6000  # rough cap to keep ~10 folders of output
        if self.log_results.document().blockCount() > MAX_LINES:
            cursor = self.log_results.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def on_folder_done(self, base_path: str, per_folder_dir: str):
        self.append_results_log(f"[Folder done] {base_path} → {per_folder_dir}")

    def on_all_done(self, ok: bool, msg: str):
        self.append_results_log(f"=== Run finished: {msg} ===")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)


# Small helper since QInputDialog.getText has an awkward signature to patch inline
def QInputDialog_getText(parent, title, label, default=""):
    from PyQt5.QtWidgets import QInputDialog
    text, ok = QInputDialog.getText(parent, title, label, text=default)
    return text, ok


# ---------- Main ----------
# gui_multi_folders.py  (Block 4/4)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
