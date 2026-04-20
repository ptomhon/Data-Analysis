#!/usr/bin/env python3
"""
rolling_average_fids_vscode.py

VS Code "Run" friendly script that:
- Prompts you to pick a BASE folder containing numbered subfolders 1..N, each with fid.csv
  e.g. D:\WSU\Raw Data\Spinsolve-1.4T_13C\...\[FOLDERNUMBER]\fid.csv
- Prompts you to pick an OUTPUT ROOT folder where results are written to [OUTPUT]\k\fid.csv
- Asks for:
    * start folder number (default 1)
    * end folder number (default 200)
    * rolling window size (default 5)
- Reads each fid.csv once, verifies all time axes match within a small tolerance,
  and writes headerless CSVs (time_ms,real,imag) for each rolling window.
- Skips windows with missing files or mismatched time axes; issues are logged to a file in OUTPUT ROOT.

CSV assumptions (robust to header/no-header):
- Column 0: time in milliseconds (ms)
- Column 1: real
- Column 2: imag

Author: ChatGPT
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------- Defaults (you can change these) ----------
DEFAULT_START = 1
DEFAULT_END = 150
DEFAULT_WINDOW = 10
ABS_TOL_MS = 1e-9  # time-axis equality tolerance in ms
INPUT_FILENAME = "fid.csv"
OUTPUT_FILENAME = "fid.csv"  # headerless
LOG_NAME = "rolling_average_log.txt"

# If you prefer not to use pickers, set these to absolute paths and they'll be used.
HARDCODED_BASE = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2026-02-19\260219-103744 7degCarbon-Cells (PYR70_2)"
HARDCODED_OUTPUT = r"C:\Users\pmtom\Downloads"
# -----------------------------------------------------


def read_fid_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a fid.csv, returning (time_ms, real, imag) as numpy arrays.
    Tries to be robust to header vs no-header and column naming.
    """
    # Try with header
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 3:
            raise ValueError("Less than 3 columns with header parse; retrying without header.")
        colmap = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if "time" in lc:
                colmap[c] = "time_ms"
            elif "real" in lc:
                colmap[c] = "real"
            elif "imag" in lc or "imaginary" in lc:
                colmap[c] = "imag"
        df = df.rename(columns=colmap)
        if not {"time_ms", "real", "imag"}.issubset(df.columns):
            raise ValueError("Could not identify required columns from header; retrying without header.")
        df = df[["time_ms", "real", "imag"]]
    except Exception:
        # Fallback: no header
        df = pd.read_csv(path, header=None, names=["time_ms", "real", "imag"])
        if df.shape[1] > 3:
            df = df.iloc[:, :3]
    # Ensure numeric
    for col in ["time_ms", "real", "imag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_ms", "real", "imag"])

    time_ms = df["time_ms"].to_numpy()
    real = df["real"].to_numpy()
    imag = df["imag"].to_numpy()
    return time_ms, real, imag


def average_arrays(time_list: List[np.ndarray],
                   real_list: List[np.ndarray],
                   imag_list: List[np.ndarray],
                   atol_ms: float) -> pd.DataFrame:
    """
    Validate identical time axis and return averaged DataFrame.
    """
    ref_time = time_list[0]
    npts = ref_time.size

    for idx, t in enumerate(time_list[1:], start=2):
        if t.size != npts:
            raise ValueError(f"Time axis length mismatch at file #{idx}: expected {npts}, got {t.size}")
        if not np.allclose(t, ref_time, atol=atol_ms, rtol=0.0):
            diffs = np.max(np.abs(t - ref_time))
            raise ValueError(f"Time axis mismatch at file #{idx}. Max |Δt|={diffs:.3e} ms exceeds atol={atol_ms}.")

    sum_real = np.zeros_like(real_list[0], dtype=np.float64)
    sum_imag = np.zeros_like(imag_list[0], dtype=np.float64)
    for r in real_list:
        sum_real += r
    for im in imag_list:
        sum_imag += im

    n = len(real_list)
    avg_real = sum_real / n
    avg_imag = sum_imag / n

    return pd.DataFrame({"time_ms": ref_time, "real": avg_real, "imag": avg_imag})


def log_append(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def run_with_ui() -> None:
    """
    Use tkinter dialogs to get base folder, output root, and numeric parameters.
    """
    base_dir: Optional[Path]
    out_root: Optional[Path]
    start_num: int = DEFAULT_START
    end_num: int = DEFAULT_END
    window: int = DEFAULT_WINDOW

    if HARDCODED_BASE is not None and HARDCODED_OUTPUT is not None:
        base_dir = Path(HARDCODED_BASE).expanduser().resolve()
        out_root = Path(HARDCODED_OUTPUT).expanduser().resolve()
    else:
        # UI pickers
        try:
            import tkinter as tk
            from tkinter import filedialog, simpledialog, messagebox
        except Exception as e:
            raise RuntimeError(
                "tkinter is not available. Install it or set HARDCODED_BASE/HARDCODED_OUTPUT."
            ) from e

        root = tk.Tk()
        root.withdraw()

        base = filedialog.askdirectory(title="Select BASE folder (contains numbered subfolders with fid.csv)")
        if not base:
            messagebox.showinfo("Rolling Average FIDs", "No base folder selected. Exiting.")
            return
        base_dir = Path(base).expanduser().resolve()

        out = filedialog.askdirectory(title="Select OUTPUT ROOT folder (averages written here)")
        if not out:
            messagebox.showinfo("Rolling Average FIDs", "No output folder selected. Exiting.")
            return
        out_root = Path(out).expanduser().resolve()

        # Ask for numbers
        start_num_val = simpledialog.askinteger("Start folder number",
                                                f"Enter start folder number (default {DEFAULT_START}):",
                                                initialvalue=DEFAULT_START,
                                                minvalue=1)
        if start_num_val is None:
            messagebox.showinfo("Rolling Average FIDs", "No start number provided. Exiting.")
            return
        start_num = start_num_val

        end_num_val = simpledialog.askinteger("End folder number",
                                              f"Enter end folder number (default {DEFAULT_END}):",
                                              initialvalue=DEFAULT_END,
                                              minvalue=start_num)
        if end_num_val is None:
            messagebox.showinfo("Rolling Average FIDs", "No end number provided. Exiting.")
            return
        end_num = end_num_val

        window_val = simpledialog.askinteger("Rolling window size",
                                             f"Enter window size (default {DEFAULT_WINDOW}):",
                                             initialvalue=DEFAULT_WINDOW,
                                             minvalue=1)
        if window_val is None:
            messagebox.showinfo("Rolling Average FIDs", "No window size provided. Exiting.")
            return
        window = window_val

    # Validate ranges
    if end_num < start_num:
        raise ValueError("end number must be >= start number.")
    if window < 1:
        raise ValueError("window size must be >= 1.")
    total = end_num - start_num + 1
    if total < window:
        raise ValueError("Range too small for the specified window size.")

    # Read all available fid.csv once
    data_cache: dict[int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    missing: List[int] = []
    for i in range(start_num, end_num + 1):
        p = base_dir / str(i) / INPUT_FILENAME
        try:
            if p.is_file():
                data_cache[i] = read_fid_csv(p)
            else:
                data_cache[i] = None
                missing.append(i)
        except Exception:
            data_cache[i] = None
            missing.append(i)

    log_path = (out_root / LOG_NAME)
    try:
        log_path.unlink(missing_ok=True)  # start fresh if exists
    except Exception:
        pass

    if missing:
        log_append(log_path, f"Missing or unreadable files (skipped in any affected windows): {missing}")

    # Rolling windows
    first_window_start = start_num
    last_window_start = end_num - window + 1
    out_count = 0
    for k, w_start in enumerate(range(first_window_start, last_window_start + 1), start=1):
        window_indices = list(range(w_start, w_start + window))
        if any(data_cache[idx] is None for idx in window_indices):
            log_append(log_path, f"Window {k} ({window_indices}) skipped due to missing file(s).")
            continue

        times = []
        reals = []
        imags = []
        for idx in window_indices:
            t, r, im = data_cache[idx]  # type: ignore
            times.append(t)
            reals.append(r)
            imags.append(im)

        try:
            df = average_arrays(times, reals, imags, atol_ms=ABS_TOL_MS)
        except Exception as e:
            log_append(log_path, f"Window {k} ({window_indices}) skipped: {type(e).__name__}: {e}")
            continue

        out_dir = out_root / str(k)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / OUTPUT_FILENAME
        df.to_csv(out_file, index=False, header=False)
        out_count += 1

    # UI summary if using tkinter
    if HARDCODED_BASE is None or HARDCODED_OUTPUT is None:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            message = f"Completed.\nWindows written: {out_count}\nLog file: {log_path}"
            messagebox.showinfo("Rolling Average FIDs", message)
        except Exception:
            print(f"Completed. Windows written: {out_count}. Log: {log_path}")


def main() -> None:
    run_with_ui()


if __name__ == "__main__":
    main()
