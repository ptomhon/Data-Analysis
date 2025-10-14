#!/usr/bin/env python3
"""
average_fids_vscode.py

Run-friendly script for VS Code (no terminal args needed).
- When you press "Run Python File" in VS Code, this script will:
  1) Open a file picker so you can choose N fid.csv files.
  2) (Optional) Ask where to save the averaged output CSV.
  3) Average the real/imag columns (time axis must match).
  4) Save a CSV with columns: time_ms,real,imag.

Expected CSV format (robust to header/no-header):
- Column 0: time in milliseconds (ms)
- Column 1: real
- Column 2: imag

You can also hardcode file paths in FILE_PATHS below (leave empty to use the picker).

Author: ChatGPT
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------- User-adjustable defaults ----------
DEFAULT_N = 5            # Use up to N files if more are selected
ABS_TOL_MS = 1e-9        # Time-axis equality tolerance in ms
DEFAULT_OUT_NAME = "averaged_fid.csv"

# If you prefer not to use the picker, put absolute paths here:
FILE_PATHS: List[str] = [
    r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-09-12\250912-173908 7degCarbon-Cells (Pyr_4)\3\fid.csv",
    r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-09-12\250912-173908 7degCarbon-Cells (Pyr_4)\4\fid.csv",
    r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-09-12\250912-173908 7degCarbon-Cells (Pyr_4)\5\fid.csv",
    r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-09-12\250912-173908 7degCarbon-Cells (Pyr_4)\6\fid.csv",
    r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-09-12\250912-173908 7degCarbon-Cells (Pyr_4)\7\fid.csv",
]

# ---------- End user defaults ----------


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


def average_fids(paths: List[Path], atol_ms: float = ABS_TOL_MS) -> pd.DataFrame:
    if len(paths) == 0:
        raise ValueError("No input files provided.")

    # Reference file
    ref_time, ref_real, ref_imag = read_fid_csv(paths[0])
    npts = ref_time.size
    sum_real = np.array(ref_real, dtype=np.float64)
    sum_imag = np.array(ref_imag, dtype=np.float64)

    used_files = 1
    for p in paths[1:]:
        t, r, im = read_fid_csv(p)
        if t.size != npts:
            raise ValueError(f"Time axis length mismatch in '{p.name}': expected {npts}, got {t.size}")
        if not np.allclose(t, ref_time, atol=atol_ms, rtol=0.0):
            diffs = np.max(np.abs(t - ref_time))
            raise ValueError(f"Time axis mismatch in '{p.name}'. Max |Δt|={diffs:.3e} ms exceeds atol={atol_ms}.")
        sum_real += r
        sum_imag += im
        used_files += 1

    avg_real = sum_real / used_files
    avg_imag = sum_imag / used_files

    return pd.DataFrame({"time_ms": ref_time, "real": avg_real, "imag": avg_imag})


def run_with_picker():
    """
    Use a file picker (tkinter) to choose input files and output path.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError(
            "tkinter is not available in this Python environment. "
            "Either install it or hardcode FILE_PATHS."
        ) from e

    root = tk.Tk()
    root.withdraw()

    filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
    in_paths = filedialog.askopenfilenames(
        title="Select fid.csv files to average (time axes must match)",
        filetypes=filetypes
    )
    if not in_paths:
        messagebox.showinfo("Average FIDs", "No files selected. Exiting.")
        return

    selected = [Path(p) for p in in_paths]
    if len(selected) > DEFAULT_N:
        # Use only first DEFAULT_N files
        selected = selected[:DEFAULT_N]
        messagebox.showinfo(
            "Average FIDs",
            f"Using the first {DEFAULT_N} files selected."
        )

    # Suggest output in the directory of the first input
    default_dir = selected[0].parent
    out_path = filedialog.asksaveasfilename(
        title="Save averaged FID CSV",
        initialdir=str(default_dir),
        initialfile=DEFAULT_OUT_NAME,
        defaultextension=".csv",
        filetypes=filetypes
    )
    if not out_path:
        out_path = str(default_dir / DEFAULT_OUT_NAME)

    try:
        df = average_fids(selected, atol_ms=ABS_TOL_MS)
        df.to_csv(out_path, index=False, header=False)
        messagebox.showinfo("Average FIDs", f"Wrote averaged FID to:\n{out_path}")
    except Exception as e:
        messagebox.showerror("Average FIDs - Error", f"{type(e).__name__}: {e}")


def main():
    # If FILE_PATHS has entries, use those; otherwise pop up a file picker
    if FILE_PATHS:
        paths = [Path(p).expanduser().resolve() for p in FILE_PATHS]
        if len(paths) > DEFAULT_N:
            paths = paths[:DEFAULT_N]
        df = average_fids(paths, atol_ms=ABS_TOL_MS)

        # Default output next to first file
        out_path = paths[0].parent / DEFAULT_OUT_NAME
        df.to_csv(out_path, index=False, header=False)
        print(f"Wrote averaged FID to: {out_path}")
        print("Files used:")
        for p in paths:
            print(f"  - {p}")
    else:
        run_with_picker()


if __name__ == "__main__":
    main()
