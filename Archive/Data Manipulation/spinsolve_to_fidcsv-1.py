#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Magritek Spinsolve FIDs to fid.csv using nmrglue (latest).
- Tries data.1d first (time-domain FID), then nmr_fid.dx (JCAMP-DX).
- Builds time axis from UDIC spectral width (dt = 1 / sw).
- Writes: <base>/<folder_number>/fid.csv  (no header) as: time,real,imag
"""

import argparse
from pathlib import Path
import numpy as np

import nmrglue as ng  # pip install -U nmrglue

PREFERRED = ("data.1d", "nmr_fid.dx", "fid.1d")  # pt1 not supported by nmrglue

def read_spinsolve_fid(folder: Path) -> tuple[np.ndarray, dict]:
    """Read FID from a Spinsolve run folder via nmrglue, returning (complex_fid, dic)."""
    last_err = None
    for specfile in PREFERRED:
        try:
            dic, data = ng.spinsolve.read(dir=folder.as_posix(), specfile=specfile)
            # Ensure 1D complex array
            arr = np.asarray(data)
            if arr.ndim > 1:
                arr = np.squeeze(arr)
            if np.iscomplexobj(arr):
                c = arr.astype(np.complex128)
            else:
                # If reader returned real-only numeric types, make imag zeros.
                # If two-column real/imag was returned (unlikely via nmrglue), handle that too.
                if arr.ndim == 2 and arr.shape[-1] == 2:
                    c = arr[..., 0].astype(float) + 1j * arr[..., 1].astype(float)
                else:
                    c = arr.astype(float).astype(np.complex128)
            return c, dic
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not read Spinsolve FID in {folder}: {last_err}")

def infer_dt_seconds(dic: dict, data: np.ndarray) -> float:
    """Use nmrglue's universal dictionary to get spectral width (Hz) and compute dt."""
    udic = ng.spinsolve.guess_udic(dic, data)  # contains 'sw', 'time', etc. for axis 0
    sw = float(udic[0].get("sw", 0.0))  # Hz
    if sw and sw > 0:
        return 1.0 / sw
    # Fallback: 1.0s if missing (should be rare if acqu.par or dx header exists)
    return 1.0

def convert_folder(folder: Path, verbose=True) -> bool:
    """Convert one numbered folder to fid.csv. Returns True on success."""
    try:
        fid, dic = read_spinsolve_fid(folder)
    except Exception as e:
        if verbose:
            print(f"[SKIP] {folder.name}: {e}")
        return False

    dt = infer_dt_seconds(dic, fid)
    t = np.arange(fid.size, dtype=float) * dt * 1000.0 # ms

    out = folder / "fid.csv"
    with out.open("w", encoding="utf-8") as f:
        for tt, v in zip(t, fid):
            f.write(f"{tt:.12g},{v.real:.12g},{v.imag:.12g}\n")
    if verbose:
        print(f"[WRITE] {out}")
    return True

def parse_folders(args) -> list[int]:
    if args.folders:
        return sorted({int(x) for x in args.folders})
    return list(range(args.start, args.end + 1))

def main():
    ap = argparse.ArgumentParser(description="Convert Spinsolve FIDs to fid.csv using nmrglue")
    ap.add_argument("--base", required=True, help="Base directory containing numbered subfolders")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--folders", nargs="+", help="Explicit folder numbers (e.g. 1 5 10)")
    g.add_argument("--start", type=int, help="Start folder number (inclusive)")
    ap.add_argument("--end", type=int, help="End folder number (inclusive) [required with --start]")
    ap.add_argument("-q", "--quiet", action="store_true", help="Suppress per-folder logs")
    args = ap.parse_args()

    if args.start is not None and args.end is None:
        ap.error("--end is required when using --start")

    base = Path(args.base)
    verbose = not args.quiet
    nums = parse_folders(args)

    n_ok = 0
    for n in nums:
        folder = base / str(n)
        if not folder.is_dir():
            if verbose:
                print(f"[MISS] {folder} (not a directory)")
            continue
        try:
            n_ok += int(convert_folder(folder, verbose=verbose))
        except Exception as e:
            if verbose:
                print(f"[ERROR] {folder}: {e}")

    if verbose:
        print(f"Done. Converted {n_ok} / {len(nums)} folders.")

if __name__ == "__main__":
    # === USER CONFIG (edit these) ===
    base = r"D:\WSU\Raw Data\Jing's data_13C\2025-12-02\251202-144243 Carbon (ref wait for T1)"
    folders = None          # e.g. [1, 5, 10] for specific runs
    start, end = 1, 2     # Range of folders to process if folders=None
    verbose = True
    # ===============================

    from pathlib import Path

    nums = folders if folders is not None else list(range(start, end+1))
    n_ok = 0
    for n in nums:
        folder = Path(base) / str(n)
        if not folder.is_dir():
            if verbose:
                print(f"[MISS] {folder} (not a directory)")
            continue
        try:
            ok = convert_folder(folder, verbose=verbose)
            n_ok += int(ok)
        except Exception as e:
            if verbose:
                print(f"[ERROR] {folder}: {e}")
    print(f"Done. Converted {n_ok} / {len(nums)} folders.")

