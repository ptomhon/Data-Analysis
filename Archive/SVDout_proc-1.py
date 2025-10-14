import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def compute_metrics(csv_path, time_cutoff=None, xlim=None, ylim=None, export=False):
    raw = pd.read_csv(csv_path, header=None)

    # Build column names from row0 type + row2 substrate
    colnames = []
    for i, (ctype, substrate) in enumerate(zip(raw.iloc[0], raw.iloc[2])):
        if i == 0:
            colnames.append("Time")
        else:
            if ctype == "Integral":
                colnames.append(f"Integral_{substrate}")
            else:
                colnames.append(None)

    df = raw.iloc[3:].copy()
    df.columns = colnames
    df = df.loc[:, [c for c in df.columns if c is not None]]
    df = df.apply(pd.to_numeric, errors="coerce")

    time = df["Time"]
    lactate = df["Integral_lactate"]
    pyruvate = df["Integral_pyruvate"]

    # safe ratio
    pyruvate_safe = pyruvate.mask(pyruvate == 0)
    ratio = lactate / pyruvate_safe

    # Apply time cutoff
    if time_cutoff is not None:
        mask = time <= time_cutoff
        sub_time = time[mask]
        sub_lac = lactate[mask]
        sub_ratio = ratio[mask]
    else:
        sub_time, sub_lac, sub_ratio = time, lactate, ratio

    idx_ratio = sub_ratio.idxmax()
    idx_lac = sub_lac.idxmax()

    results = {
        "peak_ratio_lac/pyr": float(ratio.loc[idx_ratio]),
        "peak_lactate_integral": float(lactate.loc[idx_lac]),
        "time_at_peak_ratio": float(time.loc[idx_ratio]),
        "time_at_peak_lactate": float(time.loc[idx_lac]),
        "idx_peak_ratio": int(idx_ratio),
        "idx_peak_lactate": int(idx_lac),
        "time_cutoff_used": time_cutoff,
    }

    # Plot lactate vs time
    plt.figure(figsize=(6,4))
    plt.plot(time, lactate, label="Lactate integral", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Lactate Integral")
    if xlim: plt.xlim(0, xlim)
    if ylim: plt.ylim(-1, ylim)
    plt.title("Lactate vs Time")
    plt.legend()
    plt.tight_layout()

    if export:
        csv_path = Path(csv_path)
        outdir = csv_path.parent
        name = csv_path.stem
        # pull from first "(" onward
        if "(" in name:
            name_part = name[name.index("("):]
        else:
            name_part = "_" + name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        outstem = f"{name_part}_proc_at{timestamp}"
        out_csv = outdir / f"{outstem}.csv"
        out_png = outdir / f"{outstem}.png"
        out_pdf = outdir / f"{outstem}.pdf"

        # save results
        pd.DataFrame([results]).to_csv(out_csv, index=False)
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        print(f"Exported results: {out_csv}")
        print(f"Exported plot: {out_png}")
        print(f"Exported plot: {out_pdf}")

    plt.show()

    return results

if __name__ == "__main__":
    path = r"D:\WSU\Projects\Eukaryote Experiments\2025-08 Pilot Leukemia Cell Experiments\Data Analysis\2025-08-26 SVD Proc-Int\integrated_data_250826-164823 (PYR70_7).csv"
    results = compute_metrics(
        path,
        time_cutoff=60.0,   # seconds
        xlim=400,            # x-axis max
        ylim=50,            # y-axis max
        export=True         # export CSV + plot
    )
    for k,v in results.items():
        print(f"{k}: {v}")
