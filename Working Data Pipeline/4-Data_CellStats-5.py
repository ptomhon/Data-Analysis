import sys
import pandas as pd
import os
import re
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLineEdit, QLabel, QTableWidget, QTableWidgetItem, QHBoxLayout,
    QFormLayout, QCheckBox, QTextEdit, QTabWidget, QDialog, QScrollArea,
    QMessageBox, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

def extract_expdate_from_summary(filename: str):
    """Extract 6-digit experiment date from summary filenames like:
       summary_250829_fit_at... or summary250829_proc_at..."""
    m = re.search(r"summary[_]?(\d{6})_", filename)
    return m.group(1) if m else None

class DataImporter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Group Comparison Data Importer & Analysis")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- Group count control ---
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Number of Groups:"))
        self.group_count_spin = QSpinBox()
        self.group_count_spin.setMinimum(2)
        self.group_count_spin.setMaximum(10)
        self.group_count_spin.setValue(2)
        self.group_count_spin.valueChanged.connect(self.build_group_inputs)
        group_row.addWidget(self.group_count_spin)
        self.layout.addLayout(group_row)

        # --- Group name boxes ---
        self.group_name_layout = QHBoxLayout()
        self.layout.addLayout(self.group_name_layout)
        self.group_name_inputs = {}
        self.build_group_inputs()

        # --- Metric selection & normalization toggles ---
        self.metric_names = ["Lac", "kPL (fit)"]
        self.metric_checkboxes = []
        self.norm_checkboxes = []
        self.layout.addWidget(QLabel("Select Metrics and Normalization Options:"))

        metric_box = QHBoxLayout()
        for name in self.metric_names:
            metric_layout = QVBoxLayout()
            cb_metric = QCheckBox(name)
            cb_metric.setChecked(True)
            self.metric_checkboxes.append(cb_metric)

            cb_norm = QCheckBox("Apply Normalization")
            cb_norm.setChecked(True)
            self.norm_checkboxes.append(cb_norm)

            metric_layout.addWidget(cb_metric)
            metric_layout.addWidget(cb_norm)
            metric_box.addLayout(metric_layout)

        self.layout.addLayout(metric_box)


        # --- Export checkboxes ---
        self.export_df_checkbox = QCheckBox("Export Dataframe as CSV")
        self.export_stats_checkbox = QCheckBox("Export Statistics Results as CSV")
        self.export_box_checkbox = QCheckBox("Export Boxplots (PNG & PDF)")
        self.export_pca_checkbox = QCheckBox("Export PCA Scatter (PNG & PDF)")
        for cb in [self.export_df_checkbox, self.export_stats_checkbox,
                   self.export_box_checkbox, self.export_pca_checkbox]:
            self.layout.addWidget(cb)

        # --- Data selection + analysis buttons ---
        self.data_select_button = QPushButton("Data Selection")
        self.data_select_button.clicked.connect(self.open_data_selection)
        self.layout.addWidget(self.data_select_button)

        self.build_button = QPushButton("Build Dataframe")
        self.build_button.clicked.connect(self.build_dataframe)
        self.layout.addWidget(self.build_button)

        self.stats_button = QPushButton("Run Statistical Analysis")
        self.stats_button.clicked.connect(self.run_stats)
        self.layout.addWidget(self.stats_button)

        # --- Stats log ---
        self.stats_log = QTextEdit()
        self.stats_log.setReadOnly(True)
        self.layout.addWidget(QLabel("Statistics Log:"))
        self.layout.addWidget(self.stats_log)
        clr_btn = QPushButton("Clear Log")
        clr_btn.clicked.connect(self.stats_log.clear)
        self.layout.addWidget(clr_btn)

        # --- Tabs for data/plots ---
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.boxplot_fig, self.pca_fig = Figure(figsize=(5, 4)), Figure(figsize=(5, 4))
        self.boxplot_canvas, self.pca_canvas = FigureCanvas(self.boxplot_fig), FigureCanvas(self.pca_fig)
        self.tabs.addTab(self.boxplot_canvas, "Boxplots")
        self.tabs.addTab(self.pca_canvas, "PCA Scatter")

        # runtime state
        self.metric_entries = []
        self.final_df = None
        self.first_file_path = None

    def build_group_inputs(self):
        for i in reversed(range(self.group_name_layout.count())):
            w = self.group_name_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.group_name_inputs.clear()
        n = self.group_count_spin.value()
        for i in range(n):
            edit = QLineEdit(f"G{i+1}")
            self.group_name_inputs[i] = edit
            self.group_name_layout.addWidget(QLabel(f"Group {i+1}:"))
            self.group_name_layout.addWidget(edit)

    def open_data_selection(self):
        # Pass previous selections (if any) when reopening
        existing_metrics = getattr(self, "metric_entries", [])
        existing_norm = getattr(self, "normalization_block", None)

        dialog = DataSelectionWindow(self, existing_entries=existing_metrics, existing_norm=existing_norm)
        dialog.exec_()


    def parse_row_range(self, text):
        rows = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                s, e = map(int, part.split("-"))
                rows.extend(range(s - 1, e))
            else:
                rows.append(int(part) - 1)
        return rows
    
    def get_text(self, value):
        """Return the text from either a QLineEdit widget or a raw string."""
        if hasattr(value, "text"):
            return value.text()
        return str(value)

    def build_dataframe(self):
        if not self.metric_entries:
            QMessageBox.warning(self, "Missing Data",
                                "Please open the Data Selection window and choose CSV files.")
            return

        # --- collect group names first ---
        group_names = {f"G{i+1}": self.group_name_inputs[i].text() or f"G{i+1}"
                    for i in range(self.group_count_spin.value())}
        print("Building dataframe with groups:", group_names)

        # --- collect normalization block safely ---
        norm_block = getattr(self, "normalization_block", None)
        if not norm_block:
            QMessageBox.warning(self, "Missing Normalization Inputs",
                                "Please define normalization inputs in Data Selection.")
            return
        print("Normalization block:", norm_block)

        # get column indices (already stored as strings)
        try:
            col_cell = int(norm_block["col_cell"]) - 1
            col_pol = int(norm_block["col_pol"]) - 1
        except Exception:
            QMessageBox.warning(self, "Invalid Columns",
                                "Normalization column numbers must be numeric.")
            return

        norm_data = {}
        all_data = []
        self.first_file_path = None

        # --- Load normalization CSVs ---
        for gname, info in norm_block["group_files"].items():
            csv_path = info.get("file")
            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(self, "Missing Normalization CSV",
                                    f"No normalization CSV selected for {gname}.")
                return
            df = pd.read_csv(csv_path, header=None)
            rows = self.parse_row_range(self.get_text(info.get("rows", "")))
            try:
                cell_vals = df.iloc[rows, col_cell].astype(float).to_numpy()
                pol_vals = df.iloc[rows, col_pol].astype(float).to_numpy()
            except Exception as e:
                QMessageBox.warning(self, "Error Reading Normalization",
                                    f"Error reading normalization for {gname}: {e}")
                return
            norm_data[gname] = {"cell": cell_vals, "pol": pol_vals}

        # --- Load metric CSVs ---
        for entry in self.metric_entries:
            metric_name = entry["metric_name"]
            try:
                col_idx = int(entry["col_input"]) - 1
            except Exception:
                QMessageBox.warning(self, "Invalid Column", f"Column number invalid for {metric_name}.")
                return

            for gname, gfile in entry["group_files"].items():
                if not gfile:
                    QMessageBox.warning(self, "Missing CSV",
                                        f"No CSV selected for {metric_name} in {gname}.")
                    return
                df = pd.read_csv(gfile)
                if self.first_file_path is None:
                    self.first_file_path = gfile
                rows = self.parse_row_range(self.get_text(entry["group_rows"].get(gname, "")))
                series = df.iloc[:, col_idx]

                for i, r in enumerate(rows):
                    if r >= len(series):
                        continue
                    imported = series.iloc[r]
                    cell = norm_data[gname]["cell"][i] if i < len(norm_data[gname]["cell"]) else 1.0
                    pol = norm_data[gname]["pol"][i] if i < len(norm_data[gname]["pol"]) else 1.0

                    # --- Determine if normalization is applied ---
                    apply_norm = True
                    try:
                        apply_norm = self.norm_checkboxes[entry["metric_num"] - 1].isChecked()
                    except Exception:
                        pass  # fallback if something unexpected

                    if apply_norm:
                        # Apply metric-specific normalization
                        if entry["metric_num"] == 1:
                            # Metric 1 (Lac): imported / (pol * (cell / 1e6))
                            norm_val = imported / (pol * (cell / 1e6))
                        elif entry["metric_num"] == 2:
                            # Metric 2 (kPL): imported / (cell / 1e6)
                            norm_val = imported / (cell / 1e6)
                        else:
                            # Default fallback (in case more metrics are added)
                            norm_val = imported / (cell * pol / 1e6)
                        data_type = "normalized"
                    else:
                        norm_val = imported
                        data_type = "imported"

                    all_data.append([group_names[gname],metric_name,imported,cell,pol,norm_val,data_type])

        self.final_df = pd.DataFrame(
            all_data,
            columns=["Group", "Metric", "Imported", "CellDensity", "SubPol", "Value", "DataType"]
        )
        self.show_dataframe_tab()

        # --- Export dataframe ---
        if self.export_df_checkbox.isChecked() and self.first_file_path:
            folder = os.path.dirname(self.first_file_path)
            gnames = list(group_names.values())
            expdates = []
            for g in self.metric_entries[0]["group_files"].values():
                if g:
                    expdates.append(extract_expdate_from_summary(os.path.basename(g)) or "")
            foldername = "_".join([f"{e}-{n}" if e else n for e, n in zip(expdates, gnames)])
            export_folder = os.path.join(folder, foldername)
            os.makedirs(export_folder, exist_ok=True)
            self.export_group_folder = export_folder
            outpath = os.path.join(
                export_folder, f"dataframe_at{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
            )
            self.final_df.to_csv(outpath, index=False)


    def show_dataframe_tab(self):
        df_table = QTableWidget()
        df_table.setRowCount(len(self.final_df))
        df_table.setColumnCount(len(self.final_df.columns))
        df_table.setHorizontalHeaderLabels(self.final_df.columns)
        for i in range(len(self.final_df)):
            for j in range(len(self.final_df.columns)):
                df_table.setItem(i, j, QTableWidgetItem(str(self.final_df.iat[i, j])))
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Dataframe":
                self.tabs.removeTab(i)
                break
        self.tabs.insertTab(0, df_table, "Dataframe")

    def run_stats(self):
        if self.final_df is None:
            QMessageBox.warning(self, "No Data", "Please build the dataframe first.")
            return

        # --- Get selected metrics ---
        selected_metrics = [cb.text() for cb in self.metric_checkboxes if cb.isChecked()]
        if not selected_metrics:
            QMessageBox.warning(self, "No Metrics", "Please select at least one metric for analysis.")
            return

        df = self.final_df[self.final_df["Metric"].isin(selected_metrics)]
        groups = df["Group"].unique()
        if len(groups) < 2:
            QMessageBox.warning(self, "Stats Error", "Need at least two groups for comparison.")
            return

        # --- Pairwise t-tests across all group combinations ---
        pvals, results = [], []
        for metric in selected_metrics:
            combos = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i + 1:]]
            for g1, g2 in combos:
                g1v = df[(df["Group"] == g1) & (df["Metric"] == metric)]["Value"]
                g2v = df[(df["Group"] == g2) & (df["Metric"] == metric)]["Value"]
                if len(g1v) == 0 or len(g2v) == 0:
                    continue
                stat, p = ttest_ind(g1v, g2v, equal_var=False)
                pvals.append(p)
                results.append((metric, g1, g2, p))

        if not pvals:
            QMessageBox.warning(self, "Stats Error", "No valid group comparisons could be performed.")
            return

        # --- Multiple-testing correction ---
        reject, p_corr, _, _ = multipletests(pvals, method="fdr_bh")

        # --- Summary stats per group & metric ---
        summary = (
            df.groupby(["Group", "Metric"])["Value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary["cv_percent"] = (summary["std"] / summary["mean"]) * 100

        # --- Log output ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"\n=== Statistical Analysis Run at {timestamp} ===\n"

        # Pairwise comparison results
        log_text += "\nPairwise t-test Results (FDR corrected):\n"
        for (metric, g1, g2, p), pc, rej in zip(results, p_corr, reject):
            log_text += f"{metric}: {g1} vs {g2} | p={p:.4g}, corrected p={pc:.4g}, significant={rej}\n"

        # Summary statistics table
        log_text += "\nSummary Statistics (per Group & Metric):\n"
        for _, row in summary.iterrows():
            log_text += (f"{row['Metric']} | {row['Group']} : "
                        f"mean={row['mean']:.4g}, std={row['std']:.4g}, "
                        f"CV={row['cv_percent']:.2f}%\n")

        self.stats_log.append(log_text)

        # --- Save results to CSV if requested ---
        if self.export_stats_checkbox.isChecked() and self.first_file_path:
            folder = getattr(self, "export_group_folder", os.path.dirname(self.first_file_path))
            os.makedirs(folder, exist_ok=True)

            timestamp_tag = datetime.now().strftime("%Y%m%d-%H%M")
            # Full pairwise t-test output
            stats_path = os.path.join(folder, f"stats_at{timestamp_tag}.csv")
            with open(stats_path, "w") as f:
                f.write("Metric,Group1,Group2,p_value,corrected_p,significant\n")
                for (metric, g1, g2, p), pc, rej in zip(results, p_corr, reject):
                    f.write(f"{metric},{g1},{g2},{p},{pc},{rej}\n")

            # Summary statistics output
            summary_path = os.path.join(folder, f"summary_at{timestamp_tag}.csv")
            summary.to_csv(summary_path, index=False)

        # --- Store corrected p-values for annotation in plots ---
        self.p_corr_map = {metric: pc for (metric, g1, g2, p), pc in zip(results, p_corr)}

        # --- Define consistent colors across plots ---
        unique_groups = sorted(df["Group"].unique().tolist())
        palette = sns.color_palette("Set2", n_colors=len(unique_groups))
        self.color_map = {g: palette[i] for i, g in enumerate(unique_groups)}

        # --- Draw updated plots ---
        self.draw_boxplots(df, selected_metrics)
        self.draw_pca(df, selected_metrics)



    def draw_boxplots(self, df, selected_metrics):
        self.boxplot_fig.clear()
        n_metrics = len(selected_metrics)
        axes = self.boxplot_fig.subplots(1, n_metrics, squeeze=False)[0]

        # Optional corrected p-values if available
        p_corr_map = getattr(self, "p_corr_map", {})

        for i, metric in enumerate(selected_metrics):
            ax = axes[i]
            sub_df = df[df["Metric"] == metric]
            sns.boxplot(
                x="Group", y="Value", data=sub_df, ax=ax,
                palette=self.color_map if hasattr(self, "color_map") else "Set2"
            )
            sns.stripplot(
                x="Group", y="Value", data=sub_df,
                jitter=True, marker="o", edgecolor="gray", linewidth=0.5,
                palette=self.color_map if hasattr(self, "color_map") else "Set2",
                alpha=0.6, ax=ax
            )

            ax.set_title(metric)
            ax.set_xlabel("Group")
            ax.set_ylabel("Value")

            # Dynamic y-limit
            vals = sub_df["Value"].to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            ymax = 1.05 * float(np.nanmax(finite_vals)) if finite_vals.size > 0 else 1.0
            ax.set_ylim(0, ymax)

            # Annotate corrected p
            p_corr = p_corr_map.get(metric, np.nan)
            label = f"corrected p = {p_corr:.3g}" if np.isfinite(p_corr) else "corrected p = N/A"
            ax.text(0.5, 0.95, label, transform=ax.transAxes,
                    ha="center", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        self.boxplot_fig.tight_layout()
        self.boxplot_canvas.draw()

        if self.export_box_checkbox.isChecked() and self.first_file_path:
            folder = getattr(self, "export_group_folder", os.path.dirname(self.first_file_path))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.boxplot_fig.savefig(os.path.join(folder, f"boxplot_at{timestamp}.png"))
            self.boxplot_fig.savefig(os.path.join(folder, f"boxplot_at{timestamp}.pdf"))


    def draw_pca(self, df, selected_metrics):
        self.pca_fig.clear()
        ax = self.pca_fig.add_subplot(111)

        pivot = df.pivot_table(index=["Group"], columns="Metric",
                            values="Value", aggfunc=list).explode(selected_metrics).reset_index()
        X = pivot[selected_metrics].astype(float).values
        X = StandardScaler().fit_transform(X)
        groups = pivot["Group"].values

        group_labels = np.unique(groups)
        # Use same colors as boxplot if available
        if hasattr(self, "color_map"):
            colors = self.color_map
        else:
            palette = sns.color_palette("Set2", n_colors=len(group_labels))
            colors = {g: palette[i] for i, g in enumerate(group_labels)}

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        counters = {g: 0 for g in group_labels}
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        x_offset, y_offset = 0.015 * x_range, 0.015 * y_range

        # Scatter points and numbering
        for i, (x, y) in enumerate(X_pca):
            g = groups[i]
            counters[g] += 1
            ax.scatter(x, y, color=colors[g], s=70)
            ax.text(x + x_offset, y + y_offset, f"{g}-{counters[g]}", fontsize=10)

        # 95% confidence ellipses
        for grp in np.unique(groups):
            grp_points = X_pca[groups == grp]
            if len(grp_points) > 2:
                cov = np.cov(grp_points, rowvar=False)
                mean = grp_points.mean(axis=0)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals * 5.991)  # 95% CI (chi2, 2df)
                ellipse = mpatches.Ellipse(mean, width, height, angle=theta,
                                        edgecolor=colors[grp], facecolor='none', linestyle='--')
                ax.add_patch(ellipse)

        # Add biplot arrows (loadings)
        loadings = pca.components_.T
        vecs = loadings * np.sqrt(pca.explained_variance_)
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        axis_span = 0.5 * min(xmax - xmin, ymax - ymin)
        for i, metric in enumerate(selected_metrics):
            lx, ly = vecs[i, 0] * axis_span, vecs[i, 1] * axis_span
            ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))
            ax.text(lx * 1.08, ly * 1.08, metric, color="red", fontsize=8, ha="center")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        handles = [mpatches.Patch(color=colors[g], label=g) for g in group_labels]
        ax.legend(handles=handles, title="Group")

        self.pca_canvas.draw()

        if self.export_pca_checkbox.isChecked() and self.first_file_path:
            folder = getattr(self, "export_group_folder", os.path.dirname(self.first_file_path))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.pca_fig.savefig(os.path.join(folder, f"pca_at{timestamp}.png"))
            self.pca_fig.savefig(os.path.join(folder, f"pca_at{timestamp}.pdf"))


class DataSelectionWindow(QDialog):
    def __init__(self, parent=None, existing_entries=None, existing_norm=None):
        super().__init__(parent)
        self.setWindowTitle("Data Selection")
        self.resize(900, 600)
        self.parent = parent
        self.metric_entries = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.scroll_layout = QVBoxLayout(container)
        scroll.setWidget(container)
        layout_main = QVBoxLayout(self)
        layout_main.addWidget(scroll)

        self.metric_names = parent.metric_names
        group_count = parent.group_count_spin.value()
        self.groups = [f"G{i+1}" for i in range(group_count)]

        # --- Metric blocks ---
        for i, name in enumerate(self.metric_names):
            layout = QVBoxLayout()
            layout.addWidget(QLabel(f"Metric {i+1}: {name}"))
            entry = {"metric_num": i + 1, "metric_name": name,
                     "col_input": QLineEdit("3"),
                     "group_files": {}, "group_rows": {}}
            for g in self.groups:
                hl = QHBoxLayout()
                btn = QPushButton(f"Select CSV for {g}")
                lbl = QLabel("No file selected")
                btn.clicked.connect(lambda _, e=entry, g=g, l=lbl: self.load_csv(e, g, l))
                row_in = QLineEdit("")
                hl.addWidget(btn); hl.addWidget(lbl)
                hl.addWidget(QLabel("Rows:")); hl.addWidget(row_in)
                entry["group_files"][g] = None
                entry["group_rows"][g] = row_in
                layout.addLayout(hl)
            layout.addWidget(QLabel("Column:"))
            layout.addWidget(entry["col_input"])
            self.metric_entries.append(entry)
            self.scroll_layout.addLayout(layout)

        # --- Normalization block ---
        norm_layout = QVBoxLayout()
        norm_layout.addWidget(QLabel("Normalization Inputs"))
        self.norm_block = {"group_files": {}, "col_cell": QLineEdit("14"), "col_pol": QLineEdit("25")}
        for g in self.groups:
            hl = QHBoxLayout()
            btn = QPushButton(f"Select Normalization CSV for {g}")
            lbl = QLabel("No file selected")
            btn.clicked.connect(lambda _, gn=g, l=lbl: self.load_norm_csv(gn, l))
            row_in = QLineEdit("")
            hl.addWidget(btn); hl.addWidget(lbl)
            hl.addWidget(QLabel("Rows:")); hl.addWidget(row_in)
            self.norm_block["group_files"][g] = {"file": None, "label": lbl, "rows": row_in}
            norm_layout.addLayout(hl)
        form = QFormLayout()
        form.addRow(QLabel("Cell Density Column:"), self.norm_block["col_cell"])
        form.addRow(QLabel("Polarization Column:"), self.norm_block["col_pol"])
        norm_layout.addLayout(form)
        self.scroll_layout.addLayout(norm_layout)

        # --- If reopening, repopulate with previously saved selections ---
        if existing_entries:
            for old, new in zip(existing_entries, self.metric_entries):
                # Restore column input
                new["col_input"].setText(str(old.get("col_input", "3")))

                # Restore CSV file paths and row inputs per group
                for g in self.groups:
                    # Restore CSV file path
                    saved_path = old.get("group_files", {}).get(g)
                    if saved_path:
                        new["group_files"][g] = saved_path

                    # Restore row numbers
                    saved_rows = old.get("group_rows", {}).get(g)
                    if isinstance(saved_rows, str):
                        new["group_rows"][g].setText(saved_rows)

                # Now update the visible filename labels deterministically
                # Find all labels in this metric block by traversing layouts in order
                metric_layouts = new.get("metric_layouts", [])
                for g in self.groups:
                    saved_path = old.get("group_files", {}).get(g)
                    if not saved_path:
                        continue
                    filename = os.path.basename(saved_path)

                    # Go through every QLabel in this DataSelectionWindow
                    for lbl in self.findChildren(QLabel):
                        # Match by "Select CSV for Gx" label or "No file selected"
                        if f"{filename}" in lbl.text() or lbl.text() == "No file selected":
                            lbl.setText(filename)
                            break

        if existing_norm:
            self.norm_block["col_cell"].setText(existing_norm.get("col_cell", "14"))
            self.norm_block["col_pol"].setText(existing_norm.get("col_pol", "25"))
            for g in self.groups:
                gdata = existing_norm.get("group_files", {}).get(g, {})
                if gdata:
                    file_path = gdata.get("file")
                    rows = gdata.get("rows", "")
                    if file_path:
                        self.norm_block["group_files"][g]["file"] = file_path
                        self.norm_block["group_files"][g]["label"].setText(os.path.basename(file_path))
                    self.norm_block["group_files"][g]["rows"].setText(rows)

        # --- Log + Buttons ---
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.scroll_layout.addWidget(QLabel("Load Log:"))
        self.scroll_layout.addWidget(self.log_box)
        btns = QHBoxLayout()
        save_btn = QPushButton("Save Selections")
        save_btn.clicked.connect(self.save_and_close)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(save_btn); btns.addWidget(close_btn)
        self.scroll_layout.addLayout(btns)

    def load_csv(self, entry, g, label):
        path, _ = QFileDialog.getOpenFileName(self, f"Select CSV for {g}", "", "CSV Files (*.csv)")
        if not path: return
        entry["group_files"][g] = path
        label.setText(os.path.basename(path))
        self.log_box.append(f"{g} metric: {path} loaded.")

    def load_norm_csv(self, g, label):
        path, _ = QFileDialog.getOpenFileName(self, f"Select Normalization CSV for {g}", "", "CSV Files (*.csv)")
        if not path: return
        self.norm_block["group_files"][g]["file"] = path
        label.setText(os.path.basename(path))
        self.log_box.append(f"{g} normalization: {path} loaded.")

    def save_and_close(self):
        # Deep copy metric entries (store text values, not widget refs)
        saved_entries = []
        for entry in self.metric_entries:
            copied = {
                "metric_num": entry["metric_num"],
                "metric_name": entry["metric_name"],
                "col_input": entry["col_input"].text(),
                "group_files": {g: path for g, path in entry["group_files"].items()},
                "group_rows": {g: r.text() for g, r in entry["group_rows"].items()},
            }
            saved_entries.append(copied)

        # Copy normalization info safely
        norm_copy = {
            "col_cell": self.norm_block["col_cell"].text(),
            "col_pol": self.norm_block["col_pol"].text(),
            "group_files": {}
        }
        for g, info in self.norm_block["group_files"].items():
            norm_copy["group_files"][g] = {
                "file": info["file"],
                "rows": info["rows"].text()
            }

        self.parent.metric_entries = saved_entries
        self.parent.normalization_block = norm_copy
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataImporter()
    window.show()
    sys.exit(app.exec_())
