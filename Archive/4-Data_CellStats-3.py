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
    QMessageBox, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
import json

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

        # Custom group names
        name_row = QHBoxLayout()
        self.g1_name_input = QLineEdit("G1")
        self.g2_name_input = QLineEdit("G2")
        name_row.addWidget(QLabel("Group 1 name:"))
        name_row.addWidget(self.g1_name_input)
        name_row.addWidget(QLabel("Group 2 name:"))
        name_row.addWidget(self.g2_name_input)
        self.layout.addLayout(name_row)

        # Metric checkboxes for statistical analysis
        self.metric_names = ["Lac/Pyr Ratio", "Lac", "Lac/Pyr Ratio (fit)", "kPL (fit)"]
        self.metric_checkboxes = []
        self.layout.addWidget(QLabel("Select Metrics for Analysis:"))
        metric_box = QHBoxLayout()
        for name in self.metric_names:
            cb = QCheckBox(name)
            cb.setChecked(True)  # default: include all
            self.metric_checkboxes.append(cb)
            metric_box.addWidget(cb)
        self.layout.addLayout(metric_box)


        # Placeholder for metric entries, filled by DataSelectionWindow
        self.metric_entries = []
        self.final_df = None
        self.first_file_path = None

        # Normalization inputs with history
        self.history_file = os.path.join(os.path.dirname(__file__), "norm_history.json")
        self.norm_history = self.load_norm_history()

        self.layout.addWidget(QLabel("Normalization Factors (select from history or enter new values)"))
        self.norm_inputs = {"G1": {}, "G2": {}}
        for group in ["G1", "G2"]:
            form = QFormLayout()
            form.addRow(QLabel(f"Group {group} Normalization:"))

            for key, label in [("cell", "Final Cell Densities:"),
                               ("conc", "Final Sub Concentrations:"),
                               ("pol", "Final Sub Polarizations:")]:
                combo = QComboBox()
                combo.setEditable(True)
                for val in self.norm_history.get(group, {}).get(key, []):
                    combo.addItem(val)
                form.addRow(label, combo)
                self.norm_inputs[group][key] = combo

            self.layout.addLayout(form)

        # Export checkboxes
        self.export_df_checkbox = QCheckBox("Export Dataframe as CSV")
        self.export_stats_checkbox = QCheckBox("Export Statistics Results as CSV")
        self.export_box_checkbox = QCheckBox("Export Boxplots (PNG & PDF)")
        self.export_pca_checkbox = QCheckBox("Export PCA Scatter (PNG & PDF)")
        for cb in [self.export_df_checkbox, self.export_stats_checkbox, self.export_box_checkbox, self.export_pca_checkbox]:
            self.layout.addWidget(cb)

        # Data Selection Button
        self.data_select_button = QPushButton("Data Selection")
        self.data_select_button.clicked.connect(self.open_data_selection)
        self.layout.addWidget(self.data_select_button)

        # Buttons
        self.build_button = QPushButton("Build Dataframe")
        self.build_button.clicked.connect(self.build_dataframe)
        self.layout.addWidget(self.build_button)

        self.stats_button = QPushButton("Run Statistical Analysis")
        self.stats_button.clicked.connect(self.run_stats)
        self.layout.addWidget(self.stats_button)

        # Stats log
        self.stats_log = QTextEdit()
        self.stats_log.setReadOnly(True)
        self.layout.addWidget(QLabel("Statistics Log:"))
        self.layout.addWidget(self.stats_log)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        self.layout.addWidget(self.clear_log_button)

        # Tab widget for Dataframe + Plots
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.boxplot_fig, self.pca_fig = Figure(figsize=(5, 4)), Figure(figsize=(5, 4))
        self.boxplot_canvas, self.pca_canvas = FigureCanvas(self.boxplot_fig), FigureCanvas(self.pca_fig)
        self.tabs.addTab(self.boxplot_canvas, "Boxplots")
        self.tabs.addTab(self.pca_canvas, "PCA Scatter")

    # --- Normalization history ---
    def load_norm_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_norm_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.norm_history, f, indent=2)

    def update_norm_history(self):
        for group in ["G1", "G2"]:
            if group not in self.norm_history:
                self.norm_history[group] = {}
            for key in ["cell", "conc", "pol"]:
                combo: QComboBox = self.norm_inputs[group][key]
                val = combo.currentText().strip()
                if not val:
                    continue
                history = self.norm_history[group].get(key, [])
                if val in history:
                    history.remove(val)
                history.insert(0, val)
                self.norm_history[group][key] = history[:5]
        self.save_norm_history()

    def open_data_selection(self):
        dialog = DataSelectionWindow(self, existing_entries=self.metric_entries)
        dialog.exec_()

    # --- Data build + analysis ---
    def build_dataframe(self):
        if not self.metric_entries:
            QMessageBox.warning(self, "Missing Data", 
                                "Please open the Data Selection window and choose CSV files.")
            return

        self.update_norm_history()
        all_data = []
        names = self.get_group_names()
        self.group_names = names
        factors = {}

        # Collect normalization factors
        for group in ["G1", "G2"]:
            cells = self.parse_factors(self.norm_inputs[group]["cell"].currentText())
            concs = self.parse_factors(self.norm_inputs[group]["conc"].currentText())
            pols = self.parse_factors(self.norm_inputs[group]["pol"].currentText())
            factors[group] = {"cell": cells, "conc": concs, "pol": pols}

        self.first_file_path = None

        # Determine mode: same or different CSV
        same_mode = True
        if hasattr(self, "metric_entries") and self.metric_entries:
            same_mode = self.parent_same_csv_mode()

        for entry in self.metric_entries:
            metric_name = entry["metric_name"]
            metric_num = entry["metric_num"]
            col_idx = int(entry["col_input"].text()) - 1
            use_norm = entry["norm_checkbox"].isChecked()

            if same_mode:
                if not entry["g1_file"]:
                    QMessageBox.warning(self, "Missing CSV", 
                                        f"No CSV selected for {metric_name}.")
                    return
                df = pd.read_csv(entry["g1_file"])
                if metric_num == 1 and not self.first_file_path:
                    self.first_file_path = entry["g1_file"]

                series = df.iloc[:, col_idx]
                for group, row_input in [("G1", entry["g1_input"]), ("G2", entry["g2_input"])]:
                    rows = self.parse_row_range(row_input.text())
                    for i, r in enumerate(rows):
                        imported = series.iloc[r]
                        cell = factors[group]["cell"][i]
                        conc = factors[group]["conc"][i]
                        pol = factors[group]["pol"][i]
                        norm_val, status = self.normalize_value(metric_num, imported, cell, conc, pol, use_norm)
                        group_label = names[group]
                        all_data.append([group_label, metric_name, imported, cell, conc, pol, norm_val, status])

            else:  # Different CSV mode
                if not entry["g1_file"] or not entry["g2_file"]:
                    QMessageBox.warning(self, "Missing CSV", 
                                        f"Both G1 and G2 CSVs must be selected for {metric_name}.")
                    return

                # --- Group 1 ---
                df1 = pd.read_csv(entry["g1_file"])
                if metric_num == 1 and not self.first_file_path:
                    self.first_file_path = entry["g1_file"]
                series1 = df1.iloc[:, col_idx]
                rows1 = self.parse_row_range(entry["g1_input"].text())
                for i, r in enumerate(rows1):
                    imported = series1.iloc[r]
                    cell = factors["G1"]["cell"][i]
                    conc = factors["G1"]["conc"][i]
                    pol = factors["G1"]["pol"][i]
                    norm_val, status = self.normalize_value(metric_num, imported, cell, conc, pol, use_norm)
                    group_label = names["G1"]
                    all_data.append([group_label, metric_name, imported, cell, conc, pol, norm_val, status])

                # --- Group 2 ---
                df2 = pd.read_csv(entry["g2_file"])
                series2 = df2.iloc[:, col_idx]
                rows2 = self.parse_row_range(entry["g2_input"].text())
                for i, r in enumerate(rows2):
                    imported = series2.iloc[r]
                    cell = factors["G2"]["cell"][i]
                    conc = factors["G2"]["conc"][i]
                    pol = factors["G2"]["pol"][i]
                    norm_val, status = self.normalize_value(metric_num, imported, cell, conc, pol, use_norm)
                    group_label = names["G2"]
                    all_data.append([group_label, metric_name, imported, cell, conc, pol, norm_val, status])

        # Build dataframe
        self.final_df = pd.DataFrame(
            all_data, 
            columns=["Group", "Metric", "Imported", "CellDensity", "SubConc", "SubPol", "Value", "NormApplied"]
        )

        # Show dataframe in tab
        self.show_dataframe_tab()

        # Export dataframe if requested
        if self.export_df_checkbox.isChecked() and self.first_file_path:
            # --- Build experiment/date-based subfolder ---
            folder = getattr(self, "export_group_folder", os.path.dirname(self.first_file_path))
            names = self.get_group_names()
            g1_name, g2_name = names["G1"], names["G2"]

            # Extract expdates from any loaded CSV paths (proc/fit summaries)
            g1_files = [e["g1_file"] for e in self.metric_entries if e.get("g1_file")]
            g2_files = [e["g2_file"] for e in self.metric_entries if e.get("g2_file")]
            expdate1 = extract_expdate_from_summary(os.path.basename(g1_files[0])) if g1_files else None
            expdate2 = extract_expdate_from_summary(os.path.basename(g2_files[0])) if g2_files else None

            part1 = f"{expdate1}-{g1_name}" if expdate1 else g1_name
            part2 = f"{expdate2}-{g2_name}" if expdate2 else g2_name
            group_folder = os.path.join(folder, f"{part1}_{part2}")
            os.makedirs(group_folder, exist_ok=True)
            self.export_group_folder = group_folder  # store for other exports

            # --- Export the dataframe ---
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_path = os.path.join(group_folder, f"dataframe_at{timestamp}.csv")
            self.final_df.to_csv(export_path, index=False)


    def parent_same_csv_mode(self):
        if self.metric_entries and "g1_file" in self.metric_entries[0]:
            return all(e["g1_file"] == e["g2_file"] for e in self.metric_entries if e["g1_file"])
        return True

    def normalize_value(self, metric_num, imported, cell, conc, pol, use_norm):
        if use_norm:
            if metric_num in (1, 3):
                return imported / (cell / 1e6), "normalized"
            elif metric_num == 2:
                return imported / (pol * (cell / 1e6)), "normalized"
            elif metric_num == 4:
                return imported / (cell / 1e6), "normalized"
        else:
            if metric_num == 2:
                return imported / 1000.0, "raw_scaled"
            else:
                return imported, "raw"

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
            return
        selected_metrics = [cb.text() for cb in self.parent_checkboxes() if cb.isChecked()]
        if not selected_metrics:
            return

        df = self.final_df[self.final_df["Metric"].isin(selected_metrics)]
        names = self.get_group_names()
        g1_label, g2_label = names["G1"], names["G2"]

        pvals, results = [], []
        for metric in selected_metrics:
            g1 = df[(df["Group"] == g1_label) & (df["Metric"] == metric)]["Value"]
            g2 = df[(df["Group"] == g2_label) & (df["Metric"] == metric)]["Value"]
            if len(g1) == 0 or len(g2) == 0:
                continue
            stat, p = ttest_ind(g1, g2, equal_var=False)
            status = df[df["Metric"] == metric]["NormApplied"].iloc[0]
            pvals.append(p)
            results.append((metric, p, status))

        if not pvals:
            return

        reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"\n=== Stats Run at {timestamp} ===\n"
        for (metric, p, status), p_corr, rej in zip(results, pvals_corr, reject):
            log_text += f"{metric} ({status}): p={p:.4f}, corrected p={p_corr:.4f}, significant={rej}\n"
        self.stats_log.append(log_text)

        if self.export_stats_checkbox.isChecked() and self.first_file_path:
            folder = getattr(self, "export_group_folder", os.path.dirname(self.first_file_path))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_path = os.path.join(folder, f"stats_at{timestamp}.csv")
            with open(export_path, "w") as f:
                f.write("Metric,p_value,corrected_p,significant,NormApplied\n")
                for (metric, p, status), p_corr, rej in zip(results, pvals_corr, reject):
                    f.write(f"{metric},{p},{p_corr},{rej},{status}\n")

        p_corr_map = {metric: p_corr for (metric, _p, _s), p_corr in zip(results, pvals_corr)}
        self.draw_boxplots(df, selected_metrics, p_corr_map)
        self.draw_pca(df, selected_metrics)

    def parent_checkboxes(self):
        if hasattr(self, "metric_checkboxes"):
            return self.metric_checkboxes
        return []

    def draw_boxplots(self, df, selected_metrics, p_corr_map=None):
        if p_corr_map is None:
            p_corr_map = {}
        self.boxplot_fig.clear()
        n_metrics = len(selected_metrics)
        axes = self.boxplot_fig.subplots(1, n_metrics, squeeze=False)[0]

        for i, metric in enumerate(selected_metrics):
            ax = axes[i]
            sub_df = df[df["Metric"] == metric]
            sns.boxplot(x="Group", y="Value", data=sub_df, ax=ax, palette="Set2")
            sns.stripplot(
                x="Group", y="Value", data=sub_df,
                jitter=True, marker="o", edgecolor="gray", linewidth=0.5,
                color="black", alpha=0.6, ax=ax
            )
            ax.set_title(metric)
            ax.set_xlabel("Group")
            ax.set_ylabel("Value")
            vals = sub_df["Value"].to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            ymax = 1.05 * float(np.nanmax(finite_vals)) if finite_vals.size > 0 else 1.0
            ax.set_ylim(0, ymax)
            p_corr = p_corr_map.get(metric, np.nan)
            label = f"corrected p = {p_corr:.3g}" if np.isfinite(p_corr) else "corrected p = N/A"
            ax.text(0.5, 0.95, label, transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

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

        pivot = df.pivot_table(index=["Group"], columns="Metric", values="Value", aggfunc=list).explode(selected_metrics).reset_index()
        X = pivot[selected_metrics].astype(float).values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        groups = pivot["Group"].values

        group_labels = np.unique(groups)
        palette = sns.color_palette("Set2", n_colors=len(group_labels))
        colors = {g: palette[i] for i, g in enumerate(group_labels)}

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        counters = {g: 0 for g in group_labels}
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        x_offset, y_offset = 0.015 * x_range, 0.015 * y_range

        for i, (x, y) in enumerate(X_pca):
            g = groups[i]
            counters[g] += 1
            ax.scatter(x, y, color=colors.get(g, "gray"), s=70)
            ax.text(x + x_offset, y + y_offset, f"{g}-{counters[g]}", fontsize=10)

        for grp in np.unique(groups):
            grp_points = X_pca[groups == grp]
            if len(grp_points) > 2:
                cov = np.cov(grp_points, rowvar=False)
                mean = grp_points.mean(axis=0)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals * 5.991)
                ellipse = mpatches.Ellipse(mean, width, height, angle=theta,
                                           edgecolor=colors.get(grp, "gray"),
                                           facecolor='none', linestyle='--')
                ax.add_patch(ellipse)

        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        ax.set_xlim(min(xmin, 0), max(xmax, 0))
        ax.set_ylim(min(ymin, 0), max(ymax, 0))
        loadings = pca.components_.T
        vecs = loadings * np.sqrt(pca.explained_variance_)
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

    def parse_row_range(self, text):
        rows = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-"))
                rows.extend(range(start - 1, end))
            else:
                rows.append(int(part) - 1)
        return rows

    def parse_factors(self, text):
        tokens = re.split(r'[,\s]+', text.strip())
        vals = [float(x) for x in tokens if x]
        return vals if vals else [1.0, 1.0, 1.0]

    def get_group_names(self):
        g1 = (self.g1_name_input.text() or "G1").strip()
        g2 = (self.g2_name_input.text() or "G2").strip()
        return {"G1": g1, "G2": g2}

    def clear_log(self):
        self.stats_log.clear()


class DataSelectionWindow(QDialog):
    def __init__(self, parent=None, existing_entries=None):
        super().__init__(parent)
        self.setWindowTitle("Data Selection")
        self.resize(800, 400)

        self.parent = parent
        self.metric_entries = []
        self.existing_entries = existing_entries

        outer_layout = QVBoxLayout(self)

        self.same_csv_checkbox = QCheckBox("Group 1 and Group 2 use the same CSV")
        self.same_csv_checkbox.setChecked(True)
        self.same_csv_checkbox.stateChanged.connect(self.update_metric_blocks)
        outer_layout.addWidget(self.same_csv_checkbox)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        self.metric_names = ["Lac/Pyr Ratio", "Lac", "Lac/Pyr Ratio (fit)", "kPL (fit)"]
        self.default_columns = [2, 3, 12, 3]
        self.default_g1_rows = "1-3"
        self.default_g2_rows = "4-6"

        for i, name in enumerate(self.metric_names):
            block = self.create_metric_block(i + 1, name, str(self.default_columns[i]))
            self.scroll_layout.addLayout(block)

        if self.existing_entries:
            for old, new in zip(self.existing_entries, self.metric_entries):
                for k in ["g1_file", "g2_file"]:
                    new[k] = old[k]
                    if new[k]:
                        lbl = new["g1_label"] if k == "g1_file" else new["g2_label"]
                        lbl.setText(os.path.basename(new[k]))
                new["col_input"].setText(old["col_input"].text())
                new["g1_input"].setText(old["g1_input"].text())
                new["g2_input"].setText(old["g2_input"].text())
                new["norm_checkbox"].setChecked(old["norm_checkbox"].isChecked())

        # File load log
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("CSV load history will appear here...")
        outer_layout.addWidget(QLabel("CSV Load Log:"))
        outer_layout.addWidget(self.log_box)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Selections")
        save_btn.clicked.connect(self.save_and_close)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        outer_layout.addLayout(btn_row)

    def create_metric_block(self, metric_num, metric_name, default_col):
        layout = QHBoxLayout()
        entry = {
            "metric_num": metric_num,
            "metric_name": metric_name,
            "col_input": QLineEdit(default_col),
            "g1_input": QLineEdit(self.default_g1_rows),
            "g2_input": QLineEdit(self.default_g2_rows),
            "g1_file": None,
            "g2_file": None,
            "g1_label": QLabel("No file selected"),
            "g2_label": QLabel("No file selected"),
            "norm_checkbox": QCheckBox("Apply Normalization")
        }
        entry["norm_checkbox"].setChecked(True)

        entry["same_btn"] = QPushButton(f"Select CSV for {metric_name}")
        entry["same_btn"].clicked.connect(lambda _, e=entry: self.load_csv(e, "both"))

        entry["g1_btn"] = QPushButton(f"Select G1 CSV for {metric_name}")
        entry["g1_btn"].clicked.connect(lambda _, e=entry: self.load_csv(e, "g1"))

        entry["g2_btn"] = QPushButton(f"Select G2 CSV for {metric_name}")
        entry["g2_btn"].clicked.connect(lambda _, e=entry: self.load_csv(e, "g2"))

        layout.addWidget(entry["same_btn"])
        layout.addWidget(entry["g1_btn"])
        layout.addWidget(entry["g1_label"])
        layout.addWidget(entry["g2_btn"])
        layout.addWidget(entry["g2_label"])
        layout.addWidget(QLabel("Column:"))
        layout.addWidget(entry["col_input"])
        layout.addWidget(QLabel("G1 Rows:"))
        layout.addWidget(entry["g1_input"])
        layout.addWidget(QLabel("G2 Rows:"))
        layout.addWidget(entry["g2_input"])
        layout.addWidget(entry["norm_checkbox"])

        self.metric_entries.append(entry)
        return layout

    def update_metric_blocks(self):
        same_csv = self.same_csv_checkbox.isChecked()
        for entry in self.metric_entries:
            entry["same_btn"].setVisible(same_csv)
            entry["g1_btn"].setVisible(not same_csv)
            entry["g2_btn"].setVisible(not same_csv)
            entry["g1_label"].setVisible(not same_csv)
            entry["g2_label"].setVisible(not same_csv)

    def load_csv(self, entry, target):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        filename = os.path.basename(file_path)
        if target == "both":
            entry["g1_file"] = file_path
            entry["g2_file"] = file_path
            entry["g1_label"].setText(filename)
            entry["g2_label"].setText(filename)
        elif target == "g1":
            entry["g1_file"] = file_path
            entry["g1_label"].setText(filename)
        elif target == "g2":
            entry["g2_file"] = file_path
            entry["g2_label"].setText(filename)

        # Append to log box inside the Data Selection window
        self.log_box.append(f"{file_path} loaded.")

    def save_and_close(self):
        self.parent.metric_entries = self.metric_entries
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataImporter()
    window.show()
    sys.exit(app.exec_())
