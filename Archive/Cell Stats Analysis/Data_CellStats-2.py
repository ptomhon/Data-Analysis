import sys
import pandas as pd
import os
import re
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLineEdit, QLabel, QTableWidget, QTableWidgetItem, QHBoxLayout,
    QFormLayout, QCheckBox, QTextEdit, QTabWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


class DataImporter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Group Comparison Data Importer & Analysis")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Custom group names
        name_row = QHBoxLayout()
        self.g1_name_input = QLineEdit()
        self.g1_name_input.setText("G1")
        self.g2_name_input = QLineEdit()
        self.g2_name_input.setText("G2")
        name_row.addWidget(QLabel("Group 1 name:"))
        name_row.addWidget(self.g1_name_input)
        name_row.addWidget(QLabel("Group 2 name:"))
        name_row.addWidget(self.g2_name_input)
        self.layout.addLayout(name_row)

        self.metric_entries = []
        self.dataframes = []
        self.first_file_path = None
        self.final_df = None

        # Metric defaults (no "Norm")
        self.metric_names = [
            "Lac/Pyr Ratio",
            "Lac",
            "Lac/Pyr Ratio (fit)",
            "kPL (fit)"
        ]
        self.default_columns = [2, 3, 12, 3]  # 1-indexed
        self.default_g1_rows = "1-3"
        self.default_g2_rows = "4-6"

        # Metric checkboxes for analysis
        self.metric_checkboxes = []
        self.layout.addWidget(QLabel("Select Metrics for Analysis:"))
        metric_box = QHBoxLayout()
        for name in self.metric_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.metric_checkboxes.append(cb)
            metric_box.addWidget(cb)
        self.layout.addLayout(metric_box)

        # Add 4 metric blocks
        for i in range(4):
            block = self.create_metric_block(i+1, self.metric_names[i], str(self.default_columns[i]))
            self.layout.addLayout(block)

        # Normalization inputs
        self.layout.addWidget(QLabel("Normalization Factors (whitespace, comma, tab, or newline separated)"))
        self.norm_inputs = {"G1": {}, "G2": {}}
        for group in ["G1", "G2"]:
            form = QFormLayout()
            form.addRow(QLabel(f"Group {group} Normalization:"))

            cell_input = QLineEdit()
            conc_input = QLineEdit()
            pol_input = QLineEdit()

            form.addRow("Final Cell Densities:", cell_input)
            form.addRow("Final Sub Concentrations:", conc_input)
            form.addRow("Final Sub Polarizations:", pol_input)

            self.layout.addLayout(form)
            self.norm_inputs[group]["cell"] = cell_input
            self.norm_inputs[group]["conc"] = conc_input
            self.norm_inputs[group]["pol"] = pol_input

        # Export checkboxes
        self.export_df_checkbox = QCheckBox("Export Dataframe as CSV")
        self.export_stats_checkbox = QCheckBox("Export Statistics Results as CSV")
        self.export_box_checkbox = QCheckBox("Export Boxplots (PNG & PDF)")
        self.export_pca_checkbox = QCheckBox("Export PCA Scatter (PNG & PDF)")
        self.layout.addWidget(self.export_df_checkbox)
        self.layout.addWidget(self.export_stats_checkbox)
        self.layout.addWidget(self.export_box_checkbox)
        self.layout.addWidget(self.export_pca_checkbox)

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

        # Clear log button
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        self.layout.addWidget(self.clear_log_button)

        # Tab widget for Dataframe + Plots
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Add Boxplot and PCA tabs (empty initially)
        self.boxplot_fig = Figure(figsize=(5, 4))
        self.boxplot_canvas = FigureCanvas(self.boxplot_fig)
        self.tabs.addTab(self.boxplot_canvas, "Boxplots")

        self.pca_fig = Figure(figsize=(5, 4))
        self.pca_canvas = FigureCanvas(self.pca_fig)
        self.tabs.addTab(self.pca_canvas, "PCA Scatter")

    def create_metric_block(self, metric_num, default_name, default_col):
        layout = QHBoxLayout()
        btn = QPushButton(f"Select CSV for {default_name}")
        label = QLabel("No file selected")
        btn.clicked.connect(lambda _, n=metric_num, lbl=label: self.load_csv(n, lbl))
        layout.addWidget(btn)
        layout.addWidget(label)

        col_input = QLineEdit()
        col_input.setText(default_col)
        layout.addWidget(QLabel("Column:"))
        layout.addWidget(col_input)

        g1_input = QLineEdit()
        g1_input.setText(self.default_g1_rows)
        layout.addWidget(QLabel("G1 Rows:"))
        layout.addWidget(g1_input)

        g2_input = QLineEdit()
        g2_input.setText(self.default_g2_rows)
        layout.addWidget(QLabel("G2 Rows:"))
        layout.addWidget(g2_input)

        norm_cb = QCheckBox("Apply Normalization")
        norm_cb.setChecked(True)
        layout.addWidget(norm_cb)

        self.metric_entries.append({
            "file": None,
            "file_label": label,
            "col_input": col_input,
            "g1_input": g1_input,
            "g2_input": g2_input,
            "metric_name": default_name,
            "metric_num": metric_num,
            "norm_checkbox": norm_cb
        })

        return layout

    def load_csv(self, metric_num, label_widget):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.metric_entries[metric_num-1]["file"] = file_path
            filename = os.path.basename(file_path)
            label_widget.setText(filename)

    def parse_row_range(self, text):
        rows = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-"))
                rows.extend(range(start-1, end))
            else:
                rows.append(int(part)-1)
        return rows

    def parse_factors(self, text): 
        tokens = re.split(r'[,\s]+', text.strip())
        vals = [float(x) for x in tokens if x]
        if not vals:   # if user left input blank
            vals = [1.0, 1.0, 1.0]  # default to 3 values of 1.0
        return vals
    
    def get_group_names(self):
        g1 = (self.g1_name_input.text() or "G1").strip()
        g2 = (self.g2_name_input.text() or "G2").strip()
        return {"G1": g1, "G2": g2}

    def apply_normalization(self, metric_num, imported, cell, conc, pol):
        if metric_num == 1:  # Lac/Pyr Ratio
            return (imported) / (cell / 1e6)
        elif metric_num == 2:  # Lac
            return imported / (pol * (cell / 1e6))
        elif metric_num == 3:  # Lac/Pyr Ratio (fit)
            return (imported) / (cell / 1e6)
        elif metric_num == 4:  # kPL (fit)
            return imported / (cell / 1e6)
        else:
            return imported

    def build_dataframe(self):
        all_data = []
        names = self.get_group_names()
        self.group_names = names  # keep for other methods
        factors = {}
        for group in ["G1", "G2"]:
            cells = self.parse_factors(self.norm_inputs[group]["cell"].text())
            concs = self.parse_factors(self.norm_inputs[group]["conc"].text())
            pols = self.parse_factors(self.norm_inputs[group]["pol"].text())
            factors[group] = {"cell": cells, "conc": concs, "pol": pols}

        self.first_file_path = None
        for entry in self.metric_entries:
            if not entry["file"]:
                continue
            if entry["metric_num"] == 1 and not self.first_file_path:
                self.first_file_path = entry["file"]
            df = pd.read_csv(entry["file"])
            col_idx = int(entry["col_input"].text()) - 1
            series = df.iloc[:, col_idx]
            metric_name = entry["metric_name"]
            metric_num = entry["metric_num"]
            use_norm = entry["norm_checkbox"].isChecked()
            for group, row_input in [("G1", entry["g1_input"]), ("G2", entry["g2_input"])]:
                rows = self.parse_row_range(row_input.text())
                for i, r in enumerate(rows):
                    imported = series.iloc[r]
                    cell = factors[group]["cell"][i]
                    conc = factors[group]["conc"][i]
                    pol = factors[group]["pol"][i]
                    if use_norm:
                        norm_val = self.apply_normalization(metric_num, imported, cell, conc, pol)
                        status = "normalized"
                    else:
                        if metric_num == 2:  # Lac special case
                            norm_val = imported / 1000.0
                            status = "raw_scaled"
                        else:
                            norm_val = imported
                            status = "raw"
                    group_label = names[group]  # map "G1"/"G2" -> custom label
                    all_data.append([group_label, metric_name, imported, cell, conc, pol, norm_val, status])

        self.final_df = pd.DataFrame(all_data, columns=["Group", "Metric", "Imported", "CellDensity", "SubConc", "SubPol", "Value", "NormApplied"])
        self.dataframes.append(self.final_df)

        # Create or refresh Dataframe tab
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

        if self.export_df_checkbox.isChecked() and self.first_file_path:
            folder = os.path.dirname(self.first_file_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_path = os.path.join(folder, f"dataframe_at{timestamp}.csv")
            self.final_df.to_csv(export_path, index=False)

    def run_stats(self):
        if self.final_df is None:
            return
        selected_metrics = [cb.text() for cb in self.metric_checkboxes if cb.isChecked()]
        if not selected_metrics:
            return

        df = self.final_df[self.final_df["Metric"].isin(selected_metrics)]

        # Use custom group labels for selections
        names = self.get_group_names()
        g1_label = names["G1"]
        g2_label = names["G2"]

        # Univariate t-tests
        pvals = []
        results = []
        for metric in selected_metrics:
            g1 = df[(df["Group"] == g1_label) & (df["Metric"] == metric)]["Value"]
            g2 = df[(df["Group"] == g2_label) & (df["Metric"] == metric)]["Value"]
            stat, p = ttest_ind(g1, g2, equal_var=False)
            status = df[df["Metric"] == metric]["NormApplied"].iloc[0]
            pvals.append(p)
            results.append((metric, p, status))

        reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

        # Log results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"\n=== Stats Run at {timestamp} ===\n"
        for (metric, p, status), p_corr, rej in zip(results, pvals_corr, reject):
            log_text += f"{metric} ({status}): p={p:.4f}, corrected p={p_corr:.4f}, significant={rej}\n"
        self.stats_log.append(log_text)

        # Export stats if requested
        if self.export_stats_checkbox.isChecked() and self.first_file_path:
            folder = os.path.dirname(self.first_file_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_path = os.path.join(folder, f"stats_at{timestamp}.csv")
            with open(export_path, "w") as f:
                f.write("Metric,p_value,corrected_p,significant,NormApplied\n")
                for (metric, p, status), p_corr, rej in zip(results, pvals_corr, reject):
                    f.write(f"{metric},{p},{p_corr},{rej},{status}\n")

        # Build a dict for annotation: metric -> corrected p
        p_corr_map = {metric: p_corr for (metric, _p, _s), p_corr in zip(results, pvals_corr)}

        # Pass corrected p map to the boxplots
        self.draw_boxplots(df, selected_metrics, p_corr_map)
        self.draw_pca(df, selected_metrics)


    def draw_boxplots(self, df, selected_metrics, p_corr_map=None):
        if p_corr_map is None:
            p_corr_map = {}
        self.boxplot_fig.clear()
        n_metrics = len(selected_metrics)
        axes = self.boxplot_fig.subplots(1, n_metrics, squeeze=False)[0]  # row of subplots

        for i, metric in enumerate(selected_metrics):
            ax = axes[i]
            sub_df = df[df["Metric"] == metric]

            # Draw per-metric box/strip
            sns.boxplot(x="Group", y="Value", data=sub_df, ax=ax, palette="Set2")
            sns.stripplot(
                x="Group", y="Value", data=sub_df,
                jitter=True, marker="o", edgecolor="gray", linewidth=0.5,
                color="black", alpha=0.6, ax=ax
            )

            # Titles/labels
            ax.set_title(metric)
            ax.set_xlabel("Group")
            ax.set_ylabel("Value")

            # Y-limits: start at 0, set a sensible max per subplot
            vals = sub_df["Value"].to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size > 0:
                vmax = float(np.nanmax(finite_vals))
                ymax = 1.05 * vmax if vmax > 0 else 1.0
            else:
                ymax = 1.0
            ax.set_ylim(0, ymax)

            # Annotate corrected p (FDR-BH)
            p_corr = p_corr_map.get(metric, np.nan)
            label = f"corrected p = {p_corr:.3g}" if np.isfinite(p_corr) else "corrected p = N/A"
            ax.text(
                0.5, 0.95, label,
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
            )

        self.boxplot_fig.tight_layout()
        self.boxplot_canvas.draw()

        if self.export_box_checkbox.isChecked() and self.first_file_path:
            folder = os.path.dirname(self.first_file_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.boxplot_fig.savefig(os.path.join(folder, f"boxplot_at{timestamp}.png"))
            self.boxplot_fig.savefig(os.path.join(folder, f"boxplot_at{timestamp}.pdf"))



    def draw_pca(self, df, selected_metrics):
        self.pca_fig.clear()
        ax = self.pca_fig.add_subplot(111)

        # Prepare data (unchanged lines around pivot)
        pivot = df.pivot_table(index=["Group"], columns="Metric", values="Value", aggfunc=list).explode(selected_metrics).reset_index()
        X = pivot[selected_metrics].astype(float).values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        groups = pivot["Group"].values

        # Dynamic colors for however the groups are named
        group_labels = np.unique(groups)
        palette = sns.color_palette("Set2", n_colors=len(group_labels))
        colors = {g: palette[i] for i, g in enumerate(group_labels)}

        # Track counters for repeats
        group_counters = {g: 0 for g in group_labels}

        # PCA (unchanged)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Axis ranges for offsets (unchanged)
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        x_offset = 0.015 * x_range
        y_offset = 0.015 * y_range

        # Scatter with labels like "PAR-1", "CTRL-2" (using custom names)
        for i, (x, y) in enumerate(X_pca):
            g = groups[i]
            group_counters[g] += 1
            ax.scatter(x, y, color=colors.get(g, "gray"), s=70)
            ax.text(x + x_offset, y + y_offset, f"{g}-{group_counters[g]}",
                    fontsize=12, ha="left", va="bottom")


        # 95% confidence ellipses with relative scaling
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()

        for grp in np.unique(groups):
            grp_points = X_pca[groups == grp]
            if len(grp_points) > 2:
                cov = np.cov(grp_points, rowvar=False)
                mean = grp_points.mean(axis=0)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals * 5.991)  # 95% CI scaling

                # # Ensure ellipse is visible relative to data spread
                # min_width = 0.05 * x_range if x_range > 0 else 0.1
                # min_height = 0.05 * y_range if y_range > 0 else 0.1
                # width = max(width, min_width)
                # height = max(height, min_height)

                ellipse = mpatches.Ellipse(mean, width, height, angle=theta,
                                           edgecolor=colors.get(grp, "gray"),
                                           facecolor='none', linestyle='--', zorder=2)
                ax.add_patch(ellipse)

        # --- Robust biplot arrows ---
        # 1) Ensure the origin is inside the view (arrow base is (0,0))
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        ax.set_xlim(min(xmin, 0), max(xmax, 0))
        ax.set_ylim(min(ymin, 0), max(ymax, 0))
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()

        # 2) Classic biplot scaling: loadings * sqrt(explained_variance)
        loadings = pca.components_.T                         # shape: (n_vars, 2)
        vecs = loadings * np.sqrt(pca.explained_variance_)   # in score units

        # 3) Arrow length as a fraction of current axis span
        axis_span = 1.0 * min(xmax - xmin, ymax - ymin)     # adjust 0.35 if you want longer/shorter

        for i, metric in enumerate(selected_metrics):
            lx = vecs[i, 0] * axis_span
            ly = vecs[i, 1] * axis_span

            ax.annotate(
                "", xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="red"),
                zorder=5
            )
            ax.text(lx * 1.08, ly * 1.08, metric,
                    color="red", fontsize=8, ha="center", va="center", zorder=5)

        # Axis labels with explained variance
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")

        # Legend
        handles = [mpatches.Patch(color=colors[g], label=g) for g in group_labels]
        ax.legend(handles=handles, title="Group")

        self.pca_canvas.draw()

        # Export if requested
        if self.export_pca_checkbox.isChecked() and self.first_file_path:
            folder = os.path.dirname(self.first_file_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.pca_fig.savefig(os.path.join(folder, f"pca_at{timestamp}.png"))
            self.pca_fig.savefig(os.path.join(folder, f"pca_at{timestamp}.pdf"))



    def clear_log(self):
        self.stats_log.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataImporter()
    window.show()
    sys.exit(app.exec_())
