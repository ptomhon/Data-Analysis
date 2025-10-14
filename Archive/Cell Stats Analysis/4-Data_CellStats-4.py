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

        # --- Normalization layout (must exist before build_group_inputs) ---
        self.layout.addWidget(QLabel("Normalization Factors (per group)"))
        self.norm_layout = QVBoxLayout()
        self.layout.addLayout(self.norm_layout)

        # Initialize normalization input dictionary *before* first build
        self.norm_inputs = {}

        # Initialize groups and norm inputs
        self.build_group_inputs()

        # --- Metric selection ---
        self.metric_names = ["Lac", "kPL (fit)"]
        self.metric_checkboxes = []
        self.layout.addWidget(QLabel("Select Metrics for Analysis:"))
        metric_box = QHBoxLayout()
        for name in self.metric_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.metric_checkboxes.append(cb)
            metric_box.addWidget(cb)
        self.layout.addLayout(metric_box)

        # --- Normalization history setup ---
        self.history_file = os.path.join(os.path.dirname(__file__), "norm_history.json")
        self.norm_history = self.load_norm_history()


        # --- Export checkboxes ---
        self.export_df_checkbox = QCheckBox("Export Dataframe as CSV")
        self.export_stats_checkbox = QCheckBox("Export Statistics Results as CSV")
        self.export_box_checkbox = QCheckBox("Export Boxplots (PNG & PDF)")
        self.export_pca_checkbox = QCheckBox("Export PCA Scatter (PNG & PDF)")
        for cb in [self.export_df_checkbox, self.export_stats_checkbox, self.export_box_checkbox, self.export_pca_checkbox]:
            self.layout.addWidget(cb)

        # --- Data selection + buttons ---
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
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        self.layout.addWidget(self.clear_log_button)

        # --- Tabbed output for data and plots ---
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.boxplot_fig, self.pca_fig = Figure(figsize=(5, 4)), Figure(figsize=(5, 4))
        self.boxplot_canvas, self.pca_canvas = FigureCanvas(self.boxplot_fig), FigureCanvas(self.pca_fig)
        self.tabs.addTab(self.boxplot_canvas, "Boxplots")
        self.tabs.addTab(self.pca_canvas, "PCA Scatter")

        # Store metric entries
        self.metric_entries = []
        self.final_df = None
        self.first_file_path = None


    def build_group_inputs(self):
        # clear
        for i in reversed(range(self.group_name_layout.count())):
            item = self.group_name_layout.itemAt(i).widget()
            if item:
                item.setParent(None)
        self.group_name_inputs.clear()

        # rebuild
        n = self.group_count_spin.value()
        for i in range(n):
            name_edit = QLineEdit(f"G{i+1}")
            self.group_name_inputs[i] = name_edit
            self.group_name_layout.addWidget(QLabel(f"Group {i+1} name:"))
            self.group_name_layout.addWidget(name_edit)

        self.build_norm_inputs()

    def build_norm_inputs(self):
        # clear
        for i in reversed(range(self.norm_layout.count())):
            item = self.norm_layout.itemAt(i)
            if item is not None and item.layout() is not None:
                while item.layout().count():
                    w = item.layout().takeAt(0).widget()
                    if w:
                        w.setParent(None)
                item.layout().setParent(None)

        self.norm_inputs.clear()
        n = self.group_count_spin.value()

        for i in range(n):
            gname = f"G{i+1}"
            form = QFormLayout()
            form.addRow(QLabel(f"Group {gname} normalization inputs:"))

            # Normalization CSV
            norm_csv_btn = QPushButton("Select Normalization CSV")
            norm_csv_lbl = QLabel("No file selected")
            norm_csv_btn.clicked.connect(lambda _, lbl=norm_csv_lbl, name=gname: self.load_norm_csv(lbl, name))
            form.addRow(norm_csv_btn, norm_csv_lbl)

            # Default column/row inputs
            col_cell = QLineEdit("14")
            col_pol = QLineEdit("25")
            row_input = QLineEdit("")
            form.addRow(QLabel("Cell Density Column:"), col_cell)
            form.addRow(QLabel("Polarization Column:"), col_pol)
            form.addRow(QLabel("Rows (comma/range):"), row_input)

            self.norm_inputs[gname] = {
                "csv": None,
                "csv_label": norm_csv_lbl,
                "col_cell": col_cell,
                "col_pol": col_pol,
                "rows": row_input,
            }
            self.norm_layout.addLayout(form)

    def load_norm_csv(self, label_widget, group_name):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Normalization CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        label_widget.setText(os.path.basename(file_path))
        self.norm_inputs[group_name]["csv"] = file_path

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

    def open_data_selection(self):
        dialog = DataSelectionWindow(self)
        dialog.exec_()

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
        return vals if vals else [1.0]

    def build_dataframe(self):
        if not self.metric_entries:
            QMessageBox.warning(self, "Missing Data", "Please open the Data Selection window and choose CSV files.")
            return

        all_data = []
        group_names = {f"G{i+1}": self.group_name_inputs[i].text() or f"G{i+1}" for i in range(self.group_count_spin.value())}
        self.first_file_path = None

        # --- Gather normalization data for each group ---
        norm_data = {}
        for gname, ninfo in self.norm_inputs.items():
            csv_path = ninfo["csv"]
            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(self, "Missing Normalization CSV", f"No normalization CSV selected for {gname}.")
                return
            df = pd.read_csv(csv_path)
            rows = self.parse_row_range(ninfo["rows"].text())
            try:
                cell_vals = df.iloc[rows, int(ninfo["col_cell"].text()) - 1].astype(float).to_numpy()
                pol_vals = df.iloc[rows, int(ninfo["col_pol"].text()) - 1].astype(float).to_numpy()
            except Exception as e:
                QMessageBox.warning(self, "Error Reading CSV", f"Error reading normalization for {gname}: {e}")
                return
            norm_data[gname] = {"cell": cell_vals, "pol": pol_vals}

        # --- Import each metric × group ---
        for entry in self.metric_entries:
            metric_name = entry["metric_name"]
            metric_num = entry["metric_num"]
            col_idx = int(entry["col_input"].text()) - 1
            for gname, gfile in entry["group_files"].items():
                if not gfile:
                    QMessageBox.warning(self, "Missing CSV", f"No CSV selected for {metric_name} in {gname}.")
                    return
                df = pd.read_csv(gfile)
                if self.first_file_path is None:
                    self.first_file_path = gfile
                rows = self.parse_row_range(entry["group_rows"][gname].text())
                series = df.iloc[:, col_idx]
                for i, r in enumerate(rows):
                    imported = series.iloc[r]
                    cell = norm_data[gname]["cell"][i] if i < len(norm_data[gname]["cell"]) else 1.0
                    pol = norm_data[gname]["pol"][i] if i < len(norm_data[gname]["pol"]) else 1.0
                    # Normalize by cell density × polarization
                    norm_val = imported / (cell * pol / 1e6)
                    all_data.append([group_names[gname], metric_name, imported, cell, pol, norm_val])

        # --- Build dataframe ---
        self.final_df = pd.DataFrame(
            all_data,
            columns=["Group", "Metric", "Imported", "CellDensity", "SubPol", "Value"]
        )
        self.show_dataframe_tab()

        # --- Optional export ---
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
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_path = os.path.join(export_folder, f"dataframe_at{timestamp}.csv")
            self.final_df.to_csv(export_path, index=False)

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
        selected_metrics = [cb.text() for cb in self.metric_checkboxes if cb.isChecked()]
        if not selected_metrics:
            return

        df = self.final_df[self.final_df["Metric"].isin(selected_metrics)]
        group_labels = df["Group"].unique()
        if len(group_labels) < 2:
            QMessageBox.warning(self, "Stats Error", "Need at least two groups for comparison.")
            return

        pvals, results = [], []
        for metric in selected_metrics:
            combos = [(g1, g2) for i, g1 in enumerate(group_labels) for g2 in group_labels[i + 1:]]
            for g1, g2 in combos:
                g1v = df[(df["Group"] == g1) & (df["Metric"] == metric)]["Value"]
                g2v = df[(df["Group"] == g2) & (df["Metric"] == metric)]["Value"]
                if len(g1v) == 0 or len(g2v) == 0:
                    continue
                stat, p = ttest_ind(g1v, g2v, equal_var=False)
                pvals.append(p)
                results.append((metric, g1, g2, p))

        if not pvals:
            return
        reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
        log_text = f"\n=== Stats Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        for (metric, g1, g2, p), p_corr, rej in zip(results, pvals_corr, reject):
            log_text += f"{metric} {g1} vs {g2}: p={p:.4f}, corrected={p_corr:.4f}, significant={rej}\n"
        self.stats_log.append(log_text)

        if self.export_stats_checkbox.isChecked() and self.first_file_path:
            folder = getattr(self, "export_group_folder", os.path.dirname(self.first_file_path))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_path = os.path.join(folder, f"stats_at{timestamp}.csv")
            with open(export_path, "w") as f:
                f.write("Metric,Group1,Group2,p_value,corrected_p,significant\n")
                for (metric, g1, g2, p), p_corr, rej in zip(results, pvals_corr, reject):
                    f.write(f"{metric},{g1},{g2},{p},{p_corr},{rej}\n")

        self.draw_boxplots(df, selected_metrics)
        self.draw_pca(df, selected_metrics)

    def draw_boxplots(self, df, selected_metrics):
        self.boxplot_fig.clear()
        n = len(selected_metrics)
        axes = self.boxplot_fig.subplots(1, n, squeeze=False)[0]
        for i, metric in enumerate(selected_metrics):
            ax = axes[i]
            sub = df[df["Metric"] == metric]
            sns.boxplot(x="Group", y="Value", data=sub, ax=ax, palette="Set2")
            sns.stripplot(x="Group", y="Value", data=sub, ax=ax, color="black", jitter=True, alpha=0.6)
            ax.set_title(metric)
            ax.set_xlabel("Group")
            ax.set_ylabel("Value")
        self.boxplot_fig.tight_layout()
        self.boxplot_canvas.draw()

    def draw_pca(self, df, selected_metrics):
        self.pca_fig.clear()
        ax = self.pca_fig.add_subplot(111)
        pivot = df.pivot_table(index=["Group"], columns="Metric", values="Value", aggfunc=list).explode(selected_metrics).reset_index()
        X = pivot[selected_metrics].astype(float).values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        groups = pivot["Group"].values
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X)
        colors = sns.color_palette("Set2", n_colors=len(np.unique(groups)))
        for i, g in enumerate(np.unique(groups)):
            pts = Xp[groups == g]
            ax.scatter(pts[:, 0], pts[:, 1], label=g, color=colors[i])
        ax.legend()
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        self.pca_canvas.draw()

    def clear_log(self):
        self.stats_log.clear()

class DataSelectionWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Selection")
        self.resize(900, 500)
        self.parent = parent
        self.metric_entries = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.scroll_layout = QVBoxLayout(container)
        scroll.setWidget(container)

        self.metric_names = parent.metric_names
        group_count = parent.group_count_spin.value()
        self.groups = [f"G{i+1}" for i in range(group_count)]

        for i, name in enumerate(self.metric_names):
            layout = QVBoxLayout()
            layout.addWidget(QLabel(f"Metric {i+1}: {name}"))
            entry = {"metric_num": i + 1, "metric_name": name, "col_input": QLineEdit("3"), "group_files": {}, "group_rows": {}}
            for g in self.groups:
                hl = QHBoxLayout()
                btn = QPushButton(f"Select CSV for {g}")
                lbl = QLabel("No file selected")
                btn.clicked.connect(lambda _, e=entry, g=g, l=lbl: self.load_csv(e, g, l))
                row_in = QLineEdit("")
                hl.addWidget(btn)
                hl.addWidget(lbl)
                hl.addWidget(QLabel("Rows:"))
                hl.addWidget(row_in)
                entry["group_files"][g] = None
                entry["group_rows"][g] = row_in
                layout.addLayout(hl)
            layout.addWidget(QLabel("Column:"))
            layout.addWidget(entry["col_input"])
            self.metric_entries.append(entry)
            self.scroll_layout.addLayout(layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.scroll_layout.addWidget(QLabel("CSV Load Log:"))
        self.scroll_layout.addWidget(self.log_box)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Selections")
        save_btn.clicked.connect(self.save_and_close)
        btn_row.addWidget(save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        self.scroll_layout.addLayout(btn_row)
        layout_main = QVBoxLayout(self)
        layout_main.addWidget(scroll)

    def load_csv(self, entry, group_name, label):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select CSV for {group_name}", "", "CSV Files (*.csv)")
        if not file_path:
            return
        entry["group_files"][group_name] = file_path
        label.setText(os.path.basename(file_path))
        self.log_box.append(f"{group_name}: {file_path} loaded.")

    def save_and_close(self):
        self.parent.metric_entries = self.metric_entries
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataImporter()
    window.show()
    sys.exit(app.exec_())
