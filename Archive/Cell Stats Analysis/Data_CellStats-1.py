import sys
import pandas as pd
import os
import re
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLineEdit, QLabel, QTableWidget, QTableWidgetItem, QHBoxLayout, QFormLayout, QCheckBox
)


class DataImporter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Group Comparison Data Importer (4 Metrics, 2 Groups)")
        self.layout = QVBoxLayout()

        self.metric_entries = []
        self.dataframes = []

        # Metric defaults
        self.metric_names = [
            "Norm Lac/Pyr Ratio",
            "Norm Lac",
            "Norm Lac/Pyr Ratio (fit)",
            "kPL (fit)"
        ]
        self.default_columns = [2, 3, 12, 3]  # 1-indexed
        self.default_g1_rows = "1-3"
        self.default_g2_rows = "4-6"

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

        # CSV export option
        self.export_checkbox = QCheckBox("Export Dataframe as CSV")
        self.layout.addWidget(self.export_checkbox)

        # Button to build dataframe
        self.build_button = QPushButton("Build Dataframe")
        self.build_button.clicked.connect(self.build_dataframe)
        self.layout.addWidget(self.build_button)

        # Table to display dataframe
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.setLayout(self.layout)

    def create_metric_block(self, metric_num, default_name, default_col):
        layout = QHBoxLayout()

        # File selector
        btn = QPushButton(f"Select CSV for {default_name}")
        label = QLabel("No file selected")
        btn.clicked.connect(lambda _, n=metric_num, lbl=label: self.load_csv(n, lbl))
        layout.addWidget(btn)
        layout.addWidget(label)

        # Column input
        col_input = QLineEdit()
        col_input.setText(default_col)
        layout.addWidget(QLabel("Column:"))
        layout.addWidget(col_input)

        # Group 1 rows input
        g1_input = QLineEdit()
        g1_input.setText(self.default_g1_rows)
        layout.addWidget(QLabel("G1 Rows:"))
        layout.addWidget(g1_input)

        # Group 2 rows input
        g2_input = QLineEdit()
        g2_input.setText(self.default_g2_rows)
        layout.addWidget(QLabel("G2 Rows:"))
        layout.addWidget(g2_input)

        self.metric_entries.append({
            "file": None,
            "file_label": label,
            "col_input": col_input,
            "g1_input": g1_input,
            "g2_input": g2_input,
            "metric_name": default_name,
            "metric_num": metric_num
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
        # Split on any whitespace, comma, or tab
        tokens = re.split(r'[,\s]+', text.strip())
        return [float(x) for x in tokens if x]

    def apply_normalization(self, metric_num, imported, cell, conc, pol):
        if metric_num == 1:  # Norm Lac/Pyr Ratio
            return (imported * conc) / (cell / 1e6)
        elif metric_num == 2:  # Norm Lac
            return imported / (conc * pol * (cell / 1e6))
        elif metric_num == 3:  # Norm Lac/Pyr Ratio (fit)
            return (imported * conc) / (cell / 1e6)
        elif metric_num == 4:  # kPL (fit) -> no normalization
            return imported
        else:
            return imported

    def build_dataframe(self):
        all_data = []

        # Parse normalization factors
        factors = {}
        for group in ["G1", "G2"]:
            cells = self.parse_factors(self.norm_inputs[group]["cell"].text())
            concs = self.parse_factors(self.norm_inputs[group]["conc"].text())
            pols = self.parse_factors(self.norm_inputs[group]["pol"].text())
            factors[group] = {"cell": cells, "conc": concs, "pol": pols}

        # Track first file path for export location
        first_file_path = None

        for entry in self.metric_entries:
            if not entry["file"]:
                continue

            if entry["metric_num"] == 1 and not first_file_path:
                first_file_path = entry["file"]

            df = pd.read_csv(entry["file"])
            col_idx = int(entry["col_input"].text()) - 1
            series = df.iloc[:, col_idx]

            metric_name = entry["metric_name"]
            metric_num = entry["metric_num"]

            for group, row_input in [("G1", entry["g1_input"]), ("G2", entry["g2_input"])]:
                rows = self.parse_row_range(row_input.text())
                for i, r in enumerate(rows):
                    imported = series.iloc[r]
                    cell = factors[group]["cell"][i]
                    conc = factors[group]["conc"][i]
                    pol = factors[group]["pol"][i]
                    norm_val = self.apply_normalization(metric_num, imported, cell, conc, pol)
                    all_data.append([group, metric_name, imported, cell, conc, pol, norm_val])

        final_df = pd.DataFrame(all_data, columns=["Group", "Metric", "Imported", "CellDensity", "SubConc", "SubPol", "Value"])
        self.dataframes.append(final_df)
        self.show_dataframe(final_df)

        # Export CSV if requested
        if self.export_checkbox.isChecked() and first_file_path:
            folder = os.path.dirname(first_file_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            export_name = f"dataframe_at{timestamp}.csv"
            export_path = os.path.join(folder, export_name)
            final_df.to_csv(export_path, index=False)
            print(f"Exported dataframe to {export_path}")

    def show_dataframe(self, df):
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataImporter()
    window.show()
    sys.exit(app.exec_())
