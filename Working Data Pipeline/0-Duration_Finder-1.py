import sys
import os
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QTextEdit,
    QSpinBox
)
from PyQt5.QtCore import Qt


class DurationFinderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Duration Finder (acqu.par)")
        self.resize(600, 450)

        # Store last results for export
        self.last_results = []

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Parent folder selection row ---
        folder_layout = QHBoxLayout()
        self.lbl_parent = QLabel("Parent folder:")
        self.lbl_parent_path = QLabel("<none selected>")
        self.lbl_parent_path.setStyleSheet("color: gray;")
        self.lbl_parent_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        btn_browse = QPushButton("Browse...")

        folder_layout.addWidget(self.lbl_parent)
        folder_layout.addWidget(self.lbl_parent_path, stretch=1)
        folder_layout.addWidget(btn_browse)

        main_layout.addLayout(folder_layout)

        # --- Total Timepoints row ---
        tp_layout = QHBoxLayout()
        tp_label = QLabel("Total Timepoints:")
        self.spin_timepoints = QSpinBox()
        self.spin_timepoints.setRange(1, 100000)
        self.spin_timepoints.setValue(150)
        self.spin_timepoints.setFixedWidth(100)

        tp_layout.addWidget(tp_label)
        tp_layout.addWidget(self.spin_timepoints)
        tp_layout.addStretch(1)

        main_layout.addLayout(tp_layout)

        # --- Subfolder list ---
        self.subfolder_list = QListWidget()
        self.subfolder_list.setSelectionMode(QListWidget.MultiSelection)
        main_layout.addWidget(QLabel("Subfolders in parent (select one or more):"))
        main_layout.addWidget(self.subfolder_list, stretch=1)

        # --- Bottom buttons ---
        bottom_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_find = QPushButton("Find durations")
        self.btn_export = QPushButton("Export timesteps")
        self.btn_export.setEnabled(False)

        bottom_layout.addWidget(self.btn_select_all)
        bottom_layout.addWidget(self.btn_clear_sel)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.btn_find)
        bottom_layout.addWidget(self.btn_export)

        main_layout.addLayout(bottom_layout)

        # Connections
        btn_browse.clicked.connect(self.choose_parent_folder)
        self.btn_select_all.clicked.connect(self.select_all_subfolders)
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        self.btn_find.clicked.connect(self.find_durations)
        self.btn_export.clicked.connect(self.export_timesteps)

    # ---------------- Helper methods ---------------- #

    def choose_parent_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder", ""
        )
        if folder:
            self.lbl_parent_path.setText(folder)
            self.lbl_parent_path.setStyleSheet("color: black;")
            self.populate_subfolders(folder)
            # Reset results when folder changes
            self.last_results = []
            self.btn_export.setEnabled(False)

    def populate_subfolders(self, parent_folder):
        self.subfolder_list.clear()
        try:
            entries = sorted(
                [
                    name for name in os.listdir(parent_folder)
                    if os.path.isdir(os.path.join(parent_folder, name))
                ]
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error listing subfolders:\n{e}")
            return

        if not entries:
            QMessageBox.information(
                self, "No subfolders",
                "No subfolders were found in the selected parent folder."
            )
            return

        for name in entries:
            item = QListWidgetItem(name)
            self.subfolder_list.addItem(item)

    def select_all_subfolders(self):
        for i in range(self.subfolder_list.count()):
            item = self.subfolder_list.item(i)
            item.setSelected(True)

    def clear_selection(self):
        self.subfolder_list.clearSelection()

    def find_durations(self):
        parent = self.lbl_parent_path.text()
        if not parent or parent.startswith("<"):
            QMessageBox.warning(
                self, "No parent folder",
                "Please select a parent folder first."
            )
            return

        selected_items = self.subfolder_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No subfolders selected",
                "Please select one or more subfolders."
            )
            return

        # Collect as list of (experiment_name, value) tuples
        results = []
        for item in selected_items:
            subfolder_name = item.text()
            subfolder_path = os.path.join(parent, subfolder_name)

            acqu_path = self.find_acqu_par(subfolder_path)
            if not acqu_path:
                results.append((subfolder_name, "acqu.par NOT FOUND"))
                continue

            duration = self.extract_duration(acqu_path)
            if duration is None:
                results.append((subfolder_name, "DURATION NOT FOUND"))
            else:
                results.append((subfolder_name, duration))

        if not results:
            QMessageBox.information(
                self, "No results",
                "No durations found."
            )
            return

        # Store results and enable export
        self.last_results = results
        self.btn_export.setEnabled(True)

        self.show_results_dialog(results)

    def find_acqu_par(self, folder_path):
        """Recursively search for acqu.par within folder_path."""
        for root, dirs, files in os.walk(folder_path):
            if "acqu.par" in files:
                return os.path.join(root, "acqu.par")
        return None

    def extract_duration(self, acqu_path):
        """
        Read acqu.par and find a line like:
        duration = XXX or XXX.X or XXX.XX etc.
        Returns the string as found (not converted), or None.
        """
        pattern = re.compile(r"^\s*duration\s*=\s*([0-9]*\.?[0-9]+)\s*$")

        try:
            with open(acqu_path, "r") as f:
                for line in f:
                    m = pattern.match(line)
                    if m:
                        return m.group(1)
        except Exception as e:
            print(f"Error reading {acqu_path}: {e}")
            return None

        return None

    def export_timesteps(self):
        """
        Export timestep values (duration / total_timepoints) to a text file
        named timestep_<parent_folder_name>.txt in the parent folder.
        """
        parent = self.lbl_parent_path.text()
        if not parent or parent.startswith("<"):
            QMessageBox.warning(
                self, "No parent folder",
                "Please select a parent folder first."
            )
            return

        if not self.last_results:
            QMessageBox.warning(
                self, "No results",
                "Please run 'Find durations' first."
            )
            return

        total_timepoints = self.spin_timepoints.value()

        # Compute timesteps, skipping experiments with errors
        timestep_lines = []
        skipped = []
        for name, value in self.last_results:
            try:
                duration_val = float(value)
                timestep = duration_val / total_timepoints
                timestep_lines.append(f"{timestep:.4f}")
            except (ValueError, TypeError):
                skipped.append(f"{name}: {value}")

        if not timestep_lines:
            QMessageBox.warning(
                self, "No valid durations",
                "No valid numeric durations were found to export."
            )
            return

        # Build filename from the parent folder name (which is a date)
        parent_folder_name = os.path.basename(parent)
        filename = f"timestep_{parent_folder_name}.txt"
        export_path = os.path.join(parent, filename)

        try:
            with open(export_path, "w") as f:
                f.write("\n".join(timestep_lines) + "\n")
        except Exception as e:
            QMessageBox.critical(
                self, "Export error",
                f"Could not write file:\n{export_path}\n\n{e}"
            )
            return

        msg = f"Exported {len(timestep_lines)} timestep(s) to:\n{export_path}"
        if skipped:
            msg += f"\n\nSkipped {len(skipped)} experiment(s) with errors:\n"
            msg += "\n".join(skipped)

        QMessageBox.information(self, "Export complete", msg)

    def show_results_dialog(self, results):
        """
        Show a dialog with two columns (two QTextEdits side-by-side):
        - Left: experiment names
        - Right: durations/errors

        So the user can copy only one column at a time.
        """
        dialog = QWidget(self, Qt.Dialog)
        dialog.setWindowTitle("Durations (copy one column at a time)")
        main_layout = QVBoxLayout(dialog)

        info_label = QLabel(
            "Left: experiment names    |    Right: durations / errors\n"
            "Use Ctrl+A then Ctrl+C in ONE box to copy only that column."
        )
        main_layout.addWidget(info_label)

        # Horizontal layout for the two text boxes
        columns_layout = QHBoxLayout()

        # Build column strings
        exp_lines = []
        val_lines = []
        for name, value in results:
            exp_lines.append(str(name))
            val_lines.append(str(value))

        exp_text = "\n".join(exp_lines)
        val_text = "\n".join(val_lines)

        # Left column: experiments
        txt_exp = QTextEdit()
        txt_exp.setPlainText(exp_text)
        txt_exp.setReadOnly(True)
        txt_exp.setPlaceholderText("Experiment names")
        columns_layout.addWidget(txt_exp)

        # Right column: durations/errors
        txt_val = QTextEdit()
        txt_val.setPlainText(val_text)
        txt_val.setReadOnly(True)
        txt_val.setPlaceholderText("Durations / errors")
        columns_layout.addWidget(txt_val)

        main_layout.addLayout(columns_layout)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.close)
        main_layout.addWidget(btn_close, alignment=Qt.AlignRight)

        dialog.resize(600, 350)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


def main():
    app = QApplication(sys.argv)
    win = DurationFinderGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
