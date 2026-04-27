# ===== Standard Library =====
import sys
import os
from datetime import datetime

# ===== Third-Party =====
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QTextEdit, QTabWidget, QMessageBox, QFileDialog,
    QScrollArea, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence


class PasteTableWidget(QTableWidget):
    """QTableWidget with copy-paste-delete support for multi-cell operations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def keyPressEvent(self, event):
        """Handle keyboard events for copy/paste/delete."""
        if event.matches(QKeySequence.Paste):
            self.paste_from_clipboard()
        elif event.matches(QKeySequence.Copy):
            self.copy_to_clipboard()
        elif event.matches(QKeySequence.Delete) or event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected_cells()
        else:
            super().keyPressEvent(event)
    
    def delete_selected_cells(self):
        """Clear contents of all selected cells."""
        selected_ranges = self.selectedRanges()
        
        for sel_range in selected_ranges:
            for row in range(sel_range.topRow(), sel_range.bottomRow() + 1):
                for col in range(sel_range.leftColumn(), sel_range.rightColumn() + 1):
                    item = self.item(row, col)
                    if item:
                        item.setText("")
    
    def paste_from_clipboard(self):
        """Paste clipboard data into table starting from current selection."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        
        if not text:
            return
        
        # Get starting cell
        current = self.currentItem()
        if current:
            start_row = current.row()
            start_col = current.column()
        else:
            # If no cell selected, check for selected ranges
            selected = self.selectedRanges()
            if selected:
                start_row = selected[0].topRow()
                start_col = selected[0].leftColumn()
            else:
                start_row = 0
                start_col = 0
        
        # Parse clipboard text (tab-separated columns, newline-separated rows)
        rows = text.strip().split('\n')
        
        for i, row_text in enumerate(rows):
            target_row = start_row + i
            if target_row >= self.rowCount():
                break
            
            # Split by tab (Excel/spreadsheet format)
            cells = row_text.split('\t')
            
            for j, cell_text in enumerate(cells):
                target_col = start_col + j
                if target_col >= self.columnCount():
                    break
                
                # Get or create table item
                item = self.item(target_row, target_col)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(target_row, target_col, item)
                
                item.setText(cell_text.strip())
    
    def copy_to_clipboard(self):
        """Copy selected cells to clipboard."""
        selected = self.selectedRanges()
        if not selected:
            return
        
        # Get the bounding rectangle of selection
        top_row = min(r.topRow() for r in selected)
        bottom_row = max(r.bottomRow() for r in selected)
        left_col = min(r.leftColumn() for r in selected)
        right_col = max(r.rightColumn() for r in selected)
        
        rows = []
        for row in range(top_row, bottom_row + 1):
            row_data = []
            for col in range(left_col, right_col + 1):
                item = self.item(row, col)
                row_data.append(item.text() if item else "")
            rows.append('\t'.join(row_data))
        
        clipboard = QApplication.clipboard()
        clipboard.setText('\n'.join(rows))


class DataStatsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Group Comparison - Direct Data Entry")
        self.resize(1100, 900)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Runtime state
        self.data_tables = {}  # variable_name -> QTableWidget
        self.final_df = None
        self.output_folder = None
        
        # Undo state stack (stores snapshots of table data)
        self.undo_stack = []
        self.max_undo_levels = 10
        
        # === Configuration Section ===
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)
        
        # Row 1: Counts
        counts_layout = QHBoxLayout()
        
        counts_layout.addWidget(QLabel("Number of Groups:"))
        self.group_count_spin = QSpinBox()
        self.group_count_spin.setMinimum(2)
        self.group_count_spin.setMaximum(10)
        self.group_count_spin.setValue(2)
        self.group_count_spin.valueChanged.connect(self.update_group_inputs)
        counts_layout.addWidget(self.group_count_spin)
        
        counts_layout.addWidget(QLabel("Number of Variables:"))
        self.var_count_spin = QSpinBox()
        self.var_count_spin.setMinimum(1)
        self.var_count_spin.setMaximum(10)
        self.var_count_spin.setValue(2)
        self.var_count_spin.valueChanged.connect(self.update_variable_inputs)
        counts_layout.addWidget(self.var_count_spin)
        
        counts_layout.addWidget(QLabel("Number of Rows:"))
        self.row_count_spin = QSpinBox()
        self.row_count_spin.setMinimum(1)
        self.row_count_spin.setMaximum(100)
        self.row_count_spin.setValue(5)
        counts_layout.addWidget(self.row_count_spin)
        
        counts_layout.addStretch()
        config_layout.addLayout(counts_layout)
        
        # Row 2: Group names
        self.group_names_layout = QHBoxLayout()
        self.group_names_layout.addWidget(QLabel("Group Names:"))
        self.group_name_inputs = []
        config_layout.addLayout(self.group_names_layout)
        
        # Row 3: Variable names
        self.var_names_layout = QHBoxLayout()
        self.var_names_layout.addWidget(QLabel("Variable Names:"))
        self.var_name_inputs = []
        config_layout.addLayout(self.var_names_layout)
        
        # Initialize name inputs
        self.update_group_inputs()
        self.update_variable_inputs()
        
        self.layout.addWidget(config_group)
        
        # === Generate/Clear/Undo Tables Buttons ===
        tables_btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Data Tables")
        self.generate_btn.clicked.connect(self.generate_tables)
        tables_btn_layout.addWidget(self.generate_btn)
        
        self.clear_tables_btn = QPushButton("Clear Tables")
        self.clear_tables_btn.clicked.connect(self.clear_tables)
        tables_btn_layout.addWidget(self.clear_tables_btn)
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_action)
        self.undo_btn.setEnabled(False)  # Disabled until there's something to undo
        tables_btn_layout.addWidget(self.undo_btn)
        
        tables_btn_layout.addStretch()
        self.layout.addLayout(tables_btn_layout)
        
        # === Data Tables Section (scrollable) ===
        self.tables_scroll = QScrollArea()
        self.tables_scroll.setWidgetResizable(True)
        self.tables_container = QWidget()
        self.tables_layout = QVBoxLayout(self.tables_container)
        self.tables_scroll.setWidget(self.tables_container)
        self.tables_scroll.setMinimumHeight(250)
        self.layout.addWidget(self.tables_scroll)
        
        # === Export Options ===
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout()
        export_group.setLayout(export_layout)
        
        self.export_df_checkbox = QCheckBox("Export Dataframe as CSV")
        self.export_stats_checkbox = QCheckBox("Export Statistics Results as CSV")
        self.export_box_checkbox = QCheckBox("Export Boxplots (PNG & PDF)")
        self.export_pca_checkbox = QCheckBox("Export PCA Scatter (PNG & PDF)")
        
        for cb in [self.export_df_checkbox, self.export_stats_checkbox,
                   self.export_box_checkbox, self.export_pca_checkbox]:
            export_layout.addWidget(cb)
        
        self.layout.addWidget(export_group)
        
        # === Statistics Options ===
        stats_options_layout = QHBoxLayout()
        stats_options_layout.addWidget(QLabel("FDR Correction:"))
        self.fdr_within_var_checkbox = QCheckBox("Apply within each variable only")
        self.fdr_within_var_checkbox.setChecked(False)  # Default: across all variables
        self.fdr_within_var_checkbox.setToolTip(
            "Unchecked (default): FDR correction across all comparisons from all variables.\n"
            "Checked: FDR correction applied separately within each variable."
        )
        stats_options_layout.addWidget(self.fdr_within_var_checkbox)
        stats_options_layout.addStretch()
        self.layout.addLayout(stats_options_layout)
        
        # === Plot Options ===
        plot_options_layout = QHBoxLayout()
        plot_options_layout.addWidget(QLabel("Boxplot Y-max:"))
        self.ymax_input = QLineEdit()
        self.ymax_input.setPlaceholderText("auto")
        self.ymax_input.setMaximumWidth(80)
        self.ymax_input.setToolTip("Leave empty for automatic Y-axis maximum, or enter a value to override.")
        plot_options_layout.addWidget(self.ymax_input)
        plot_options_layout.addStretch()
        self.layout.addLayout(plot_options_layout)
        
        # === Output Folder Selection ===
        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Select Output Folder")
        self.folder_btn.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(self.folder_btn)
        self.folder_label = QLabel("No folder selected")
        folder_layout.addWidget(self.folder_label)
        folder_layout.addStretch()
        self.layout.addLayout(folder_layout)
        
        # === Action Buttons ===
        action_layout = QHBoxLayout()
        self.build_btn = QPushButton("Build Dataframe")
        self.build_btn.clicked.connect(self.build_dataframe)
        action_layout.addWidget(self.build_btn)
        
        self.stats_btn = QPushButton("Run Statistical Analysis")
        self.stats_btn.clicked.connect(self.run_stats)
        action_layout.addWidget(self.stats_btn)
        
        self.layout.addLayout(action_layout)
        
        # === Statistics Log ===
        self.stats_log = QTextEdit()
        self.stats_log.setReadOnly(True)
        self.stats_log.setMaximumHeight(150)
        self.layout.addWidget(QLabel("Statistics Log:"))
        self.layout.addWidget(self.stats_log)
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.stats_log.clear)
        self.layout.addWidget(clear_btn)
        
        # === Tabs for Plots ===
        self.tabs = QTabWidget()
        self.boxplot_fig = Figure(figsize=(6, 4))
        self.pca_fig = Figure(figsize=(6, 4))
        self.boxplot_canvas = FigureCanvas(self.boxplot_fig)
        self.pca_canvas = FigureCanvas(self.pca_fig)
        self.tabs.addTab(self.boxplot_canvas, "Boxplots")
        self.tabs.addTab(self.pca_canvas, "PCA Scatter")
        self.layout.addWidget(self.tabs)
    
    def update_group_inputs(self):
        """Rebuild group name input fields when count changes."""
        # Clear existing inputs (skip the label at index 0)
        while self.group_names_layout.count() > 1:
            item = self.group_names_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        
        self.group_name_inputs = []
        n = self.group_count_spin.value()
        
        for i in range(n):
            edit = QLineEdit(f"Group{i+1}")
            edit.setMaximumWidth(100)
            self.group_name_inputs.append(edit)
            self.group_names_layout.addWidget(edit)
        
        self.group_names_layout.addStretch()
    
    def update_variable_inputs(self):
        """Rebuild variable name input fields when count changes."""
        # Clear existing inputs (skip the label at index 0)
        while self.var_names_layout.count() > 1:
            item = self.var_names_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        
        self.var_name_inputs = []
        n = self.var_count_spin.value()
        
        for i in range(n):
            edit = QLineEdit(f"Var{i+1}")
            edit.setMaximumWidth(100)
            self.var_name_inputs.append(edit)
            self.var_names_layout.addWidget(edit)
        
        self.var_names_layout.addStretch()
    
    def save_undo_state(self):
        """Save current table state to undo stack."""
        if not self.data_tables:
            # Save empty state with current configuration
            state = {
                'tables_data': {},
                'group_names': [edit.text() for edit in self.group_name_inputs],
                'var_names': [edit.text() for edit in self.var_name_inputs],
                'n_rows': self.row_count_spin.value(),
                'has_tables': False
            }
        else:
            # Save current table contents
            tables_data = {}
            for var_name, table in self.data_tables.items():
                table_contents = []
                for r in range(table.rowCount()):
                    row_data = []
                    for c in range(table.columnCount()):
                        item = table.item(r, c)
                        row_data.append(item.text() if item else "")
                    table_contents.append(row_data)
                tables_data[var_name] = {
                    'contents': table_contents,
                    'headers': [table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else f"Col{c}" 
                               for c in range(table.columnCount())]
                }
            
            state = {
                'tables_data': tables_data,
                'group_names': [edit.text() for edit in self.group_name_inputs],
                'var_names': [edit.text() for edit in self.var_name_inputs],
                'n_rows': self.row_count_spin.value(),
                'has_tables': True
            }
        
        self.undo_stack.append(state)
        
        # Limit stack size
        if len(self.undo_stack) > self.max_undo_levels:
            self.undo_stack.pop(0)
        
        self.undo_btn.setEnabled(True)
    
    def undo_action(self):
        """Restore previous table state."""
        if not self.undo_stack:
            QMessageBox.information(self, "Nothing to Undo", "No previous state available.")
            return
        
        state = self.undo_stack.pop()
        
        # Update undo button state
        self.undo_btn.setEnabled(len(self.undo_stack) > 0)
        
        if not state['has_tables']:
            # Restore to no-tables state - clear current tables
            while self.tables_layout.count():
                item = self.tables_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.data_tables = {}
            self.stats_log.append("Undo: Restored to empty state.")
            return
        
        # Restore configuration
        for i, name in enumerate(state['group_names']):
            if i < len(self.group_name_inputs):
                self.group_name_inputs[i].setText(name)
        
        for i, name in enumerate(state['var_names']):
            if i < len(self.var_name_inputs):
                self.var_name_inputs[i].setText(name)
        
        # Check if we need to regenerate tables or just restore data
        current_vars = set(self.data_tables.keys())
        saved_vars = set(state['tables_data'].keys())
        
        if current_vars != saved_vars:
            # Need to regenerate tables with saved structure
            while self.tables_layout.count():
                item = self.tables_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.data_tables = {}
            
            for var_name, var_data in state['tables_data'].items():
                var_group = QGroupBox(f"Variable: {var_name}")
                var_layout = QVBoxLayout()
                var_group.setLayout(var_layout)
                
                n_rows = len(var_data['contents'])
                n_cols = len(var_data['headers'])
                
                table = PasteTableWidget()
                table.setRowCount(n_rows)
                table.setColumnCount(n_cols)
                table.setHorizontalHeaderLabels(var_data['headers'])
                table.setVerticalHeaderLabels([str(i+1) for i in range(n_rows)])
                
                # Restore cell contents
                for r, row_data in enumerate(var_data['contents']):
                    for c, cell_text in enumerate(row_data):
                        table.setItem(r, c, QTableWidgetItem(cell_text))
                
                table.resizeColumnsToContents()
                for c in range(n_cols):
                    if table.columnWidth(c) < 80:
                        table.setColumnWidth(c, 80)
                
                var_layout.addWidget(table)
                self.tables_layout.addWidget(var_group)
                self.data_tables[var_name] = table
        else:
            # Just restore data to existing tables
            for var_name, var_data in state['tables_data'].items():
                if var_name in self.data_tables:
                    table = self.data_tables[var_name]
                    for r, row_data in enumerate(var_data['contents']):
                        for c, cell_text in enumerate(row_data):
                            if r < table.rowCount() and c < table.columnCount():
                                item = table.item(r, c)
                                if item:
                                    item.setText(cell_text)
                                else:
                                    table.setItem(r, c, QTableWidgetItem(cell_text))
        
        self.stats_log.append("Undo: Restored previous state.")
    
    def generate_tables(self):
        """Generate data entry tables based on current configuration."""
        # Save current state before generating new tables
        self.save_undo_state()
        
        # Clear existing tables
        while self.tables_layout.count():
            item = self.tables_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.data_tables = {}
        
        # Get configuration
        group_names = [edit.text().strip() or f"Group{i+1}" 
                       for i, edit in enumerate(self.group_name_inputs)]
        var_names = [edit.text().strip() or f"Var{i+1}" 
                     for i, edit in enumerate(self.var_name_inputs)]
        n_rows = self.row_count_spin.value()
        
        # Create one table per variable
        for var_name in var_names:
            # Group box for this variable
            var_group = QGroupBox(f"Variable: {var_name}")
            var_layout = QVBoxLayout()
            var_group.setLayout(var_layout)
            
            # Create table with paste support
            table = PasteTableWidget()
            table.setRowCount(n_rows)
            table.setColumnCount(len(group_names))
            table.setHorizontalHeaderLabels(group_names)
            
            # Set row headers as row numbers
            table.setVerticalHeaderLabels([str(i+1) for i in range(n_rows)])
            
            # Initialize empty cells
            for r in range(n_rows):
                for c in range(len(group_names)):
                    table.setItem(r, c, QTableWidgetItem(""))
            
            # Adjust column widths
            table.resizeColumnsToContents()
            for c in range(len(group_names)):
                if table.columnWidth(c) < 80:
                    table.setColumnWidth(c, 80)
            
            var_layout.addWidget(table)
            self.tables_layout.addWidget(var_group)
            self.data_tables[var_name] = table
        
        self.stats_log.append(f"Generated {len(var_names)} tables with {len(group_names)} groups and {n_rows} rows each.")
        self.stats_log.append("Tip: Select a cell and use Ctrl+V to paste data from Excel/spreadsheet.")
    
    def clear_tables(self):
        """Clear all data from existing tables."""
        if not self.data_tables:
            QMessageBox.information(self, "No Tables", "No tables to clear.")
            return
        
        # Save current state before clearing
        self.save_undo_state()
        
        for var_name, table in self.data_tables.items():
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    item = table.item(r, c)
                    if item:
                        item.setText("")
        
        self.stats_log.append("All tables cleared.")
    
    def select_output_folder(self):
        """Select folder for exports."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.folder_label.setText(folder)
            self.stats_log.append(f"Output folder set: {folder}")
    
    def build_dataframe(self):
        """Build combined dataframe from all data tables."""
        if not self.data_tables:
            QMessageBox.warning(self, "No Tables", 
                                "Please generate data tables first.")
            return
        
        # Get current group names from inputs
        group_names = [edit.text().strip() or f"Group{i+1}" 
                       for i, edit in enumerate(self.group_name_inputs)]
        
        all_data = []
        
        for var_name, table in self.data_tables.items():
            n_rows = table.rowCount()
            n_cols = table.columnCount()
            
            for col_idx in range(n_cols):
                group_name = group_names[col_idx] if col_idx < len(group_names) else f"Group{col_idx+1}"
                
                for row_idx in range(n_rows):
                    item = table.item(row_idx, col_idx)
                    if item is None:
                        continue
                    
                    text = item.text().strip()
                    if not text:
                        continue
                    
                    # Try to parse as number
                    try:
                        value = float(text)
                        all_data.append({
                            "Group": group_name,
                            "Variable": var_name,
                            "Value": value
                        })
                    except ValueError:
                        # Skip non-numeric entries
                        self.stats_log.append(f"Warning: Skipped non-numeric value '{text}' in {var_name}, {group_name}")
        
        if not all_data:
            QMessageBox.warning(self, "No Data", 
                                "No valid numeric data found in tables.")
            return
        
        self.final_df = pd.DataFrame(all_data)
        self.show_dataframe_tab()
        
        self.stats_log.append(f"Built dataframe with {len(self.final_df)} data points.")
        
        # Export dataframe if requested
        if self.export_df_checkbox.isChecked():
            if not self.output_folder:
                QMessageBox.warning(self, "No Output Folder", 
                                    "Please select an output folder for exports.")
                return
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            outpath = os.path.join(self.output_folder, f"dataframe_at{timestamp}.csv")
            self.final_df.to_csv(outpath, index=False)
            self.stats_log.append(f"Dataframe exported to: {outpath}")
    
    def show_dataframe_tab(self):
        """Display the built dataframe in a tab."""
        df_table = QTableWidget()
        df_table.setRowCount(len(self.final_df))
        df_table.setColumnCount(len(self.final_df.columns))
        df_table.setHorizontalHeaderLabels(self.final_df.columns.tolist())
        
        for i in range(len(self.final_df)):
            for j, col in enumerate(self.final_df.columns):
                val = self.final_df.iloc[i, j]
                df_table.setItem(i, j, QTableWidgetItem(str(val)))
        
        # Remove existing Dataframe tab if present
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Dataframe":
                self.tabs.removeTab(i)
                break
        
        self.tabs.insertTab(0, df_table, "Dataframe")
        self.tabs.setCurrentIndex(0)
    
    def run_stats(self):
        """Run statistical analysis on the built dataframe."""
        if self.final_df is None or self.final_df.empty:
            QMessageBox.warning(self, "No Data", 
                                "Please build the dataframe first.")
            return
        
        df = self.final_df
        groups = df["Group"].unique()
        variables = df["Variable"].unique()
        
        if len(groups) < 2:
            QMessageBox.warning(self, "Stats Error", 
                                "Need at least two groups for comparison.")
            return
        
        # Check FDR correction mode
        fdr_within_variable = self.fdr_within_var_checkbox.isChecked()
        
        # === Collect all pairwise t-tests ===
        raw_results = []  # Will store (var, g1, g2, p_raw)
        
        for var in variables:
            combos = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]
            
            for g1, g2 in combos:
                g1_vals = df[(df["Group"] == g1) & (df["Variable"] == var)]["Value"]
                g2_vals = df[(df["Group"] == g2) & (df["Variable"] == var)]["Value"]
                
                if len(g1_vals) < 2 or len(g2_vals) < 2:
                    continue
                
                stat, p = ttest_ind(g1_vals, g2_vals, equal_var=False)
                raw_results.append((var, g1, g2, p))
        
        if not raw_results:
            QMessageBox.warning(self, "Stats Error", 
                                "No valid group comparisons could be performed. Need at least 2 values per group.")
            return
        
        # === Apply FDR correction based on selected mode ===
        all_results = []  # Will store (var, g1, g2, p_raw, p_corrected, reject)
        
        if fdr_within_variable:
            # FDR correction within each variable separately
            fdr_mode_text = "within each variable"
            
            # Group results by variable
            var_results = {}
            for (var, g1, g2, p) in raw_results:
                if var not in var_results:
                    var_results[var] = []
                var_results[var].append((g1, g2, p))
            
            # Apply FDR within each variable
            for var, comparisons in var_results.items():
                pvals = [p for (g1, g2, p) in comparisons]
                
                if len(pvals) > 1:
                    reject, p_corr, _, _ = multipletests(pvals, method="fdr_bh")
                else:
                    p_corr = pvals
                    reject = [pvals[0] < 0.05]
                
                for (g1, g2, p_raw), pc, rej in zip(comparisons, p_corr, reject):
                    all_results.append((var, g1, g2, p_raw, pc, rej))
        else:
            # FDR correction across all variables (default)
            fdr_mode_text = "across all variables"
            
            pvals = [p for (var, g1, g2, p) in raw_results]
            
            if len(pvals) > 1:
                reject, p_corr, _, _ = multipletests(pvals, method="fdr_bh")
            else:
                p_corr = pvals
                reject = [pvals[0] < 0.05]
            
            for (var, g1, g2, p_raw), pc, rej in zip(raw_results, p_corr, reject):
                all_results.append((var, g1, g2, p_raw, pc, rej))
        
        # === Summary statistics ===
        summary = (
            df.groupby(["Group", "Variable"])["Value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["cv_percent"] = (summary["std"] / summary["mean"]) * 100
        
        # === Log output ===
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"\n=== Statistical Analysis Run at {timestamp} ===\n"
        log_text += f"FDR correction mode: {fdr_mode_text}\n"
        
        log_text += "\nPairwise t-test Results (FDR corrected):\n"
        for (var, g1, g2, p_raw, pc, rej) in all_results:
            log_text += f"  {var}: {g1} vs {g2} | p={p_raw:.4g}, corrected p={pc:.4g}, significant={rej}\n"
        
        log_text += "\nSummary Statistics (per Group & Variable):\n"
        for _, row in summary.iterrows():
            log_text += (f"  {row['Variable']} | {row['Group']}: "
                         f"n={int(row['count'])}, mean={row['mean']:.4g}, "
                         f"std={row['std']:.4g}, CV={row['cv_percent']:.2f}%\n")
        
        self.stats_log.append(log_text)
        
        # === Export statistics if requested ===
        if self.export_stats_checkbox.isChecked():
            if not self.output_folder:
                QMessageBox.warning(self, "No Output Folder", 
                                    "Please select an output folder for exports.")
                return
            
            timestamp_tag = datetime.now().strftime("%Y%m%d-%H%M")
            
            # Pairwise results
            stats_path = os.path.join(self.output_folder, f"stats_at{timestamp_tag}.csv")
            with open(stats_path, "w") as f:
                f.write("Variable,Group1,Group2,p_value,corrected_p,significant\n")
                for (var, g1, g2, p_raw, pc, rej) in all_results:
                    f.write(f"{var},{g1},{g2},{p_raw},{pc},{rej}\n")
            
            # Summary statistics
            summary_path = os.path.join(self.output_folder, f"summary_at{timestamp_tag}.csv")
            summary.to_csv(summary_path, index=False)
            
            self.stats_log.append(f"Statistics exported to: {stats_path}")
        
        # === Store p-values for plot annotation ===
        # Store the first comparison's corrected p for each variable (for annotation)
        self.p_corr_map = {}
        for (var, g1, g2, p_raw, pc, rej) in all_results:
            if var not in self.p_corr_map:
                self.p_corr_map[var] = pc
        
        # === Define consistent colors (preserve input order) ===
        # Use the order from the group name inputs, filtered to groups present in data
        input_group_order = [edit.text().strip() or f"Group{i+1}"
                             for i, edit in enumerate(self.group_name_inputs)]
        df_groups = set(df["Group"].unique().tolist())
        unique_groups = [g for g in input_group_order if g in df_groups]
        # Append any groups not covered by inputs (fallback)
        for g in df["Group"].unique():
            if g not in unique_groups:
                unique_groups.append(g)
        palette = sns.color_palette("Set2", n_colors=len(unique_groups))
        self.color_map = {g: palette[i] for i, g in enumerate(unique_groups)}

        # === Preserve variable order from input fields ===
        input_var_order = [edit.text().strip() or f"Var{i+1}"
                           for i, edit in enumerate(self.var_name_inputs)]
        df_vars = set(df["Variable"].unique().tolist())
        ordered_variables = [v for v in input_var_order if v in df_vars]
        for v in df["Variable"].unique():
            if v not in ordered_variables:
                ordered_variables.append(v)

        # === Draw plots ===
        self.draw_boxplots(df, ordered_variables, unique_groups)
        self.draw_pca(df, ordered_variables)
    
    def draw_boxplots(self, df, variables, group_order=None):
        """Draw boxplots for each variable."""
        self.boxplot_fig.clear()
        n_vars = len(variables)
        axes = self.boxplot_fig.subplots(1, n_vars, squeeze=False)[0]
        
        # Check for manual Y-max override
        ymax_text = self.ymax_input.text().strip()
        manual_ymax = None
        if ymax_text:
            try:
                manual_ymax = float(ymax_text)
            except ValueError:
                self.stats_log.append(f"Warning: Invalid Y-max value '{ymax_text}', using auto.")
        
        for i, var in enumerate(variables):
            ax = axes[i]
            sub_df = df[df["Variable"] == var]
            
            # Use input-order for groups, filtered to those present in this variable's data
            present_groups = set(sub_df["Group"].unique())
            if group_order is not None:
                plot_order = [g for g in group_order if g in present_groups]
            else:
                plot_order = sorted(present_groups)
            
            sns.boxplot(
                x="Group", y="Value", data=sub_df, ax=ax,
                palette=self.color_map, order=plot_order
            )
            sns.stripplot(
                x="Group", y="Value", data=sub_df, ax=ax,
                jitter=True, marker="o", edgecolor="gray", linewidth=0.5,
                palette=self.color_map, alpha=0.6,
                order=plot_order
            )
            
            ax.set_title(var)
            ax.set_xlabel("Group")
            ax.set_ylabel("Value")
            
            # Y-axis limits: minimum always 0, maximum auto or manual
            vals = sub_df["Value"].to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            
            if manual_ymax is not None:
                ax.set_ylim(0, manual_ymax)
            elif finite_vals.size > 0:
                ymax = float(np.nanmax(finite_vals))
                margin = 0.1 * ymax if ymax > 0 else 0.1
                ax.set_ylim(0, ymax + margin)
            else:
                ax.set_ylim(0, 1)  # Default if no data
            
            # Annotate corrected p
            p_corr = self.p_corr_map.get(var, np.nan)
            label = f"corrected p = {p_corr:.3g}" if np.isfinite(p_corr) else "corrected p = N/A"
            ax.text(0.5, 0.95, label, transform=ax.transAxes,
                    ha="center", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        
        self.boxplot_fig.tight_layout()
        self.boxplot_canvas.draw()
        
        # Export if requested
        if self.export_box_checkbox.isChecked() and self.output_folder:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.boxplot_fig.savefig(os.path.join(self.output_folder, f"boxplot_at{timestamp}.png"), dpi=300)
            self.boxplot_fig.savefig(os.path.join(self.output_folder, f"boxplot_at{timestamp}.pdf"))
            self.stats_log.append(f"Boxplots exported.")
    
    def draw_pca(self, df, variables):
        """Draw PCA scatter plot."""
        self.pca_fig.clear()
        ax = self.pca_fig.add_subplot(111)
        
        # Pivot data: each row is one observation, columns are variables
        # Need to handle the fact that each group may have different numbers of observations
        
        # Get unique groups
        groups = df["Group"].unique()
        
        # Build a matrix where rows are samples and columns are variables
        # We'll pair up observations within each group by their order
        
        pca_data = []
        pca_groups = []
        
        for group in groups:
            group_df = df[df["Group"] == group]
            
            # Get values for each variable in this group
            var_values = {}
            for var in variables:
                var_values[var] = group_df[group_df["Variable"] == var]["Value"].values
            
            # Find minimum length (some variables might have fewer observations)
            min_len = min(len(v) for v in var_values.values()) if var_values else 0
            
            if min_len == 0:
                continue
            
            # Create rows (one per observation)
            for i in range(min_len):
                row = [var_values[var][i] for var in variables]
                pca_data.append(row)
                pca_groups.append(group)
        
        if len(pca_data) < 2:
            ax.text(0.5, 0.5, "Insufficient data for PCA", 
                    transform=ax.transAxes, ha="center", va="center")
            self.pca_canvas.draw()
            return
        
        X = np.array(pca_data)
        groups_arr = np.array(pca_groups)
        
        # Check for sufficient variance
        if X.shape[1] < 2:
            ax.text(0.5, 0.5, "Need at least 2 variables for PCA", 
                    transform=ax.transAxes, ha="center", va="center")
            self.pca_canvas.draw()
            return
        
        # Standardize
        X_scaled = StandardScaler().fit_transform(X)
        
        # PCA
        n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        if X_pca.shape[1] < 2:
            ax.text(0.5, 0.5, "Insufficient dimensions for 2D PCA", 
                    transform=ax.transAxes, ha="center", va="center")
            self.pca_canvas.draw()
            return
        
        # Plot setup
        group_labels = sorted(np.unique(groups_arr))
        counters = {g: 0 for g in group_labels}
        
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        x_offset = 0.015 * x_range if x_range > 0 else 0.1
        y_offset = 0.015 * y_range if y_range > 0 else 0.1
        
        # Scatter points with labels
        for i, (x, y) in enumerate(X_pca):
            g = groups_arr[i]
            counters[g] += 1
            ax.scatter(x, y, color=self.color_map[g], s=70)
            ax.text(x + x_offset, y + y_offset, f"{g}-{counters[g]}", fontsize=9)
        
        # 95% confidence ellipses
        for grp in group_labels:
            grp_points = X_pca[groups_arr == grp]
            if len(grp_points) > 2:
                cov = np.cov(grp_points, rowvar=False)
                mean = grp_points.mean(axis=0)
                
                # Handle potential singular covariance matrix
                try:
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    width, height = 2 * np.sqrt(np.maximum(vals, 0) * 5.991)  # 95% CI
                    ellipse = mpatches.Ellipse(mean, width, height, angle=theta,
                                               edgecolor=self.color_map[grp], 
                                               facecolor='none', linestyle='--')
                    ax.add_patch(ellipse)
                except Exception:
                    pass  # Skip ellipse if calculation fails
        
        # Biplot arrows (loadings)
        loadings = pca.components_.T
        vecs = loadings * np.sqrt(pca.explained_variance_)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        axis_span = 0.5 * min(xmax - xmin, ymax - ymin)
        
        for i, var in enumerate(variables):
            lx, ly = vecs[i, 0] * axis_span, vecs[i, 1] * axis_span
            ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))
            ax.text(lx * 1.1, ly * 1.1, var, color="red", fontsize=8, ha="center")
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title("PCA - All Variables")
        
        handles = [mpatches.Patch(color=self.color_map[g], label=g) for g in group_labels]
        ax.legend(handles=handles, title="Group", loc="best")
        
        self.pca_fig.tight_layout()
        self.pca_canvas.draw()
        
        # Export if requested
        if self.export_pca_checkbox.isChecked() and self.output_folder:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            self.pca_fig.savefig(os.path.join(self.output_folder, f"pca_at{timestamp}.png"), dpi=300)
            self.pca_fig.savefig(os.path.join(self.output_folder, f"pca_at{timestamp}.pdf"))
            self.stats_log.append(f"PCA plot exported.")


# ===== Entrypoint =====
def main():
    app = QApplication(sys.argv)
    window = DataStatsGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
