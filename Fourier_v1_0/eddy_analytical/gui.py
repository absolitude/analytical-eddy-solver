from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    # QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from .plotting import write_result_reports
from .worker import CalculationWorker

logger = logging.getLogger(__name__)


def _spin(value: float, minimum: float, maximum: float, step: float, decimals: int = 4) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


def _int_spin(value: int, minimum: int, maximum: int, step: int = 1) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


class ResultWindow(QMainWindow):
    """A tabbed viewer for the generated interactive Plotly HTML reports."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Eddy Current Results")
        self.resize(1450, 950)
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

    def set_reports(self, reports: dict[str, Path]) -> None:
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()

        for title, path in reports.items():
            if not path.exists():
                continue
            view = QWebEngineView(self.tabs)
            view.load(QUrl.fromLocalFile(str(path.resolve())))
            self.tabs.addTab(view, title)



class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D Eddy Current Analysis (Fourier) v1.0")
        self.resize(1120, 850)
        self.worker: CalculationWorker | None = None
        self.last_result = None
        self.active_parameters: dict[str, Any] | None = None
        self.output_dir = Path.cwd() / "analytical_results"
        self.report_paths: dict[str, Path] = {}
        self.result_window: ResultWindow | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Geometry selector.
        shape_form = QFormLayout()
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Circular", "Ellipse", "Flat pipe (only for 180 degree rotation symmetry)"])
        self.shape_combo.currentTextChanged.connect(self.update_geometry_fields)
        shape_form.addRow("Boundary shape:", self.shape_combo)
        root.addLayout(shape_form)

        forms_row = QHBoxLayout()
        root.addLayout(forms_row)

        # Geometry controls. Only the controls relevant to the selected shape
        # are visible; the numerical and wire settings are shared.
        geometry_form = QFormLayout()
        self.radius_label = QLabel("Radius R (mm):")
        self.radius_input = _spin(4.0, 0.1, 500.0, 0.5, 3)
        geometry_form.addRow(self.radius_label, self.radius_input)

        self.a_label = QLabel("Semi-major axis a (mm):")
        self.a_input = _spin(20.0, 0.1, 500.0, 0.5, 3)
        geometry_form.addRow(self.a_label, self.a_input)
        self.b_ellipse_label = QLabel("Semi-minor axis b (mm):")
        self.b_ellipse_input = _spin(4.0, 0.1, 500.0, 0.5, 3)
        geometry_form.addRow(self.b_ellipse_label, self.b_ellipse_input)

        self.flat_gap_label = QLabel("Full gap 2b (mm):")
        self.flat_gap_input = _spin(8.0, 0.2, 1000.0, 1.0, 3)
        geometry_form.addRow(self.flat_gap_label, self.flat_gap_input)

        self.flat_range_label = QLabel("Full x-direction width 2xmax (mm):")
        self.flat_range_input = _spin(400.0, 20.0, 10000.0, 20.0, 2)
        geometry_form.addRow(self.flat_range_label, self.flat_range_input)
        self.flat_points_label = QLabel("Spatial points (even):")
        self.flat_points_input = _int_spin(4096, 256, 32768, 256)
        geometry_form.addRow(self.flat_points_label, self.flat_points_input)
        forms_row.addLayout(geometry_form)

        physics_form = QFormLayout()
        self.conductivity_input = _spin(1.25e6, 1.0, 1.0e9, 1.0e5, 1)
        self.thickness_input = _spin(5.0, 0.01, 1000.0, 0.5, 3)
        self.pulse_length_input = _spin(5.0, 0.01, 10000.0, 0.5, 3)
        self.pulse_type_combo = QComboBox()
        self.pulse_type_combo.addItems(["half_sine"])
        physics_form.addRow("Conductivity (S/m):", self.conductivity_input)
        physics_form.addRow("Coating thickness (μm):", self.thickness_input)
        physics_form.addRow("Pulse length T (μs):", self.pulse_length_input)
        physics_form.addRow("Pulse type:", self.pulse_type_combo)
        forms_row.addLayout(physics_form)

        numerical_form = QFormLayout()
        self.cutoff_input = _spin(50.0, 1.0, 500.0, 5.0, 1)
        self.frequency_points_input = _int_spin(501, 3, 20001, 2)
        self.time_points_input = _int_spin(251, 2, 20001, 2)
        self.x_axis_points_input = _int_spin(101, 2, 20001, 1)
        self.x_axis_points_input.setToolTip(
            "Number of horizontal-axis observation points (nopxa). An odd number is "
            "recommended so that x=0 is included. The output range is geometry-dependent."
        )
        self.time_points_input.setToolTip(
            "Number of time samples (noptd) on the common grid t = linspace(-T, +T, noptd). "
            "Use the same T and noptd in the Green solver for point-by-point comparison."
        )
        self.order_input = _int_spin(10, 1, 100, 1)
        self.boundary_points_input = _int_spin(1024, 128, 20000, 128)
        self.fit_order_input = _int_spin(20, 1, 30, 1)
        # self.bem_check_input = QCheckBox("Run Green's-function cross-check (Ellipse only)")
        # self.bem_check_input.setChecked(True)
        # self.bem_check_input.setToolTip(
        #     "Run an independent straight-panel boundary-integral calculation and compare "
        #     "Fourier coefficients and the on-axis magnetic field."
        # )
        # self.bem_panels_input = _int_spin(160, 64, 4096, 32)
        # self.bem_panels_input.setToolTip(
        #     "Number of straight panels used by the Green's-function reference calculation. "
        #     "This is independent of the analytical boundary sample count."
        # )
        numerical_form.addRow("Frequency cutoff C:", self.cutoff_input)
        numerical_form.addRow("Frequency points (odd, nopfd):", self.frequency_points_input)
        numerical_form.addRow("Time points (noptd):", self.time_points_input)
        numerical_form.addRow("Horizontal-axis points (nopxa):", self.x_axis_points_input)
        numerical_form.addRow("Fourier order (circle/ellipse):", self.order_input)
        numerical_form.addRow("Boundary samples (circle/ellipse):", self.boundary_points_input)
        numerical_form.addRow("Polynomial fit order:", self.fit_order_input)
        # numerical_form.addRow(self.bem_check_input)
        # numerical_form.addRow("BEM boundary panels:", self.bem_panels_input)
        forms_row.addLayout(numerical_form)
        # self.bem_check_input.toggled.connect(self.bem_panels_input.setEnabled)

        # External wire configuration.
        wire_header = QHBoxLayout()
        wire_header.addWidget(QLabel("External z-directed wires"))
        wire_header.addStretch(1)
        self.wire_preset_combo = QComboBox()
        self.wire_preset_combo.addItems([
            "BESSY-II type"
        ])
        self.apply_preset_button = QPushButton("Apply preset")
        self.add_wire_button = QPushButton("Add wire")
        self.remove_wire_button = QPushButton("Remove selected")
        wire_header.addWidget(self.wire_preset_combo)
        wire_header.addWidget(self.apply_preset_button)
        wire_header.addWidget(self.add_wire_button)
        wire_header.addWidget(self.remove_wire_button)
        root.addLayout(wire_header)

        self.wire_table = QTableWidget(8, 3)
        self.wire_table.setHorizontalHeaderLabels(["x (mm)", "y (mm)", "Current (A)"])
        self.wire_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.wire_table)
        self.apply_wire_preset("BESSY-II type")

        self.apply_preset_button.clicked.connect(
            lambda: self.apply_wire_preset(self.wire_preset_combo.currentText())
        )
        self.add_wire_button.clicked.connect(self.add_wire_row)
        self.remove_wire_button.clicked.connect(self.remove_wire_rows)

        action_row = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setMinimumHeight(36)
        self.open_output_button = QPushButton("Choose output folder")
        self.open_output_button.clicked.connect(self.choose_output_folder)
        self.view_results_button = QPushButton("View HTML reports")
        self.view_results_button.setEnabled(False)
        self.view_results_button.clicked.connect(self.open_results_window)
        action_row.addWidget(self.calculate_button)
        action_row.addWidget(self.open_output_button)
        action_row.addWidget(self.view_results_button)
        root.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.output_label = QLabel(f"Output: {self.output_dir}")
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.output_label)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: #666; padding: 6px;")
        root.addWidget(self.note_label)

        self.calculate_button.clicked.connect(self.start_calculation)
        self.update_geometry_fields(self.shape_combo.currentText())

    def update_geometry_fields(self, shape: str) -> None:
        is_circle = shape == "Circular"
        is_ellipse = shape == "Ellipse"
        is_flat = shape == "Flat pipe (only for 180 degree rotation symmetry)"
        for widget, visible in (
            (self.radius_label, is_circle),
            (self.radius_input, is_circle),
            (self.a_label, is_ellipse),
            (self.a_input, is_ellipse),
            (self.b_ellipse_label, is_ellipse),
            (self.b_ellipse_input, is_ellipse),
            (self.flat_gap_label, is_flat),
            (self.flat_gap_input, is_flat),
            (self.flat_range_label, is_flat),
            (self.flat_range_input, is_flat),
            (self.flat_points_label, is_flat),
            (self.flat_points_input, is_flat),
        ):
            widget.setVisible(visible)

        self.order_input.setEnabled(not is_flat)
        self.boundary_points_input.setEnabled(not is_flat)
        # self.bem_check_input.setEnabled(is_ellipse)
        # self.bem_panels_input.setEnabled(is_ellipse and self.bem_check_input.isChecked())
        if is_circle:
            note = (
                "Circular analytical solution: independent azimuthal Fourier modes. "
                "The m=0 gauge/mean-current component is excluded from the induced-current modes."
            )
        elif is_ellipse:
            note = (
                "Elliptical solution: cosine and sine mode-coupling systems are solved separately. "
                "The cosine m=0 row imposes zero net longitudinal current on the closed coating."
            )
        else:
            note = (
                "Flat-pipe solution: two infinite plates at y=±b, where 2b is the entered full gap. "
                "The internal spatial FFT uses the finite window from -xmax to +xmax, where "
                "2*xmax is the entered full x-direction width. Field and plate-current outputs "
                "are evaluated at nopxa points over ±0.9*xmax by direct Fourier-series synthesis; "
                "nopxa does not change the internal FFT resolution. The external wires must have "
                "zero net current. Check convergence by increasing the full x-direction width "
                "and the internal spatial FFT points."
            )
        self.note_label.setText(note)

    def apply_wire_preset(self, preset: str) -> None:
        current = 1600.0
        sign = 1.0
        wires: list[tuple[float, float, float]] = []

        if preset == "BESSY-II type":
            for x, y in [(6, 6), (6, -6), (-6, 6), (-6, -6)]:
                wires.append((x, y, sign * current))
            for x, y in [(10, 10), (10, -10), (-10, 10), (-10, -10)]:
                wires.append((x, y, -sign * current))
        elif preset == "ALBA-II type":
            wires = [
                (6, 6, current), (6, -6, current),
                (-6, 6, -current), (-6, -6, -current),
                (10, 10, -current), (10, -10, -current),
                (-10, 10, current), (-10, -10, current),
            ]
        elif preset == "SOLEIL Topology D type":
            main = 1000.0
            outer = 1000.0
            wires = [
                (0, 4.5, main), (0, -4.5, main),
                (2.4, 4.5, -main), (2.4, -4.5, -main),
                (-2.4, 4.5, -main), (-2.4, -4.5, -main),
                (4.8, 4.5, main), (4.8, -4.5, main),
                (-4.8, 4.5, main), (-4.8, -4.5, main),
                (0, 17, -outer), (0, -17, -outer),
            ]
        else:
            if self.wire_table.rowCount() == 0:
                wires = [(20.0, 0.0, 1000.0)]
            else:
                return

        self.wire_table.setRowCount(len(wires))
        for row, wire in enumerate(wires):
            for col, value in enumerate(wire):
                self.wire_table.setItem(row, col, QTableWidgetItem(f"{value:g}"))

    def add_wire_row(self) -> None:
        row = self.wire_table.rowCount()
        self.wire_table.insertRow(row)
        for col, value in enumerate((20.0, 0.0, 1000.0)):
            self.wire_table.setItem(row, col, QTableWidgetItem(str(value)))

    def remove_wire_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.wire_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.wire_table.removeRow(row)

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", str(self.output_dir)
        )
        if folder:
            self.output_dir = Path(folder)
            self.output_label.setText(f"Output: {self.output_dir}")

    def collect_parameters(self) -> dict[str, Any]:
        nfreq = self.frequency_points_input.value()
        if nfreq % 2 == 0:
            nfreq += 1
            self.frequency_points_input.setValue(nfreq)

        nopxa = self.x_axis_points_input.value()
        if nopxa < self.fit_order_input.value() + 1:
            raise ValueError(
                "Horizontal-axis points (nopxa) must be at least "
                "polynomial fit order + 1."
            )

        wires = []
        for row in range(self.wire_table.rowCount()):
            items = [self.wire_table.item(row, col) for col in range(3)]
            if any(item is None for item in items):
                raise ValueError(f"Wire row {row + 1} is incomplete")
            x, y, current = (float(item.text()) for item in items)
            wires.append({"x_mm": x, "y_mm": y, "current_A": current})
        if not wires:
            raise ValueError("At least one external wire is required")

        p: dict[str, Any] = {
            "shape": self.shape_combo.currentText(),
            "conductivity_Sm": self.conductivity_input.value(),
            "thickness_um": self.thickness_input.value(),
            "pulse_length_us": self.pulse_length_input.value(),
            "pulse_type": self.pulse_type_combo.currentText(),
            "frequency_cutoff_C": self.cutoff_input.value(),
            "frequency_points": nfreq,
            "time_points": self.time_points_input.value(),
            "x_axis_points": self.x_axis_points_input.value(),
            "fourier_order": self.order_input.value(),
            "boundary_points": self.boundary_points_input.value(),
            "polynomial_fit_order": self.fit_order_input.value(),
            # "run_bem_validation": (
            #     self.shape_combo.currentText() == "Ellipse"
            #     and self.bem_check_input.isChecked()
            # ),
            # "bem_boundary_points": self.bem_panels_input.value(),
            "run_bem_validation": False,
            "bem_boundary_points": 160,
            "wires": wires,
        }
        shape = p["shape"]
        if shape == "Circular":
            p["radius_mm"] = self.radius_input.value()
        elif shape == "Ellipse":
            p["a_mm"] = self.a_input.value()
            p["b_mm"] = self.b_ellipse_input.value()
            if p["a_mm"] <= p["b_mm"]:
                raise ValueError("Ellipse requires a > b. For a circle, select Circular.")
            # if p["run_bem_validation"]:
            #     minimum_bem_panels = 8 * (p["fourier_order"] + 1)
            #     if p["bem_boundary_points"] < minimum_bem_panels:
            #         raise ValueError(
            #             "BEM boundary panels must be at least "
            #             f"8*(Fourier order+1) = {minimum_bem_panels}."
            #         )
        elif shape == "Flat pipe (only for 180 degree rotation symmetry)":
            p["half_gap_mm"] = self.flat_gap_input.value()/ 2.0
            p["spatial_range_mm"] = self.flat_range_input.value()/ 2.0
            p["spatial_points"] = self.flat_points_input.value()
        return p

    def start_calculation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            parameters = self.collect_parameters()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        self.active_parameters = parameters
        self.report_paths = {}
        self.view_results_button.setEnabled(False)
        self.calculate_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Calculating {parameters['shape']} analytical solution...")
        self.worker = CalculationWorker(parameters, self)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.completed.connect(self.calculation_completed)
        self.worker.failed.connect(self.calculation_failed)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def calculation_completed(self, result) -> None:
        self.last_result = result
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            csv_paths = result.save_csv(self.output_dir)
            p = self.active_parameters or self.collect_parameters()
            # bem_comparison = getattr(result, "bem_comparison", None)
            # if bem_comparison is not None:
            #     csv_paths.extend(bem_comparison.save_csv(self.output_dir, result))
            self.report_paths = write_result_reports(
                result,
                p,
                self.output_dir,
                fit_order=p["polynomial_fit_order"],
            )
            with (self.output_dir / "run_parameters.json").open("w", encoding="utf-8") as stream:
                json.dump(p, stream, indent=2, ensure_ascii=False)

            status_text = (
                f"Completed: {p['shape']}, {len(result.omega_rad_s)} temporal frequency points. "
                f"Saved {len(csv_paths)} CSV files and {len(self.report_paths)} interactive HTML reports."
            )
            # bem_error = getattr(result, "bem_comparison_error", None)
            # if bem_comparison is not None:
            #     status_text += (
            #         " Green's-function cross-check completed "
            #         f"({bem_comparison.panel_count} BEM panels)."
            #     )
            # if bem_error:
            #     status_text += f" Green's-function cross-check failed: {bem_error}"
            # self.status_label.setText(status_text)
            # if bem_error:
            #     QMessageBox.warning(
            #         self,
            #         "Green's-function cross-check failed",
            #         "The analytical result was completed and saved, but the optional "
            #         f"BEM cross-check failed:\n\n{bem_error}",
            #     )
            self.status_label.setText(status_text)
            self.view_results_button.setEnabled(True)
            self.open_results_window()
        except Exception as exc:
            logger.exception("Failed to save or plot results")
            QMessageBox.critical(self, "Output error", str(exc))

    def open_results_window(self) -> None:
        if not self.report_paths:
            QMessageBox.information(self, "No results", "Please complete a calculation first.")
            return
        if self.result_window is None:
            self.result_window = ResultWindow(self)
        self.result_window.set_reports(self.report_paths)
        self.result_window.show()
        self.result_window.raise_()
        self.result_window.activateWindow()

    def calculation_failed(self, message: str) -> None:
        logger.error("Calculation failed:\n%s", message)
        QMessageBox.critical(self, "Calculation failed", message)
        self.status_label.setText("Calculation failed. See the log for details.")

    def worker_finished(self) -> None:
        self.calculate_button.setEnabled(True)
        self.open_output_button.setEnabled(True)
        self.worker = None


def run_gui() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
