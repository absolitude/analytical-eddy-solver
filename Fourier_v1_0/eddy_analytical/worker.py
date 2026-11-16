from __future__ import annotations

import traceback
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from .physics import Wire
from .solvers.circular import calculate_circular_case
from .solvers.elliptical import calculate_elliptical_case
from .solvers.flat_pipe import calculate_flat_pipe_case
# from .bem_validation import compare_elliptical_solution_to_bem


class CalculationWorker(QThread):
    progress = pyqtSignal(int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, parameters: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.parameters = dict(parameters)

    def run(self) -> None:
        try:
            self.progress.emit(0)
            p = self.parameters
            wires = [Wire(**item) for item in p["wires"]]
            self.progress.emit(5)
            common = dict(
                conductivity_Sm=p["conductivity_Sm"],
                thickness_um=p["thickness_um"],
                pulse_length_us=p["pulse_length_us"],
                frequency_cutoff_C=p["frequency_cutoff_C"],
                frequency_points=p["frequency_points"],
                time_points=p["time_points"],
                x_axis_points=p.get("x_axis_points"),
                wires=wires,
                pulse_type=p["pulse_type"],
                progress=self.progress.emit,
            )
            shape = p["shape"]
            if shape == "Circular":
                result = calculate_circular_case(
                    radius_mm=p["radius_mm"],
                    fourier_order=p["fourier_order"],
                    boundary_points=p["boundary_points"],
                    **common,
                )
            elif shape == "Ellipse":
                # run_bem = bool(p.get("run_bem_validation", False))
                # if run_bem:
                #     # Reserve the final part of the progress bar for the
                #     # independent boundary-integral cross-check.
                #     analytical_common = dict(common)
                #     analytical_common["progress"] = lambda value: self.progress.emit(
                #         int(round(0.65 * value))
                #     )
                # else:
                #     analytical_common = common

                result = calculate_elliptical_case(
                    a_mm=p["a_mm"],
                    b_mm=p["b_mm"],
                    fourier_order=p["fourier_order"],
                    boundary_points=p["boundary_points"],
                    **common,
                )

                # if run_bem:
                #     try:
                #         comparison = compare_elliptical_solution_to_bem(
                #             analytical_result=result,
                #             wires=wires,
                #             conductivity_Sm=p["conductivity_Sm"],
                #             thickness_um=p["thickness_um"],
                #             panel_count=p["bem_boundary_points"],
                #             progress=lambda value: self.progress.emit(
                #                 65 + int(round(0.35 * value))
                #             ),
                #         )
                #         result.bem_comparison = comparison
                #         result.bem_comparison_error = None
                #     except Exception as exc:
                #         # A failed independent check should not discard a
                #         # successfully completed analytical calculation.
                #         result.bem_comparison = None
                #         result.bem_comparison_error = str(exc)
                #         self.progress.emit(100)
            elif shape == "Flat pipe (only for 180 degree rotation symmetry)":
                result = calculate_flat_pipe_case(
                    half_gap_mm=p["half_gap_mm"],
                    spatial_range_mm=p["spatial_range_mm"],
                    spatial_points=p["spatial_points"],
                    **common,
                )
            else:
                raise ValueError(f"Unsupported boundary shape: {shape}")
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
