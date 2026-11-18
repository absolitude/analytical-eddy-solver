# Green Eddy-Current Solver
# Analytical calculation of eddy currents in thin conductive coatings
# of accelerator vacuum chambers.
#
# Copyright (C) 2026 Hao-Wen Luo.
# Copyright (C) 2026 National Synchrotron Radiation Research Center (NSRRC).
#
# This file is part of Green Eddy-Current Solver.
#
# Green Eddy-Current Solver is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 3.
#
# Green Eddy-Current Solver is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Green Eddy-Current Solver.
# If not, see <https://www.gnu.org/licenses/>.

import sys
import numpy as np
import pandas as pd
from scipy import linalg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QLineEdit, QComboBox, QPushButton, QProgressBar,
                             QTableWidget, QTableWidgetItem, QSpinBox, QLabel,QFileDialog, QCheckBox,QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor
import webbrowser
import os
import math
import logging
import hashlib
import json

from thin_wall_eddy_solver import (
    component_from_point_dicts,
    make_parallel_plate_components,
    load_boundary_csv,
    build_boundary_mesh,
    build_log_integral_matrix,
    vector_potential_from_wires,
    solve_frequency,
    SpectrumSolution,
    build_B_field_kernel,
    inverse_fourier,
    half_sine_pulse_spectrum,
    rectangular_pulse_spectrum,
)


# 設置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('eddy_current.log', encoding='utf-8'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# 常數
mu_0 = 4 * np.pi * 1e-7  # 真空磁導率 (H/m)

class ComputeThread(QThread):
    progress = pyqtSignal(int)
    calc_progress = pyqtSignal(int)  # Calculate
    post_progress = pyqtSignal(int) # Post-process
    pre_finished = pyqtSignal(object, object, object, object, object)
    calc_finished = pyqtSignal(object)
    post_finished = pyqtSignal(
        object, object, object, object,
        object, object, object, object
    )
    calc_finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, params, task="pre", task_args=()):
        super().__init__()
        self.params = params
        self.task = task
        self.task_args = task_args

    def run(self):

        try:

            if self.task == "pre":

                self.pre_process()

            elif self.task == "calculate":

                # 保留舊的獨立 Calculate 功能，方便測試
                self.calculate(*self.task_args)

            elif self.task == "post":

                # 保留舊的獨立 Post-process 功能，方便測試
                self.post_process(*self.task_args)

            elif self.task == "full":

                # 取得全部需要的輸入資料
                mesh, drive_A, omega, f = self.task_args

                logger.info("Starting full calculation pipeline")

                # -----------------------------------------
                # Stage 1: Calculate
                # -----------------------------------------

                solution = self.calculate(
                    mesh,
                    drive_A,
                    omega
                )

                # 如果 Calculate 失敗，就停止，不執行 Post-process
                if solution is None:
                    return

                logger.info("Calculate stage completed")

                # -----------------------------------------
                # Stage 2: Post-process
                # -----------------------------------------

                self.post_process(
                    solution,
                    mesh,
                    omega,
                    f
                )

                logger.info("Full calculation pipeline completed")

            else:

                raise ValueError(
                    f"Unknown task: {self.task}"
                )

        except Exception as e:

            import traceback

            logger.error(
                f"Worker error:\n{traceback.format_exc()}"
            )

            self.error.emit(
                f"Worker error: {str(e)}"
            )

    def get_custom_boundary(self, file_path):
        # 讀取 CSV
        df = pd.read_csv(file_path)
        # 轉換為 points 格式
        points = [{'x': row['x'], 'y': row['y']} for _, row in df.iterrows()]
        return points

    def calculate_generic_weights(self, points, is_closed=True):
        # 1. 轉成 Numpy 陣列加速： (N, 2)
        coords = np.array([[p['x'], p['y']] for p in points])
        n = len(coords)
        
        # 2. 計算點與點之間的向量與距離
        # diffs[i] 代表從點 i 到點 i+1 的距離
        diffs = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        
        if is_closed:
            # 封閉路徑：補上最後一點到第一點的距離，形成一個環
            dist_last_to_first = np.sqrt(np.sum((coords[-1] - coords[0])**2))
            full_diffs = np.append(diffs, dist_last_to_first)
            
            # 權重是「前一段距離的一半」加「後一段距離的一半」
            # np.roll(full_diffs, 1) 會把 [d1, d2, d3, d_last] 變成 [d_last, d1, d2, d3]
            weights = (full_diffs + np.roll(full_diffs, 1)) / 2
        else:
            # 非封閉路徑 (如平行板)：頭尾點權重只有一邊
            weights = np.zeros(n)
            weights[0] = diffs[0] / 2
            weights[-1] = diffs[-1] / 2
            # 中間點權重是前後線段各一半
            weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2
            
        return weights  # 單位換算 (mm -> m)
    
    def handle_load_file(self):
        import os
        # 1. 直接移除 options = QFileDialog.Options() 這行

        # 2. 修改 getOpenFileName 的呼叫方式
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Boundary File", 
            "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
            # 這裡不需要傳入 options 參數
        )
        
        if file_path:
            self.custom_file_path = file_path
            # 更新顯示讀取成功的訊息
            print(f"Successfully loaded: {file_path}")
            # 如果你有一個 label 用來顯示路徑，可以在這更新
            # self.status_label.setText(f"File: {os.path.basename(file_path)}")


    def pre_process(self):
        try:
            self.progress.emit(0)
            p = self.params
            shape = p['shape']
            a, b = p['a'], p['b']
            T = p['pulse_length'] * 1e-6
            C = p['C']
            nopfd = p['nopfd']

            omega_max = C * np.pi / T
            omega = np.linspace(-omega_max, omega_max, nopfd)

            if p['pulse_type'] == 'half_sine':
                f = half_sine_pulse_spectrum(omega, T)
            else:
                f = rectangular_pulse_spectrum(omega, T)

            # 先建立「幾何 component」，再離散成 panels。
            # 不可把上下平行板合併成一條 open polyline，否則會錯誤連接端點。
            if shape == 'parallel plates':
                components = make_parallel_plate_components(
                    width_mm=a,
                    gap_mm=b,
                    points_per_plate=p['nophor_paraplates'],
                )
                discretized_points = [
                    {'x': float(x), 'y': float(y)}
                    for comp in components
                    for x, y in comp.points_mm
                ]

            elif shape == 'custom':
                custom_path = p.get('custom_path')
                if not custom_path:
                    raise ValueError('Please load a boundary file first!')
                # x,y 兩欄的舊 CSV 預設是一條封閉曲線；
                # 新格式可用 component/closed/current_group 等欄位表示多段邊界。
                components = load_boundary_csv(custom_path)
                discretized_points = [
                    {'x': float(x), 'y': float(y)}
                    for comp in components
                    for x, y in comp.points_mm
                ]

            else:
                discretized_points = self.get_discretized_boundary(
                    shape, a, b,
                    p['nopell'], p['nophor_rect'],
                    p['nophor_paraplates'], p['nopver'], p['nophfc'],
                    p.get('custom_path')
                )
                components = [
                    component_from_point_dicts(
                        discretized_points,
                        closed=True,
                        name=shape,
                    )
                ]

            mesh = build_boundary_mesh(components)
            if mesh.n_panels < 1:
                raise ValueError('Boundary mesh contains no panels')

            # 外加線電流的 Az 在 panel collocation points 評估。
            drive_A = vector_potential_from_wires(
                mesh.midpoints_m,
                p['wires'],
            )

            omega_list = [
                {'omega': float(w), 'f': float(f[i])}
                for i, w in enumerate(omega)
            ]

            self.progress.emit(100)
            self.pre_finished.emit(
                omega_list, f, discretized_points, mesh, drive_A
            )

        except Exception as exc:
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(f"Pre-process error: {exc}")

    def calculate(self, mesh, drive_A, omega):
        try:
            self.calc_progress.emit(0)
            omega = np.asarray(omega, dtype=float)
            drive_A = np.asarray(drive_A, dtype=complex)

            # 幾何 Green kernel 與頻率無關，只建立一次。
            H = build_log_integral_matrix(mesh)
            nfreq = len(omega)
            npanels = mesh.n_panels

            K_all = np.zeros((nfreq, npanels), dtype=complex)
            phi_all = np.zeros_like(K_all)
            offsets_all = np.zeros(
                (nfreq, mesh.n_constraints), dtype=complex
            )

            conductivity = float(self.params['conductivity'])
            thickness_m = float(self.params['thickness']) * 1e-6

            last_progress = 0

            for k, w in enumerate(omega):

                result = solve_frequency(
                    w,
                    drive_A,
                    mesh,
                    H,
                    conductivity,
                    thickness_m,
                )

                K_all[k, :] = result.current_density
                phi_all[k, :] = result.induced_A
                offsets_all[k, :] = result.A_offsets

                progress = int((k + 1) / nfreq * 100)

                # 只有百分比改變才發出 signal
                if progress > last_progress:
                    self.calc_progress.emit(progress)
                    last_progress = progress

            solution = SpectrumSolution(
                omega=omega,
                drive_A=drive_A,
                current_density=K_all,
                induced_A=phi_all,
                A_offsets=offsets_all,
                log_kernel=H,
            )

            logger.info(
                "New solver finished: %d frequencies, %d panels, %d constraints",
                nfreq, npanels, mesh.n_constraints
            )
            self.calc_finished.emit(solution)
            return solution

        except Exception as exc:
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(f"Calculate error: {exc}")

    def compute_time_domain_by_ploy(self, sol, discretized_points, weights, omega, f, Az, params):
        try:
            logger.info("Starting compute_time_domain_by_ploy")
            shape = params['shape']
            a = params['a']
            thickness = params['thickness'] * 1e-6
            conductivity = params['conductivity']
            T = params['pulse_length'] * 1e-6
            C = params['C']
            nopfd = params['nopfd']
            noptd = params['noptd']
            nfts = params['nfts']
            nopxa = params['nopxa']
            last_progress = -1

            # x 軸範圍
            if shape == 'ellipse':
                x_range = a
            else:
                x_range = a / 2
            listxa = np.linspace(-0.9 * x_range, 0.9 * x_range, nopxa)
            logger.debug(f"x-axis range: {listxa[0]:.2f} to {listxa[-1]:.2f}, nopxa={nopxa}")

            # 檢查內存需求
            if nopxa * nopfd > 1e7 or nopxa * noptd > 1e7:
                logger.warning(f"Large array sizes: listFSploy={nopxa*nopfd}, by_time_ploy={nopxa*noptd}. Consider reducing parameters.")
                self.error.emit("Array sizes too large. Please reduce nopxa, nopfd, or noptd.")
                return [], [], [], []

            # 初始化數組
            listFSploy = np.zeros((nopxa, nopfd), dtype=complex)
            by_time_ploy = np.zeros((nopxa, noptd))
            logger.debug(f"listFSploy shape: {listFSploy.shape}, by_time_ploy shape: {by_time_ploy.shape}")

            # 計算頻域 By
            logger.info("Computing frequency domain By")

            # 1. 預先將 discretized_points 轉為 NumPy 陣列 (只算一次)
            px = np.array([p['x'] for p in discretized_points]) / 1000  # 轉為公尺
            py = np.array([p['y'] for p in discretized_points]) / 1000
            w = np.array(weights)

            # 2. 將觀察點 listxa 也轉為陣列
            xa = np.array(listxa) / 1000  # 轉為公尺

            # 3. 利用廣播計算所有 (i, j) 組合的 dx 與 r (這是幾何常數矩陣)
            # xa[:, np.newaxis] 形態為 (nopxa, 1)
            # px[np.newaxis, :] 形態為 (1, num_points)
            DX = xa[:, np.newaxis] - px[np.newaxis, :]  # 結果是 (nopxa, num_points) 的矩陣
            DY = py[np.newaxis, :]                      # (1, num_points)
            R = DX**2 + DY**2
            R = np.maximum(R, 1e-10)                    # 防止除以零

            # 4. 預算幾何因子： weights * dx / r
            # 這是固定不變的物理結構因子，大小為 (nopxa, num_points)
            geometry_factor = (w[np.newaxis, :] * DX) / R

            # 5. 開始頻率迴圈 (只剩一層 k)
            for k in range(nopfd):
                phi_k = np.array(sol[k])
                # 計算 (phi + Az) * weights * dx / r
                # 利用矩陣與向量相乘 (Dot product) 一次算完所有觀察點 i
                # (nopxa, num_points) @ (num_points,) -> (nopxa,)
                source_term = phi_k + Az
                sum_vals = geometry_factor @ source_term
                
                # 計算係數並存入結果
                coeff = -1j * thickness * mu_0 * conductivity * omega[k] / (2 * np.pi)
                listFSploy[:, k] = coeff * sum_vals
                
                progress = int(k / nopfd * 50)
                if progress != last_progress:
                    self.post_progress.emit(progress)
                    last_progress = progress
                    QThread.msleep(10)
            logger.debug(f"listFSploy min={np.min(listFSploy):.2e}, max={np.max(listFSploy):.2e}")
            finite_count = np.sum(np.isfinite(listFSploy))
            logger.debug(f"listFSploy finite values: {finite_count}/{listFSploy.size}")
            if finite_count < listFSploy.size:
                non_finite_indices = np.where(~np.isfinite(listFSploy))
                logger.warning(f"Non-finite values in listFSploy at indices: {list(zip(non_finite_indices[0], non_finite_indices[1]))}")

            # 計算時域 By
            logger.info("Computing time domain By")

            complex_signal = f * listFSploy
            t = np.linspace(-T, T, noptd)
            for i in range(nopxa):
                phase = -np.outer(omega, t)
                S = complex_signal[i, :]
                re_S = np.real(S)[:, np.newaxis]
                im_S = np.imag(S)[:, np.newaxis]
                
                # 一次計算該位置 i 的所有時間點
                combined = re_S * np.cos(phase) + im_S * np.sin(phase)
                by_time_ploy[i, :] = (C / (T * nopfd)) * np.sum(combined, axis=0)

                last_progress = 50
                progress = int(50 + i / nopxa * 50)
                if progress != last_progress:
                    self.post_progress.emit(progress)
                    last_progress = progress
                    QThread.msleep(10)
            logger.debug(f"by_time_ploy min={np.min(by_time_ploy):.2e}, max={np.max(by_time_ploy):.2e}")

            # 多項式擬合（修正：固定時間，對 x 擬合）
            logger.info("Performing polynomial fitting on x for each time point")
            by_coeffs_ploy = []  # 每個時間點的係數，len = noptd
            t = np.linspace(-T, T, noptd)

            for ti_idx in range(noptd):
                y_data = by_time_ploy[:, ti_idx]  # 固定時間 ti_idx 的 B_y(x)
                
                if not np.all(np.isfinite(y_data)):
                    logger.warning(f"Non-finite values at time index {ti_idx}")
                    by_coeffs_ploy.append([0.0] * (nfts + 1))
                    continue
                
                if np.max(np.abs(y_data)) < 1e-10:
                    logger.debug(f"Near zero field at time index {ti_idx}")
                    by_coeffs_ploy.append([0.0] * (nfts + 1))
                    continue
                
                try:
                    # 對 x 擬合，x 單位轉 m（物理正確）
                    coeffs = np.polyfit(listxa / 1000, y_data, nfts)
                    by_coeffs_ploy.append(coeffs.tolist())
                except np.linalg.LinAlgError:
                    logger.warning(f"Fitting failed at time index {ti_idx}")
                    by_coeffs_ploy.append([0.0] * (nfts + 1))
                except Exception as e:
                    logger.error(f"Error in fitting at time index {ti_idx}: {e}")
                    by_coeffs_ploy.append([0.0] * (nfts + 1))
            logger.debug(f"by_coeffs_ploy length={len(by_coeffs_ploy)}")

            logger.info("compute_time_domain_by_ploy completed")

            by_time_ploy_hash = hashlib.md5(by_time_ploy.tobytes()).hexdigest()
            logger.debug(f"by_time_ploy hash before return: {by_time_ploy_hash}")
            finite_count = np.sum(np.isfinite(by_time_ploy))
            logger.debug(f"by_time_ploy finite values: {finite_count}/{by_time_ploy.size}")
            if finite_count < by_time_ploy.size:
                non_finite_indices = np.where(~np.isfinite(by_time_ploy))
                logger.warning(f"Non-finite values in by_time_ploy at indices: {list(zip(non_finite_indices[0], non_finite_indices[1]))}")
            return listxa, listFSploy, by_time_ploy, by_coeffs_ploy
        except Exception as e:
            logger.error(f"compute_time_domain_by_ploy error: {str(e)}", exc_info=True)
            self.error.emit(f"compute_time_domain_by_ploy error: {str(e)}")
            return [], [], [], []

    def compute_ellipse_fourier(self, values, order):
        """
        Fourier decomposition of a complex quantity distributed
        on an elliptical boundary.

        Parameters
        ----------
        values:
            Complex array, shape (N_frequency, N_panels).
            In this application, values = solution.induced_A.

        order:
            Maximum Fourier order.

        Returns
        -------
        cos_coeffs:
            Shape (order+1, N_frequency).

        sin_coeffs:
            Shape (order+1, N_frequency).
        """

        values = np.asarray(values, dtype=complex)

        if values.ndim != 2:
            raise ValueError("values must be a 2D array")

        nfreq, npanels = values.shape

        if npanels < 3:
            raise ValueError("Not enough panels for Fourier analysis")

        max_order = (npanels - 1) // 2

        if order < 0 or order > max_order:
            raise ValueError(
                f"Fourier order must be between 0 and {max_order}"
            )

        # Ellipse panel midpoint angles.
        # Original ellipse vertices are equally spaced in theta.
        # The panel midpoint corresponds to theta_i + delta_theta/2.
        theta = (
            np.arange(npanels) + 0.5
        ) * (2 * np.pi / npanels)

        # Fourier modes
        m = np.arange(order + 1)

        phase = np.outer(m, theta)

        cos_matrix = np.cos(phase)
        sin_matrix = np.sin(phase)

        # Keep the original normalization: 2/N
        cos_coeffs = (
            2.0 / npanels
        ) * (values @ cos_matrix.T).T

        # Correct normalization for m = 0
        cos_coeffs[0, :] *= 0.5

        sin_coeffs = (
            2.0 / npanels
        ) * (values @ sin_matrix.T).T

        # The zeroth sine coefficient is exactly zero.
        sin_coeffs[0, :] = 0.0

        return cos_coeffs, sin_coeffs

    def save_boundary_panel_geometry(self, mesh, output_path="boundary_panels.csv"):
        """Save the exact panel geometry used by the Green solver.

        Panel index follows the column order of SpectrumSolution.current_density.
        Coordinates are exported in mm; panel length is also exported in mm.
        """
        n_panels = mesh.n_panels
        component_ids = np.asarray(mesh.component_ids, dtype=int)
        panel_groups = list(mesh.panel_current_group)
        constrained_groups = set(mesh.constraint_labels)

        if len(component_ids) != n_panels or len(panel_groups) != n_panels:
            raise ValueError("Boundary mesh metadata length does not match panel count")

        component_names = []
        for component_id in component_ids:
            if 0 <= component_id < len(mesh.component_names):
                component_names.append(str(mesh.component_names[component_id]))
            else:
                component_names.append(f"component_{component_id}")

        starts_mm = np.asarray(mesh.starts_m, dtype=float) * 1e3
        ends_mm = np.asarray(mesh.ends_m, dtype=float) * 1e3
        midpoints_mm = np.asarray(mesh.midpoints_m, dtype=float) * 1e3
        lengths_mm = np.asarray(mesh.lengths_m, dtype=float) * 1e3

        panel_df = pd.DataFrame({
            "panel_index": np.arange(n_panels, dtype=int),
            "component_index": component_ids,
            "component": component_names,
            "current_group": [str(group) for group in panel_groups],
            "zero_net_current_constraint": [group in constrained_groups for group in panel_groups],
            "x_start_mm": starts_mm[:, 0],
            "y_start_mm": starts_mm[:, 1],
            "x_end_mm": ends_mm[:, 0],
            "y_end_mm": ends_mm[:, 1],
            "x_mid_mm": midpoints_mm[:, 0],
            "y_mid_mm": midpoints_mm[:, 1],
            "length_mm": lengths_mm,
        })
        panel_df.to_csv(output_path, index=False)
        logger.info("Saved %s (%d boundary panels)", output_path, n_panels)
        return panel_df

    def save_boundary_current_time(
        self, solution, mesh, omega, pulse_spectrum, time_s,
        output_path="boundary_current_time.csv"
    ):
        """Reconstruct K_z(panel, t) from the saved frequency-domain solution.

        This uses the same two-sided inverse Fourier routine and the same pulse
        spectrum/time grid as the beam-axis field calculation. No interpolation
        is applied: each current column remains associated with its original panel.
        """
        omega = np.asarray(omega, dtype=float).reshape(-1)
        pulse_spectrum = np.asarray(pulse_spectrum, dtype=complex).reshape(-1)
        time_s = np.asarray(time_s, dtype=float).reshape(-1)
        K_frequency = np.asarray(solution.current_density, dtype=complex)

        expected_shape = (len(omega), mesh.n_panels)
        if K_frequency.shape != expected_shape:
            raise ValueError(
                f"Current spectrum shape {K_frequency.shape} does not match {expected_shape}"
            )
        if pulse_spectrum.shape != (len(omega),):
            raise ValueError("Pulse spectrum length does not match frequency grid")
        if len(time_s) < 1:
            raise ValueError("Time grid must contain at least one point")
        if not (np.all(np.isfinite(omega)) and np.all(np.isfinite(pulse_spectrum))
                and np.all(np.isfinite(time_s)) and np.all(np.isfinite(K_frequency))):
            raise ValueError("Current reconstruction input contains non-finite values")

        # inverse_fourier returns (N_time, N_panel); transpose to one row per panel.
        K_time = inverse_fourier(
            omega,
            pulse_spectrum[:, None] * K_frequency,
            time_s,
        ).T

        if K_time.shape != (mesh.n_panels, len(time_s)):
            raise ValueError(
                f"Reconstructed current shape {K_time.shape} is not "
                f"({mesh.n_panels}, {len(time_s)})"
            )

        # Include panel metadata in the same table so panel columns cannot become
        # detached from the geometric locations when the CSV is used later.
        component_ids = np.asarray(mesh.component_ids, dtype=int)
        panel_groups = list(mesh.panel_current_group)
        constrained_groups = set(mesh.constraint_labels)
        component_names = [
            str(mesh.component_names[i]) if 0 <= i < len(mesh.component_names)
            else f"component_{i}"
            for i in component_ids
        ]
        starts_mm = np.asarray(mesh.starts_m, dtype=float) * 1e3
        ends_mm = np.asarray(mesh.ends_m, dtype=float) * 1e3
        midpoints_mm = np.asarray(mesh.midpoints_m, dtype=float) * 1e3
        lengths_mm = np.asarray(mesh.lengths_m, dtype=float) * 1e3

        data = {
            "panel_index": np.arange(mesh.n_panels, dtype=int),
            "component_index": component_ids,
            "component": component_names,
            "current_group": [str(group) for group in panel_groups],
            "zero_net_current_constraint": [group in constrained_groups for group in panel_groups],
            "x_start_mm": starts_mm[:, 0],
            "y_start_mm": starts_mm[:, 1],
            "x_end_mm": ends_mm[:, 0],
            "y_end_mm": ends_mm[:, 1],
            "x_mid_mm": midpoints_mm[:, 0],
            "y_mid_mm": midpoints_mm[:, 1],
            "length_mm": lengths_mm,
        }
        for it, time_value in enumerate(time_s):
            # 9-digit scientific notation avoids duplicate headings at normal
            # GUI time resolutions while preserving a readable time label.
            data[f"K_t_{time_value:.9e}_s"] = K_time[:, it]

        pd.DataFrame(data).to_csv(output_path, index=False)
        logger.info(
            "Saved %s: %d panels x %d time points (K_z in A/m)",
            output_path, mesh.n_panels, len(time_s)
        )
        return K_time

    def post_process(self, solution, mesh, omega, f):
        try:
            self.post_progress.emit(0)
            params = self.params
            shape = params['shape']
            T = params['pulse_length'] * 1e-6
            noptd = params['noptd']
            nopxa = params['nopxa']
            nfts = params['nfts']

            # Export the panel geometry and time-domain boundary current from
            # the already-computed Green solution. These are diagnostic/output
            # files only; they do not modify the solver result or field calculation.
            t = np.linspace(-T, T, noptd)
            self.save_boundary_panel_geometry(mesh)
            self.save_boundary_current_time(solution, mesh, omega, f, t)

            if shape == 'ellipse':
                x_range = params['a']
            else:
                x_range = params['a'] / 2.0
            listxa = np.linspace(-0.9 * x_range, 0.9 * x_range, nopxa)

            # 觀察點位於 y=0，單位轉成 m。
            obs = np.column_stack((listxa * 1e-3, np.zeros(nopxa)))
            _, By_kernel = build_B_field_kernel(mesh, obs)

            # K(omega) -> induced By(omega)，shape: (n_frequency, n_x)
            By_omega = solution.current_density @ By_kernel.T

            # 套用外加 pulse 的頻譜，再做 inverse Fourier transform。
            t = np.linspace(-T, T, noptd)
            By_time_t_x = inverse_fourier(
                omega,
                np.asarray(f)[:, None] * By_omega,
                t,
            )
            by_time_ploy = By_time_t_x.T  # shape: (n_x, n_time)
            listFSploy = By_omega.T        # shape: (n_x, n_frequency)

            # 對每個時間點的 By(x) 做 polynomial fit；x 軸使用 m。
            by_coeffs_ploy = []
            for it in range(noptd):
                ydata = by_time_ploy[:, it]
                if not np.all(np.isfinite(ydata)):
                    by_coeffs_ploy.append([0.0] * (nfts + 1))
                    continue
                if len(listxa) < nfts + 1:
                    raise ValueError(
                        f"nopxa={len(listxa)} is insufficient for polynomial order {nfts}"
                    )
                coeffs = np.polyfit(listxa * 1e-3, ydata, nfts)
                by_coeffs_ploy.append(coeffs.tolist())

            # =====================================================
            # Ellipse Fourier Analysis
            # =====================================================

            listFS = []
            listTS = []

            listFS_sin = []
            listTS_sin = []

            if shape == 'ellipse':

                logger.info("Starting ellipse Fourier analysis")

                # Check geometry
                if len(mesh.component_names) != 1:
                    raise ValueError(
                        "Ellipse Fourier analysis requires one boundary component"
                    )

                if mesh.n_panels != params['nopell']:
                    raise ValueError(
                        "Ellipse panel number does not match nopell"
                    )

                # -------------------------------------------------
                # 1. Fourier decomposition of induced vector potential
                # -------------------------------------------------

                A_cos, A_sin = self.compute_ellipse_fourier(
                    solution.induced_A,
                    params['order']
                )

                logger.info(
                    f"Ellipse Fourier decomposition finished: "
                    f"order={params['order']}, "
                    f"frequency points={len(omega)}"
                )

                for m in range(params['order'] + 1):

                    cos_amp = np.max(np.abs(A_cos[m]))
                    sin_amp = np.max(np.abs(A_sin[m]))

                    logger.info(
                        f"Fourier m={m}: "
                        f"Cos max={cos_amp:.4e}, "
                        f"Sin max={sin_amp:.4e}"
                    )

                # A_cos and A_sin:
                # shape = (order+1, N_frequency)

                # -------------------------------------------------
                # 2. Time-domain reconstruction
                # -------------------------------------------------

                t = np.linspace(-T, T, noptd)

                # Multiply by external pulse spectrum
                S_cos = np.asarray(f)[:, None] * A_cos.T
                S_sin = np.asarray(f)[:, None] * A_sin.T

                # Inverse Fourier transform
                # -------------------------------------------------
                # 2. Time-domain reconstruction
                # -------------------------------------------------

                A_cos_time = inverse_fourier(
                    omega,
                    S_cos,
                    t
                ).T

                # Check whether sine coefficients are physically significant
                cos_scale = np.max(np.abs(S_cos))
                sin_scale = np.max(np.abs(S_sin))

                sin_ratio = sin_scale / max(cos_scale, 1e-30)

                logger.info(
                    f"Fourier sine/cosine spectrum ratio = {sin_ratio:.3e}"
                )

                # -------------------------------------------------
                # Sine coefficients may theoretically vanish
                # for symmetric wire configurations.
                # Avoid rejecting numerical round-off errors.
                # -------------------------------------------------

                if sin_ratio < 1e-10:

                    A_sin_time = np.zeros_like(A_cos_time)

                    logger.info(
                        "Sine coefficients are negligible. "
                        "Set time-domain sine coefficients to zero."
                    )

                else:

                    A_sin_time = inverse_fourier(
                        omega,
                        S_sin,
                        t
                    ).T

                # -------------------------------------------------
                # 3. Convert to list format for existing GUI
                # -------------------------------------------------

                listFS = A_cos.tolist()
                listFS_sin = A_sin.tolist()

                listTS = A_cos_time.tolist()
                listTS_sin = A_sin_time.tolist()

                logger.info("Ellipse Fourier time reconstruction finished")

            # =====================================================
            # Finish
            # =====================================================

            self.post_progress.emit(100)

            self.post_finished.emit(
                listFS,
                listTS,
                listxa,
                listFSploy,
                by_time_ploy,
                by_coeffs_ploy,
                listFS_sin,
                listTS_sin
            )

        except Exception as exc:
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(f"Post-process error: {exc}")


    def get_discretized_boundary(self, shape, a, b, nopell, nophor, nophor_paraplates, nopver, nophfc, custom_path=None):
        points = []
        if shape == 'ellipse':
            theta = np.linspace(0, 2 * np.pi, nopell, endpoint=False)
            points = [{'x': a * np.cos(t), 'y': b * np.sin(t), 'theta': t} for t in theta]
        elif shape == 'rectangle':
            if nophor < 3 or nopver < 3:
                raise ValueError("nophor must be >= 3 and nopver >= 3 for rectangle")
            x_top = np.linspace(-a/2, a/2, nophor)[1:-1]
            y_right = np.linspace(b/2, -b/2, nopver)
            x_bottom = np.linspace(a/2, -a/2, nophor)[1:-1]
            y_left = np.linspace(-b/2, b/2, nopver)
            points = ([{'x': x, 'y': b/2} for x in x_top] +
                      [{'x': a/2, 'y': y} for y in y_right] +
                      [{'x': x, 'y': -b/2} for x in x_bottom] +
                      [{'x': -a/2, 'y': y} for y in y_left])
        elif shape == 'parallel plates':
            if nophor_paraplates < 3:
                raise ValueError("nophor must be >= 3  for parallel plates")
            x_top = np.linspace(-a/2, a/2, nophor_paraplates)
            x_bottom = np.linspace(a/2, -a/2, nophor_paraplates)
            points = ([{'x': x, 'y': b/2} for x in x_top] +
                      [{'x': x, 'y': -b/2} for x in x_bottom])
        elif shape == 'racetrack':
            if nophor < 3 or nophfc < 3:
                raise ValueError("nophor must be >= 3 and nophfc >= 3 for racetrack")
            straight_len = (a - b) / 2
            theta_right = np.linspace(-np.pi/2, np.pi/2, nophfc)
            x_top = np.linspace(straight_len, -straight_len, nophor)[1:-1]
            theta_left = np.linspace(np.pi/2, 3*np.pi/2, nophfc)
            x_bottom = np.linspace(-straight_len, straight_len, nophor)[1:-1]
            points = ([{'x': straight_len + (b/2)*np.cos(t), 'y': (b/2)*np.sin(t)} for t in theta_right] +
                      [{'x': x, 'y': b/2} for x in x_top] +
                      [{'x': -straight_len + (b/2)*np.cos(t), 'y': (b/2)*np.sin(t)} for t in theta_left] +
                      [{'x': x, 'y': -b/2} for x in x_bottom])
        elif shape == 'custom':
            # 這裡建議用 try-except 包起來，防止 CSV 格式不對導致程式當掉
            try:
                # 假設 CSV 格式為兩欄：x, y
                coords = np.loadtxt(custom_path, delimiter=',', skiprows=1)
                points = [{'x': row[0], 'y': row[1]} for row in coords]
                return points
            except Exception as e:
                print(f"讀取 CSV 失敗: {e}")
                return []

        logger.debug(f"discretized_points: len={len(points)}")
        return points

    def get_boundary_weights(self, shape, a, b, nopell, nophor, nophor_paraplates,nopver, nophfc, is_closed, discretized_points=None, **kwargs):
        weights = []
        if shape == 'ellipse':
            theta = np.linspace(0, 2 * np.pi, nopell, endpoint=False)
            delta_theta = 2 * np.pi / nopell
            theta_mid = theta + delta_theta / 2
            weights = [np.sqrt(a**2 * np.sin(t)**2 + b**2 * np.cos(t)**2) * delta_theta
                       for t in theta_mid]
        elif shape == 'rectangle':
            if nophor < 3 or nopver < 3:
                raise ValueError("nophor must be >= 3 and nopver >= 3 for rectangle")
            w_hor = a / (nophor - 1) 
            w_ver = b / (nopver - 1) 
            w_cor = (w_hor + w_ver) / 2 
            weights = ([w_hor] * (nophor - 2) + [w_cor] + [w_ver] * (nopver - 2) + [w_cor] +
                       [w_hor] * (nophor - 2) + [w_cor] + [w_ver] * (nopver - 2) + [w_cor])
        elif shape == 'parallel plates':
            if nophor_paraplates < 3 :
                raise ValueError("nophor must be >= 3 for parallel plates")
            w_hor = a / (nophor_paraplates -1) 
            w_cor = (w_hor) / 2
            weights = ([w_cor] + [w_hor] * (nophor_paraplates -2) + [w_cor] * 2 + 
                       [w_hor] * (nophor_paraplates -2) + [w_cor])
        elif shape == 'racetrack':
            if nophor < 3 or nophfc < 3:
                raise ValueError("nophor must be >= 3 and nophfc >= 3 for racetrack")
            w_straight = (a - b) / (nophor - 1)
            w_arc = (np.pi * b / 2) / (nophfc - 1)
            w_joint = (w_straight + w_arc) / 2
            weights = ([w_joint] + [w_arc] * (nophfc - 2) + [w_joint] + [w_straight] * (nophor - 2) + 
                       [w_joint] + [w_arc] * (nophfc - 2) + [w_joint] + [w_straight] * (nophor - 2))
        elif shape == 'custom':
            # 你的通用公式邏輯 (向量化實作)
            coords = np.array([[p['x'], p['y']] for p in discretized_points])
            
            # 計算點與點之間的歐幾里得距離
            # np.diff 計算 [x2-x1, x3-x2, ...]
            diffs = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            
            if is_closed:
                # 封閉環：補上最後一點到第一點的距離
                dist_last_to_first = np.sqrt(np.sum((coords[-1] - coords[0])**2))
                # d_full = [d1-2, d2-3, ..., d_last-1]
                d_full = np.append(diffs, dist_last_to_first)
                # 每個點的權重是 (前段距離 + 後段距離) / 2
                weights = (d_full + np.roll(d_full, 1)) / 2
            else:
                # 非封閉 (如平行板)
                weights = np.zeros(len(coords))
                weights[0] = diffs[0] / 2
                weights[-1] = diffs[-1] / 2
                weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2

        weights = np.array(weights) / 1000
        logger.debug(f"Weights range: min={np.min(weights):.2e}, max={np.max(weights):.2e}")
        return weights

    def get_vector_potential(self, points, wires):
        Az = np.zeros(len(points))
        for i, point in enumerate(points):
            for wire in wires:
                dx = (point['x'] - wire['x']) / 1000
                dy = (point['y'] - wire['y']) / 1000
                r = np.sqrt(dx*dx + dy*dy) or 1e-10
                Az[i] -= (mu_0 * wire['I'] / (2 * np.pi)) * np.log(r)
        return Az

    def compute_fourier_coefficients(self, sol, nopell, order, points):
        # 將 sol 轉為 numpy 陣列 (nopfd, nopell)
        sol_array = np.array(sol) 
        theta = np.linspace(0, 2 * np.pi, nopell, endpoint=False)
        
        # 建立 m 的序列 (order + 1, 1)
        m_values = np.arange(order + 1)[:, np.newaxis]
        
        # 建立 cos(m * theta) 矩陣，形狀為 (order + 1, nopell)
        cos_matrix = np.cos(m_values * theta)
        
        # 利用矩陣乘法一次算完！
        # (order + 1, nopell) @ (nopell, nopfd) -> (order + 1, nopfd)
        # 注意：sol_array 需要轉置來配合矩陣乘法
        sum_im = cos_matrix @ sol_array.T
        
        if not np.all(np.isfinite(sum_im)):
            raise ValueError("Non-finite values in listFS computation")
            
        # 直接套用係數並轉回 list (如果後續需要的話)
        listFS_all = (2 / nopell * sum_im).tolist()
        
        return listFS_all

    def reconstruct_time_series(self, listFS_all, omega, f, T, C, noptd):
        # 轉為 numpy 陣列 (order + 1, nopfd)
        FS_all = np.array(listFS_all)
        t = np.linspace(-T, T, noptd)
        num_omega = len(omega)
        
        # 1. 計算相位矩陣 (num_omega, noptd)
        # -omega[i] * t[j] 的所有組合
        phases = -np.outer(omega, t)
        
        # 2. 準備複數訊號 S = f * FS (考慮到 f 是頻率權重)
        # S 的形狀為 (order + 1, num_omega)
        S = f * FS_all
        
        # 3. 核心向量化運算：
        # 我們利用實部與虛部的展開公式：
        # Re(S) * cos(phase) + Im(S) * sin(phase)
        # 這裡可以用矩陣乘法一次解決
        # (order + 1, num_omega) @ (num_omega, noptd) -> (order + 1, noptd)
        results = (np.real(S) @ np.cos(phases)) + (np.imag(S) @ np.sin(phases))
        
        if not np.all(np.isfinite(results)):
            raise ValueError("Non-finite values in listTS computation")
            
        # 套用常數係數
        scaling = C / (T * num_omega)
        listTS_all = (results * scaling).tolist()
        
        return listTS_all 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Eddy Current Analysis (Green's Function) v1.2")
        self.setGeometry(100, 100, 900, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        form_container = QHBoxLayout()
        layout.addLayout(form_container)

        left_form_layout = QFormLayout()
        form_container.addLayout(left_form_layout)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(['ellipse', 'rectangle', 'racetrack', 'parallel plates', 'custom'])
        self.shape_combo.currentTextChanged.connect(self.update_labels)
        left_form_layout.addRow("Boundary Shape:", self.shape_combo)

        self.a_input = QLineEdit("20")
        self.b_input = QLineEdit("4")
        self.a_label = QLabel("x semi-axis (mm):")
        self.b_label = QLabel("y semi-axis (mm):")
        left_form_layout.addRow(self.a_label, self.a_input)
        left_form_layout.addRow(self.b_label, self.b_input)

        self.thickness_input = QLineEdit("5")
        self.conductivity_input = QLineEdit("1.25e6")
        left_form_layout.addRow("Thickness (μm):", self.thickness_input)
        left_form_layout.addRow("Conductivity (S/m):", self.conductivity_input)

        self.skin_depth_label = QLabel("N/A")
        self.ratio_label = QLabel("N/A")
        left_form_layout.addRow("Skin Depth (μm):", self.skin_depth_label)
        left_form_layout.addRow("Ratio (thickness/skin_depth):", self.ratio_label)

        right_form_layout = QFormLayout()
        form_container.addLayout(right_form_layout)

        self.pulse_length_input = QLineEdit("5")
        self.pulse_type_combo = QComboBox()
        #self.pulse_type_combo.addItems(['half_sine', 'box'])
        self.pulse_type_combo.addItems(['half_sine'])
        self.freq_range_C_input = QLineEdit("50")
        self.nopfd_input = QLineEdit("501")
        self.noptd_input = QLineEdit("251")
        self.nfts_input = QLineEdit("20")
        self.nopxa_input = QLineEdit("100")
        right_form_layout.addRow("Pulse Length (μs):", self.pulse_length_input)
        right_form_layout.addRow("Pulse Type:", self.pulse_type_combo)
        right_form_layout.addRow("Frequency Range C:", self.freq_range_C_input)
        right_form_layout.addRow("Number of Freq Points (nopfd):", self.nopfd_input)
        right_form_layout.addRow("Number of Time Points (noptd):", self.noptd_input)
        right_form_layout.addRow("Number of Fitting Terms (nfts):", self.nfts_input)
        right_form_layout.addRow("Number of x-axis Points (nopxa):", self.nopxa_input)

        remaining_form_layout = QFormLayout()
        layout.addLayout(remaining_form_layout)

        self.nopell_input = QLineEdit("160")
        self.nophor_rect_input = QLineEdit("65")
        self.nophor_paraplates_input = QLineEdit("65")
        self.nopver_input = QLineEdit("15")
        self.nophor_race_input = QLineEdit("65")
        self.nophfc_input = QLineEdit("15")
        self.order_input = QLineEdit("10")
        self.order_chose_input = QLineEdit("4")
        self.nopell_widget = QWidget()
        nopell_layout = QHBoxLayout(self.nopell_widget)
        nopell_layout.addWidget(QLabel("Ellipse Points (nopell):"))
        nopell_layout.addWidget(self.nopell_input)
        self.nophor_rect_widget = QWidget()
        nophor_rect_layout = QHBoxLayout(self.nophor_rect_widget)
        nophor_rect_layout.addWidget(QLabel("Horizontal Points (nophor):"))
        nophor_rect_layout.addWidget(self.nophor_rect_input)
        self.nopver_widget = QWidget()
        nopver_layout = QHBoxLayout(self.nopver_widget)
        nopver_layout.addWidget(QLabel("Vertical Points (nopver):"))
        nopver_layout.addWidget(self.nopver_input)
        self.nophor_paraplates_widget = QWidget()
        nophor_paraplates_layout = QHBoxLayout(self.nophor_paraplates_widget)
        nophor_paraplates_layout.addWidget(QLabel("Horizontal Points (nophor):"))
        nophor_paraplates_layout.addWidget(self.nophor_paraplates_input)
        self.nophor_race_widget = QWidget()
        nophor_race_layout = QHBoxLayout(self.nophor_race_widget)
        nophor_race_layout.addWidget(QLabel("Horizontal Points (nophor):"))
        nophor_race_layout.addWidget(self.nophor_race_input)
        self.nophfc_widget = QWidget()
        nophfc_layout = QHBoxLayout(self.nophfc_widget)
        nophfc_layout.addWidget(QLabel("Half-Circle Points (nophfc):"))
        nophfc_layout.addWidget(self.nophfc_input)
        self.order_widget = QWidget()
        order_layout = QHBoxLayout(self.order_widget)
        order_layout.addWidget(QLabel("Fourier Order:"))
        order_layout.addWidget(self.order_input)
        self.order_chose_widget = QWidget()
        order_chose_layout = QHBoxLayout(self.order_chose_widget)
        order_chose_layout.addWidget(QLabel("Fourier Order for plot:"))
        order_chose_layout.addWidget(self.order_chose_input)
        remaining_form_layout.addRow(self.nopell_widget)
        remaining_form_layout.addRow(self.nophor_rect_widget)
        remaining_form_layout.addRow(self.nophor_paraplates_widget)
        remaining_form_layout.addRow(self.nopver_widget)
        remaining_form_layout.addRow(self.nophor_race_widget)
        remaining_form_layout.addRow(self.nophfc_widget)
        remaining_form_layout.addRow(self.order_widget)
        remaining_form_layout.addRow(self.order_chose_widget)

        self.bessy_fields_widget = QWidget()
        bessy_layout = QFormLayout(self.bessy_fields_widget)
        self.rect1_length_input = QLineEdit("12")
        self.rect1_width_input = QLineEdit("12")
        self.rect2_length_input = QLineEdit("20")
        self.rect2_width_input = QLineEdit("20")
        self.current_magnitude_input = QLineEdit("1600")
        self.current_sign_combo = QComboBox()
        self.current_sign_combo.addItems(["Mode 1", "Mode 2"])
        bessy_layout.addRow("Rect 1 Length (mm):", self.rect1_length_input)
        bessy_layout.addRow("Rect 1 Width (mm):", self.rect1_width_input)
        bessy_layout.addRow("Rect 2 Length (mm):", self.rect2_length_input)
        bessy_layout.addRow("Rect 2 Width (mm):", self.rect2_width_input)
        bessy_layout.addRow("Current Magnitude (A):", self.current_magnitude_input)
        bessy_layout.addRow("Current Sign:", self.current_sign_combo)
        remaining_form_layout.addRow(self.bessy_fields_widget)
        self.bessy_fields_widget.hide()

        self.topologyd_fields_widget = QWidget()
        topologyd_layout = QFormLayout(self.topologyd_fields_widget)
        self.topologydrect1_length_input = QLineEdit("2.4")
        self.topologydrect2_length_input = QLineEdit("4.8")
        self.topologyd_width_input = QLineEdit("9")
        self.topologydout1x_input = QLineEdit("0")
        self.topologydout1y_input = QLineEdit("17")
        self.topologydout2x_input = QLineEdit("0")
        self.topologydout2y_input = QLineEdit("-17")
        self.topodcurrent_magnitude_input = QLineEdit("1000")
        self.topodoutcurrent_magnitude_input = QLineEdit("1000")
        self.topodcurrent_sign_combo = QComboBox()
        self.topodcurrent_sign_combo.addItems(["Mode 1", "Mode 2"])
        topologyd_layout.addRow("Inner Distance (mm):", self.topologydrect1_length_input)
        topologyd_layout.addRow("Outer Distance (mm):", self.topologydrect2_length_input)
        topologyd_layout.addRow("Width (mm):", self.topologyd_width_input)
        topologyd_layout.addRow("Top Outer x (mm):", self.topologydout1x_input)
        topologyd_layout.addRow("Top Outer y (mm):", self.topologydout1y_input)
        topologyd_layout.addRow("Bottom Outer x (mm):", self.topologydout2x_input)
        topologyd_layout.addRow("Bottom Outer y (mm):", self.topologydout2y_input)
        topologyd_layout.addRow("Main Current Magnitude (A):", self.topodcurrent_magnitude_input)
        topologyd_layout.addRow("Outer Current Magnitude (A):", self.topodoutcurrent_magnitude_input)
        topologyd_layout.addRow("Current Sign:", self.topodcurrent_sign_combo)
        remaining_form_layout.addRow(self.topologyd_fields_widget)
        self.topologyd_fields_widget.hide()

        self.wire_mode_combo = QComboBox()
        self.wire_mode_combo.addItems(['BESSY-II type','ALBA-II type','SOLEIL Topology D type','User defined'])
        self.wire_mode_combo.currentTextChanged.connect(self.update_wire_input)
        remaining_form_layout.addRow("Wire Mode:", self.wire_mode_combo)

        self.wire_table = QTableWidget()
        self.wire_table.setRowCount(1)
        self.wire_table.setColumnCount(3)
        self.wire_table.setHorizontalHeaderLabels(['x (mm)', 'y (mm)', 'Current (A)'])
        self.wire_table.setItem(0, 0, QTableWidgetItem("20"))
        self.wire_table.setItem(0, 1, QTableWidgetItem("0"))
        self.wire_table.setItem(0, 2, QTableWidgetItem("1000"))
        layout.addWidget(QLabel("Wires:"))
        layout.addWidget(self.wire_table)

        wire_count_layout = QHBoxLayout()
        wire_count_layout.addWidget(QLabel("Number of Wires:"))
        self.wire_count_input = QSpinBox()
        self.wire_count_input.setMinimum(1)
        self.wire_count_input.setValue(1)
        self.wire_count_input.valueChanged.connect(self.update_wire_table)
        wire_count_layout.addWidget(self.wire_count_input)
        layout.addLayout(wire_count_layout)

        button_layout = QHBoxLayout()
        self.pre_button = QPushButton("Pre-process")
        self.calc_button = QPushButton("Calculate")
        self.pre_button.setEnabled(True)
        self.calc_button.setEnabled(False)
        self.pre_button.clicked.connect(self.start_pre_process)
        self.calc_button.clicked.connect(self.start_calculate)
        button_layout.addWidget(self.pre_button)
        button_layout.addWidget(self.calc_button)
        layout.addLayout(button_layout)

        # ==========================================
        # Calculation progress
        # ==========================================

        self.progress_label = QLabel("Pre-process / Calculate")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)


        # ==========================================
        # Post-processing progress
        # ==========================================

        self.post_progress_label = QLabel("Post-processing")

        self.post_progress_bar = QProgressBar()
        self.post_progress_bar.setMinimum(0)
        self.post_progress_bar.setMaximum(100)
        self.post_progress_bar.setValue(0)

        layout.addWidget(self.post_progress_label)
        layout.addWidget(self.post_progress_bar)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.conductivity_input.textChanged.connect(self.update_derived_fields)
        self.pulse_length_input.textChanged.connect(self.update_derived_fields)
        self.thickness_input.textChanged.connect(self.update_derived_fields)
        self.rect1_length_input.textChanged.connect(self.update_wire_input)
        self.rect1_width_input.textChanged.connect(self.update_wire_input)
        self.rect2_length_input.textChanged.connect(self.update_wire_input)
        self.rect2_width_input.textChanged.connect(self.update_wire_input)
        self.current_magnitude_input.textChanged.connect(self.update_wire_input)
        self.current_sign_combo.currentTextChanged.connect(self.update_wire_input)
        self.topologydrect1_length_input.textChanged.connect(self.update_wire_input)
        self.topologydrect2_length_input.textChanged.connect(self.update_wire_input)
        self.topologyd_width_input.textChanged.connect(self.update_wire_input)
        self.topologydout1x_input.textChanged.connect(self.update_wire_input)
        self.topologydout1y_input.textChanged.connect(self.update_wire_input)
        self.topologydout2x_input.textChanged.connect(self.update_wire_input)
        self.topologydout2y_input.textChanged.connect(self.update_wire_input)
        self.topodcurrent_magnitude_input.textChanged.connect(self.update_wire_input)
        self.topodoutcurrent_magnitude_input.textChanged.connect(self.update_wire_input)
        self.topodcurrent_sign_combo.currentTextChanged.connect(self.update_wire_input)

        form_container.setSpacing(20)
        left_form_layout.setVerticalSpacing(10)
        right_form_layout.setVerticalSpacing(10)
        remaining_form_layout.setVerticalSpacing(10)

        self.update_labels()
        self.update_wire_input()
        self.update_derived_fields()

        self.pre_done = False
        self.calc_done = False
        self.pre_data = None
        self.sol = None
        self.params = None
        self.thread = None

        # 新增「儲存設定」按鈕（可放在 layout 最後）
        # save_btn = QPushButton("Save Settings")
        # save_btn.clicked.connect(self.save_settings)
        # layout.addWidget(save_btn)  # 或加到其他位置

        # 加兩個按鈕（可放在 layout 最後或工具列）
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        load_btn = QPushButton("Load Settings")
        load_btn.clicked.connect(self.load_settings)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(load_btn)
        layout.addLayout(btn_layout)

        btn_layout2 = QHBoxLayout()
        # 1. 檔案讀取按鈕
        self.loadbound_btn = QPushButton("Load Boundary File")
        self.loadbound_btn.clicked.connect(self.handle_load_file)
        # 2. 是否封閉的勾選框
        # self.is_closed_cb = QCheckBox("Closed Boundary (e.g., Loop)")
        # self.is_closed_cb.setChecked(True)  # 預設勾選，因為大部分形狀是封閉的
        btn_layout2.addWidget(self.loadbound_btn)
        # btn_layout2.addWidget(self.is_closed_cb)
        layout.addLayout(btn_layout2)

        # 3. 儲存檔案路徑的變數
        self.custom_file_path = None

        # 程式啟動時自動載入
        # self.load_settings()
    
    def handle_load_file(self):
        # 1. 刪除或註解掉 options = QFileDialog.Options()
        
        # 2. 修改 getOpenFileName，把 options=options 刪掉
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Boundary File", 
            "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            self.custom_file_path = file_path
            print(f"成功載入路徑: {file_path}")
            # 如果你有一個 QLabel 顯示路徑，也可以在這裡更新它

    def get_all_settings(self):
        settings = {}
        
        # 安全取值（如果欄位不存在，就用預設值）
        def safe_text(attr_name, default=""):
            widget = getattr(self, attr_name, None)
            return widget.text() if widget and hasattr(widget, 'text') else str(default)
        
        # 固定欄位（請根據您的實際名稱調整）
        settings['shape'] = self.shape_combo.currentText()
        settings['a'] = safe_text('a_input', '20')
        settings['b'] = safe_text('b_input', '4')
        settings['thickness'] = safe_text('thickness_input', '5')
        settings['conductivity'] = safe_text('conductivity_input', '1.25e6')
        settings['pulse_length'] = safe_text('pulse_length_input', '50')
        settings['freq_range_C'] = safe_text('freq_range_C_input', '50')  # ← 這裡如果沒有 C_input，就會用 "1.0"
        settings['nopfd'] = safe_text('nopfd_input', '501') 
        settings['noptd'] = safe_text('noptd_input', '251')
        settings['nfts'] = safe_text('nfts_input', '20')
        settings['nopxa'] = safe_text('nopxa_input', '100')
        settings['order'] = safe_text('order_input', '4')
        settings['order_chose'] = safe_text('order_chose_input', '4')
        settings['nopell'] = safe_text('nopell_input', '160')
        settings['nophor_rect'] = safe_text('nophor_rect_input', '65')
        settings['nophor_paraplates'] = safe_text('nophor_paraplates_input', '65')
        settings['nopver'] = safe_text('nopver_input', '15')
        settings['nophor_race'] = safe_text('nophor_race_input', '65')
        settings['nophfc'] = safe_text('nophfc_input', '15') 
        
        # Wire Mode 相關
        settings['wire_mode'] = self.wire_mode_combo.currentText()
        settings['wire_count'] = self.wire_count_input.value()
        
        # User defined wires（如果有）
        if settings['wire_mode'] == 'User defined':
            wires = []
            for row in range(self.wire_table.rowCount()):
                wire = {}
                x_item = self.wire_table.item(row, 0)
                y_item = self.wire_table.item(row, 1)
                I_item = self.wire_table.item(row, 2)
                wire['x'] = x_item.text() if x_item else "0"
                wire['y'] = y_item.text() if y_item else "0"
                wire['I'] = I_item.text() if I_item else "0"
                wires.append(wire)
            settings['user_wires'] = wires
        
        # BESSY-II 模式（如果有）
        if settings['wire_mode'] == 'BESSY-II type':
            settings['rect1_length'] = safe_text('rect1_length_input', '12')
            settings['rect1_width'] = safe_text('rect1_width_input', '12')
            settings['rect2_length'] = safe_text('rect2_length_input', '20')
            settings['rect2_width'] = safe_text('rect2_width_input', '20')
            settings['current_magnitude'] = safe_text('current_magnitude_input', '1600')
            settings['current_sign'] = self.current_sign_combo.currentText()
        
        # ALBA-II 模式（如果有）
        if settings['wire_mode'] == 'ALBA-II type':
            settings['rect1_length'] = safe_text('rect1_length_input', '12')
            settings['rect1_width'] = safe_text('rect1_width_input', '12')
            settings['rect2_length'] = safe_text('rect2_length_input', '20')
            settings['rect2_width'] = safe_text('rect2_width_input', '20')
            settings['current_magnitude'] = safe_text('current_magnitude_input', '1600')
            settings['current_sign'] = self.current_sign_combo.currentText()
        
        # Type D 模式（如果有）
        if settings['wire_mode'] == 'SOLEIL Topology D type':
            settings['topologydrect1_length'] = safe_text('topologydrect1_length_input', '2.4')
            settings['topologydrect2_length'] = safe_text('topologydrect2_length_input', '4.8')
            settings['topologyd_width'] = safe_text('topologyd_width_input', '9')
            settings['topologydout1x'] = safe_text('topologydout1x_input', '0')
            settings['topologydout1y'] = safe_text('topologydout1y_input', '17')
            settings['topologydout2x'] = safe_text('topologydout2x_input', '0')
            settings['topologydout2y'] = safe_text('topologydout2y_input', '-17')
            settings['topodcurrent_magnitude'] = safe_text('topodcurrent_magnitude_input', '1000')
            settings['topodoutcurrent_magnitude'] = safe_text('topodoutcurrent_magnitude_input', '1000')
            settings['topodcurrent_sign'] = self.topodcurrent_sign_combo.currentText()
        
        return settings 

    def save_settings(self):
        """儲存所有設定到使用者選擇的 JSON 檔"""
        settings = self.get_all_settings()
        
        # 開啟儲存對話框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            "settings.json",  # 預設檔名
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:  # 使用者取消
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            logger.info(f"Settings saved to {file_path}")
            self.statusBar().showMessage(f"Settings saved to {os.path.basename(file_path)}", 5000)
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")
            self.statusBar().showMessage(f"Save failed: {str(e)}", 5000)

    def load_settings(self):
        """載入使用者選擇的 JSON 設定檔"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Settings",
            "",  # 預設從目前目錄開始
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:  # 使用者取消
            logger.info("No file selected for loading settings")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # 載入固定欄位（與之前相同） 
            self.shape_combo.setCurrentText(settings.get('shape', 'ellipse'))
            self.thickness_input.setText(settings.get('thickness', '5'))
            # ... 其他固定欄位 ...
            self.order_input.setText(settings.get('order', '4'))
            self.order_chose_input.setText(settings.get('order_chose', '4'))

            self.a_input.setText(settings.get('a', '20'))
            self.b_input.setText(settings.get('b', '4'))
            
            self.conductivity_input.setText(settings.get('conductivity', '1.25e6'))
            self.pulse_length_input.setText(settings.get('pulse_length', '5'))
            self.freq_range_C_input.setText(settings.get('freq_range_C', '50'))
            self.nopfd_input.setText(settings.get('nopfd', '501'))
            self.noptd_input.setText(settings.get('noptd', '251'))
            self.nfts_input.setText(settings.get('nfts', '20'))
            self.nopxa_input.setText(settings.get('nopxa', '100'))
            self.nopell_input.setText(settings.get('nopell', '160'))
            self.nophor_rect_input.setText(settings.get('nophor_rect', '65'))
            self.nophor_paraplates_input.setText(settings.get('nophor_paraplates', '65'))
            self.nopver_input.setText(settings.get('nopver', '15'))
            self.nophor_race_input.setText(settings.get('nophor_race', '65'))
            self.nophfc_input.setText(settings.get('nophfc', '15'))
            
            # Wire Mode
            self.wire_mode_combo.setCurrentText(settings.get('wire_mode', 'BESSY-II type'))
            
            if settings['wire_mode'] == 'BESSY-II type':
                self.rect1_length_input.setText(settings.get('rect1_length', '12'))
                self.rect1_width_input.setText(settings.get('rect1_width', '12'))
                self.rect2_length_input.setText(settings.get('rect2_length', '20'))
                self.rect2_width_input.setText(settings.get('rect2_width', '20'))
                self.current_magnitude_input.setText(settings.get('current_magnitude', '20'))
                self.current_sign_combo.setCurrentText(settings.get('current_sign', 'Mode 1'))
                self.update_wire_input()  # 觸發更新表格

            elif settings['wire_mode'] == 'ALBA-II type':
                self.rect1_length_input.setText(settings.get('rect1_length', '12'))
                self.rect1_width_input.setText(settings.get('rect1_width', '12'))
                self.rect2_length_input.setText(settings.get('rect2_length', '20'))
                self.rect2_width_input.setText(settings.get('rect2_width', '20'))
                self.current_magnitude_input.setText(settings.get('current_magnitude', '20'))
                self.current_sign_combo.setCurrentText(settings.get('current_sign', 'Mode 1'))
                self.update_wire_input()  # 觸發更新表格

            elif settings['wire_mode'] == 'SOLEIL Topology D type':
                self.topologydrect1_length_input.setText(settings.get('topologydrect1_length', '2.4'))
                self.topologydrect2_length_input.setText(settings.get('topologydrect2_length', '4.8'))
                self.topologyd_width_input.setText(settings.get('topologyd_width', '9'))
                self.topologydout1x_input.setText(settings.get('topologydout1x', '0'))
                self.topologydout1y_input.setText(settings.get('topologydout1y', '17'))
                self.topologydout2x_input.setText(settings.get('topologydout2x', '0'))
                self.topologydout2y_input.setText(settings.get('topologydout2y', '-17'))
                self.topodcurrent_magnitude_input.setText(settings.get('topodcurrent_magnitude', '1000'))
                self.topodoutcurrent_magnitude_input.setText(settings.get('topodoutcurrent_magnitude', '1000'))
                self.topodcurrent_sign_combo.setCurrentText(settings.get('topodcurrent_sign', 'Mode 1'))
                self.update_wire_input()  # 觸發更新表格
            
            elif settings['wire_mode'] == 'User defined':
                self.wire_count_input.setValue(settings.get('wire_count', 1))
                self.wire_table.setRowCount(0)
                self.wire_table.setRowCount(settings['wire_count'])
                user_wires = settings.get('user_wires', [])
                for row, wire in enumerate(user_wires[:settings['wire_count']]):
                    self.wire_table.setItem(row, 0, QTableWidgetItem(wire.get('x', '0')))
                    self.wire_table.setItem(row, 1, QTableWidgetItem(wire.get('y', '0')))
                    self.wire_table.setItem(row, 2, QTableWidgetItem(wire.get('I', '0')))
                logger.info(f"Loaded {len(user_wires)} user-defined wires")
            
            logger.info(f"Settings loaded from {file_path}")
            self.statusBar().showMessage(f"Settings loaded from {os.path.basename(file_path)}", 5000)
        
        except Exception as e:
            logger.error(f"Failed to load settings: {str(e)}")
            self.statusBar().showMessage(f"Load failed: {str(e)}", 5000)

    def update_derived_fields(self):
        try:
            conductivity = float(self.conductivity_input.text())
            pulse_length = float(self.pulse_length_input.text())
            thickness = float(self.thickness_input.text())
            if conductivity <= 0 or pulse_length <= 0 or thickness <= 0:
                raise ValueError("Non-positive values")
            skin_depth = math.sqrt((pulse_length * 1e-6) / (mu_0 * conductivity * math.pi)) * 1e6
            ratio = thickness / skin_depth
            self.skin_depth_label.setText(f"{skin_depth:.2f}")
            self.ratio_label.setText(f"{ratio:.3f}")
        except (ValueError, TypeError):
            self.skin_depth_label.setText("N/A")
            self.ratio_label.setText("N/A")

    def update_labels(self):
        shape = self.shape_combo.currentText()
        if shape == 'ellipse':
            self.a_label.setText("x semi-axis (mm):")
            self.b_label.setText("y semi-axis (mm):")
            self.nopell_widget.show()
            self.nophor_rect_widget.hide()
            self.nophor_paraplates_widget.hide()
            self.nopver_widget.hide()
            self.nophor_race_widget.hide()
            self.nophfc_widget.hide()
            self.order_widget.show()
            self.order_chose_widget.show()
        elif shape == 'rectangle':
            self.a_label.setText("Width (mm):")
            self.b_label.setText("Height (mm):")
            self.nopell_widget.hide()
            self.nophor_rect_widget.show()
            self.nophor_paraplates_widget.hide()
            self.nopver_widget.show()
            self.nophor_race_widget.hide()
            self.nophfc_widget.hide()
            self.order_widget.hide()
            self.order_chose_widget.hide()
        elif shape == 'parallel plates':
            self.a_label.setText("Width (mm):")
            self.b_label.setText("Height (mm):")
            self.nopell_widget.hide()
            self.nophor_rect_widget.hide()
            self.nophor_paraplates_widget.show()
            self.nopver_widget.hide()
            self.nophor_race_widget.hide()
            self.nophfc_widget.hide()
            self.order_widget.hide()
            self.order_chose_widget.hide()
        elif shape == 'custom':
            self.a_label.setText("x range for polynomial fit (mm)")
            self.b_label.setText(" ")
            self.nopell_widget.hide()
            self.nophor_rect_widget.hide()
            self.nophor_paraplates_widget.hide()
            self.nopver_widget.hide()
            self.nophor_race_widget.hide()
            self.nophfc_widget.hide()
            self.order_widget.hide()
            self.order_chose_widget.hide() 
        else:
            self.a_label.setText("Total Width (mm):")
            self.b_label.setText("Height (mm):")
            self.nopell_widget.hide()
            self.nophor_rect_widget.hide()
            self.nophor_paraplates_widget.hide()
            self.nopver_widget.hide()
            self.nophor_race_widget.show()
            self.nophfc_widget.show()
            self.order_widget.hide()
            self.order_chose_widget.hide()

    def update_wire_input(self):
        mode = self.wire_mode_combo.currentText()
        if mode == 'BESSY-II type':
            self.wire_table.setEnabled(False)
            self.wire_count_input.setEnabled(False)
            self.bessy_fields_widget.show()
            self.topologyd_fields_widget.hide() 
            try:
                rect1_length = float(self.rect1_length_input.text())
                rect1_width = float(self.rect1_width_input.text())
                rect2_length = float(self.rect2_length_input.text())
                rect2_width = float(self.rect2_width_input.text())
                current_magnitude = float(self.current_magnitude_input.text())
                sign = self.current_sign_combo.currentText()
                rect1_sign = 1 if sign == "Mode 1" else -1
                rect2_sign = -rect1_sign

                bessy_wires = [
                    {'x': rect1_length/2, 'y': rect1_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': rect1_length/2, 'y': -rect1_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': -rect1_length/2, 'y': rect1_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': -rect1_length/2, 'y': -rect1_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': rect2_length/2, 'y': rect2_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': rect2_length/2, 'y': -rect2_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': -rect2_length/2, 'y': rect2_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': -rect2_length/2, 'y': -rect2_width/2, 'I': rect2_sign * current_magnitude}
                ]
                self.wire_table.setRowCount(8)
                for i, wire in enumerate(bessy_wires):
                    self.wire_table.setItem(i, 0, QTableWidgetItem(f"{wire['x']:.2f}"))
                    self.wire_table.setItem(i, 1, QTableWidgetItem(f"{wire['y']:.2f}"))
                    self.wire_table.setItem(i, 2, QTableWidgetItem(f"{wire['I']:.2f}"))
                self.wire_count_input.setValue(8)
            except ValueError:
                self.status_label.setText("Invalid BESSY-II input values")
                self.wire_table.setRowCount(0)
        elif mode == 'ALBA-II type':
            self.wire_table.setEnabled(False)
            self.wire_count_input.setEnabled(False)
            self.bessy_fields_widget.show()
            self.topologyd_fields_widget.hide()
            try:
                rect1_length = float(self.rect1_length_input.text())
                rect1_width = float(self.rect1_width_input.text())
                rect2_length = float(self.rect2_length_input.text())
                rect2_width = float(self.rect2_width_input.text())
                current_magnitude = float(self.current_magnitude_input.text())
                sign = self.current_sign_combo.currentText()
                rect1_sign = 1 if sign == "Mode 1" else -1
                rect2_sign = -rect1_sign

                alba_wires = [
                    {'x': rect1_length/2, 'y': rect1_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': rect1_length/2, 'y': -rect1_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': -rect1_length/2, 'y': rect1_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': -rect1_length/2, 'y': -rect1_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': rect2_length/2, 'y': rect2_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': rect2_length/2, 'y': -rect2_width/2, 'I': rect2_sign * current_magnitude},
                    {'x': -rect2_length/2, 'y': rect2_width/2, 'I': rect1_sign * current_magnitude},
                    {'x': -rect2_length/2, 'y': -rect2_width/2, 'I': rect1_sign * current_magnitude}
                ]
                self.wire_table.setRowCount(8)
                for i, wire in enumerate(alba_wires):
                    self.wire_table.setItem(i, 0, QTableWidgetItem(f"{wire['x']:.2f}"))
                    self.wire_table.setItem(i, 1, QTableWidgetItem(f"{wire['y']:.2f}"))
                    self.wire_table.setItem(i, 2, QTableWidgetItem(f"{wire['I']:.2f}"))
                self.wire_count_input.setValue(8)
            except ValueError:
                self.status_label.setText("Invalid ALBA-II input values")
                self.wire_table.setRowCount(0)
        
        elif mode == 'SOLEIL Topology D type':
            self.wire_table.setEnabled(False)
            self.wire_count_input.setEnabled(False)
            self.bessy_fields_widget.hide()
            self.topologyd_fields_widget.show()
            try:
                topologydrect1_length = float(self.topologydrect1_length_input.text())
                topologydrect2_length = float(self.topologydrect2_length_input.text())
                topologyd_width = float(self.topologyd_width_input.text())
                topologydout1x = float(self.topologydout1x_input.text())
                topologydout1y = float(self.topologydout1y_input.text())
                topologydout2x = float(self.topologydout2x_input.text())
                topologydout2y = float(self.topologydout2y_input.text())
                topodcurrent_magnitude = float(self.topodcurrent_magnitude_input.text())
                topodoutcurrent_magnitude = float(self.topodoutcurrent_magnitude_input.text())
                topodcurrent_sign = self.topodcurrent_sign_combo.currentText()
                tds1_sign = 1 if topodcurrent_sign == "Mode 1" else -1
                tds2_sign = -tds1_sign

                topologyd_wires = [
                    {'x': 0, 'y': topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                    {'x': 0, 'y': -topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                    {'x': topologydrect1_length, 'y': topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                    {'x': topologydrect1_length, 'y': -topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                    {'x': -topologydrect1_length, 'y': topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                    {'x': -topologydrect1_length, 'y': -topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                    {'x': topologydrect2_length, 'y': topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                    {'x': topologydrect2_length, 'y': -topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                    {'x': -topologydrect2_length, 'y': topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                    {'x': -topologydrect2_length, 'y': -topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                    {'x': topologydout1x, 'y': topologydout1y, 'I': tds2_sign * topodoutcurrent_magnitude},
                    {'x': topologydout2x, 'y': topologydout2y, 'I': tds2_sign * topodoutcurrent_magnitude}
                ]
                self.wire_table.setRowCount(12)
                for i, wire in enumerate(topologyd_wires):
                    self.wire_table.setItem(i, 0, QTableWidgetItem(f"{wire['x']:.2f}"))
                    self.wire_table.setItem(i, 1, QTableWidgetItem(f"{wire['y']:.2f}"))
                    self.wire_table.setItem(i, 2, QTableWidgetItem(f"{wire['I']:.2f}"))
                self.wire_count_input.setValue(12)
            except ValueError:
                self.status_label.setText("Invalid SOLEIL Topology D input values")
                self.wire_table.setRowCount(0)

        else:
            self.wire_table.setEnabled(True)
            self.wire_count_input.setEnabled(True)
            self.bessy_fields_widget.hide()
            self.topologyd_fields_widget.hide()
            self.update_wire_table()

    def update_wire_table(self):
        if self.wire_mode_combo.currentText() == 'BESSY-II type':
            return
        num_wires = self.wire_count_input.value()
        self.wire_table.setRowCount(num_wires)
        for i in range(num_wires):
            if not self.wire_table.item(i, 0):
                self.wire_table.setItem(i, 0, QTableWidgetItem(str(20 + i*5)))
                self.wire_table.setItem(i, 1, QTableWidgetItem("0"))
                self.wire_table.setItem(i, 2, QTableWidgetItem(str(1000 if i % 2 == 0 else -1000)))

    def get_params(self):
        try:
            # 檢查：如果選了 custom 卻沒路徑，直接在主執行緒噴警告
            if self.shape_combo.currentText() == 'custom' and not self.custom_file_path:
                QMessageBox.warning(self, "Warning", "Please load a boundary file first!")
                return None # 回傳 None 代表失敗
            params = {
                'shape': self.shape_combo.currentText(),
                'a': float(self.a_input.text()),
                'b': float(self.b_input.text()),
                'thickness': float(self.thickness_input.text()),
                'conductivity': float(self.conductivity_input.text()),
                'pulse_length': float(self.pulse_length_input.text()),
                'pulse_type': self.pulse_type_combo.currentText(),
                'C': float(self.freq_range_C_input.text()),
                'nopfd': int(self.nopfd_input.text()),
                'noptd': int(self.noptd_input.text()),
                'nfts': int(self.nfts_input.text()),
                'nopxa': int(self.nopxa_input.text()),
                'nopell': int(self.nopell_input.text()),
                'nophor_rect': int(self.nophor_rect_input.text()),
                'nophor_paraplates': int(self.nophor_paraplates_input.text()),
                'nopver': int(self.nopver_input.text()),
                'nophor_race': int(self.nophor_race_input.text()),
                'nophfc': int(self.nophfc_input.text()),
                'order': int(self.order_input.text()),
                'order_chose': int(self.order_chose_input.text()),
                'wires': [],
                # 'is_closed': self.is_closed_cb.isChecked(),      # 讀取勾選框 Boolean 值
                'is_closed': int(1),      # 讀取勾選框 Boolean 值
                'custom_path': self.custom_file_path
            }
            if self.wire_mode_combo.currentText() == 'BESSY-II type':
                try:
                    rect1_length = float(self.rect1_length_input.text())
                    rect1_width = float(self.rect1_width_input.text())
                    rect2_length = float(self.rect2_length_input.text())
                    rect2_width = float(self.rect2_width_input.text())
                    current_magnitude = float(self.current_magnitude_input.text())
                    sign = self.current_sign_combo.currentText()
                    rect1_sign = 1 if sign == "Mode 1" else -1
                    rect2_sign = -rect1_sign
                    params['wires'] = [
                        {'x': rect1_length/2, 'y': rect1_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': rect1_length/2, 'y': -rect1_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': -rect1_length/2, 'y': rect1_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': -rect1_length/2, 'y': -rect1_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': rect2_length/2, 'y': rect2_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': rect2_length/2, 'y': -rect2_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': -rect2_length/2, 'y': rect2_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': -rect2_length/2, 'y': -rect2_width/2, 'I': rect2_sign * current_magnitude}
                    ]
                except ValueError:
                    raise ValueError("Invalid BESSY-II input values")
            elif self.wire_mode_combo.currentText() == 'ALBA-II type':
                try:
                    rect1_length = float(self.rect1_length_input.text())
                    rect1_width = float(self.rect1_width_input.text())
                    rect2_length = float(self.rect2_length_input.text())
                    rect2_width = float(self.rect2_width_input.text())
                    current_magnitude = float(self.current_magnitude_input.text())
                    sign = self.current_sign_combo.currentText()
                    rect1_sign = 1 if sign == "Mode 1" else -1
                    rect2_sign = -rect1_sign
                    params['wires'] = [
                        {'x': rect1_length/2, 'y': rect1_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': rect1_length/2, 'y': -rect1_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': -rect1_length/2, 'y': rect1_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': -rect1_length/2, 'y': -rect1_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': rect2_length/2, 'y': rect2_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': rect2_length/2, 'y': -rect2_width/2, 'I': rect2_sign * current_magnitude},
                        {'x': -rect2_length/2, 'y': rect2_width/2, 'I': rect1_sign * current_magnitude},
                        {'x': -rect2_length/2, 'y': -rect2_width/2, 'I': rect1_sign * current_magnitude}
                    ]
                except ValueError:
                    raise ValueError("Invalid ALBA-II input values")
            elif self.wire_mode_combo.currentText() == 'SOLEIL Topology D type':
                try:
                    topologydrect1_length = float(self.topologydrect1_length_input.text())
                    topologydrect2_length = float(self.topologydrect2_length_input.text())
                    topologyd_width = float(self.topologyd_width_input.text())
                    topologydout1x = float(self.topologydout1x_input.text())
                    topologydout1y = float(self.topologydout1y_input.text())
                    topologydout2x = float(self.topologydout2x_input.text())
                    topologydout2y = float(self.topologydout2y_input.text())
                    topodcurrent_magnitude = float(self.topodcurrent_magnitude_input.text())
                    topodoutcurrent_magnitude = float(self.topodoutcurrent_magnitude_input.text())
                    topodcurrent_sign = self.topodcurrent_sign_combo.currentText()
                    tds1_sign = 1 if topodcurrent_sign == "Mode 1" else -1
                    tds2_sign = -tds1_sign
                    params['wires'] = [
                        {'x': 0, 'y': topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                        {'x': 0, 'y': -topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                        {'x': topologydrect1_length, 'y': topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                        {'x': topologydrect1_length, 'y': -topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                        {'x': -topologydrect1_length, 'y': topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                        {'x': -topologydrect1_length, 'y': -topologyd_width/2, 'I': tds2_sign * topodcurrent_magnitude},
                        {'x': topologydrect2_length, 'y': topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                        {'x': topologydrect2_length, 'y': -topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                        {'x': -topologydrect2_length, 'y': topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                        {'x': -topologydrect2_length, 'y': -topologyd_width/2, 'I': tds1_sign * topodcurrent_magnitude},
                        {'x': topologydout1x, 'y': topologydout1y, 'I': tds2_sign * topodoutcurrent_magnitude},
                        {'x': topologydout2x, 'y': topologydout2y, 'I': tds2_sign * topodoutcurrent_magnitude}
                    ]
                except ValueError:
                    raise ValueError("Invalid SOLEIL Topology D input values")
            else:
                for i in range(self.wire_table.rowCount()):
                    x = float(self.wire_table.item(i, 0).text())
                    y = float(self.wire_table.item(i, 1).text())
                    I = float(self.wire_table.item(i, 2).text())
                    params['wires'].append({'x': x, 'y': y, 'I': I})

            if params['a'] <= 0 or params['b'] <= 0 or params['thickness'] <= 0 or \
               params['conductivity'] <= 0 or params['pulse_length'] <= 0 or \
               params['C'] <= 0 or params['nopfd'] < 1 or params['noptd'] < 1 or \
               params['nfts'] < 1 or params['nopxa'] < 1 or params['nopell'] < 1 or \
               params['nophor_rect'] < 1 or params['nopver'] < 1 or \
               params['nophor_paraplates'] < 1 or \
               params['nophor_race'] < 1 or params['nophfc'] < 1 or params['order'] < 0:
                raise ValueError("Invalid input parameters")
            if params['shape'] == 'racetrack' and params['a'] <= params['b']:
                raise ValueError("Total width must be greater than height for racetrack")
            return params
        except ValueError as e:
            raise ValueError(f"Invalid input: {str(e)}")

    def start_pre_process(self):
        try:
            self.params = self.get_params()
            self.pre_done = False
            self.calc_done = False
            self.pre_data = None
            self.sol = None
            self.pre_button.setEnabled(False)
            self.calc_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_label.setText("Pre-processing...")

            self.thread = ComputeThread(self.params, task="pre")
            self.thread.progress.connect(self.update_progress)
            self.thread.error.connect(self.handle_error)
            self.thread.pre_finished.connect(self.pre_computation_finished)
            self.thread.start()

        except ValueError as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.pre_button.setEnabled(True)

    def start_calculate(self):

        if not self.pre_done:

            self.status_label.setText(
                "Error: Pre-process not completed"
            )

            return

        try:

            # ======================================
            # 1. Update GUI
            # ======================================

            self.pre_button.setEnabled(False)
            self.calc_button.setEnabled(False)

            self.progress_bar.setValue(0)
            self.post_progress_bar.setValue(0)

            self.progress_label.setText(
                "Calculation progress"
            )

            self.post_progress_label.setText(
                "Post-processing progress"
            )

            self.status_label.setText(
                "Calculating..."
            )

            self.calc_done = False

            # ======================================
            # 2. Retrieve pre-process data
            # ======================================

            omega = np.asarray(
                [item['omega'] for item in self.pre_data[0]],
                dtype=float
            )

            f = np.asarray(
                self.pre_data[1],
                dtype=float
            )

            mesh = self.pre_data[3]
            drive_A = self.pre_data[4]

            # ======================================
            # 3. Create one worker for both stages
            # ======================================

            self.thread = ComputeThread(
                self.params,
                task="full",
                task_args=(
                    mesh,
                    drive_A,
                    omega,
                    f
                )
            )

            # ======================================
            # 4. Connect signals
            # ======================================

            self.thread.calc_progress.connect(
                self.update_progress
            )

            self.thread.post_progress.connect(
                self.update_post_progress
            )

            self.thread.calc_finished.connect(
                self.calc_computation_finished
            )

            self.thread.post_finished.connect(
                self.post_computation_finished
            )

            self.thread.error.connect(
                self.handle_error
            )

            # ======================================
            # 5. Start
            # ======================================

            self.thread.start()

        except Exception as e:

            self.status_label.setText(
                f"Error: {str(e)}"
            )

            self.pre_button.setEnabled(True)
            self.calc_button.setEnabled(True)

    def start_post_process(self):

        if not self.calc_done:
            self.status_label.setText(
                "Error: Calculate not completed"
            )
            return

        try:
            self.pre_button.setEnabled(False)
            self.calc_button.setEnabled(False)

            self.progress_bar.setValue(0)
            self.status_label.setText("Post-processing...")

            # 1. 取得頻率陣列
            omega = np.asarray(
                [item['omega'] for item in self.pre_data[0]],
                dtype=float
            )

            # 2. 取得 pulse spectrum
            f = np.asarray(
                self.pre_data[1],
                dtype=float
            )

            # 3. 取得 boundary mesh
            mesh = self.pre_data[3]

            # 4. 建立 Post-processing Worker
            self.thread = ComputeThread(
                self.params,
                task="post",
                task_args=(
                    self.sol,
                    mesh,
                    omega,
                    f
                )
            )

            # 5. Signal connections
            self.thread.progress.connect(
                self.update_progress
            )

            self.thread.error.connect(
                self.handle_error
            )

            self.thread.post_finished.connect(
                self.post_computation_finished
            )

            # 6. 啟動背景執行緒
            self.thread.start()

        except Exception as e:

            self.status_label.setText(
                f"Error: {str(e)}"
            )

            self.post_button.setEnabled(True)

    def update_progress(self, value):

        self.progress_bar.setValue(value)

        # 不要呼叫 QApplication.processEvents()
        # Qt signal 會自動將 GUI 更新排入主執行緒

        logger.debug(f"Progress updated to {value}%")

    def update_post_progress(self, value):

        self.post_progress_bar.setValue(value)

        logger.debug(
            f"Post-processing progress: {value}%"
        )

    def handle_error(self, message):

        self.status_label.setText(
            f"Error: {message}"
        )

        logger.error(f"Error: {message}")

        self.pre_button.setEnabled(True)

        self.calc_button.setEnabled(
            self.pre_done
        )

        if self.thread:

            self.thread.quit()
            self.thread.wait()

            self.thread = None

    def pre_computation_finished(
        self, omega_list, f, discretized_points, mesh, drive_A
    ):
        try:
            self.pre_data = (
                omega_list,
                f,
                discretized_points,
                mesh,
                drive_A
            )
            self.plot_pre_results(
                omega_list, f, discretized_points, mesh, drive_A,
                self.params
            )
            self.pre_done = True
            self.pre_button.setEnabled(True)
            self.calc_button.setEnabled(True)
            self.status_label.setText("Pre-processing finished")
            self.status_label.setText(
            "Pre-processing finished. Ready to calculate."
        )
        except Exception as exc:
            self.handle_error(f"Pre-process plotting error: {exc}")
        finally:
            if self.thread:
                self.thread.quit()
                self.thread.wait()
                self.thread = None

    def calc_computation_finished(self, solution):

        try:

            self.sol = solution

            # 儲存 Kz 與 induced Az 頻域結果
            self.plot_calc_results(
                solution,
                self.params
            )

            self.calc_done = True

            self.status_label.setText(
                "Calculation finished. Post-processing..."
            )

            logger.info(
                "Calculation stage finished. Continuing post-processing."
            )

        except Exception as e:

            self.handle_error(
                f"Calculation result processing error: {str(e)}"
            )

    def post_computation_finished(
        self,
        listFS,
        listTS,
        listxa,
        listFSploy,
        by_time_ploy,
        by_coeffs_ploy,
        listFS_sin,
        listTS_sin
    ):

        try:

            self.plot_post_results(
                listFS,
                listTS,
                listxa,
                listFSploy,
                by_time_ploy,
                by_coeffs_ploy,
                self.params,
                listFS_sin,
                listTS_sin
            )

            self.status_label.setText(
                "All calculations completed successfully"
            )

            self.calc_done = True

            logger.info(
                "Full calculation and post-processing completed"
            )

            self.pre_button.setEnabled(True)
            self.calc_button.setEnabled(True)

        except Exception as e:

            self.handle_error(
                f"Post-process plotting error: {str(e)}"
            )

        finally:

            if self.thread:

                self.thread.quit()
                self.thread.wait()

                self.thread = None

    def plot_pre_results(self, omega_list, f, discretized_points, mesh, drive_A, params):
        shape = params['shape']
        a = params['a']
        b = params['b']
        T = params['pulse_length'] * 1e-6
        C = params['C']
        nopfd = params['nopfd']
        noptd = params['noptd']
        nfts = params['nfts']
        wires = params['wires']
        pulse_type = params['pulse_type']

        omega = [d['omega'] for d in omega_list]
        fig = make_subplots(rows=4, cols=2,
                            subplot_titles=("Boundary", "Magnetic Field", "Frequency Domain",
                                           "Time Domain", "Discretized Boundary", "Vector Potential",
                                           "Polynomial Fitting Comparison", ""),
            horizontal_spacing=0.2
            )
        

        N = 100
        if shape == 'ellipse':
            theta = np.linspace(0, 2*np.pi, N)
            x = a * np.cos(theta)
            y = b * np.sin(theta)
            x_range_min, x_range_max = -a, a
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Boundary'), row=1, col=1)
        elif shape == 'rectangle':
            x_top = np.linspace(-a/2, a/2, N//4)
            y_right = np.linspace(b/2, -b/2, N//4)
            x_bottom = np.linspace(a/2, -a/2, N//4)
            y_left = np.linspace(-b/2, b/2, N//4)
            x = np.concatenate([x_top, [a/2]*len(y_right), x_bottom, [-a/2]*len(y_left)])
            y = np.concatenate([[b/2]*len(x_top), y_right, [-b/2]*len(x_bottom), y_left])
            x_range_min, x_range_max = -a/2, a/2
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Boundary'), row=1, col=1)
        elif shape == 'parallel plates':
            num_per_plate = N // 2
            if N % 2 == 1:
                num_per_plate += 1  # 奇數時上板多一點
            
            x_top = np.linspace(-a/2, a/2, num_per_plate)
            x_bottom = np.linspace(a/2, -a/2, num_per_plate)  # 方向相反，維持逆時針
            fig.add_trace(go.Scatter(x=x_top, y=[b/2] * len(x_top), mode='lines',
                             line=dict(color='#87CEFA', width=3), name='Upper Plate'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_bottom, y=[-b/2] * len(x_bottom), mode='lines',
                                    line=dict(color='#87CEFA', width=3), name='Lower Plate'), row=1, col=1)
            x = np.concatenate([x_top, x_bottom])
            y = np.concatenate([[b/2] * len(x_top), [-b/2] * len(x_bottom)])
            x_range_min, x_range_max = -a/2, a/2
        elif shape == 'custom':
            all_xy = np.array([[p['x'], p['y']] for p in discretized_points])
            x = all_xy[:, 0]
            y = all_xy[:, 1]
            x_range_min, x_range_max = np.min(x), np.max(x)

            for ic, component_name in enumerate(mesh.component_names):
                mask = mesh.component_ids == ic
                starts = mesh.starts_m[mask] * 1000.0
                ends = mesh.ends_m[mask] * 1000.0
                if len(starts) == 0:
                    continue
                plot_x = np.concatenate((starts[:, 0], ends[-1:, 0]))
                plot_y = np.concatenate((starts[:, 1], ends[-1:, 1]))
                fig.add_trace(go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    mode='lines+markers',
                    name=f'Custom: {component_name}',
                    line=dict(color='orange'),
                ), row=1, col=1)

        else:
            straight_len = (a - b)/2
            theta_right = np.linspace(-np.pi/2, np.pi/2, N//4)
            x_top = np.linspace(straight_len, -straight_len, N//4)
            theta_left = np.linspace(np.pi/2, 3*np.pi/2, N//4)
            x_bottom = np.linspace(-straight_len, straight_len, N//4)
            x = np.concatenate([
                straight_len + (b/2)*np.cos(theta_right),
                x_top,
                -straight_len + (b/2)*np.cos(theta_left),
                x_bottom
            ])
            y = np.concatenate([
                (b/2)*np.sin(theta_right),
                [b/2]*len(x_top),
                (b/2)*np.sin(theta_left),
                [-b/2]*len(x_bottom)
            ])
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Boundary'), row=1, col=1)
            x_range_min, x_range_max = -a/2, a/2

        axis_range = max(a, b, max([abs(w['x']) for w in wires] + [abs(w['y']) for w in wires])) * 1.5
        
        fig.add_trace(go.Scatter(x=[w['x'] for w in wires], y=[w['y'] for w in wires],
                                mode='markers', marker=dict(size=10, color=['red' if w['I'] > 0 else 'blue' for w in wires]),
                                name='Wires'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[-axis_range, axis_range], y=[0, 0], mode='lines', line=dict(color='gray', dash='dash'),
                                name='x-axis'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 0], y=[-axis_range, axis_range], mode='lines', line=dict(color='gray', dash='dash'),
                                name='y-axis'), row=1, col=1)

        x_points = np.linspace(x_range_min, x_range_max, 100)
        By = [sum(mu_0 * w['I'] * (x/1000 - w['x']/1000) / (2 * np.pi * ((x/1000 - w['x']/1000)**2 + (w['y']/1000)**2 or 1e-10))
                  for w in wires) for x in x_points]
        fig.add_trace(go.Scatter(x=x_points, y=By, mode='lines', name='By', line=dict(color='blue')),
                      row=1, col=2)

        try:
            abs_By = np.abs(By)
            pos_mask = x_points > 0
            neg_mask = x_points < 0
            if np.any(pos_mask) and np.any(neg_mask):
                pos_indices = np.where(pos_mask)[0]
                neg_indices = np.where(neg_mask)[0]
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    pos_idx = pos_indices[np.argmax(abs_By[pos_indices])]
                    neg_idx = neg_indices[np.argmax(abs_By[neg_indices])]
                    pos_x = x_points[pos_idx]
                    neg_x = x_points[neg_idx]
                    fig.add_trace(go.Scatter(x=[pos_x, pos_x], y=[min(By), max(By)],
                                            mode='lines', line=dict(color='red', dash='dash'),
                                            name=f'Max |By| (x={pos_x:.2f} mm)'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=[neg_x, neg_x], y=[min(By), max(By)],
                                            mode='lines', line=dict(color='red', dash='dash'),
                                            name=f'Max |By| (x={neg_x:.2f} mm)'), row=1, col=2)
                    fig.add_annotation(x=pos_x, y=max(By), text=f'x={pos_x:.2f}', showarrow=True,
                                      arrowhead=1, ax=20, ay=-30, row=1, col=2)
                    fig.add_annotation(x=neg_x, y=max(By), text=f'x={neg_x:.2f}', showarrow=True,
                                      arrowhead=1, ax=-20, ay=-30, row=1, col=2)
        except Exception as e:
            logger.error(f"Failed to mark max |By| points: {str(e)}")

        try:
            if shape == 'ellipse':
                fit_mask = np.abs(x_points) < a
            else:
                fit_mask = np.abs(x_points) < a/2

            fit_x = x_points[fit_mask]
            fit_By = np.array(By)[fit_mask]

            if len(fit_x) < nfts + 1:
                raise ValueError(f"Not enough points ({len(fit_x)}) for polynomial fitting with {nfts} terms")

            coeffs = np.polyfit(fit_x, fit_By, nfts)
            fitted_By = np.polyval(coeffs, fit_x)

            # 產生多項式字串（使用科學記號 .2e，保留正負號）
            poly_terms = []
            for i, c in enumerate(coeffs[::-1]):  # coeffs[::-1] 從常數項到最高次
                if abs(c) > 1e-10:  # 濾掉噪音項（可調門檻）
                    if i == 0:
                        term = f"{c:.2e}"  # 常數項
                    elif i == 1:
                        term = f"{c:+.2e}x"  # 一次項
                    else:
                        term = f"{c:+.2e}x^{i}"  # 高次項
                    poly_terms.append(term)

            poly_str = " + ".join(poly_terms) if poly_terms else "0"

            # 自動換行函數（每 60 個字元左右換行，可調整 width）
            def wrap_text(text, width=60):
                lines = []
                current_line = ""
                for char in text:
                    if len(current_line) + 1 <= width:
                        current_line += char
                    else:
                        lines.append(current_line)
                        current_line = char
                if current_line:
                    lines.append(current_line)
                return "<br>".join(lines)  # 用 <br> 讓 Plotly 換行

            poly_str_wrapped = wrap_text(poly_str)

            # 加註解：左對齊 + 換行 + 背景 + 科學記號
            fig.add_annotation(
                xref="paper", yref="paper",
                x=-0.05, y=0.05,              # 從左邊 2% 開始，避免左邊跑出
                xanchor="left", yanchor="bottom",
                text=f"<b>Polynomial fit:</b><br>{poly_str_wrapped}",
                showarrow=False,
                font=dict(size=11, family="Courier New, monospace"),  # 等寬字體，科學記號對齊好看
                align="left",                 # 文字左對齊
                bgcolor="rgba(255,255,255,0.9)",  # 白色半透明背景
                bordercolor="gray",
                borderwidth=1,
                borderpad=6,
                row=4, col=1
            )

            # 畫原始與擬合曲線（保持不變）
            fig.add_trace(go.Scatter(x=fit_x, y=fit_By, mode='lines', name='Original By',
                                    line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=fit_x, y=fitted_By, mode='lines', name='Fitted Polynomial',
                                    line=dict(color='orange', dash='dash')), row=4, col=1)
            fit_data = {
                'x': fit_x,
                'By': fit_By,
                'fitted_By': fitted_By
            }
            for i, c in enumerate(coeffs[::-1]):
                fit_data[f'c_{i}'] = [c] * len(fit_x)
            fit_df = pd.DataFrame(fit_data)
            fit_df.to_csv('fitting_results.csv', index=False)
            logger.info("Saved fitting results to fitting_results.csv")
        except Exception as e:
            logger.error(f"Failed to perform polynomial fitting: {str(e)}")

        fig.add_trace(go.Scatter(x=omega, y=f, mode='lines', name='f(ω)', line=dict(color='green')),
                      row=2, col=1)

        t = np.linspace(-T, T, noptd)
        F = [C / (T * nopfd) * sum(f[i] * np.cos(omega[i] * ti) for i in range(nopfd)) for ti in t]
        fig.add_trace(go.Scatter(x=t, y=F, mode='lines', name='F(t)', line=dict(color='purple')),
                      row=2, col=2)

        fig.add_trace(go.Scatter(x=[p['x'] for p in discretized_points], y=[p['y'] for p in discretized_points],
                                mode='markers', marker=dict(size=5, color='red'), name='Discretized Points'),
                      row=3, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(len(drive_A)),
            y=drive_A,
            mode='lines',
            name='Drive Az',
            line=dict(color='orange')
        ), row=3, col=2)

        fig.update_layout(
            height=1200, width=800,
            showlegend=True,
            title_text="Pre-process Results"
        )
        fig.update_xaxes(title_text="x (mm)", row=1, col=1)
        fig.update_yaxes(title_text="y (mm)", row=1, col=1)
        fig.update_xaxes(title_text="x (mm)", row=1, col=2)
        fig.update_yaxes(title_text="By (T)", row=1, col=2)
        fig.update_xaxes(title_text="ω (rad/s)", row=2, col=1)
        fig.update_yaxes(title_text="f(ω) (s)", row=2, col=1)
        fig.update_xaxes(title_text="t (s)", row=2, col=2)
        fig.update_yaxes(title_text="F(t)", row=2, col=2)
        fig.update_xaxes(title_text="x (mm)", row=3, col=1)
        fig.update_yaxes(title_text="y (mm)", row=3, col=1)
        fig.update_xaxes(title_text="Point Index", row=3, col=2)
        fig.update_yaxes(title_text="Az (Wb/m)", row=3, col=2)
        fig.update_xaxes(title_text="x (mm)", row=4, col=1)
        fig.update_yaxes(title_text="By (T)", row=4, col=1)

        try:
            omega_df = pd.DataFrame(omega_list)
            omega_df.to_csv('omega_list.csv', index=False)
            logger.info("Saved omega_list to omega_list.csv")
        except Exception as e:
            logger.error(f"Failed to save omega_list.csv: {str(e)}")

        try:
            f_df = pd.DataFrame({'f': f})
            f_df.to_csv('f.csv', index=False)
            logger.info("Saved f to f.csv")
        except Exception as e:
            logger.error(f"Failed to save f.csv: {str(e)}")

        try:
            if shape == 'ellipse':
                points_df = pd.DataFrame({
                    'x': [p['x'] for p in discretized_points],
                    'y': [p['y'] for p in discretized_points],
                    'theta': [p['theta'] for p in discretized_points]
                })
            else:
                points_df = pd.DataFrame({
                    'x': [p['x'] for p in discretized_points],
                    'y': [p['y'] for p in discretized_points]
                })
            points_df.to_csv('discretized_points.csv', index=False)
            logger.info("Saved discretized_points to discretized_points.csv")
        except Exception as e:
            logger.error(f"Failed to save discretized_points.csv: {str(e)}")

        try:
            drive_A = np.asarray(drive_A, dtype=complex)

            drive_A_df = pd.DataFrame({
                'drive_A_real': drive_A.real,
                'drive_A_imag': drive_A.imag
            })

            drive_A_df.to_csv('drive_A.csv', index=False)

            logger.info("Saved drive_A.csv")

        except Exception as e:
            logger.error(
                f"Failed to save drive_A.csv: {str(e)}"
            )

        try:
            html_file = os.path.abspath("pre_plots.html")
            fig.write_html(html_file)
            webbrowser.open(f"file://{html_file}")
            logger.info("Saved and opened pre_plots.html")
        except Exception as e:
            logger.error(f"Failed to save or open pre_plots.html: {str(e)}")

    def plot_calc_results(self, solution, params):
        K = np.asarray(solution.current_density, dtype=complex)
        phi = np.asarray(solution.induced_A, dtype=complex)

        if K.shape[0] != params['nopfd']:
            raise ValueError(
                f"K spectrum frequency count mismatch: {K.shape[0]} != {params['nopfd']}"
            )
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(phi)):
            raise ValueError("Solver result contains non-finite values")

        data = {}
        for i in range(K.shape[1]):
            data[f'K_{i}_real'] = K[:, i].real
            data[f'K_{i}_imag'] = K[:, i].imag
            data[f'Aeddy_{i}_real'] = phi[:, i].real
            data[f'Aeddy_{i}_imag'] = phi[:, i].imag
        pd.DataFrame(data).to_csv('sol.csv', index=False)
        logger.info("Saved K_z and induced A_z spectra to sol.csv")

    def plot_post_results(
            self,
            listFS,
            listTS,
            listxa,
            listFSploy,
            by_time_ploy,
            by_coeffs_ploy,
            params,
            listFS_sin=None,
            listTS_sin=None
        ):
        logger.info("Starting plot_post_results")
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots

        except ImportError as e:
            logger.error(f"Plotly not available: {e}")
            self.error.emit("Plotly library is required for post-process plotting. Please install plotly.")
            return

        logger.info("Starting plot_post_results")
        shape = params['shape']
        T = params['pulse_length'] * 1e-6
        noptd = params['noptd']
        nopxa = params['nopxa']
        nfts = params['nfts']
        nopfd = params['nopfd']
        order = params['order']
        order_chose = params['order_chose']
        C = params['C']
        
        # 計算 omega_max 並生成 omega
        omega_max = C * np.pi / T
        omega = params.get('omega', np.linspace(-omega_max, omega_max, nopfd))
        if len(omega) != nopfd:
            logger.error(f"omega length {len(omega)} does not match nopfd {nopfd}, using zero array")
            omega = np.zeros(nopfd)
        logger.debug(f"omega range: {omega[0]:.2e} to {omega[-1]:.2e}, length={len(omega)}")

        fig = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{}, {}],
                [{}, {}],
                [{}, {}],
                [{"colspan": 2}, None],
            ],
            subplot_titles=(
                "Fourier Coefficients (Ellipse)",
                "Time Series (Ellipse)",
                "By Frequency Domain (Real)",
                "By Frequency Domain (Imag)",
                "By Time Domain",
                "Boundary current evolution",
                "Induced By(x,t) evolution",
            ),
            horizontal_spacing=0.15,
            vertical_spacing=0.10,
        )
        # 清除所有現有 trace，確保乾淨的子圖
        fig.data = []
        logger.debug("Cleared all existing traces in fig")

        # 保存傅立葉數據（僅橢圓）
        # =====================================================
        # Ellipse Fourier Results
        # =====================================================

        if shape == 'ellipse' and len(listFS) > 0:

            logger.info("Plotting ellipse Fourier results")

            FS_cos = np.asarray(listFS, dtype=complex)
            FS_sin = np.asarray(listFS_sin, dtype=complex)

            TS_cos = np.asarray(listTS, dtype=float)
            TS_sin = np.asarray(listTS_sin, dtype=float)

            # Dimensions
            num_modes = FS_cos.shape[0]

            t = np.linspace(-T, T, noptd)

            # -------------------------------------------------
            # 1. Save frequency-domain Fourier coefficients
            # -------------------------------------------------

            rows = []

            for component_name, coefficient_array in [
                ("cos", FS_cos),
                ("sin", FS_sin)
            ]:

                for m in range(num_modes):

                    for k in range(nopfd):

                        value = coefficient_array[m, k]

                        rows.append({
                            'component': component_name,
                            'm': m,
                            'omega': omega[k],
                            'real': value.real,
                            'imag': value.imag
                        })

            df_fourier = pd.DataFrame(rows)

            df_fourier.to_csv(
                'ellipse_fourier_frequency.csv',
                index=False
            )

            logger.info("Saved ellipse_fourier_frequency.csv")

            # -------------------------------------------------
            # 2. Save time-domain Fourier coefficients
            # -------------------------------------------------

            time_data = {'time_s': t}

            for m in range(num_modes):

                time_data[f'A_cos_m{m}'] = TS_cos[m]
                time_data[f'A_sin_m{m}'] = TS_sin[m]

            df_time = pd.DataFrame(time_data)

            df_time.to_csv(
                'ellipse_fourier_time.csv',
                index=False
            )

            logger.info("Saved ellipse_fourier_time.csv")

            # -------------------------------------------------
            # 3. Plot selected Fourier order
            # -------------------------------------------------

            if 0 <= order_chose < num_modes:

                m = order_chose

                # Frequency domain
                for component_name, coefficient in [
                    ("Cosine", FS_cos[m]),
                    ("Sine", FS_sin[m])
                ]:

                    fig.add_trace(
                        go.Scatter(
                            x=omega,
                            y=np.real(coefficient),
                            mode='lines',
                            name=f'{component_name} Real (m={m})'
                        ),
                        row=1,
                        col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=omega,
                            y=np.imag(coefficient),
                            mode='lines',
                            name=f'{component_name} Imag (m={m})'
                        ),
                        row=1,
                        col=1
                    )

                # Time domain
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=TS_cos[m],
                        mode='lines',
                        name=f'Cosine Time (m={m})'
                    ),
                    row=1,
                    col=2
                )

                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=TS_sin[m],
                        mode='lines',
                        name=f'Sine Time (m={m})'
                    ),
                    row=1,
                    col=2
                )

            else:

                logger.warning(
                    f"order_chose={order_chose} is out of range"
                )

        # 強制輸出 listFSploy 到 CSV（僅虛部）
        logger.debug(f"plot_post_results: nopxa={nopxa}, nopfd={nopfd}")
        logger.debug(f"listFSploy type: {type(listFSploy)}, shape: {getattr(listFSploy, 'shape', 'N/A')}, size: {getattr(listFSploy, 'size', 'N/A')}")
        if isinstance(listFSploy, list):
            logger.warning("listFSploy is a list, attempting to convert to np.ndarray")
            try:
                listFSploy = np.array(listFSploy, dtype=complex)
                logger.debug(f"Converted listFSploy shape: {listFSploy.shape}")
            except Exception as e:
                logger.error(f"Failed to convert listFSploy to np.ndarray: {str(e)}")
                listFSploy = np.zeros((nopxa, nopfd), dtype=complex)

        try:
            # 確保 listFSploy 是正確的 numpy 陣列
            if not isinstance(listFSploy, np.ndarray) or listFSploy.shape != (nopxa, nopfd):
                logger.warning(f"listFSploy shape {listFSploy.shape if hasattr(listFSploy, 'shape') else 'N/A'} does not match expected ({nopxa}, {nopfd}), using zero array")
                listFSploy = np.zeros((nopxa, nopfd), dtype=complex)

            # 提取實部與虛部
            real_part = np.real(listFSploy)
            imag_part = np.imag(listFSploy)

            logger.debug(f"listFSploy_real min: {np.min(real_part):.2e}, max: {np.max(real_part):.2e}")
            logger.debug(f"listFSploy_imag min: {np.min(imag_part):.2e}, max: {np.max(imag_part):.2e}")

            # 檢查有限值
            finite_real = np.sum(np.isfinite(real_part))
            finite_imag = np.sum(np.isfinite(imag_part))
            logger.debug(f"Real finite values: {finite_real}/{real_part.size}, Imag finite values: {finite_imag}/{imag_part.size}")

            if finite_real < real_part.size:
                non_finite_real = np.where(~np.isfinite(real_part))
                logger.warning(f"Non-finite real values at indices: {list(zip(non_finite_real[0], non_finite_real[1]))}")

            if finite_imag < imag_part.size:
                non_finite_imag = np.where(~np.isfinite(imag_part))
                logger.warning(f"Non-finite imag values at indices: {list(zip(non_finite_imag[0], non_finite_imag[1]))}")

            # 強制儲存 listFSploy 到 CSV（實部 + 虛部）
            try:
                listFSploy_df = pd.DataFrame({
                    'x_mm': listxa,
                    **{f'By_f_{omega[j]:.2e}_real': real_part[:, j] for j in range(nopfd)},
                    **{f'By_f_{omega[j]:.2e}_imag': imag_part[:, j] for j in range(nopfd)}
                })
                listFSploy_df.to_csv('listFSploy_real_imag.csv', index=False)
                logger.info(f"Saved listFSploy_real_imag.csv with {listFSploy_df.shape[1]} columns "
                            f"(x_mm + {nopfd} real + {nopfd} imag)")
            except Exception as e:
                logger.error(f"Failed to save listFSploy_real_imag.csv: {str(e)}")
                # 備案：只存 x_mm
                pd.DataFrame({'x_mm': listxa}).to_csv('listFSploy_real_imag.csv', index=False)
                with open('listFSploy_real_imag.csv', 'a') as f:
                    f.write(f"\nError: Failed to save listFSploy data: {str(e)}\n")
                logger.info("Saved partial listFSploy_real_imag.csv with error message")

            selected_indices = [0]  # 第一點

            # 中間點數（可調整 3~5 個）
            num_middle = 4  # 您可以改成 3 或 5
            #if nopxa > 2:
            #    step = (nopxa - 1) // (num_middle + 1)
            #    for k in range(1, num_middle + 1):
            #        idx = k * step
            #        if idx < nopxa - 1 and idx not in selected_indices:
            #            selected_indices.append(idx)

            # 最後一點
            #if nopxa > 1 and (nopxa - 1) not in selected_indices:
            #   selected_indices.append(nopxa - 1)

            # 去重並排序（確保順序）
            #selected_indices = sorted(set(selected_indices))


            target_count = num_middle + 2 

            if nopxa >= target_count:
                # 產生從 0 到 nopxa-1 之間，均勻分布的 target_count 個整數索引
                selected_indices = np.linspace(0, nopxa - 1, target_count).round().astype(int).tolist()
                # 轉成 set 再轉回 list 是為了確保若 nopxa 太小時不會有重複索引
                selected_indices = sorted(list(set(selected_indices)))
            else:
                # 如果總點數比要選的還少，就全選
                selected_indices = list(range(nopxa))

            # 繪製頻域 By（實部 + 虛部，使用 omega 作為橫軸）
            logger.info("Plotting By Frequency Domain (Real + Imag) with omega as x-axis")
            traces_added_real = 0
            traces_added_imag = 0

            for i in selected_indices:
                # 畫實部
                if np.all(np.isfinite(real_part[i, :])):
                    fig.add_trace(go.Scatter(x=omega, y=real_part[i, :], mode='lines',
                                            name=f'By Real at x={listxa[i]:.2f} mm'),
                                row=2, col=1)  # 您可以自己改 row/col
                    traces_added_real += 1
                else:
                    logger.warning(f"Skipping real part plot for x={listxa[i]:.2f} mm due to non-finite values")

                # 畫虛部（您原本的）
                if np.all(np.isfinite(imag_part[i, :])):
                    fig.add_trace(go.Scatter(x=omega, y=imag_part[i, :], mode='lines',
                                            name=f'By Imag at x={listxa[i]:.2f} mm'),
                                row=2, col=2)  # 您可以自己改 row/col
                    traces_added_imag += 1
                else:
                    logger.warning(f"Skipping imag part plot for x={listxa[i]:.2f} mm due to non-finite values")

            logger.info(f"Added {traces_added_real} real traces and {traces_added_imag} imag traces to By Frequency Domain")

        except Exception as e:
            logger.error(f"Failed to save or plot listFSploy: {str(e)}")
            listFSploy_df = pd.DataFrame({'x_mm': listxa})
            listFSploy_df.to_csv('listFSploy.csv', index=False)
            with open('listFSploy.csv', 'a') as f:
                f.write(f"\nError: Failed to save listFSploy data: {str(e)}\n")
            logger.info("Saved partial listFSploy.csv with error message")

        # 強制輸出 by_time_ploy 到 CSV
        t = np.linspace(-T, T, noptd)
        try:
            # 將 by_time_ploy 轉換為 np.ndarray
            if isinstance(by_time_ploy, list):
                logger.warning("by_time_ploy is a list, attempting to convert to np.ndarray")
                by_time_ploy = np.array(by_time_ploy, dtype=float)
                logger.debug(f"Converted by_time_ploy shape: {by_time_ploy.shape}")
            elif not isinstance(by_time_ploy, np.ndarray):
                logger.error(f"by_time_ploy is invalid type: {type(by_time_ploy)}, using zero array")
                by_time_ploy = np.zeros((nopxa, noptd))

            # 確保形狀正確
            if by_time_ploy.shape != (nopxa, noptd):
                logger.warning(f"by_time_ploy shape {by_time_ploy.shape} does not match expected ({nopxa}, {noptd}), reshaping or padding")
                by_time_ploy = np.zeros((nopxa, noptd))  # 形狀不符時用零填充

            # 記錄數據狀態
            logger.debug(f"by_time_ploy min: {np.min(by_time_ploy):.2e}, max: {np.max(by_time_ploy):.2e}")
            finite_count = np.sum(np.isfinite(by_time_ploy))
            logger.debug(f"by_time_ploy finite values: {finite_count}/{by_time_ploy.size}")
            if finite_count < by_time_ploy.size:
                non_finite_indices = np.where(~np.isfinite(by_time_ploy))
                logger.warning(f"Non-finite values in by_time_ploy at indices: {list(zip(non_finite_indices[0], non_finite_indices[1]))}")

            # 生成 CSV
            by_time_df = pd.DataFrame({
                'x_mm': listxa,
                **{f'By_t_{t[j]:.2e}_s': by_time_ploy[:, j] for j in range(noptd)}
            })
            by_time_df.to_csv('by_time_ploy.csv', index=False)
            logger.info("Saved by_time_ploy.csv")

            # 儲存時間序列 t 到獨立 CSV
            time_df = pd.DataFrame({
                'time_s': t
            })
            time_df.to_csv('t.csv', index=False)
            logger.info("Saved t.csv")



            # 繪製有效數據 - 強制包含第一點和最後一點 + 中間均勻選點
            logger.info("Plotting By(x,t) at selected x positions (including first and last)")
            

            # 現在畫圖
            traces_added = 0
            for i in selected_indices:
                if np.all(np.isfinite(by_time_ploy[i, :])):
                    fig.add_trace(go.Scatter(x=t, y=by_time_ploy[i, :], mode='lines',
                                            name=f'By at x={listxa[i]:.2f} mm ({"first" if i==0 else "last" if i==nopxa-1 else "middle"})'),
                                row=3, col=1)
                    traces_added += 1
                else:
                    logger.warning(f"Skipping plot for x={listxa[i]:.2f} mm due to non-finite values")

            logger.info(f"Added {traces_added} By(x,t) traces at selected x positions")
        except Exception as e:
            logger.error(f"Failed to save by_time_ploy.csv: {str(e)}")
            by_time_df = pd.DataFrame({'x_mm': listxa})
            by_time_df.to_csv('by_time_ploy.csv', index=False)
            with open('by_time_ploy.csv', 'a') as f:
                f.write(f"\nError: Failed to save By time domain data: {str(e)}\n")
            logger.info("Saved partial by_time_ploy.csv with error message")


        # =====================================================
        # Boundary current evolution: Kz(panel, t)
        # =====================================================
        try:
            panel_df = pd.read_csv("boundary_panels.csv")
            current_df = pd.read_csv("boundary_current_time.csv")

            # Keep the panel ordering consistent.
            panel_df = panel_df.sort_values("panel_index").reset_index(drop=True)
            current_df = current_df.sort_values("panel_index").reset_index(drop=True)

            if not np.array_equal(
                panel_df["panel_index"].to_numpy(),
                current_df["panel_index"].to_numpy()
            ):
                raise ValueError("Panel indices do not match")

            # Extract all time-dependent Kz columns.
            k_columns = [
                c for c in current_df.columns
                if c.startswith("K_t_") and c.endswith("_s")
            ]

            K_boundary_time = current_df[k_columns].to_numpy(dtype=float)

            # Recover time from the column names.
            time_boundary_us = np.array([
                float(c[4:-2]) for c in k_columns
            ]) * 1e6

            panel_index = panel_df["panel_index"].to_numpy()

            fig.add_trace(
                go.Heatmap(
                    x=time_boundary_us,
                    y=panel_index,
                    z=K_boundary_time,
                    colorscale="RdBu_r",
                    zmid=0.0,
                    colorbar=dict(
                        title="Kz (A/m)",
                        len=0.16,
                        y=0.36,
                        yanchor="middle"
                    ),
                    name="Boundary Kz",
                    hovertemplate=(
                        "Time = %{x:.3f} μs<br>"
                        "Panel = %{y}<br>"
                        "Kz = %{z:.4e} A/m"
                        "<extra></extra>"
                    )
                ),
                row=3, col=2
            )

        except Exception as e:
            logger.error(
                f"Failed to plot boundary current evolution: {str(e)}"
            )

        # =====================================================
        # Induced By(x,t) evolution
        # =====================================================
        try:
            x_heatmap_mm = np.asarray(listxa, dtype=float)
            t_heatmap_us = np.asarray(t, dtype=float) * 1e6
            By_heatmap = np.asarray(by_time_ploy, dtype=float)

            if By_heatmap.shape != (
                len(x_heatmap_mm), len(t_heatmap_us)
            ):
                raise ValueError(
                    f"By heatmap shape {By_heatmap.shape} does not match "
                    f"({len(x_heatmap_mm)}, {len(t_heatmap_us)})"
                )

            fig.add_trace(
                go.Heatmap(
                    x=t_heatmap_us,
                    y=x_heatmap_mm,
                    z=By_heatmap,
                    colorscale="RdBu_r",
                    zmid=0.0,
                    colorbar=dict(
                        title="By (T)",
                        len=0.16,
                        y=0.0875,
                        yanchor="middle"
                    ),
                    name="Induced By(x,t)",
                    hovertemplate=(
                        "Time = %{x:.3f} μs<br>"
                        "x = %{y:.3f} mm<br>"
                        "By = %{z:.4e} T"
                        "<extra></extra>"
                    )
                ),
                row=4, col=1
            )

        except Exception as e:
            logger.error(
                f"Failed to plot induced By(x,t) evolution: {str(e)}"
            )

        
        # 存 CSV：時間 vs 多極係數
        try:
            coeff_names = [f'c{i}' for i in range(nfts, -1, -1)]  # c_nfts 到 c0
            df = pd.DataFrame(by_coeffs_ploy, columns=coeff_names)
            df['time_us'] = t * 1e6  # 時間轉 μs，更易讀
            df = df[['time_us'] + coeff_names]  # 時間放第一欄
            df.to_csv('by_fitted_coeffs_vs_time.csv', index=False)
            logger.info("Saved by_fitted_coeffs_vs_time.csv")
        except Exception as e:
            logger.error(f"Failed to save by_fitted_coeffs_vs_time.csv: {str(e)}")

        # 畫圖：每個多極係數 vs 時間（子圖）
        try:
            
            fig_multipole = make_subplots(
                rows=nfts + 1, cols=1,
                shared_xaxes=True,
                subplot_titles=[f'Fitted Coefficient c{i} (order {i}) vs Time' for i in range(nfts, -1, -1)],
                vertical_spacing=0.02
            )
            
            colors = px.colors.qualitative.Plotly
            for order in range(nfts + 1):  # order = 0 是最高次 c_nfts
                row = order + 1
                power = nfts - order
                coeff_data = [coeffs[order] for coeffs in by_coeffs_ploy]  # 第 order 項係數
                fig_multipole.add_trace(
                    go.Scatter(
                        x=t * 1e6, y=coeff_data,
                        mode='lines',
                        name=f'c{nfts - order} (order {nfts - order})',
                        line=dict(color=colors[order % len(colors)])
                    ),
                    row=row, col=1
                )
                fig_multipole.update_yaxes(
                    title_text=rf"Coefficient ($\mathrm{{T}}/\mathrm{{m}}^{{{power}}}$)",
                    row=row, col=1
                )
            
            fig_multipole.update_layout(
                height=300 * (nfts + 1),
                title="Multipole Coefficients vs Time",
                showlegend=False
            )
            fig_multipole.update_xaxes(title_text="Time (μs)", row=nfts + 1, col=1)
            
            # 加到您的主 fig 或單獨顯示
            # fig.add_traces(fig_multipole.data)  # 如果想加到主圖
            fig_multipole.show()  # 單獨開新視窗顯示
            
        except Exception as e:
            logger.error(f"Failed to plot multipole coefficients: {str(e)}")

        fig.update_layout(
            height=1300, width=1000,
            showlegend=True,
            title_text="Post-process Results"
        )
        fig.update_xaxes(title_text="Frequency (Hz-rad)", row=1, col=1)
        fig.update_yaxes(title_text="Fourier Coefficients", row=1, col=1)
        fig.update_xaxes(title_text="t (s)", row=1, col=2)
        fig.update_yaxes(title_text="Time Series", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz-rad)", row=2, col=1)
        fig.update_yaxes(title_text="Im(By) (Hz)", row=2, col=1)  # 更新 y 軸標籤
        fig.update_xaxes(title_text="Frequency (Hz-rad)", row=2, col=2)
        fig.update_yaxes(title_text="Re(By) (Hz)", row=2, col=2)
        fig.update_xaxes(title_text="t (s)", row=3, col=1)
        fig.update_yaxes(title_text="By (T)", row=3, col=1)

        fig.update_xaxes(title_text="Time (μs)", row=3, col=2)
        fig.update_yaxes(title_text="Boundary panel index", row=3, col=2)

        fig.update_xaxes(title_text="Time (μs)", row=4, col=1)
        fig.update_yaxes(title_text="x (mm)", row=4, col=1)

        try:
            html_file = os.path.abspath("post_plots.html")
            fig.write_html(html_file)
            webbrowser.open(f"file://{html_file}")
            logger.info("Saved and opened post_plots.html")
        except Exception as e:
            logger.error(f"Failed to save or open post_plots.html: {str(e)}")

        # 開新視窗畫全部 m 的子圖
        # =====================================================
        # Plot all ellipse Fourier orders
        # =====================================================

        if shape == 'ellipse' and len(listFS) > 0:

            logger.info("Generating all ellipse Fourier modes")

            FS_cos = np.asarray(listFS, dtype=complex)
            FS_sin = np.asarray(listFS_sin, dtype=complex)

            TS_cos = np.asarray(listTS, dtype=float)
            TS_sin = np.asarray(listTS_sin, dtype=float)

            num_modes = FS_cos.shape[0]

            t = np.linspace(-T, T, noptd)

            vertical_spacing = min(
                0.04,
                0.8 / max(num_modes - 1, 1)
            )

            fig_all = make_subplots(
                rows=num_modes,
                cols=2,
                subplot_titles=[
                    title
                    for m in range(num_modes)
                    for title in [
                        f"m={m} Frequency Domain",
                        f"m={m} Time Domain"
                    ]
                ],
                vertical_spacing=vertical_spacing,
                horizontal_spacing=0.08
            )

            for m in range(num_modes):

                # ---------------------------------------------
                # Frequency domain: cosine
                # ---------------------------------------------

                fig_all.add_trace(
                    go.Scatter(
                        x=omega,
                        y=FS_cos[m].real,
                        mode='lines',
                        name=f'Cos Real m={m}'
                    ),
                    row=m+1,
                    col=1
                )

                fig_all.add_trace(
                    go.Scatter(
                        x=omega,
                        y=FS_cos[m].imag,
                        mode='lines',
                        name=f'Cos Imag m={m}'
                    ),
                    row=m+1,
                    col=1
                )

                # Frequency domain: sine
                if m > 0:

                    fig_all.add_trace(
                        go.Scatter(
                            x=omega,
                            y=FS_sin[m].real,
                            mode='lines',
                            name=f'Sin Real m={m}'
                        ),
                        row=m+1,
                        col=1
                    )

                    fig_all.add_trace(
                        go.Scatter(
                            x=omega,
                            y=FS_sin[m].imag,
                            mode='lines',
                            name=f'Sin Imag m={m}'
                        ),
                        row=m+1,
                        col=1
                    )

                # ---------------------------------------------
                # Time domain
                # ---------------------------------------------

                fig_all.add_trace(
                    go.Scatter(
                        x=t,
                        y=TS_cos[m],
                        mode='lines',
                        name=f'Cos Time m={m}'
                    ),
                    row=m+1,
                    col=2
                )

                if m > 0:

                    fig_all.add_trace(
                        go.Scatter(
                            x=t,
                            y=TS_sin[m],
                            mode='lines',
                            name=f'Sin Time m={m}'
                        ),
                        row=m+1,
                        col=2
                    )

            fig_all.update_layout(
                height=300 * num_modes,
                width=1400,
                title="Ellipse Boundary Fourier Analysis",
                showlegend=True
            )

            fig_all.update_xaxes(
                title_text="Angular Frequency (rad/s)",
                row=num_modes,
                col=1
            )

            fig_all.update_xaxes(
                title_text="Time (s)",
                row=num_modes,
                col=2
            )

            fig_all.update_yaxes(
                title_text="Fourier Coefficient",
                row=num_modes,
                col=1
            )

            fig_all.update_yaxes(
                title_text="Aeddy (T*m)",
                row=num_modes,
                col=2
            )

            html_file_all = os.path.abspath(
                "all_ellipse_fourier_orders.html"
            )

            fig_all.write_html(html_file_all)

            webbrowser.open(f"file://{html_file_all}")

            logger.info(
                "Saved and opened all_ellipse_fourier_orders.html"
            )

        # 動畫：B_y(x) 隨時間 t 變化（自動開新視窗）
        logger.info("Generating B_y(x,t) animation")
        try:
            # 確保資料形狀正確
            if by_time_ploy.shape != (nopxa, noptd):
                raise ValueError(f"by_time_ploy shape {by_time_ploy.shape} != expected ({nopxa}, {noptd})")
            
            # 轉置成 (時間幀, x 點數)，每行是一條曲線
            By_xt = by_time_ploy.T  # 現在形狀 (noptd, nopxa)
            
            # x 軸座標（假設您有 listxa 或 x_points）
            x_points = np.array(listxa)  # 或您原本的 x_points
            
            # 時間軸
            t_points = np.linspace(-T, T, noptd)
            
            # 關鍵：計算全局最大絕對值，並加 10% 邊距
            global_max_abs = np.max(np.abs(By_xt)) * 1.1  # 1.1 倍安全邊距，可改成 1.2 或 1.5
            y_range = [-global_max_abs, global_max_abs]
            
            fig_anim = go.Figure()
            
            # 初始曲線（第一幀）
            fig_anim.add_trace(
                go.Scatter(
                    x=x_points,
                    y=By_xt[0],
                    mode='lines',
                    name='B_y(x)',
                    line=dict(color='blue')
                )
            )
            
            # 建立動畫框架
            frames = []
            for i in range(noptd):
                frame = go.Frame(
                    data=[
                        go.Scatter(
                            x=x_points,
                            y=By_xt[i],
                            mode='lines',
                            line=dict(color='blue')
                        )
                    ],
                    name=f"t={t_points[i]:.2e} s",
                    traces=[0]  # 更新第 0 條 trace
                )
                frames.append(frame)
            
            fig_anim.frames = frames
            
            # 動畫控制按鈕 + 滑桿
            fig_anim.update_layout(
                title_text="Eddy Current Induced B_y(x) Evolution over Time",
                xaxis_title="x (mm)",
                yaxis_title="B_y (T)",
                height=600,
                width=1000,
                yaxis_range=y_range,  # ← 這裡固定縱軸範圍（關鍵！）
                yaxis=dict(range=y_range, fixedrange=True),  # 固定 y 軸，不讓使用者縮放改變
                updatemenus=[{
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                            'fromcurrent': True, 'mode': 'immediate'}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                            'mode': 'immediate', 'transition': {'duration': 0}}],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'font': {'size': 16},
                        'prefix': 'Time: ',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [[f.name], {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }],
                            'label': f"t={t:.2e} s",
                            'method': 'animate'
                        } for f, t in zip(fig_anim.frames, t_points)
                    ]
                }]
            )
            
            # 存成 HTML 並自動開新視窗
            html_file_anim = os.path.abspath("By_x_t_animation.html")
            fig_anim.write_html(html_file_anim)
            webbrowser.open(f"file://{html_file_anim}")
            logger.info("Saved and opened By_x_t_animation.html (B_y(x) evolution over time)")
            
        except Exception as e:
            logger.error(f"Failed to create/save B_y(x,t) animation: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())