"""Boundary-integral cross-check for the analytical elliptical solver.

This module is intentionally limited to a single, closed elliptical coating.
It implements the same straight-panel logarithmic Green-function method used
by thin_wall_eddy_solver.py, including its zero-net-longitudinal-current
constraint and the associated constant potential offset.

The BEM result is evaluated at exact points on the ellipse (not at chord
midpoints) before Fourier projection.  The BEM-induced potential is shifted
by the solved A_offset so that it uses the same zero-mean effective-potential
convention as the analytical m=0 equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .physics import MU0, Wire


@dataclass
class EllipseBEMComparison:
    panel_count: int
    psi_rad: np.ndarray
    omega_rad_s: np.ndarray
    offset_frequency: np.ndarray
    bem_induced_A_boundary_frequency: np.ndarray
    bem_effective_A_boundary_frequency: np.ndarray
    bem_current_boundary_frequency: np.ndarray
    bem_induced_A_cos_frequency: np.ndarray
    bem_induced_A_sin_frequency: np.ndarray
    bem_current_cos_frequency: np.ndarray
    bem_current_sin_frequency: np.ndarray
    bem_Bx_induced_frequency: np.ndarray
    bem_By_induced_frequency: np.ndarray
    net_current_relative_residual: np.ndarray
    equation_relative_residual: np.ndarray

    def save_csv(self, output_dir, analytical_result=None) -> list:
        from pathlib import Path
        import pandas as pd

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        rows = []
        n_modes = self.bem_induced_A_cos_frequency.shape[1]
        for mode in range(n_modes):
            for k, omega in enumerate(self.omega_rad_s):
                a = self.bem_induced_A_cos_frequency[k, mode]
                current = self.bem_current_cos_frequency[k, mode]
                raw_a = a + self.offset_frequency[k] if mode == 0 else a
                row = {
                    "component": "cos",
                    "mode_m": mode,
                    "omega_rad_s": omega,
                    "Aeddy_aligned_real_Wb_per_m": a.real,
                    "Aeddy_aligned_imag_Wb_per_m": a.imag,
                    "Aeddy_raw_real_Wb_per_m": raw_a.real,
                    "Aeddy_raw_imag_Wb_per_m": raw_a.imag,
                    "Kz_real_A_per_m": current.real,
                    "Kz_imag_A_per_m": current.imag,
                }
                if analytical_result is not None:
                    aa = analytical_result.induced_A_cos_frequency[k, mode]
                    ak = analytical_result.current_cos_frequency[k, mode]
                    row.update({
                        "Aeddy_analytical_real_Wb_per_m": aa.real,
                        "Aeddy_analytical_imag_Wb_per_m": aa.imag,
                        "Aeddy_aligned_difference_abs_Wb_per_m": abs(a - aa),
                        "Aeddy_aligned_difference_relative": abs(a - aa) / max(abs(a), abs(aa), 1e-30),
                        "Kz_analytical_real_A_per_m": ak.real,
                        "Kz_analytical_imag_A_per_m": ak.imag,
                        "Kz_difference_abs_A_per_m": abs(current - ak),
                        "Kz_difference_relative": abs(current - ak) / max(abs(current), abs(ak), 1e-30),
                    })
                rows.append(row)
        for mode in range(self.bem_induced_A_sin_frequency.shape[1]):
            for k, omega in enumerate(self.omega_rad_s):
                a = self.bem_induced_A_sin_frequency[k, mode]
                current = self.bem_current_sin_frequency[k, mode]
                row = {
                    "component": "sin",
                    "mode_m": mode + 1,
                    "omega_rad_s": omega,
                    "Aeddy_aligned_real_Wb_per_m": a.real,
                    "Aeddy_aligned_imag_Wb_per_m": a.imag,
                    "Aeddy_raw_real_Wb_per_m": a.real,
                    "Aeddy_raw_imag_Wb_per_m": a.imag,
                    "Kz_real_A_per_m": current.real,
                    "Kz_imag_A_per_m": current.imag,
                }
                if analytical_result is not None:
                    aa = analytical_result.induced_A_sin_frequency[k, mode]
                    ak = analytical_result.current_sin_frequency[k, mode]
                    row.update({
                        "Aeddy_analytical_real_Wb_per_m": aa.real,
                        "Aeddy_analytical_imag_Wb_per_m": aa.imag,
                        "Aeddy_aligned_difference_abs_Wb_per_m": abs(a - aa),
                        "Aeddy_aligned_difference_relative": abs(a - aa) / max(abs(a), abs(aa), 1e-30),
                        "Kz_analytical_real_A_per_m": ak.real,
                        "Kz_analytical_imag_A_per_m": ak.imag,
                        "Kz_difference_abs_A_per_m": abs(current - ak),
                        "Kz_difference_relative": abs(current - ak) / max(abs(current), abs(ak), 1e-30),
                    })
                rows.append(row)
        path = out / "ellipse_bem_fourier.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)

        diagnostic = pd.DataFrame({
            "omega_rad_s": self.omega_rad_s,
            "A_offset_real_Wb_per_m": self.offset_frequency.real,
            "A_offset_imag_Wb_per_m": self.offset_frequency.imag,
            "net_current_relative_residual": self.net_current_relative_residual,
            "equation_relative_residual": self.equation_relative_residual,
        })
        path = out / "ellipse_bem_diagnostics.csv"
        diagnostic.to_csv(path, index=False)
        paths.append(path)
        return paths


def _log_primitive(q: np.ndarray, abs_v: np.ndarray, reference_length_m: float) -> np.ndarray:
    rho2 = q * q + abs_v * abs_v
    log_r = 0.5 * np.log(np.maximum(rho2, np.finfo(float).tiny)) - np.log(reference_length_m)
    return q * (log_r - 1.0) + abs_v * np.arctan2(q, abs_v)


def _panel_geometry(a_m: float, b_m: float, panel_count: int):
    psi_vertices = 2.0 * np.pi * np.arange(panel_count) / panel_count
    points = np.column_stack((a_m * np.cos(psi_vertices), b_m * np.sin(psi_vertices)))
    starts = points
    ends = np.roll(points, -1, axis=0)
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-15):
        raise ValueError("BEM ellipse mesh contains a zero-length panel")
    tangents = vectors / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    midpoints = 0.5 * (starts + ends)
    return starts, ends, lengths, tangents, normals, midpoints


def _log_integral_matrix(targets: np.ndarray, lengths: np.ndarray,
                         tangents: np.ndarray, normals: np.ndarray,
                         midpoints: np.ndarray, reference_length_m: float = 1.0) -> np.ndarray:
    delta = targets[:, None, :] - midpoints[None, :, :]
    u = np.einsum("ijk,jk->ij", delta, tangents)
    v = np.einsum("ijk,jk->ij", delta, normals)
    av = np.abs(v)
    half = 0.5 * lengths[None, :]
    q_lo = u - half
    q_hi = u + half
    return (
        _log_primitive(q_hi, av, reference_length_m)
        - _log_primitive(q_lo, av, reference_length_m)
    )


def _wire_vector_potential(points_m: np.ndarray, wires: Sequence[Wire],
                           reference_length_m: float = 1.0) -> np.ndarray:
    az = np.zeros(points_m.shape[0], dtype=float)
    for wire in wires:
        wx = wire.x_mm * 1e-3
        wy = wire.y_mm * 1e-3
        radius = np.linalg.norm(points_m - np.array([wx, wy]), axis=1)
        if np.any(radius < 1e-14):
            raise ValueError("An external wire lies on a BEM boundary collocation point")
        az -= MU0 * wire.current_A / (2.0 * np.pi) * np.log(radius / reference_length_m)
    return az


def _b_field_kernels(observation_points_m: np.ndarray, lengths: np.ndarray,
                     tangents: np.ndarray, normals: np.ndarray,
                     midpoints: np.ndarray):
    delta = observation_points_m[:, None, :] - midpoints[None, :, :]
    u = np.einsum("ijk,jk->ij", delta, tangents)
    v = np.einsum("ijk,jk->ij", delta, normals)
    half = 0.5 * lengths[None, :]
    q1 = u - half
    q2 = u + half
    av = np.abs(v)
    den1 = q1 * q1 + v * v
    den2 = q2 * q2 + v * v
    on_panel = (av < 1e-13) & (np.abs(u) < half - 1e-13)
    if np.any(on_panel):
        raise ValueError("A B-field observation point lies on a BEM panel")
    tiny = np.finfo(float).tiny
    int_t = 0.5 * np.log(np.maximum(den2, tiny) / np.maximum(den1, tiny))
    int_n = np.zeros_like(int_t)
    mask = av > 1e-15
    int_n[mask] = np.sign(v[mask]) * (
        np.arctan2(q2[mask], av[mask]) - np.arctan2(q1[mask], av[mask])
    )
    int_dx = int_t * tangents[None, :, 0] + int_n * normals[None, :, 0]
    int_dy = int_t * tangents[None, :, 1] + int_n * normals[None, :, 1]
    factor = MU0 / (2.0 * np.pi)
    return -factor * int_dy, factor * int_dx


def _fourier_coefficients(values: np.ndarray, psi: np.ndarray, order: int):
    """Project sampled values onto standard cosine/sine Fourier coefficients."""
    n = psi.size
    modes = np.arange(order + 1)
    cos_basis = np.cos(np.outer(psi, modes))
    sin_basis = np.sin(np.outer(psi, modes[1:]))
    cos_coeff = np.empty((values.shape[0], order + 1), dtype=complex)
    cos_coeff[:, 0] = np.mean(values, axis=1)
    if order:
        cos_coeff[:, 1:] = (2.0 / n) * (values @ cos_basis[:, 1:])
        sin_coeff = (2.0 / n) * (values @ sin_basis)
    else:
        sin_coeff = np.zeros((values.shape[0], 0), dtype=complex)
    return cos_coeff, sin_coeff


def compare_elliptical_solution_to_bem(
    analytical_result,
    wires: Sequence[Wire],
    conductivity_Sm: float,
    thickness_um: float,
    panel_count: int = 192,
    progress: Callable[[int], None] | None = None,
) -> EllipseBEMComparison:
    """Run the straight-panel Green-function BEM with the analytical inputs.

    The BEM uses a separate, user-selectable panel count. Its induced A is
    evaluated on exact ellipse points and the solved group offset is removed
    before comparison, matching the analytical solver's m=0 convention.
    Pulse spectrum is applied to A, K, and B only after the frequency solve.
    """
    if panel_count < 32:
        raise ValueError("BEM panel count must be at least 32")
    if conductivity_Sm <= 0 or thickness_um <= 0:
        raise ValueError("Conductivity and coating thickness must be positive")

    a_m = float(analytical_result.a_m)
    b_m = float(analytical_result.b_m)
    omega = np.asarray(analytical_result.omega_rad_s, dtype=float)
    pulse = np.asarray(analytical_result.pulse_spectrum, dtype=float)
    order = int(len(analytical_result.mode_numbers) - 1)
    if panel_count < 8 * (order + 1):
        raise ValueError(
            f"BEM panel count ({panel_count}) should be at least "
            f"8*(Fourier order+1)={8*(order+1)}"
        )

    starts, ends, lengths, tangents, normals, midpoints = _panel_geometry(
        a_m, b_m, panel_count
    )
    drive_mid = _wire_vector_potential(midpoints, wires)
    H_mid = _log_integral_matrix(midpoints, lengths, tangents, normals, midpoints)

    # Evaluate the induced A at exact ellipse parameter midpoints rather than
    # at the geometric chord midpoints used by the collocation equations.
    psi = 2.0 * np.pi * (np.arange(panel_count) + 0.5) / panel_count
    targets = np.column_stack((a_m * np.cos(psi), b_m * np.sin(psi)))
    H_eval = _log_integral_matrix(targets, lengths, tangents, normals, midpoints)

    # A single closed ellipse has one zero-net-longitudinal-current constraint.
    C = np.ones(panel_count, dtype=complex)
    B = lengths.astype(complex)
    alpha_green = MU0 / (2.0 * np.pi)
    sigma_d = float(conductivity_Sm) * float(thickness_um) * 1e-6

    nf = omega.size
    K_unpulsed = np.zeros((nf, panel_count), dtype=complex)
    phi_unpulsed = np.zeros((nf, panel_count), dtype=complex)
    offset_unpulsed = np.zeros(nf, dtype=complex)
    net_residual = np.zeros(nf, dtype=float)
    eq_residual = np.zeros(nf, dtype=float)
    eye = np.eye(panel_count, dtype=complex)

    if progress:
        progress(5)
    for k, w in enumerate(omega):
        beta = -1j * float(w) * sigma_d
        if abs(w) < 1e-30:
            continue
        lhs11 = eye + beta * alpha_green * H_mid
        lhs = np.empty((panel_count + 1, panel_count + 1), dtype=complex)
        lhs[:panel_count, :panel_count] = lhs11
        lhs[:panel_count, panel_count] = C
        lhs[panel_count, :panel_count] = B
        lhs[panel_count, panel_count] = 0.0
        rhs = np.empty(panel_count + 1, dtype=complex)
        rhs[:panel_count] = beta * drive_mid
        rhs[panel_count] = 0.0
        solved = np.linalg.solve(lhs, rhs)
        current = solved[:panel_count]
        u = solved[panel_count]
        offset = u / beta
        K_unpulsed[k] = current
        offset_unpulsed[k] = offset
        phi_unpulsed[k] = -alpha_green * (H_eval @ current)

        net = np.dot(lengths, current)
        net_scale = np.dot(lengths, np.abs(current))
        net_residual[k] = abs(net) / max(net_scale, 1e-30)
        equation = lhs11 @ current + C * u - beta * drive_mid
        eq_scale = max(np.linalg.norm(beta * drive_mid), np.linalg.norm(current), 1e-30)
        eq_residual[k] = np.linalg.norm(equation) / eq_scale
        if progress:
            progress(5 + int(75 * (k + 1) / nf))

    # Align the gauge/longitudinal-voltage convention with the analytical
    # m=0 equation: A_induced,aligned = A_induced,BEM - A_offset.
    effective_phi_unpulsed = phi_unpulsed - offset_unpulsed[:, None]
    bem_phi = effective_phi_unpulsed * pulse[:, None]
    bem_current = K_unpulsed * pulse[:, None]

    a_cos, a_sin = _fourier_coefficients(bem_phi, psi, order)
    k_cos, k_sin = _fourier_coefficients(bem_current, psi, order)

    bx_kernel, by_kernel = _b_field_kernels(
        np.column_stack((analytical_result.x_observation_m,
                         np.zeros_like(analytical_result.x_observation_m))),
        lengths, tangents, normals, midpoints,
    )
    bx = (K_unpulsed @ bx_kernel.T) * pulse[:, None]
    by = (K_unpulsed @ by_kernel.T) * pulse[:, None]
    if progress:
        progress(100)

    return EllipseBEMComparison(
        panel_count=panel_count,
        psi_rad=psi,
        omega_rad_s=omega,
        offset_frequency=offset_unpulsed * pulse,
        bem_induced_A_boundary_frequency=phi_unpulsed * pulse[:, None],
        bem_effective_A_boundary_frequency=bem_phi,
        bem_current_boundary_frequency=bem_current,
        bem_induced_A_cos_frequency=a_cos,
        bem_induced_A_sin_frequency=a_sin,
        bem_current_cos_frequency=k_cos,
        bem_current_sin_frequency=k_sin,
        bem_Bx_induced_frequency=bx,
        bem_By_induced_frequency=by,
        net_current_relative_residual=net_residual,
        equation_relative_residual=eq_residual,
    )
