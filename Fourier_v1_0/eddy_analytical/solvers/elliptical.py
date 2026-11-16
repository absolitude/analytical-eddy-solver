"""Analytical thin-coating eddy-current solver for an elliptical chamber.

The implementation follows the cosine/sine separation in the user's draft,
with a Galerkin projection of the boundary scale factor h(psi).  The sign of
the sheet jump is kept consistent with the package-wide exp(+i*omega*t)
convention and with the circular solver:

    dA_out/dn - dA_in/dn = +i*mu0*sigma*d*omega*A_total.

The cosine sector includes m=0.  Its m=0 equation enforces zero net
longitudinal current on the closed coating.  The sine sector starts at m=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..physics import (
    MU0,
    Wire,
    box_spectrum,
    half_sine_spectrum,
    inverse_fourier,
    magnetic_field_from_wires,
    vector_potential_from_wires,
)


@dataclass
class EllipticalResult:
    a_m: float
    b_m: float
    conductivity_Sm: float
    thickness_m: float
    omega_rad_s: np.ndarray
    time_s: np.ndarray
    psi_rad: np.ndarray
    mode_numbers: np.ndarray
    x_observation_m: np.ndarray
    scale_factor_m: np.ndarray
    applied_A_cos: np.ndarray
    applied_A_sin: np.ndarray
    induced_A_cos_frequency: np.ndarray
    induced_A_sin_frequency: np.ndarray
    current_cos_frequency: np.ndarray
    current_sin_frequency: np.ndarray
    induced_A_boundary_frequency: np.ndarray
    current_boundary_frequency: np.ndarray
    induced_A_boundary_time: np.ndarray
    current_boundary_time: np.ndarray
    Bx_induced_frequency: np.ndarray
    By_induced_frequency: np.ndarray
    Bx_applied_frequency: np.ndarray
    By_applied_frequency: np.ndarray
    Bx_induced_time: np.ndarray
    By_induced_time: np.ndarray
    Bx_total_time: np.ndarray
    By_total_time: np.ndarray
    pulse_spectrum: np.ndarray

    def save_csv(self, output_dir: str | Path) -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        rows = []
        for j, m in enumerate(self.mode_numbers):
            for k, w in enumerate(self.omega_rad_s):
                rows.append({
                    "component": "cos",
                    "mode_m": int(m),
                    "omega_rad_s": w,
                    "Aapp": self.applied_A_cos[j],
                    "Aeddy_real": self.induced_A_cos_frequency[k, j].real,
                    "Aeddy_imag": self.induced_A_cos_frequency[k, j].imag,
                    "Kz_real": self.current_cos_frequency[k, j].real,
                    "Kz_imag": self.current_cos_frequency[k, j].imag,
                })
        for j, m in enumerate(self.mode_numbers[1:]):
            for k, w in enumerate(self.omega_rad_s):
                rows.append({
                    "component": "sin",
                    "mode_m": int(m),
                    "omega_rad_s": w,
                    "Aapp": self.applied_A_sin[j],
                    "Aeddy_real": self.induced_A_sin_frequency[k, j].real,
                    "Aeddy_imag": self.induced_A_sin_frequency[k, j].imag,
                    "Kz_real": self.current_sin_frequency[k, j].real,
                    "Kz_imag": self.current_sin_frequency[k, j].imag,
                })
        path = out / "elliptical_fourier_frequency.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)

        boundary_rows = []
        for it, t in enumerate(self.time_s):
            for ip, psi in enumerate(self.psi_rad):
                boundary_rows.append({
                    "time_s": t,
                    "psi_rad": psi,
                    "x_mm": 1e3 * self.a_m * np.cos(psi),
                    "y_mm": 1e3 * self.b_m * np.sin(psi),
                    "scale_factor_m": self.scale_factor_m[ip],
                    "Aeddy_time_Wb_per_m": self.induced_A_boundary_time[ip, it],
                    "Kz_time_A_per_m": self.current_boundary_time[ip, it],
                })
        path = out / "elliptical_boundary_time.csv"
        pd.DataFrame(boundary_rows).to_csv(path, index=False)
        written.append(path)

        field_rows = []
        for ix, x in enumerate(self.x_observation_m):
            for it, t in enumerate(self.time_s):
                field_rows.append({
                    "x_mm": 1e3 * x,
                    "y_mm": 0.0,
                    "time_s": t,
                    "Bx_eddy_T": self.Bx_induced_time[ix, it],
                    "By_eddy_T": self.By_induced_time[ix, it],
                    "Bx_total_T": self.Bx_total_time[ix, it],
                    "By_total_T": self.By_total_time[ix, it],
                })
        path = out / "elliptical_axis_field_time.csv"
        pd.DataFrame(field_rows).to_csv(path, index=False)
        written.append(path)
        return written


class EllipticalSolver:
    """Fourier/Galerkin analytical solver for a thin elliptical coating."""

    def __init__(
        self,
        a_m: float,
        b_m: float,
        conductivity_Sm: float,
        thickness_m: float,
        fourier_order: int,
        boundary_points: int = 2048,
    ) -> None:
        self.a_m = float(a_m)
        self.b_m = float(b_m)
        self.conductivity_Sm = float(conductivity_Sm)
        self.thickness_m = float(thickness_m)
        self.fourier_order = int(fourier_order)
        self.boundary_points = int(boundary_points)

        if self.a_m <= 0 or self.b_m <= 0:
            raise ValueError("Ellipse semi-axes must be positive")
        if self.a_m <= self.b_m:
            raise ValueError("This implementation assumes a >= b (major axis along x)")
        if (self.a_m - self.b_m) / self.a_m < 1e-10:
            raise ValueError("Ellipse is numerically indistinguishable from a circle; use Circular")
        if self.conductivity_Sm <= 0 or self.thickness_m <= 0:
            raise ValueError("Conductivity and thickness must be positive")
        if self.fourier_order < 1:
            raise ValueError("fourier_order must be at least 1")
        if self.boundary_points < max(128, 8 * (self.fourier_order + 1)):
            raise ValueError("boundary_points is too small for the selected Fourier order")
        if self.fourier_order >= self.boundary_points // 2:
            raise ValueError("fourier_order must be below the boundary Nyquist order")

        self.f_m = float(np.sqrt((self.a_m - self.b_m) * (self.a_m + self.b_m)))
        self.eta0 = float(np.arctanh(self.b_m / self.a_m))

    @staticmethod
    def _h_matrix(psi: np.ndarray, h: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
        """Build 4*mean(h*b_n*b_m) Galerkin matrices.

        Doubling the n=0 equation relative to the usual mean projection does
        not change its homogeneous zero-current constraint and makes Hc
        symmetric positive definite.
        """
        modes = np.arange(order + 1)
        cos_basis = np.cos(np.outer(modes, psi))
        sin_modes = np.arange(1, order + 1)
        sin_basis = np.sin(np.outer(sin_modes, psi))
        Hc = (4.0 / psi.size) * ((cos_basis * h[None, :]) @ cos_basis.T)
        Hs = (4.0 / psi.size) * ((sin_basis * h[None, :]) @ sin_basis.T)
        return Hc, Hs

    def _applied_boundary_fourier(
        self, wires: Sequence[Wire]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        psi = 2.0 * np.pi * np.arange(self.boundary_points) / self.boundary_points
        x = self.a_m * np.cos(psi)
        y = self.b_m * np.sin(psi)
        for wire in wires:
            rho2 = (wire.x_mm * 1e-3 / self.a_m) ** 2 + (wire.y_mm * 1e-3 / self.b_m) ** 2
            if rho2 <= 1.0:
                raise ValueError(
                    "For the elliptical boundary-value model, every applied wire "
                    "must be outside the ellipse."
                )
        A = vector_potential_from_wires(x, y, wires)
        modes = np.arange(self.fourier_order + 1)
        c = np.cos(np.outer(modes, psi))
        s = np.sin(np.outer(modes[1:], psi))
        A_cos = (2.0 / psi.size) * (c @ A)
        # The constant Fourier mode uses 1/N, while m >= 1 uses 2/N.
        # This is important for the absolute m=0 potential convention.
        A_cos[0] *= 0.5
        A_sin = (2.0 / psi.size) * (s @ A)
        return psi, A_cos, A_sin, np.sqrt(
            self.a_m**2 * np.sin(psi) ** 2 + self.b_m**2 * np.cos(psi) ** 2
        )

    def _radial_jump_diagonals(self) -> tuple[np.ndarray, np.ndarray]:
        """Return D_m = m*exp(m*eta0)/cosh or sinh(m*eta0).

        These are evaluated with exp(-2*m*eta0) ratios to avoid overflow and
        to retain the circular limit.
        """
        m = np.arange(1, self.fourier_order + 1, dtype=float)
        em = np.exp(-2.0 * m * self.eta0)
        Dcos = 2.0 * m / (1.0 + em)
        Dsin = 2.0 * m / (1.0 - em)
        return np.concatenate(([0.0], Dcos)), Dsin

    def _axis_basis_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Derivatives of normalized interior cosine/sine harmonic bases.

        q_m(z)=T_m(z/f)/cosh(m*eta0), r_m(z)=T_m(z/f)/sinh(m*eta0).
        Only real z=x is needed for the horizontal-axis output. Recurrences
        are scaled so the near-circular limit does not overflow.
        """
        z = np.asarray(x, dtype=complex)
        M = self.fourier_order
        qprime = np.zeros((M + 1, z.size), dtype=complex)
        rprime = np.zeros((M + 1, z.size), dtype=complex)

        q_prev = np.ones_like(z)
        q_curr = z / self.a_m
        qp_prev = np.zeros_like(z)
        qp_curr = np.full_like(z, 1.0 / self.a_m)
        qprime[1] = qp_curr

        exp_eta = np.exp(-2.0 * self.eta0)
        for m in range(1, M):
            em = np.exp(-2.0 * m * self.eta0)
            enext = em * exp_eta
            coef_a = (2.0 / (self.a_m + self.b_m)) * (1.0 + em) / (1.0 + enext)
            coef_b = exp_eta * (1.0 + em / exp_eta) / (1.0 + enext)
            q_next = coef_a * z * q_curr - coef_b * q_prev
            qp_next = coef_a * (q_curr + z * qp_curr) - coef_b * qp_prev
            q_prev, q_curr = q_curr, q_next
            qp_prev, qp_curr = qp_curr, qp_next
            qprime[m + 1] = qp_curr

        if M >= 1:
            r_prev = z / self.b_m
            rp_prev = np.full_like(z, 1.0 / self.b_m)
            rprime[1] = rp_prev
        if M >= 2:
            r_curr = (z**2 - 0.5 * self.f_m**2) / (self.a_m * self.b_m)
            rp_curr = 2.0 * z / (self.a_m * self.b_m)
            rprime[2] = rp_curr
            for m in range(2, M):
                em = np.exp(-2.0 * m * self.eta0)
                enext = em * exp_eta
                eminus = em / exp_eta
                coef_a = (2.0 / (self.a_m + self.b_m)) * (1.0 - em) / (1.0 - enext)
                coef_b = exp_eta * (1.0 - eminus) / (1.0 - enext)
                r_next = coef_a * z * r_curr - coef_b * r_prev
                rp_next = coef_a * (r_curr + z * rp_curr) - coef_b * rp_prev
                r_prev, r_curr = r_curr, r_next
                rp_prev, rp_curr = rp_curr, rp_next
                rprime[m + 1] = rp_curr
        return qprime, rprime

    def solve(
        self,
        wires: Sequence[Wire],
        omega_rad_s: np.ndarray,
        pulse_spectrum: np.ndarray,
        time_s: np.ndarray,
        observation_x_m: np.ndarray | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> EllipticalResult:
        omega = np.asarray(omega_rad_s, dtype=float)
        pulse = np.asarray(pulse_spectrum, dtype=float)
        time = np.asarray(time_s, dtype=float)
        if omega.ndim != 1 or pulse.shape != omega.shape:
            raise ValueError("omega_rad_s and pulse_spectrum must be same-length 1D arrays")
        if omega.size < 3 or not np.all(np.diff(omega) > 0):
            raise ValueError("omega_rad_s must be strictly increasing with at least 3 points")
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time_s must contain at least two points")

        psi, Aapp_cos, Aapp_sin, h = self._applied_boundary_fourier(wires)
        Hc, Hs = self._h_matrix(psi, h, self.fourier_order)
        Dcos, Dsin = self._radial_jump_diagonals()
        if progress:
            progress(10)

        alpha = 0.5 * MU0 * self.conductivity_Sm * self.thickness_m * omega
        drive_cos = pulse[:, None] * Aapp_cos[None, :]
        drive_sin = pulse[:, None] * Aapp_sin[None, :]
        Ccos = np.zeros_like(drive_cos, dtype=complex)
        Csin = np.zeros_like(drive_sin, dtype=complex)

        for k, ak in enumerate(alpha):
            if ak == 0.0:
                continue
            Mc = np.diag(Dcos.astype(complex)) + 1j * ak * Hc
            Ms = np.diag(Dsin.astype(complex)) + 1j * ak * Hs
            Ccos[k] = np.linalg.solve(Mc, -1j * ak * (Hc @ drive_cos[k]))
            Csin[k] = np.linalg.solve(Ms, -1j * ak * (Hs @ drive_sin[k]))
            if progress and (k + 1) % max(1, omega.size // 10) == 0:
                progress(10 + int(40 * (k + 1) / omega.size))

        total_cos = drive_cos + Ccos
        total_sin = drive_sin + Csin
        current_factor = (-1j * omega * self.conductivity_Sm * self.thickness_m)[:, None]
        Kcos = current_factor * total_cos
        Ksin = current_factor * total_sin

        modes = np.arange(self.fourier_order + 1)
        cos_basis = np.cos(np.outer(modes, psi))
        sin_basis = np.sin(np.outer(modes[1:], psi))
        A_boundary_freq = Ccos @ cos_basis + Csin @ sin_basis
        K_boundary_freq = Kcos @ cos_basis + Ksin @ sin_basis
        A_boundary_time = inverse_fourier(omega, A_boundary_freq, time)
        K_boundary_time = inverse_fourier(omega, K_boundary_freq, time)
        if progress:
            progress(60)

        if observation_x_m is None:
            x_obs = np.linspace(-0.9 * self.a_m, 0.9 * self.a_m, 181)
        else:
            x_obs = np.asarray(observation_x_m, dtype=float)
        if x_obs.ndim != 1 or np.any(np.abs(x_obs) >= self.a_m):
            raise ValueError("Horizontal-axis observation points must satisfy |x| < a")

        qprime, rprime = self._axis_basis_derivatives(x_obs)
        # Cartesian field from A_z = sum(c_m*Re[q_m(z)] + s_m*Im[r_m(z)]).
        # On the real x-axis q'_m and r'_m are real; equations below retain
        # complex frequency coefficients without applying a complex Re().
        Bx_eddy_freq = (
            -Ccos[:, 1:] @ qprime[1:].imag
            + Csin @ rprime[1:].real
        )
        By_eddy_freq = (
            -Ccos[:, 1:] @ qprime[1:].real
            - Csin @ rprime[1:].imag
        )
        Bx_app_static, By_app_static = magnetic_field_from_wires(
            x_obs, np.zeros_like(x_obs), wires
        )
        Bx_app_freq = pulse[:, None] * Bx_app_static[None, :]
        By_app_freq = pulse[:, None] * By_app_static[None, :]
        Bx_eddy_time = inverse_fourier(omega, Bx_eddy_freq, time)
        By_eddy_time = inverse_fourier(omega, By_eddy_freq, time)
        Bx_app_time = inverse_fourier(omega, Bx_app_freq, time)
        By_app_time = inverse_fourier(omega, By_app_freq, time)
        if progress:
            progress(100)

        return EllipticalResult(
            a_m=self.a_m,
            b_m=self.b_m,
            conductivity_Sm=self.conductivity_Sm,
            thickness_m=self.thickness_m,
            omega_rad_s=omega,
            time_s=time,
            psi_rad=psi,
            mode_numbers=modes,
            x_observation_m=x_obs,
            scale_factor_m=h,
            applied_A_cos=Aapp_cos,
            applied_A_sin=Aapp_sin,
            induced_A_cos_frequency=Ccos,
            induced_A_sin_frequency=Csin,
            current_cos_frequency=Kcos,
            current_sin_frequency=Ksin,
            induced_A_boundary_frequency=A_boundary_freq,
            current_boundary_frequency=K_boundary_freq,
            induced_A_boundary_time=A_boundary_time,
            current_boundary_time=K_boundary_time,
            Bx_induced_frequency=Bx_eddy_freq,
            By_induced_frequency=By_eddy_freq,
            Bx_applied_frequency=Bx_app_freq,
            By_applied_frequency=By_app_freq,
            Bx_induced_time=Bx_eddy_time,
            By_induced_time=By_eddy_time,
            Bx_total_time=Bx_eddy_time + Bx_app_time,
            By_total_time=By_eddy_time + By_app_time,
            pulse_spectrum=pulse,
        )


def calculate_elliptical_case(
    *,
    a_mm: float,
    b_mm: float,
    conductivity_Sm: float,
    thickness_um: float,
    pulse_length_us: float,
    frequency_cutoff_C: float,
    frequency_points: int,
    time_points: int,
    fourier_order: int,
    boundary_points: int,
    wires: Sequence[Wire],
    x_axis_points: int | None = None,
    pulse_type: str = "half_sine",
    progress: Callable[[int], None] | None = None,
) -> EllipticalResult:
    T = float(pulse_length_us) * 1e-6
    if T <= 0 or frequency_cutoff_C <= 0:
        raise ValueError("Pulse length and frequency cutoff must be positive")
    if frequency_points < 3 or frequency_points % 2 == 0:
        raise ValueError("frequency_points must be an odd integer >= 3")
    if time_points < 2:
        raise ValueError("time_points must be >= 2")
    omega_max = float(frequency_cutoff_C) * np.pi / T
    omega = np.linspace(-omega_max, omega_max, int(frequency_points))
    if pulse_type == "half_sine":
        pulse = half_sine_spectrum(omega, T)
    elif pulse_type == "rectangular":
        pulse = box_spectrum(omega, T)
    else:
        raise ValueError(f"Unsupported pulse_type: {pulse_type}")
    time = np.linspace(-T, T, int(time_points))
    solver = EllipticalSolver(
        a_m=float(a_mm) * 1e-3,
        b_m=float(b_mm) * 1e-3,
        conductivity_Sm=conductivity_Sm,
        thickness_m=float(thickness_um) * 1e-6,
        fourier_order=fourier_order,
        boundary_points=boundary_points,
    )
    observation_x_m = None
    if x_axis_points is not None:
        if int(x_axis_points) < 2:
            raise ValueError("x_axis_points must be >= 2")
        observation_x_m = np.linspace(
            -0.9 * solver.a_m, 0.9 * solver.a_m, int(x_axis_points)
        )
    return solver.solve(
        wires, omega, pulse, time, observation_x_m=observation_x_m, progress=progress
    )
