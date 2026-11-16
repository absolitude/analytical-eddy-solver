"""Analytical thin-coating eddy-current solution for a circular chamber."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..physics import (
    MU0,
    Wire,
    half_sine_spectrum,
    inverse_fourier,
    magnetic_field_from_wires,
    vector_potential_from_wires,
)


@dataclass
class CircularResult:
    radius_m: float
    conductivity_Sm: float
    thickness_m: float
    omega_rad_s: np.ndarray
    time_s: np.ndarray
    theta_rad: np.ndarray
    mode_numbers: np.ndarray
    x_observation_m: np.ndarray
    # Static applied boundary Fourier coefficients (cos/sin, shape [mode])
    applied_A_cos: np.ndarray
    applied_A_sin: np.ndarray
    # Frequency-domain coefficients, shape [frequency, mode]
    induced_A_cos_frequency: np.ndarray
    induced_A_sin_frequency: np.ndarray
    current_cos_frequency: np.ndarray
    current_sin_frequency: np.ndarray
    # Boundary distributions, frequency-domain shape [frequency, theta]
    induced_A_boundary_frequency: np.ndarray
    current_boundary_frequency: np.ndarray
    # Boundary distributions, time-domain shape [theta, time]
    induced_A_boundary_time: np.ndarray
    current_boundary_time: np.ndarray
    # Axis magnetic fields, frequency-domain shape [frequency, x]
    Bx_induced_frequency: np.ndarray
    By_induced_frequency: np.ndarray
    Bx_applied_frequency: np.ndarray
    By_applied_frequency: np.ndarray
    # Axis magnetic fields, time-domain shape [x, time]
    Bx_induced_time: np.ndarray
    By_induced_time: np.ndarray
    Bx_total_time: np.ndarray
    By_total_time: np.ndarray
    pulse_spectrum: np.ndarray

    def save_csv(self, output_dir: str | Path) -> list[Path]:
        """Save principal frequency- and time-domain results as CSV files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        rows = []
        for j, m in enumerate(self.mode_numbers):
            for k, w in enumerate(self.omega_rad_s):
                rows.append({
                    "mode_m": int(m),
                    "omega_rad_s": w,
                    "Aapp_cos": self.applied_A_cos[j],
                    "Aapp_sin": self.applied_A_sin[j],
                    "Aeddy_cos_real": self.induced_A_cos_frequency[k, j].real,
                    "Aeddy_cos_imag": self.induced_A_cos_frequency[k, j].imag,
                    "Aeddy_sin_real": self.induced_A_sin_frequency[k, j].real,
                    "Aeddy_sin_imag": self.induced_A_sin_frequency[k, j].imag,
                    "Kz_cos_real": self.current_cos_frequency[k, j].real,
                    "Kz_cos_imag": self.current_cos_frequency[k, j].imag,
                    "Kz_sin_real": self.current_sin_frequency[k, j].real,
                    "Kz_sin_imag": self.current_sin_frequency[k, j].imag,
                })
        path = out / "circular_fourier_frequency.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)

        boundary_rows = []
        for it, t in enumerate(self.time_s):
            for ip, theta in enumerate(self.theta_rad):
                boundary_rows.append({
                    "time_s": t,
                    "theta_rad": theta,
                    "x_mm": 1e3 * self.radius_m * np.cos(theta),
                    "y_mm": 1e3 * self.radius_m * np.sin(theta),
                    "Aeddy_time_Wb_per_m": self.induced_A_boundary_time[ip, it],
                    "Kz_time_A_per_m": self.current_boundary_time[ip, it],
                })
        path = out / "circular_boundary_time.csv"
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
        path = out / "circular_axis_field_time.csv"
        pd.DataFrame(field_rows).to_csv(path, index=False)
        written.append(path)

        return written


class CircularSolver:
    """Circular chamber analytical solver.

    Model assumptions:
      * infinitely long chamber and wires along z;
      * thin coating, d << skin depth;
      * non-magnetic coating with permeability mu0;
      * exp(+i*omega*t) temporal phasor convention;
      * no net longitudinal current in the closed coating, so the m=0
        (gauge/mean-potential) component is excluded from the eddy-current modes.

    For m >= 1, the induced boundary-potential coefficient C_m obeys
        C_m = -i*alpha*A_m / (m/R + i*alpha),
        alpha = mu0*sigma*d*omega/2 = d/delta^2.
    This follows from (dA/dr)_out - (dA/dr)_in = +i*2*alpha*A_total.
    """

    def __init__(
        self,
        radius_m: float,
        conductivity_Sm: float,
        thickness_m: float,
        fourier_order: int,
        boundary_points: int = 1024,
    ) -> None:
        self.radius_m = float(radius_m)
        self.conductivity_Sm = float(conductivity_Sm)
        self.thickness_m = float(thickness_m)
        self.fourier_order = int(fourier_order)
        self.boundary_points = int(boundary_points)

        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.conductivity_Sm <= 0 or self.thickness_m <= 0:
            raise ValueError("conductivity and thickness must be positive")
        if self.fourier_order < 1:
            raise ValueError("fourier_order must be at least 1")
        if self.boundary_points < max(32, 4 * (self.fourier_order + 1)):
            raise ValueError("boundary_points is too small for the selected Fourier order")
        if self.fourier_order >= self.boundary_points // 2:
            raise ValueError("fourier_order must be below the boundary Nyquist order")

    def _boundary_fourier(self, wires: Sequence[Wire]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = 2.0 * np.pi * np.arange(self.boundary_points) / self.boundary_points
        x = self.radius_m * np.cos(theta)
        y = self.radius_m * np.sin(theta)
        az = vector_potential_from_wires(x, y, wires)

        modes = np.arange(1, self.fourier_order + 1, dtype=int)
        phase = np.outer(modes, theta)
        scale = 2.0 / self.boundary_points
        a_cos = scale * (np.cos(phase) @ az)
        a_sin = scale * (np.sin(phase) @ az)
        return theta, a_cos, a_sin

    def solve(
        self,
        wires: Sequence[Wire],
        omega_rad_s: np.ndarray,
        pulse_spectrum: np.ndarray,
        time_s: np.ndarray,
        observation_x_m: np.ndarray | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> CircularResult:
        omega = np.asarray(omega_rad_s, dtype=float)
        pulse = np.asarray(pulse_spectrum, dtype=float)
        time = np.asarray(time_s, dtype=float)
        if omega.ndim != 1 or pulse.shape != omega.shape:
            raise ValueError("omega_rad_s and pulse_spectrum must be same-length 1D arrays")
        if omega.size < 3 or not np.all(np.diff(omega) > 0):
            raise ValueError("omega_rad_s must be strictly increasing with at least 3 points")
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time_s must contain at least two points")

        R = self.radius_m
        sigma = self.conductivity_Sm
        d = self.thickness_m
        for wire in wires:
            wire_radius = np.hypot(float(wire.x_mm), float(wire.y_mm)) * 1e-3
            if wire_radius <= R:
                raise ValueError(
                    "For this circular boundary-value model, all external wires "
                    "must be outside the chamber (wire radius > chamber radius)."
                )
        m = np.arange(1, self.fourier_order + 1, dtype=float)
        theta, Aapp_cos, Aapp_sin = self._boundary_fourier(wires)
        if progress is not None:
            progress(15)
        nfreq = omega.size
        nmodes = m.size

        # alpha has the sign of omega, which guarantees H(-w)=H(w)*.
        alpha = 0.5 * MU0 * sigma * d * omega
        transfer = (
            -1j * alpha[:, None]
            / (m[None, :] / R + 1j * alpha[:, None])
        )
        if progress is not None:
            progress(35)

        Aeddy_cos = transfer * Aapp_cos[None, :] * pulse[:, None]
        Aeddy_sin = transfer * Aapp_sin[None, :] * pulse[:, None]

        # Kz = -i*w*sigma*d*(Aapp + Aeddy), including pulse spectrum.
        current_factor = (-1j * omega * sigma * d)[:, None]
        Kcos = current_factor * (Aapp_cos[None, :] + transfer * Aapp_cos[None, :]) * pulse[:, None]
        Ksin = current_factor * (Aapp_sin[None, :] + transfer * Aapp_sin[None, :]) * pulse[:, None]

        phase = np.outer(m, theta)
        cos_modes = np.cos(phase)
        sin_modes = np.sin(phase)
        A_boundary_freq = Aeddy_cos @ cos_modes + Aeddy_sin @ sin_modes
        K_boundary_freq = Kcos @ cos_modes + Ksin @ sin_modes

        A_boundary_time = inverse_fourier(omega, A_boundary_freq, time)
        K_boundary_time = inverse_fourier(omega, K_boundary_freq, time)
        if progress is not None:
            progress(55)

        if observation_x_m is None:
            x_obs = np.linspace(-0.9 * R, 0.9 * R, 161)
        else:
            x_obs = np.asarray(observation_x_m, dtype=float)
        if x_obs.ndim != 1 or np.any(np.abs(x_obs) >= R):
            raise ValueError("All horizontal-axis observation points must satisfy |x| < R")

        y_obs = np.zeros_like(x_obs)
        Bx_app_static, By_app_static = magnetic_field_from_wires(x_obs, y_obs, wires)

        r = np.abs(x_obs)
        # theta=0 on +x; theta=pi on -x. At x=0, use theta=0 (r=0 terms
        # with m>1 vanish, and the m=1 term is angle independent).
        theta_obs = np.where(x_obs < 0.0, np.pi, 0.0)
        m_minus_1 = m - 1.0
        radial_factor = (m[:, None] / R) * (r[None, :] / R) ** m_minus_1[:, None]
        phase_obs = m_minus_1[:, None] * theta_obs[None, :]
        c = np.cos(phase_obs)
        s = np.sin(phase_obs)

        # Bx = (m/R)(r/R)^(m-1) [-Cc sin((m-1)theta)+Cs cos((m-1)theta)]
        # By = (m/R)(r/R)^(m-1) [-Cc cos((m-1)theta)-Cs sin((m-1)theta)]
        bx_cos_kernel = -radial_factor * s
        bx_sin_kernel = radial_factor * c
        by_cos_kernel = -radial_factor * c
        by_sin_kernel = -radial_factor * s

        Bx_eddy_freq = Aeddy_cos @ bx_cos_kernel + Aeddy_sin @ bx_sin_kernel
        By_eddy_freq = Aeddy_cos @ by_cos_kernel + Aeddy_sin @ by_sin_kernel
        Bx_applied_freq = pulse[:, None] * Bx_app_static[None, :]
        By_applied_freq = pulse[:, None] * By_app_static[None, :]
        if progress is not None:
            progress(75)

        Bx_eddy_time = inverse_fourier(omega, Bx_eddy_freq, time)
        By_eddy_time = inverse_fourier(omega, By_eddy_freq, time)
        Bx_applied_time = inverse_fourier(omega, Bx_applied_freq, time)
        By_applied_time = inverse_fourier(omega, By_applied_freq, time)
        if progress is not None:
            progress(95)
            progress(100)

        return CircularResult(
            radius_m=R,
            conductivity_Sm=sigma,
            thickness_m=d,
            omega_rad_s=omega,
            time_s=time,
            theta_rad=theta,
            mode_numbers=m.astype(int),
            x_observation_m=x_obs,
            applied_A_cos=Aapp_cos,
            applied_A_sin=Aapp_sin,
            induced_A_cos_frequency=Aeddy_cos,
            induced_A_sin_frequency=Aeddy_sin,
            current_cos_frequency=Kcos,
            current_sin_frequency=Ksin,
            induced_A_boundary_frequency=A_boundary_freq,
            current_boundary_frequency=K_boundary_freq,
            induced_A_boundary_time=A_boundary_time,
            current_boundary_time=K_boundary_time,
            Bx_induced_frequency=Bx_eddy_freq,
            By_induced_frequency=By_eddy_freq,
            Bx_applied_frequency=Bx_applied_freq,
            By_applied_frequency=By_applied_freq,
            Bx_induced_time=Bx_eddy_time,
            By_induced_time=By_eddy_time,
            Bx_total_time=Bx_eddy_time + Bx_applied_time,
            By_total_time=By_eddy_time + By_applied_time,
            pulse_spectrum=pulse,
        )


def calculate_circular_case(
    *,
    radius_mm: float,
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
) -> CircularResult:
    """Convenience entry point used by the GUI and tests."""
    T = float(pulse_length_us) * 1e-6
    if T <= 0:
        raise ValueError("pulse_length_us must be positive")
    if frequency_cutoff_C <= 0:
        raise ValueError("frequency_cutoff_C must be positive")
    if frequency_points < 3 or frequency_points % 2 == 0:
        raise ValueError("frequency_points must be an odd integer >= 3")
    if time_points < 2:
        raise ValueError("time_points must be >= 2")

    omega_max = float(frequency_cutoff_C) * np.pi / T
    omega = np.linspace(-omega_max, omega_max, int(frequency_points))
    if pulse_type == "half_sine":
        pulse = half_sine_spectrum(omega, T)
    elif pulse_type == "rectangular":
        from ..physics import box_spectrum
        pulse = box_spectrum(omega, T)
    else:
        raise ValueError(f"Unsupported pulse_type: {pulse_type}")

    time = np.linspace(-T, T, int(time_points))
    solver = CircularSolver(
        radius_m=float(radius_mm) * 1e-3,
        conductivity_Sm=float(conductivity_Sm),
        thickness_m=float(thickness_um) * 1e-6,
        fourier_order=int(fourier_order),
        boundary_points=int(boundary_points),
    )
    observation_x_m = None
    if x_axis_points is not None:
        if int(x_axis_points) < 2:
            raise ValueError("x_axis_points must be >= 2")
        observation_x_m = np.linspace(
            -0.9 * solver.radius_m, 0.9 * solver.radius_m, int(x_axis_points)
        )
    return solver.solve(
        wires=wires,
        omega_rad_s=omega,
        pulse_spectrum=pulse,
        time_s=time,
        observation_x_m=observation_x_m,
        progress=progress,
    )
