"""Analytical thin-coating solver for two infinite parallel plates.

The original derivation in the user's draft obtains a +/- spatial-wavenumber
2x2 system by using 180-degree rotational symmetry.  This implementation
uses the equivalent generalized two-sheet system at each spatial wavenumber:
the applied vector potential is sampled independently on the upper and lower
plates.  If the applied configuration has the stated 180-degree symmetry, the
lower-plate spectrum is the reflected upper spectrum and this system reduces
to the +/- formulation in the draft.

At spatial k=0, the antisymmetric sheet amplitude can scale as 1/k.  The
physical combinations (boundary potential and mid-plane Bx) remain finite.
The code evaluates those combinations using their analytic k->0 limits,
never attempting to store the divergent amplitude itself.
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


def has_180_degree_wire_symmetry(
    wires: Sequence[Wire], tolerance: float = 1e-8
) -> bool:
    """Check whether every z-directed wire has an equal-current partner at (-x,-y)."""
    unmatched = list(wires)
    scale = max(1.0, max((abs(w.current_A) for w in wires), default=1.0))
    while unmatched:
        wire = unmatched.pop(0)
        found = None
        for idx, candidate in enumerate(unmatched):
            if (
                abs(candidate.x_mm + wire.x_mm) <= tolerance
                and abs(candidate.y_mm + wire.y_mm) <= tolerance
                and abs(candidate.current_A - wire.current_A) <= tolerance * scale
            ):
                found = idx
                break
        if found is None:
            return False
        unmatched.pop(found)
    return True


def flat_pipe_mode_solution(
    spatial_wavenumber_rad_m: float,
    A_top: complex,
    A_bottom: complex,
    alpha: float,
    half_gap_m: float,
) -> tuple[complex, complex, complex, complex, complex]:
    """Solve one spatial mode and return physical field/potential combinations.

    Parameters
    ----------
    spatial_wavenumber_rad_m:
        Signed spatial wavenumber Omega [1/m]; the solver uses its magnitude
        in the plate-coupling factors and its sign in By=-i*Omega*A.
    A_top, A_bottom:
        Applied vector-potential spectra at y=+b and y=-b.  They already
        include the temporal pulse spectrum.
    alpha:
        mu0*sigma*d*omega/2 [1/m], signed with temporal omega.
    half_gap_m:
        Plate half separation b [m].

    Returns
    -------
    Aeddy_top, Aeddy_bottom, Aeddy_mid, Bxeddy_mid, Byeddy_mid
        Spectral components.  The zero-wavenumber result is the analytic
        limit of the physical combinations; no divergent antisymmetric sheet
        amplitude is returned.
    """
    Omega = float(spatial_wavenumber_rad_m)
    k = abs(Omega)
    alpha = float(alpha)
    b = float(half_gap_m)
    if b <= 0.0:
        raise ValueError("half_gap_m must be positive")
    if alpha == 0.0:
        return 0.0j, 0.0j, 0.0j, 0.0j, 0.0j

    A_s = 0.5 * (complex(A_top) + complex(A_bottom))
    A_a = 0.5 * (complex(A_top) - complex(A_bottom))

    if k == 0.0:
        # X_s -> -A_s/2; k*X_a -> -i*alpha*A_a/(1+2*i*alpha*b).
        k_Xa = -1j * alpha * A_a / (1.0 + 2j * alpha * b)
        Aeddy_mid = -A_s
        Aeddy_top = -A_s + 2.0 * b * k_Xa
        Aeddy_bottom = -A_s - 2.0 * b * k_Xa
        Bxeddy_mid = 2.0 * k_Xa
        Byeddy_mid = -1j * Omega * Aeddy_mid
        return Aeddy_top, Aeddy_bottom, Aeddy_mid, Bxeddy_mid, Byeddy_mid

    q = np.exp(-2.0 * b * k)
    denominator_s = k + 1j * alpha * (1.0 + q)
    denominator_a = k + 1j * alpha * (1.0 - q)
    X_s = -1j * alpha * A_s / denominator_s
    X_a = -1j * alpha * A_a / denominator_a

    Aeddy_top = (1.0 + q) * X_s + (1.0 - q) * X_a
    Aeddy_bottom = (1.0 + q) * X_s - (1.0 - q) * X_a
    decay_mid = np.exp(-b * k)
    Aeddy_mid = 2.0 * decay_mid * X_s
    Bxeddy_mid = 2.0 * k * decay_mid * X_a
    Byeddy_mid = -1j * Omega * Aeddy_mid
    return Aeddy_top, Aeddy_bottom, Aeddy_mid, Bxeddy_mid, Byeddy_mid


@dataclass
class FlatPipeResult:
    half_gap_m: float
    conductivity_Sm: float
    thickness_m: float
    omega_rad_s: np.ndarray
    time_s: np.ndarray
    spatial_wavenumber_rad_m: np.ndarray
    x_m: np.ndarray  # observation grid used by exported fields/currents
    x_fft_m: np.ndarray  # internal uniform FFT grid used by the spatial solver
    applied_A_top_static: np.ndarray
    applied_A_bottom_static: np.ndarray
    induced_A_top_frequency: np.ndarray
    induced_A_bottom_frequency: np.ndarray
    current_top_frequency: np.ndarray
    current_bottom_frequency: np.ndarray
    induced_A_mid_frequency: np.ndarray
    Bx_induced_frequency: np.ndarray
    By_induced_frequency: np.ndarray
    Bx_applied_frequency: np.ndarray
    By_applied_frequency: np.ndarray
    induced_A_top_time: np.ndarray
    induced_A_bottom_time: np.ndarray
    current_top_time: np.ndarray
    current_bottom_time: np.ndarray
    Bx_induced_time: np.ndarray
    By_induced_time: np.ndarray
    Bx_total_time: np.ndarray
    By_total_time: np.ndarray
    pulse_spectrum: np.ndarray

    def save_csv(self, output_dir: str | Path) -> list[Path]:
        """Save flat-pipe results without creating a huge frequency-x-time long table.

        Full arrays are preserved in a compressed NPZ file.  CSV files provide
        selected frequency traces and wide x-by-time tables similar to the
        original GUI's output style.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        # --------------------------------------------------
        # External drive vector potential on observation grid
        # --------------------------------------------------
        drive_data = {
            "x_m": self.x_m,

            "A_top_real": np.real(self.applied_A_top_static),
            "A_top_imag": np.imag(self.applied_A_top_static),

            "A_bottom_real": np.real(self.applied_A_bottom_static),
            "A_bottom_imag": np.imag(self.applied_A_bottom_static),
        }

        path = out / "fourier_drive.csv"
        pd.DataFrame(drive_data).to_csv(
            path,
            index=False,
            float_format="%.12e"
        )
        written.append(path)

        # Frequency-domain values at representative x locations.
        targets_m = np.array([0.0, -0.25 * self.x_m[-1], 0.25 * self.x_m[-1],
                              -0.5 * self.x_m[-1], 0.5 * self.x_m[-1]])
        selected = sorted({int(np.argmin(np.abs(self.x_m - value))) for value in targets_m})
        frequency_data: dict[str, np.ndarray] = {"omega_rad_s": self.omega_rad_s}
        for ix in selected:
            x_label = f"x_{1e3*self.x_m[ix]:+.3f}mm"
            for field_name, values in (
                ("Bx_eddy", self.Bx_induced_frequency),
                ("By_eddy", self.By_induced_frequency),
                ("Kz_top", self.current_top_frequency),
                ("Kz_bottom", self.current_bottom_frequency),
            ):
                frequency_data[f"{field_name}_{x_label}_real"] = values[:, ix].real
                frequency_data[f"{field_name}_{x_label}_imag"] = values[:, ix].imag
        path = out / "flat_pipe_frequency_selected_x.csv"
        pd.DataFrame(frequency_data).to_csv(path, index=False)
        written.append(path)

        # Wide format: one row per x, one column per time point and field.
        field_data: dict[str, np.ndarray] = {"x_mm": 1e3 * self.x_m}
        for time_idx, time_value in enumerate(self.time_s):
            tag = f"{1e6*time_value:+.6f}us"
            field_data[f"Bx_eddy_{tag}"] = self.Bx_induced_time[:, time_idx]
            field_data[f"By_eddy_{tag}"] = self.By_induced_time[:, time_idx]
            field_data[f"Bx_total_{tag}"] = self.Bx_total_time[:, time_idx]
            field_data[f"By_total_{tag}"] = self.By_total_time[:, time_idx]
        path = out / "flat_pipe_time_domain_fields.csv"
        pd.DataFrame(field_data).to_csv(path, index=False)
        written.append(path)

        current_data: dict[str, np.ndarray] = {"x_mm": 1e3 * self.x_m}
        for time_idx, time_value in enumerate(self.time_s):
            tag = f"{1e6*time_value:+.6f}us"
            current_data[f"Kz_top_{tag}"] = self.current_top_time[:, time_idx]
            current_data[f"Kz_bottom_{tag}"] = self.current_bottom_time[:, time_idx]
        path = out / "flat_pipe_boundary_current_time.csv"
        pd.DataFrame(current_data).to_csv(path, index=False)
        written.append(path)

        path = out / "flat_pipe_full_data.npz"
        np.savez_compressed(
            path,
            omega_rad_s=self.omega_rad_s,
            time_s=self.time_s,
            spatial_wavenumber_rad_m=self.spatial_wavenumber_rad_m,
            x_m=self.x_m,
            x_fft_m=self.x_fft_m,
            applied_A_top_static=self.applied_A_top_static,
            applied_A_bottom_static=self.applied_A_bottom_static,
            induced_A_top_frequency=self.induced_A_top_frequency,
            induced_A_bottom_frequency=self.induced_A_bottom_frequency,
            current_top_frequency=self.current_top_frequency,
            current_bottom_frequency=self.current_bottom_frequency,
            induced_A_mid_frequency=self.induced_A_mid_frequency,
            Bx_induced_frequency=self.Bx_induced_frequency,
            By_induced_frequency=self.By_induced_frequency,
            Bx_applied_frequency=self.Bx_applied_frequency,
            By_applied_frequency=self.By_applied_frequency,
            induced_A_top_time=self.induced_A_top_time,
            induced_A_bottom_time=self.induced_A_bottom_time,
            current_top_time=self.current_top_time,
            current_bottom_time=self.current_bottom_time,
            Bx_induced_time=self.Bx_induced_time,
            By_induced_time=self.By_induced_time,
            Bx_total_time=self.Bx_total_time,
            By_total_time=self.By_total_time,
            pulse_spectrum=self.pulse_spectrum,
        )
        written.append(path)
        return written



class FlatPipeSolver:
    """Two infinite parallel conducting plates at y=+/-b."""

    def __init__(
        self,
        half_gap_m: float,
        conductivity_Sm: float,
        thickness_m: float,
        spatial_range_m: float,
        spatial_points: int = 2048,
    ) -> None:
        self.half_gap_m = float(half_gap_m)
        self.conductivity_Sm = float(conductivity_Sm)
        self.thickness_m = float(thickness_m)
        self.spatial_range_m = float(spatial_range_m)
        self.spatial_points = int(spatial_points)
        if self.half_gap_m <= 0:
            raise ValueError("half_gap_m must be positive")
        if self.conductivity_Sm <= 0 or self.thickness_m <= 0:
            raise ValueError("Conductivity and thickness must be positive")
        if self.spatial_range_m <= 0:
            raise ValueError("spatial_range_m must be positive")
        if self.spatial_points < 256 or self.spatial_points % 2:
            raise ValueError("spatial_points must be an even integer >= 256")

    @staticmethod
    def _spatial_fft(values: np.ndarray, dx: float) -> np.ndarray:
        return dx * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(values)))

    @staticmethod
    def _spatial_ifft(values_hat: np.ndarray, dx: float) -> np.ndarray:
        return (
            np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(values_hat, axes=1), axis=1),
                axes=1,
            )
            / dx
        )

    @staticmethod
    def _spatial_evaluate(
        values_hat: np.ndarray,
        spatial_wavenumber_rad_m: np.ndarray,
        dx: float,
        observation_x_m: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the spatial Fourier series directly at observation points.

        This avoids interpolating the dense internal FFT grid when a separate,
        common output grid is required by the Green/Fourier cross-validation.
        ``values_hat`` has shape (n_frequency, n_spatial_fft). The result has
        shape (n_frequency, n_observation).
        """
        spectrum = np.asarray(values_hat, dtype=complex)
        k = np.asarray(spatial_wavenumber_rad_m, dtype=float)
        x_obs = np.asarray(observation_x_m, dtype=float)
        if spectrum.ndim != 2 or spectrum.shape[1] != k.size:
            raise ValueError("values_hat must have shape (n_frequency, len(k))")
        if x_obs.ndim != 1 or not np.all(np.isfinite(x_obs)):
            raise ValueError("observation_x_m must be a finite 1D array")
        if k.size % 2:
            raise ValueError("The internal spatial FFT size must be even")
        # For an even-length DFT, the negative Nyquist bin is its own partner.
        # A single complex exponential gives a spurious imaginary component
        # away from native FFT nodes. Its real, band-limited continuation is
        # represented by cos(pi*x/dx); the remaining bins use exp(+i*k*x).
        phase = np.exp(1j * np.outer(x_obs, k[1:]))
        values = (phase @ spectrum[:, 1:].T).T
        nyquist = np.cos(np.pi * x_obs / dx)
        values += spectrum[:, 0, None] * nyquist[None, :]
        return values / (k.size * dx)

    def solve(
        self,
        wires: Sequence[Wire],
        omega_rad_s: np.ndarray,
        pulse_spectrum: np.ndarray,
        time_s: np.ndarray,
        observation_x_m: np.ndarray | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> FlatPipeResult:
        omega = np.asarray(omega_rad_s, dtype=float)
        pulse = np.asarray(pulse_spectrum, dtype=float)
        time = np.asarray(time_s, dtype=float)
        if omega.ndim != 1 or pulse.shape != omega.shape:
            raise ValueError("omega_rad_s and pulse_spectrum must be same-length 1D arrays")
        if omega.size < 3 or not np.all(np.diff(omega) > 0):
            raise ValueError("omega_rad_s must be strictly increasing with at least 3 points")
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time_s must contain at least two points")
        current_sum = sum(float(w.current_A) for w in wires)
        current_scale = max(1.0, sum(abs(float(w.current_A)) for w in wires))
        if abs(current_sum) > 1e-10 * current_scale:
            raise ValueError(
                "The net external wire current must be zero for a finite spatial "
                "Fourier transform of A_z in this infinite-plate model."
            )

        N = self.spatial_points
        L = self.spatial_range_m
        dx = 2.0 * L / N
        max_wire_x = max((abs(float(w.x_mm)) * 1e-3 for w in wires), default=0.0)
        if L <= max_wire_x + 5.0 * self.half_gap_m:
            raise ValueError(
                "The spatial range xmax is too small. Use xmax > max(|wire x|) + 5*b "
                "to leave sufficient room for the applied-field tails."
            )
        nearest_plate_distance = min(
            min(abs(float(w.y_mm) * 1e-3 - self.half_gap_m),
                abs(float(w.y_mm) * 1e-3 + self.half_gap_m))
            for w in wires
        )
        if nearest_plate_distance <= 0.0:
            raise ValueError("An external wire lies on a conducting plate.")
        if dx > nearest_plate_distance / 5.0:
            raise ValueError(
                f"Spatial step dx={dx*1e3:.4g} mm is too coarse for the closest wire-to-plate "
                f"distance {nearest_plate_distance*1e3:.4g} mm. Increase spatial_points "
                "or reduce xmax (at least 5 samples across the closest distance are required)."
            )
        x_fft = (np.arange(N) - N // 2) * dx
        b = self.half_gap_m
        A_top_x_fft = vector_potential_from_wires(x_fft, np.full_like(x_fft, b), wires)
        A_bottom_x_fft = vector_potential_from_wires(x_fft, np.full_like(x_fft, -b), wires)
        A_top_hat_static = self._spatial_fft(A_top_x_fft, dx)
        A_bottom_hat_static = self._spatial_fft(A_bottom_x_fft, dx)
        spatial_omega = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        k = np.abs(spatial_omega)
        q = np.exp(-2.0 * b * k)
        if progress:
            progress(10)

        A_top_hat = pulse[:, None] * A_top_hat_static[None, :]
        A_bottom_hat = pulse[:, None] * A_bottom_hat_static[None, :]
        Aeddy_top_hat = np.zeros_like(A_top_hat, dtype=complex)
        Aeddy_bottom_hat = np.zeros_like(A_bottom_hat, dtype=complex)
        Aeddy_mid_hat = np.zeros_like(A_top_hat, dtype=complex)
        Bx_mid_hat = np.zeros_like(A_top_hat, dtype=complex)
        By_mid_hat = np.zeros_like(A_top_hat, dtype=complex)
        alpha = 0.5 * MU0 * self.conductivity_Sm * self.thickness_m * omega
        zero_k = N // 2

        for iw, ak in enumerate(alpha):
            if ak == 0.0:
                continue
            As = 0.5 * (A_top_hat[iw] + A_bottom_hat[iw])
            Aa = 0.5 * (A_top_hat[iw] - A_bottom_hat[iw])
            positive_k = k > 0.0

            den_s = k[positive_k] + 1j * ak * (1.0 + q[positive_k])
            den_a = k[positive_k] + 1j * ak * (1.0 - q[positive_k])
            Xs = -1j * ak * As[positive_k] / den_s
            Xa = -1j * ak * Aa[positive_k] / den_a
            qpos = q[positive_k]
            kpos = k[positive_k]
            decay = np.exp(-b * kpos)

            Aeddy_top_hat[iw, positive_k] = (1.0 + qpos) * Xs + (1.0 - qpos) * Xa
            Aeddy_bottom_hat[iw, positive_k] = (1.0 + qpos) * Xs - (1.0 - qpos) * Xa
            Aeddy_mid_hat[iw, positive_k] = 2.0 * decay * Xs
            Bx_mid_hat[iw, positive_k] = 2.0 * kpos * decay * Xa
            By_mid_hat[iw, positive_k] = -1j * spatial_omega[positive_k] * Aeddy_mid_hat[iw, positive_k]

            # Exact Omega=0 limit.  The antisymmetric sheet amplitude itself
            # behaves as 1/k, but these observable combinations are finite.
            As0 = As[zero_k]
            Aa0 = Aa[zero_k]
            kXa0 = -1j * ak * Aa0 / (1.0 + 2j * ak * b)
            Aeddy_mid_hat[iw, zero_k] = -As0
            Aeddy_top_hat[iw, zero_k] = -As0 + 2.0 * b * kXa0
            Aeddy_bottom_hat[iw, zero_k] = -As0 - 2.0 * b * kXa0
            Bx_mid_hat[iw, zero_k] = 2.0 * kXa0
            By_mid_hat[iw, zero_k] = 0.0j

            if progress and (iw + 1) % max(1, omega.size // 10) == 0:
                progress(10 + int(45 * (iw + 1) / omega.size))

        current_factor = (-1j * omega * self.conductivity_Sm * self.thickness_m)[:, None]
        K_top_hat = current_factor * (A_top_hat + Aeddy_top_hat)
        K_bottom_hat = current_factor * (A_bottom_hat + Aeddy_bottom_hat)

        # Select the common output grid independently of the dense FFT grid.
        # The default preserves the original full-resolution output.  When a
        # user supplies observation points, evaluate the Fourier series directly
        # instead of interpolating the inverse-FFT result.
        if observation_x_m is None:
            x_obs = x_fft.copy()
        else:
            x_obs = np.asarray(observation_x_m, dtype=float)
            if x_obs.ndim != 1 or not np.all(np.isfinite(x_obs)):
                raise ValueError("observation_x_m must be a finite 1D array")
            if x_obs.size < 2 or np.any(np.abs(x_obs) >= L):
                raise ValueError("observation_x_m must contain >=2 points inside the spatial window")

        # Evaluate each frequency-domain quantity at the selected x locations.
        Aeddy_top_freq = self._spatial_evaluate(Aeddy_top_hat, spatial_omega, dx, x_obs)
        Aeddy_bottom_freq = self._spatial_evaluate(Aeddy_bottom_hat, spatial_omega, dx, x_obs)
        Aeddy_mid_freq = self._spatial_evaluate(Aeddy_mid_hat, spatial_omega, dx, x_obs)
        K_top_freq = self._spatial_evaluate(K_top_hat, spatial_omega, dx, x_obs)
        K_bottom_freq = self._spatial_evaluate(K_bottom_hat, spatial_omega, dx, x_obs)
        Bx_eddy_freq = self._spatial_evaluate(Bx_mid_hat, spatial_omega, dx, x_obs)
        By_eddy_freq = self._spatial_evaluate(By_mid_hat, spatial_omega, dx, x_obs)

        A_top_x = vector_potential_from_wires(x_obs, np.full_like(x_obs, b), wires)
        A_bottom_x = vector_potential_from_wires(x_obs, np.full_like(x_obs, -b), wires)
        Bx_app_static, By_app_static = magnetic_field_from_wires(x_obs, np.zeros_like(x_obs), wires)
        Bx_app_freq = pulse[:, None] * Bx_app_static[None, :]
        By_app_freq = pulse[:, None] * By_app_static[None, :]
        if progress:
            progress(70)

        Aeddy_top_time = inverse_fourier(omega, Aeddy_top_freq, time)
        Aeddy_bottom_time = inverse_fourier(omega, Aeddy_bottom_freq, time)
        K_top_time = inverse_fourier(omega, K_top_freq, time)
        K_bottom_time = inverse_fourier(omega, K_bottom_freq, time)
        # A perfectly symmetric excitation can make the mid-plane Bx exactly
        # zero.  FFT round-off can leave a tiny residual whose relative
        # imaginary/real ratio is meaningless, so remove only components
        # below a strict relative numerical-noise floor.
        bx_scale = float(np.max(np.abs(Bx_eddy_freq)))
        by_scale = float(np.max(np.abs(By_eddy_freq)))
        if bx_scale < 1e-12 * max(by_scale, 1e-30):
            Bx_eddy_freq[...] = 0.0
            Bx_eddy_time = np.zeros(
                (Bx_eddy_freq.shape[1], time.size), dtype=float
            )
        else:
            Bx_eddy_time = inverse_fourier(omega, Bx_eddy_freq, time)
        By_eddy_time = inverse_fourier(omega, By_eddy_freq, time)
        Bx_app_time = inverse_fourier(omega, Bx_app_freq, time)
        By_app_time = inverse_fourier(omega, By_app_freq, time)
        if progress:
            progress(100)

        return FlatPipeResult(
            half_gap_m=b,
            conductivity_Sm=self.conductivity_Sm,
            thickness_m=self.thickness_m,
            omega_rad_s=omega,
            time_s=time,
            spatial_wavenumber_rad_m=spatial_omega,
            x_m=x_obs,
            x_fft_m=x_fft,
            applied_A_top_static=A_top_x,
            applied_A_bottom_static=A_bottom_x,
            induced_A_top_frequency=Aeddy_top_freq,
            induced_A_bottom_frequency=Aeddy_bottom_freq,
            current_top_frequency=K_top_freq,
            current_bottom_frequency=K_bottom_freq,
            induced_A_mid_frequency=Aeddy_mid_freq,
            Bx_induced_frequency=Bx_eddy_freq,
            By_induced_frequency=By_eddy_freq,
            Bx_applied_frequency=Bx_app_freq,
            By_applied_frequency=By_app_freq,
            induced_A_top_time=Aeddy_top_time,
            induced_A_bottom_time=Aeddy_bottom_time,
            current_top_time=K_top_time,
            current_bottom_time=K_bottom_time,
            Bx_induced_time=Bx_eddy_time,
            By_induced_time=By_eddy_time,
            Bx_total_time=Bx_eddy_time + Bx_app_time,
            By_total_time=By_eddy_time + By_app_time,
            pulse_spectrum=pulse,
        )


def calculate_flat_pipe_case(
    *,
    half_gap_mm: float,
    conductivity_Sm: float,
    thickness_um: float,
    pulse_length_us: float,
    frequency_cutoff_C: float,
    frequency_points: int,
    time_points: int,
    spatial_range_mm: float,
    spatial_points: int,
    wires: Sequence[Wire],
    x_axis_points: int | None = None,
    pulse_type: str = "half_sine",
    progress: Callable[[int], None] | None = None,
) -> FlatPipeResult:
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
    spatial_range_m = float(spatial_range_mm) * 1e-3
    solver = FlatPipeSolver(
        half_gap_m=float(half_gap_mm) * 1e-3,
        conductivity_Sm=conductivity_Sm,
        thickness_m=float(thickness_um) * 1e-6,
        spatial_range_m=spatial_range_m,
        spatial_points=spatial_points,
    )
    observation_x_m = None
    if x_axis_points is not None:
        if int(x_axis_points) < 2:
            raise ValueError("x_axis_points must be >= 2")
        observation_x_m = np.linspace(
            -0.9 * spatial_range_m, 0.9 * spatial_range_m, int(x_axis_points)
        )
    return solver.solve(
        wires, omega, pulse, time, observation_x_m=observation_x_m, progress=progress
    )
