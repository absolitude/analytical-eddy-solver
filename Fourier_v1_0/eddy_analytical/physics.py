"""Shared physics utilities for thin-coating eddy-current analytical solvers.

Time convention: all phasors use exp(+i omega t).
Spatial Fourier conventions are documented in the individual solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class Wire:
    """An infinitely long wire parallel to +z.

    x_mm, y_mm: transverse position in mm
    current_A: signed current in A (positive means +z)
    """
    x_mm: float
    y_mm: float
    current_A: float


def half_sine_spectrum(omega: np.ndarray, pulse_length_s: float) -> np.ndarray:
    """Fourier transform of cos(pi*t/T), |t| <= T/2, using exp(-i*w*t).

    The inverse convention is f(t) = (1/2pi) integral F(w) exp(+i*w*t) dw.
    """
    omega = np.asarray(omega, dtype=float)
    T = float(pulse_length_s)
    if T <= 0:
        raise ValueError("pulse_length_s must be positive")

    u = T * omega
    denom = np.pi**2 - u**2
    out = np.empty_like(omega, dtype=float)
    near = np.isclose(np.abs(u), np.pi, rtol=1e-9, atol=1e-12)
    out[near] = T / 2.0
    out[~near] = (
        2.0 * np.pi * T * np.cos(0.5 * u[~near]) / denom[~near]
    )
    return out


def box_spectrum(omega: np.ndarray, pulse_length_s: float) -> np.ndarray:
    """Fourier transform of a unit rectangular pulse on [-T/2, T/2]."""
    omega = np.asarray(omega, dtype=float)
    T = float(pulse_length_s)
    if T <= 0:
        raise ValueError("pulse_length_s must be positive")
    x = 0.5 * omega * T
    out = np.empty_like(x)
    small = np.abs(x) < 1e-12
    out[small] = T
    out[~small] = T * np.sin(x[~small]) / x[~small]
    return out


def vector_potential_from_wires(
    x_m: np.ndarray,
    y_m: np.ndarray,
    wires: Sequence[Wire],
    reference_length_m: float = 1.0,
) -> np.ndarray:
    """Return static A_z from z-directed infinitely long wires.

    A_z = -mu0*I/(2*pi)*ln(r/r_ref).  The reference length only fixes the
    additive gauge constant; non-zero angular Fourier modes are unchanged.
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    if x_m.shape != y_m.shape:
        raise ValueError("x_m and y_m must have identical shapes")
    if reference_length_m <= 0:
        raise ValueError("reference_length_m must be positive")

    az = np.zeros_like(x_m, dtype=float)
    for wire in wires:
        wx = float(wire.x_mm) * 1e-3
        wy = float(wire.y_mm) * 1e-3
        radius = np.hypot(x_m - wx, y_m - wy)
        if np.any(radius < 1e-12):
            raise ValueError(
                "A wire lies on an evaluation point. Move the wire away from the coating."
            )
        az += -MU0 * float(wire.current_A) / (2.0 * np.pi) * np.log(
            radius / reference_length_m
        )
    return az


def magnetic_field_from_wires(
    x_m: np.ndarray,
    y_m: np.ndarray,
    wires: Sequence[Wire],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the applied Bx, By from infinitely long +z wires."""
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    if x_m.shape != y_m.shape:
        raise ValueError("x_m and y_m must have identical shapes")

    bx = np.zeros_like(x_m, dtype=float)
    by = np.zeros_like(x_m, dtype=float)
    for wire in wires:
        dx = x_m - float(wire.x_mm) * 1e-3
        dy = y_m - float(wire.y_mm) * 1e-3
        r2 = dx * dx + dy * dy
        if np.any(r2 < 1e-24):
            raise ValueError(
                "A wire coincides with a magnetic-field observation point."
            )
        factor = MU0 * float(wire.current_A) / (2.0 * np.pi * r2)
        bx += -factor * dy
        by += factor * dx
    return bx, by


def inverse_fourier(
    omega: np.ndarray,
    spectrum: np.ndarray,
    time_s: np.ndarray,
    *,
    check_real: bool = True,
    real_tolerance: float = 2e-7,
) -> np.ndarray:
    """Numerically evaluate (1/2pi) integral S(w) exp(+i*w*t) dw.

    spectrum has shape (N_frequency, N_signal). The returned array has shape
    (N_signal, N_time). Trapezoidal weights are used at the frequency endpoints.
    """
    omega = np.asarray(omega, dtype=float)
    spectrum = np.asarray(spectrum, dtype=complex)
    time_s = np.asarray(time_s, dtype=float)

    if omega.ndim != 1 or time_s.ndim != 1:
        raise ValueError("omega and time_s must be one-dimensional")
    if spectrum.ndim == 1:
        spectrum = spectrum[:, None]
    if spectrum.ndim != 2 or spectrum.shape[0] != omega.size:
        raise ValueError("spectrum must have shape (len(omega), N_signal)")
    if omega.size < 3:
        raise ValueError("At least three frequency points are required")
    d_omega = np.diff(omega)
    if not np.allclose(d_omega, d_omega[0], rtol=1e-9, atol=1e-14):
        raise ValueError("omega must be uniformly spaced")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("spectrum contains non-finite values")

    weights = np.full(omega.size, d_omega[0], dtype=float)
    weights[[0, -1]] *= 0.5
    phase = np.exp(1j * np.outer(omega, time_s))
    result = (spectrum.T @ (phase * weights[:, None])) / (2.0 * np.pi)

    if check_real:
        max_real = float(np.max(np.abs(result.real))) if result.size else 0.0
        max_imag = float(np.max(np.abs(result.imag))) if result.size else 0.0
        scale = max(max_real, 1e-30)
        ratio = max_imag / scale
        if ratio > real_tolerance:
            raise ValueError(
                "Inverse Fourier result is not real "
                f"(max imaginary/real = {ratio:.3e}); "
                "check conjugate symmetry of the spectrum."
            )
    return result.real
