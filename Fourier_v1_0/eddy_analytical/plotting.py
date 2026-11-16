from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.polynomial import Polynomial

from .physics import Wire, inverse_fourier, magnetic_field_from_wires, vector_potential_from_wires


def _result_name(result: Any) -> str:
    return type(result).__name__


def _boundary_view(result):
    name = _result_name(result)
    if name == "CircularResult":
        return (
            np.degrees(result.theta_rad),
            "Boundary angle θ (deg)",
            result.current_boundary_time,
            result.induced_A_boundary_time,
        )
    if name == "EllipticalResult":
        return (
            np.degrees(result.psi_rad),
            "Elliptical angle ψ (deg)",
            result.current_boundary_time,
            result.induced_A_boundary_time,
        )
    if name == "FlatPipeResult":
        return (
            1e3 * result.x_m,
            "Transverse coordinate x (mm)",
            result.current_top_time,
            result.induced_A_top_time,
        )
    raise TypeError(f"Unsupported result type: {name}")


def _field_view(result):
    if _result_name(result) == "FlatPipeResult":
        return result.x_m, result.By_induced_time, result.By_total_time
    return result.x_observation_m, result.By_induced_time, result.By_total_time


def _plot_title(result) -> str:
    name = _result_name(result)
    if name == "CircularResult":
        return f"Circular chamber (R={1e3 * result.radius_m:.4g} mm)"
    if name == "EllipticalResult":
        return f"Elliptical chamber (a={1e3 * result.a_m:.4g} mm, b={1e3 * result.b_m:.4g} mm)"
    if name == "FlatPipeResult":
        return f"Infinite parallel plates (half-gap b={1e3 * result.half_gap_m:.4g} mm)"
    return "Analytical eddy-current result"


def _wires_from_parameters(parameters: dict[str, Any]) -> list[Wire]:
    return [Wire(**item) for item in parameters.get("wires", [])]


def _geometry_boundary(result):
    name = _result_name(result)
    if name == "CircularResult":
        x = result.radius_m * np.cos(result.theta_rad)
        y = result.radius_m * np.sin(result.theta_rad)
        coordinate = np.degrees(result.theta_rad)
        coordinate_label = "Boundary angle θ (deg)"
        return x, y, coordinate, coordinate_label
    if name == "EllipticalResult":
        x = result.a_m * np.cos(result.psi_rad)
        y = result.b_m * np.sin(result.psi_rad)
        coordinate = np.degrees(result.psi_rad)
        coordinate_label = "Elliptical angle ψ (deg)"
        return x, y, coordinate, coordinate_label
    x = result.x_m
    y_top = np.full_like(x, result.half_gap_m)
    y_bottom = -y_top
    return x, np.vstack((y_top, y_bottom)), x, "Transverse coordinate x (m)"


def _sample_indices(n: int, count: int = 6) -> list[int]:
    if n <= 0:
        return []
    return sorted(set(np.linspace(0, n - 1, min(n, count)).round().astype(int).tolist()))


def _write_html(fig: go.Figure, output_file: str | Path) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Store one local Plotly.js bundle shared by all HTML reports. This keeps
    # each report small while allowing fully offline interactive viewing.
    js_bundle = output_file.parent / "plotly.min.js"
    if not js_bundle.exists():
        from plotly.offline import get_plotlyjs
        js_bundle.write_text(get_plotlyjs(), encoding="utf-8")
    fig.write_html(output_file, include_plotlyjs=js_bundle.name, full_html=True)
    return output_file


def _wire_field_on_axis(x_m: np.ndarray, wires: list[Wire]):
    """Evaluate wire fields on y=0, leaving exact wire-axis singularities as NaN."""
    x_m = np.asarray(x_m, dtype=float)
    valid = np.ones(x_m.size, dtype=bool)
    for wire in wires:
        dx = x_m - wire.x_mm * 1e-3
        dy = -wire.y_mm * 1e-3
        valid &= (dx * dx + dy * dy) >= 1e-24
    bx = np.full(x_m.size, np.nan, dtype=float)
    by = np.full(x_m.size, np.nan, dtype=float)
    if np.any(valid):
        bx[valid], by[valid] = magnetic_field_from_wires(
            x_m[valid], np.zeros(np.count_nonzero(valid)), wires
        )
    return bx, by


def write_result_plot(result, output_file: str | Path) -> Path:
    """Write the general six-panel overview report."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    coord, coord_label, K_time, A_time = _boundary_view(result)
    x_obs, By_induced, By_total = _field_view(result)
    time_us = 1e6 * result.time_s
    it_peak = int(np.argmin(np.abs(result.time_s)))
    x_mm = 1e3 * x_obs

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Boundary surface current at t ≈ 0",
            "Induced and total By along y = 0 at t ≈ 0",
            "Induced By time response at selected x positions",
            "Frequency-domain response",
            "Boundary current evolution",
            "Induced By(x,t) evolution",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(x=coord, y=K_time[:, it_peak], mode="lines", name="Kz at pulse center"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_mm, y=By_induced[:, it_peak], mode="lines", name="Induced By"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=x_mm, y=By_total[:, it_peak], mode="lines", name="Total By"),
        row=1, col=2,
    )

    targets = [0.0, -0.1 * (x_obs[-1] - x_obs[0]), 0.1 * (x_obs[-1] - x_obs[0])]
    for target in targets:
        idx = int(np.argmin(np.abs(x_obs - target)))
        fig.add_trace(
            go.Scatter(
                x=time_us,
                y=By_induced[idx, :],
                mode="lines",
                name=f"Induced By at x={1e3*x_obs[idx]:.3g} mm",
            ),
            row=2, col=1,
        )

    omega = result.omega_rad_s
    if _result_name(result) in ("CircularResult", "EllipticalResult"):
        modes = result.mode_numbers
        cos_coeff = result.induced_A_cos_frequency
        for j, mode in enumerate(modes[:5]):
            fig.add_trace(
                go.Scatter(
                    x=omega, y=np.abs(cos_coeff[:, j]), mode="lines",
                    name=f"|Aeddy cos, m={mode}|",
                ),
                row=2, col=2,
            )
        if _result_name(result) == "EllipticalResult":
            for j, mode in enumerate(result.mode_numbers[1:5]):
                fig.add_trace(
                    go.Scatter(
                        x=omega,
                        y=np.abs(result.induced_A_sin_frequency[:, j]),
                        mode="lines",
                        name=f"|Aeddy sin, m={mode}|",
                        line=dict(dash="dash"),
                    ),
                    row=2, col=2,
                )
        spec_ylabel = "|Aeddy Fourier coefficient| (Wb/m)"
    else:
        idx0 = int(np.argmin(np.abs(result.x_m)))
        fig.add_trace(
            go.Scatter(
                x=omega,
                y=np.abs(result.By_induced_frequency[:, idx0]),
                mode="lines",
                name="|Induced By| at x≈0",
            ),
            row=2, col=2,
        )
        spec_ylabel = "|Induced By(x≈0)| (T)"

    fig.add_trace(
        go.Heatmap(
            x=time_us,
            y=coord,
            z=K_time,
            colorbar=dict(title="Kz (A/m)", x=0.46, y=0.19, len=0.27),
            name="Boundary Kz",
            colorscale="RdBu_r",
            zmid=0.0,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=time_us,
            y=1e3 * x_obs,
            z=By_induced,
            colorbar=dict(title="By (T)", x=1.02, y=0.19, len=0.27),
            name="Induced By(x,t)",
            colorscale="RdBu_r",
            zmid=0.0,
        ),
        row=3, col=2,
    )

    fig.update_xaxes(title_text=coord_label, row=1, col=1)
    fig.update_yaxes(title_text="Kz (A/m)", row=1, col=1)
    fig.update_xaxes(title_text="x (mm)", row=1, col=2)
    fig.update_yaxes(title_text="By (T)", row=1, col=2)
    fig.update_xaxes(title_text="Time (μs)", row=2, col=1)
    fig.update_yaxes(title_text="Induced By (T)", row=2, col=1)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=2, col=2)
    fig.update_yaxes(title_text=spec_ylabel, row=2, col=2)
    fig.update_xaxes(title_text="Time (μs)", row=3, col=1)
    fig.update_yaxes(title_text=coord_label, row=3, col=1)
    fig.update_xaxes(title_text="Time (μs)", row=3, col=2)
    fig.update_yaxes(title_text="x (mm)", row=3, col=2)

    fig.update_layout(
        title=f"Analytical Eddy Current Solver — {_plot_title(result)}",
        template="plotly_white",
        height=1250,
        width=1550,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return _write_html(fig, output_file)


def write_input_plot(result, parameters: dict[str, Any], output_file: str | Path,
                     fit_order: int = 20) -> Path:
    """Write the old pre-process-style page: geometry, wire field, pulse and applied A."""
    name = _result_name(result)
    wires = _wires_from_parameters(parameters)
    x_obs = result.x_m if name == "FlatPipeResult" else result.x_observation_m
    _, by_static = _wire_field_on_axis(x_obs, wires)

    if name == "CircularResult":
        bx, by, boundary_coordinate, boundary_label = _geometry_boundary(result)
        boundary_A = vector_potential_from_wires(bx, by, wires)
    elif name == "EllipticalResult":
        bx, by, boundary_coordinate, boundary_label = _geometry_boundary(result)
        boundary_A = vector_potential_from_wires(bx, by, wires)
    else:
        bx = result.x_m
        by = np.vstack((np.full_like(bx, result.half_gap_m), np.full_like(bx, -result.half_gap_m)))
        boundary_coordinate = result.x_m
        boundary_label = "x (m)"
        boundary_A = np.vstack((result.applied_A_top_static, result.applied_A_bottom_static))

    pulse_time = inverse_fourier(
        result.omega_rad_s,
        np.asarray(result.pulse_spectrum)[:, None],
        result.time_s,
    )[0]

    finite = np.isfinite(by_static)
    degree = max(0, min(int(fit_order), int(np.count_nonzero(finite)) - 2))
    if degree >= 0 and np.count_nonzero(finite) >= degree + 1:
        fit = Polynomial.fit(x_obs[finite], by_static[finite], degree).convert()
        fitted_by = np.full_like(by_static, np.nan)
        fitted_by[finite] = fit(x_obs[finite])
    else:
        fitted_by = np.full_like(by_static, np.nan)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Chamber boundary and external wires",
            "Applied By along y = 0 and polynomial fit",
            "Applied pulse spectrum",
            "Applied pulse waveform",
            "Applied vector potential on the boundary / plates",
            "Applied magnetic field along y = 0",
        ),
        horizontal_spacing=0.13,
        vertical_spacing=0.17,
    )

    if name == "FlatPipeResult":
        fig.add_trace(go.Scatter(x=1e3 * bx, y=1e3 * by[0], mode="lines", name="Upper plate"), row=1, col=1)
        fig.add_trace(go.Scatter(x=1e3 * bx, y=1e3 * by[1], mode="lines", name="Lower plate"), row=1, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=1e3 * np.r_[bx, bx[0]], y=1e3 * np.r_[by, by[0]], mode="lines", name="Boundary"),
            row=1, col=1,
        )
    if wires:
        fig.add_trace(
            go.Scatter(
                x=[w.x_mm for w in wires], y=[w.y_mm for w in wires], mode="markers+text",
                text=[f"{w.current_A:g} A" for w in wires], textposition="top center",
                marker=dict(size=9, color=["red" if w.current_A > 0 else "blue" for w in wires]),
                name="External wires",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(x=1e3 * x_obs, y=by_static, mode="lines", name="Applied By from wires"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=1e3 * x_obs, y=fitted_by, mode="lines", name=f"Polynomial fit (degree {degree})", line=dict(dash="dash")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=result.omega_rad_s, y=result.pulse_spectrum, mode="lines", name="Pulse spectrum f(ω)"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=1e6 * result.time_s, y=pulse_time, mode="lines", name="Pulse f(t)"),
        row=2, col=2,
    )

    if name == "FlatPipeResult":
        fig.add_trace(go.Scatter(x=1e3 * result.x_m, y=result.applied_A_top_static, mode="lines", name="Upper plate"), row=3, col=1)
        fig.add_trace(go.Scatter(x=1e3 * result.x_m, y=result.applied_A_bottom_static, mode="lines", name="Lower plate"), row=3, col=1)
        fig.update_xaxes(title_text="x (mm)", row=3, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=boundary_coordinate, y=boundary_A, mode="lines", name="Applied Az"),
            row=3, col=1,
        )
        fig.update_xaxes(title_text=boundary_label, row=3, col=1)
    fig.update_yaxes(title_text="Applied Az (Wb/m)", row=3, col=1)

    bx_static, by_static_2 = _wire_field_on_axis(x_obs, wires)
    fig.add_trace(go.Scatter(x=1e3*x_obs, y=bx_static, mode="lines", name="Applied Bx"), row=3, col=2)
    fig.add_trace(go.Scatter(x=1e3*x_obs, y=by_static_2, mode="lines", name="Applied By"), row=3, col=2)

    fig.update_xaxes(title_text="x (mm)", row=1, col=1)
    fig.update_yaxes(title_text="y (mm)", row=1, col=1)
    fig.update_xaxes(title_text="x (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Applied By (T)", row=1, col=2)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="Pulse spectrum (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (μs)", row=2, col=2)
    fig.update_yaxes(title_text="Pulse amplitude", row=2, col=2)
    fig.update_xaxes(title_text="x (mm)", row=3, col=2)
    fig.update_yaxes(title_text="Applied field (T)", row=3, col=2)
    fig.update_layout(
        title=f"Input / Pre-process — {_plot_title(result)}",
        template="plotly_white", height=1350, width=1500,
        hovermode="closest", showlegend=True,
    )
    return _write_html(fig, output_file)


def write_frequency_plot(result, output_file: str | Path, selected_mode: int | None = None) -> Path:
    """Write detailed real/imaginary frequency responses at representative positions/modes."""
    name = _result_name(result)
    omega = result.omega_rad_s
    x = result.x_m if name == "FlatPipeResult" else result.x_observation_m
    by = result.By_induced_frequency
    indices = _sample_indices(len(x), 6)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Real part of induced By(ω) at selected x positions",
            "Imaginary part of induced By(ω) at selected x positions",
            "Induced vector-potential frequency response",
            "Surface-current frequency response",
        ),
        horizontal_spacing=0.13,
        vertical_spacing=0.18,
    )
    for idx in indices:
        label = f"x={1e3*x[idx]:.3g} mm"
        fig.add_trace(go.Scatter(x=omega, y=by[:, idx].real, mode="lines", name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=omega, y=by[:, idx].imag, mode="lines", name=label), row=1, col=2)

    if name in ("CircularResult", "EllipticalResult"):
        mode_numbers = result.mode_numbers
        if selected_mode is None:
            mode_idx = min(2, len(mode_numbers) - 1)
        else:
            mode_idx = int(np.argmin(np.abs(mode_numbers - selected_mode)))
        mode = int(mode_numbers[mode_idx])
        aeddy_components = [("Cos", result.induced_A_cos_frequency[:, mode_idx])]
        current_components = [("Cos", result.current_cos_frequency[:, mode_idx])]
        if name == "EllipticalResult" and mode > 0:
            aeddy_components.append(("Sin", result.induced_A_sin_frequency[:, mode_idx - 1]))
            current_components.append(("Sin", result.current_sin_frequency[:, mode_idx - 1]))

        for component, values in aeddy_components:
            fig.add_trace(go.Scatter(x=omega, y=values.real, mode="lines", name=f"Aeddy {component} Re, m={mode}"), row=2, col=1)
            fig.add_trace(go.Scatter(x=omega, y=values.imag, mode="lines", name=f"Aeddy {component} Im, m={mode}", line=dict(dash="dash")), row=2, col=1)
        for component, values in current_components:
            fig.add_trace(go.Scatter(x=omega, y=values.real, mode="lines", name=f"Kz {component} Re, m={mode}"), row=2, col=2)
            fig.add_trace(go.Scatter(x=omega, y=values.imag, mode="lines", name=f"Kz {component} Im, m={mode}", line=dict(dash="dash")), row=2, col=2)
        left_ylabel = f"Aeddy coefficient (m={mode}, Wb/m)"
        right_ylabel = f"Kz coefficient (m={mode}, A/m)"
    else:
        idx0 = int(np.argmin(np.abs(result.x_m)))
        for label, values in (
            ("Top plate", result.induced_A_top_frequency[:, idx0]),
            ("Bottom plate", result.induced_A_bottom_frequency[:, idx0]),
        ):
            fig.add_trace(go.Scatter(x=omega, y=values.real, mode="lines", name=f"{label} Re"), row=2, col=1)
            fig.add_trace(go.Scatter(x=omega, y=values.imag, mode="lines", name=f"{label} Im", line=dict(dash="dash")), row=2, col=1)
        for label, values in (
            ("Top plate", result.current_top_frequency[:, idx0]),
            ("Bottom plate", result.current_bottom_frequency[:, idx0]),
        ):
            fig.add_trace(go.Scatter(x=omega, y=values.real, mode="lines", name=f"{label} Re"), row=2, col=2)
            fig.add_trace(go.Scatter(x=omega, y=values.imag, mode="lines", name=f"{label} Im", line=dict(dash="dash")), row=2, col=2)
        left_ylabel = f"Aeddy at x={1e3*result.x_m[idx0]:.3g} mm (Wb/m)"
        right_ylabel = f"Kz at x={1e3*result.x_m[idx0]:.3g} mm (A/m)"

    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=1, col=1)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=1, col=2)
    fig.update_yaxes(title_text="Re[By] (T)", row=1, col=1)
    fig.update_yaxes(title_text="Im[By] (T)", row=1, col=2)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=2, col=1)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=2, col=2)
    fig.update_yaxes(title_text=left_ylabel, row=2, col=1)
    fig.update_yaxes(title_text=right_ylabel, row=2, col=2)
    fig.update_layout(
        title=f"Frequency-domain details — {_plot_title(result)}",
        template="plotly_white", height=1050, width=1550,
        hovermode="closest", showlegend=True,
    )
    return _write_html(fig, output_file)


def write_fourier_modes_plot(result, output_file: str | Path) -> Path:
    """Write frequency- and time-domain Aeddy Fourier coefficients for every mode."""
    name = _result_name(result)
    if name not in ("CircularResult", "EllipticalResult"):
        raise TypeError("Boundary Fourier-mode report is only available for circular/elliptical results")

    modes = result.mode_numbers
    omega = result.omega_rad_s
    time_us = 1e6 * result.time_s
    n_modes = len(modes)
    fig = make_subplots(
        rows=n_modes,
        cols=2,
        subplot_titles=[
            title for mode in modes for title in (f"m={int(mode)} Frequency Domain", f"m={int(mode)} Time Domain")
        ],
        vertical_spacing=min(0.04, 0.8 / max(n_modes, 1)),
        horizontal_spacing=0.08,
    )

    for j, mode in enumerate(modes):
        cos_freq = result.induced_A_cos_frequency[:, j]
        cos_time = inverse_fourier(omega, cos_freq[:, None], result.time_s)[0]
        fig.add_trace(go.Scatter(x=omega, y=cos_freq.real, mode="lines", name=f"Cos Re m={mode}"), row=j+1, col=1)
        fig.add_trace(go.Scatter(x=omega, y=cos_freq.imag, mode="lines", name=f"Cos Im m={mode}"), row=j+1, col=1)
        fig.add_trace(go.Scatter(x=time_us, y=cos_time, mode="lines", name=f"Cos time m={mode}"), row=j+1, col=2)

        if name == "EllipticalResult" and int(mode) > 0:
            sin_freq = result.induced_A_sin_frequency[:, j - 1]
            sin_time = inverse_fourier(omega, sin_freq[:, None], result.time_s)[0]
            fig.add_trace(go.Scatter(x=omega, y=sin_freq.real, mode="lines", name=f"Sin Re m={mode}"), row=j+1, col=1)
            fig.add_trace(go.Scatter(x=omega, y=sin_freq.imag, mode="lines", name=f"Sin Im m={mode}"), row=j+1, col=1)
            fig.add_trace(go.Scatter(x=time_us, y=sin_time, mode="lines", name=f"Sin time m={mode}"), row=j+1, col=2)

    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=n_modes, col=1)
    fig.update_xaxes(title_text="Time (μs)", row=n_modes, col=2)
    fig.update_yaxes(title_text="Aeddy Fourier coefficient (Wb/m)")
    fig.update_layout(
        title=f"All boundary Fourier modes — {_plot_title(result)}",
        template="plotly_white", height=max(600, 300*n_modes), width=1500,
        hovermode="closest", showlegend=True,
    )
    return _write_html(fig, output_file)


def write_plate_response_plot(result, output_file: str | Path) -> Path:
    """Write the two-plate response page for the flat-pipe solution."""
    if _result_name(result) != "FlatPipeResult":
        raise TypeError("Plate-response report is only available for flat-pipe results")
    it_peak = int(np.argmin(np.abs(result.time_s)))
    x_mm = 1e3 * result.x_m
    t_us = 1e6 * result.time_s
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Top/bottom plate current at t ≈ 0",
            "Top/bottom plate induced vector potential at t ≈ 0",
            "Top plate current evolution",
            "Bottom plate current evolution",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    fig.add_trace(go.Scatter(x=x_mm, y=result.current_top_time[:, it_peak], mode="lines", name="Top Kz"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=result.current_bottom_time[:, it_peak], mode="lines", name="Bottom Kz"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=result.induced_A_top_time[:, it_peak], mode="lines", name="Top Aeddy"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_mm, y=result.induced_A_bottom_time[:, it_peak], mode="lines", name="Bottom Aeddy"), row=1, col=2)
    fig.add_trace(
        go.Heatmap(
            x=t_us, y=x_mm, z=result.current_top_time,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(
                title="Kz top",
                len=0.30,
                y=0.25,
                yanchor="middle",
                x=0.43,
            ),
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=t_us, y=x_mm, z=result.current_bottom_time,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(
                title="Kz bottom",
                len=0.30,
                y=0.25,
                yanchor="middle",
                x=1.02,
            ),
        ),
        row=2, col=2,
    )
    fig.update_xaxes(title_text="x (mm)", row=1, col=1)
    fig.update_xaxes(title_text="x (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Kz (A/m)", row=1, col=1)
    fig.update_yaxes(title_text="Aeddy (Wb/m)", row=1, col=2)
    fig.update_xaxes(title_text="Time (μs)", row=2, col=1)
    fig.update_xaxes(title_text="Time (μs)", row=2, col=2)
    fig.update_yaxes(title_text="x (mm)", row=2, col=1)
    fig.update_yaxes(title_text="x (mm)", row=2, col=2)
    fig.update_layout(
        title=f"Parallel-plate response — {_plot_title(result)}",
        template="plotly_white", height=1050, width=1500,
        hovermode="closest", showlegend=True,
    )
    return _write_html(fig, output_file)


def write_by_animation(result, output_file: str | Path, max_spatial_points: int = 600) -> Path:
    """Write an interactive animation of induced By(x,t), with a fixed vertical scale."""
    x, by_induced, _ = _field_view(result)
    x_mm = 1e3 * x
    # Downsample only the animation display for large arrays. The saved
    # CSV/NPZ data and the static plots retain the full solver resolution.
    indices = _sample_indices(len(x), max_spatial_points)
    time_indices = _sample_indices(len(result.time_s), 300)
    x_plot = x_mm[indices]
    by_plot = by_induced[np.ix_(indices, time_indices)].T
    t_us = 1e6 * result.time_s[time_indices]
    global_max = float(np.max(np.abs(by_plot))) if by_plot.size else 0.0
    global_max = max(global_max * 1.1, 1e-15)

    fig = go.Figure(
        data=[go.Scatter(x=x_plot, y=by_plot[0], mode="lines", name="Induced By(x)")]
    )
    frames = [
        go.Frame(
            data=[go.Scatter(x=x_plot, y=by_plot[it], mode="lines")],
            name=f"frame_{it}",
            traces=[0],
        )
        for it in range(len(t_us))
    ]
    fig.frames = frames
    steps = [
        {
            "args": [[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": f"{t_us[i]:.3g}",
            "method": "animate",
        }
        for i in range(len(t_us))
    ]
    fig.update_layout(
        title=f"Induced By(x) evolution over time — {_plot_title(result)}",
        template="plotly_white", height=700, width=1250,
        xaxis_title="x (mm)", yaxis_title="Induced By (T)",
        yaxis=dict(range=[-global_max, global_max], fixedrange=True),
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.1, "y": 0,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Time (μs): "},
            "steps": steps,
        }],
    )
    return _write_html(fig, output_file)


def write_multipole_fit_plot(result, output_file: str | Path, fit_order: int = 20) -> Path:
    """Fit the induced By(x) profile at every time point and plot coefficients vs time."""
    x, by_induced, _ = _field_view(result)
    x = np.asarray(x, dtype=float)
    by_induced = np.asarray(by_induced, dtype=float)
    if by_induced.shape != (x.size, result.time_s.size):
        raise ValueError("By_induced_time has an unexpected shape")
    if x.size < 3:
        raise ValueError("At least three x points are required for polynomial fitting")

    degree = max(1, min(int(fit_order), x.size - 2))
    coeffs = np.zeros((degree + 1, result.time_s.size), dtype=float)
    for it in range(result.time_s.size):
        # Polynomial.fit scales the domain internally and is more stable than
        # fitting directly with high powers of a coordinate measured in metres.
        fit = Polynomial.fit(x, by_induced[:, it], degree).convert()
        coeffs[:, it] = fit.coef

    t_us = 1e6 * result.time_s
    it_peak = int(np.argmin(np.abs(result.time_s)))
    fit_peak = Polynomial(coeffs[:, it_peak])(x)
    fit_data = {"x_m": x, "x_mm": 1e3*x, "By_induced_T": by_induced[:, it_peak], "By_fitted_T": fit_peak}
    for order in range(degree + 1):
        fit_data[f"c{order}_T_per_m{order}"] = np.full(x.size, coeffs[order, it_peak])
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(fit_data).to_csv(output_file.with_name("multipole_fit_at_pulse_center.csv"), index=False)
    coeff_data = {"time_us": t_us}
    for order in range(degree + 1):
        coeff_data[f"c{order}_T_per_m{order}"] = coeffs[order]
    pd.DataFrame(coeff_data).to_csv(output_file.with_name("multipole_coefficients_vs_time.csv"), index=False)

    fig = make_subplots(
        rows=degree + 1, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Polynomial coefficient c{order} (order {order}) vs time" for order in range(degree + 1)],
        vertical_spacing=min(0.025, 0.5 / max(degree + 1, 1)),
    )
    for order in range(degree + 1):
        fig.add_trace(
            go.Scatter(x=t_us, y=coeffs[order], mode="lines", name=f"c{order}"),
            row=order + 1, col=1,
        )
    fig.update_xaxes(title_text="Time (μs)", row=degree + 1, col=1)
    fig.update_yaxes(title_text="Coefficient (T/mⁿ)")
    fig.update_layout(
        title=f"Polynomial fit coefficients of induced By(x,t), degree {degree} — {_plot_title(result)}",
        template="plotly_white", height=max(700, 260*(degree + 1)), width=1250,
        showlegend=False,
    )
    return _write_html(fig, output_file)



def write_ellipse_bem_comparison_plot(result, output_file: str | Path) -> Path:
    """Compare the analytical ellipse solution with the Green-function BEM.

    The BEM induced potential has already had its longitudinal A_offset
    removed and has the pulse spectrum applied, matching the analytical
    solver's effective m=0 convention.
    """
    comparison = getattr(result, "bem_comparison", None)
    if comparison is None:
        raise ValueError("No BEM comparison data is attached to this result")

    omega = result.omega_rad_s
    modes = result.mode_numbers
    fig = make_subplots(
        rows=5, cols=2,
        specs=[
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=(
            "m=0 cosine coefficient — real part",
            "m=0 cosine coefficient — imaginary part",
            "Maximum relative error by Fourier mode",
            "BEM numerical diagnostics",
            "Boundary Kz — real part at representative frequency",
            "Boundary Kz — imaginary part at representative frequency",
            "Induced By on y=0 — real part",
            "Induced Bx on y=0 — real part",
            "m=0 time domain — analytical vs offset-aligned BEM",
            "m=0 time domain — raw BEM (without offset subtraction)",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    # The zero mode is the main item under investigation.  The comparison
    # uses BEM Aeddy - A_offset, not the raw BEM induced_A.
    analytic_m0 = result.induced_A_cos_frequency[:, 0]
    bem_m0 = comparison.bem_induced_A_cos_frequency[:, 0]
    fig.add_trace(go.Scatter(x=omega, y=analytic_m0.real, mode="lines", name="Analytical Re"), row=1, col=1)
    fig.add_trace(go.Scatter(x=omega, y=bem_m0.real, mode="lines", name="BEM aligned Re", line=dict(dash="dash")), row=1, col=1)
    raw_bem_m0 = bem_m0 + comparison.offset_frequency
    fig.add_trace(go.Scatter(x=omega, y=raw_bem_m0.real, mode="lines", name="BEM raw Re (before offset removal)", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=omega, y=analytic_m0.imag, mode="lines", name="Analytical Im"), row=1, col=2)
    fig.add_trace(go.Scatter(x=omega, y=bem_m0.imag, mode="lines", name="BEM aligned Im", line=dict(dash="dash")), row=1, col=2)

    # Reconstruct the m=0 coefficient in time domain.  The analytical and
    # offset-aligned BEM spectra should agree; the raw BEM curve is included
    # to show the effect of the longitudinal A_offset explicitly.
    analytical_m0_time = inverse_fourier(
        omega, analytic_m0[:, None], result.time_s
    )[0]
    bem_aligned_m0_time = inverse_fourier(
        omega, bem_m0[:, None], result.time_s
    )[0]
    
    bem_raw_m0_spectrum = bem_m0 + comparison.offset_frequency

    # Check whether the raw m=0 component is numerically negligible
    # compared with the two terms that cancel each other.
    raw_reference_scale = max(
        float(np.max(np.abs(bem_m0))),
        float(np.max(np.abs(comparison.offset_frequency))),
        1e-30,
    )

    raw_m0_relative_level = (
        float(np.max(np.abs(bem_raw_m0_spectrum))) / raw_reference_scale
    )

    if raw_m0_relative_level < 1e-10:
        # The raw m=0 component is below the reliable numerical
        # resolution of the cancellation. Treat it as zero.
        bem_raw_m0_time = np.zeros_like(result.time_s, dtype=float)
        raw_bem_label = "BEM raw m=0 time (numerically zero)"
    else:
        bem_raw_m0_time = inverse_fourier(
            omega, bem_raw_m0_spectrum[:, None], result.time_s
        )[0]
        raw_bem_label = "BEM raw m=0 time (no offset removal)"

    time_us = 1e6 * result.time_s

    fig.add_trace(go.Scatter(
        x=time_us, y=analytical_m0_time, mode="lines",
        name="Analytical m=0 time",
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=time_us, y=bem_aligned_m0_time, mode="lines",
        name="BEM aligned m=0 time", line=dict(dash="dash"),
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=time_us, y=bem_raw_m0_time, mode="lines",
        name=raw_bem_label, line=dict(dash="dot"),
    ), row=5, col=2)

    # Save the reconstructed zero-mode time traces for numerical inspection.
    import pandas as pd
    time_csv = Path(output_file).with_name("ellipse_bem_m0_time.csv")
    time_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time_us": time_us,
        "Aeddy_m0_analytical_Wb_per_m": analytical_m0_time,
        "Aeddy_m0_BEM_aligned_Wb_per_m": bem_aligned_m0_time,
        "Aeddy_m0_BEM_raw_Wb_per_m": bem_raw_m0_time,
        # Optional legacy-style coefficients. The original GUI used 2/N
        # even for m=0; the standard constant coefficient is mean(A).
        "Aeddy_m0_analytical_legacy_2mean_Wb_per_m": 2.0 * analytical_m0_time,
        "Aeddy_m0_BEM_aligned_legacy_2mean_Wb_per_m": 2.0 * bem_aligned_m0_time,
    }).to_csv(time_csv, index=False)

    # Show per-mode relative errors only for modes with a meaningful signal.
    # Symmetry-forbidden odd/sine modes are numerical zero; normalizing their
    # tiny round-off values by themselves would misleadingly show ~100% error.
    global_scale = max(
        float(np.max(np.abs(result.induced_A_cos_frequency))),
        float(np.max(np.abs(comparison.bem_induced_A_cos_frequency))),
        float(np.max(np.abs(result.induced_A_sin_frequency))) if result.induced_A_sin_frequency.size else 0.0,
        float(np.max(np.abs(comparison.bem_induced_A_sin_frequency))) if comparison.bem_induced_A_sin_frequency.size else 0.0,
        1e-30,
    )
    error_threshold = 1e-8 * global_scale
    error_labels = []
    error_values = []
    for j, mode in enumerate(modes):
        av = result.induced_A_cos_frequency[:, j]
        bv = comparison.bem_induced_A_cos_frequency[:, j]
        scale = max(float(np.max(np.abs(av))), float(np.max(np.abs(bv))))
        if scale >= error_threshold:
            error_labels.append(f"cos {int(mode)}")
            error_values.append(float(np.max(np.abs(av - bv)) / scale))
    for j, mode in enumerate(modes[1:]):
        av = result.induced_A_sin_frequency[:, j]
        bv = comparison.bem_induced_A_sin_frequency[:, j]
        scale = max(float(np.max(np.abs(av))), float(np.max(np.abs(bv))))
        if scale >= error_threshold:
            error_labels.append(f"sin {int(mode)}")
            error_values.append(float(np.max(np.abs(av - bv)) / scale))
    fig.add_trace(
        go.Bar(x=error_labels, y=error_values, name="Maximum relative coefficient error"),
        row=2, col=1,
    )

    fig.add_trace(go.Scatter(
        x=omega, y=comparison.net_current_relative_residual, mode="lines",
        name="Relative net-current residual",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=omega, y=comparison.equation_relative_residual, mode="lines",
        name="Relative equation residual", line=dict(dash="dash"),
    ), row=2, col=2)

    # Pick a positive frequency away from DC for spatial comparisons.
    positive = np.flatnonzero(omega > 0.0)
    if positive.size:
        target = 0.2 * float(np.max(omega))
        kf = int(positive[np.argmin(np.abs(omega[positive] - target))])
    else:
        kf = int(np.argmax(np.abs(omega)))
    psi = comparison.psi_rad
    cos_basis = np.cos(np.outer(psi, modes))
    sin_basis = np.sin(np.outer(psi, modes[1:]))
    analytic_k = (
        result.current_cos_frequency[kf] @ cos_basis.T
        + result.current_sin_frequency[kf] @ sin_basis.T
    )
    bem_k = comparison.bem_current_boundary_frequency[kf]
    angle_deg = np.degrees(psi)
    fig.add_trace(go.Scatter(x=angle_deg, y=analytic_k.real, mode="lines", name="Analytical Re Kz"), row=3, col=1)
    fig.add_trace(go.Scatter(x=angle_deg, y=bem_k.real, mode="lines", name="BEM Re Kz", line=dict(dash="dash")), row=3, col=1)
    fig.add_trace(go.Scatter(x=angle_deg, y=analytic_k.imag, mode="lines", name="Analytical Im Kz"), row=3, col=2)
    fig.add_trace(go.Scatter(x=angle_deg, y=bem_k.imag, mode="lines", name="BEM Im Kz", line=dict(dash="dash")), row=3, col=2)

    x_mm = 1e3 * result.x_observation_m
    by_a = result.By_induced_frequency[kf]
    by_b = comparison.bem_By_induced_frequency[kf]
    bx_a = result.Bx_induced_frequency[kf]
    bx_b = comparison.bem_Bx_induced_frequency[kf]
    fig.add_trace(go.Scatter(x=x_mm, y=by_a.real, mode="lines", name="Analytical Re By"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=by_b.real, mode="lines", name="BEM Re By", line=dict(dash="dash")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=bx_a.real, mode="lines", name="Analytical Re Bx"), row=4, col=2)
    fig.add_trace(go.Scatter(x=x_mm, y=bx_b.real, mode="lines", name="BEM Re Bx", line=dict(dash="dash")), row=4, col=2)

    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=1, col=1)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=1, col=2)
    fig.update_yaxes(title_text="Aeddy m=0 (Wb/m)", row=1, col=1)
    fig.update_yaxes(title_text="Aeddy m=0 (Wb/m)", row=1, col=2)
    fig.update_yaxes(title_text="Relative error", row=2, col=1)
    fig.update_xaxes(title_text="Fourier component", row=2, col=1)
    fig.update_xaxes(title_text="Angular frequency ω (rad/s)", row=2, col=2)
    fig.update_yaxes(title_text="Residual", type="log", row=2, col=2)
    fig.update_xaxes(title_text="Ellipse parameter ψ (deg)", row=3, col=1)
    fig.update_xaxes(title_text="Ellipse parameter ψ (deg)", row=3, col=2)
    fig.update_yaxes(title_text="Kz (A/m)", row=3, col=1)
    fig.update_yaxes(title_text="Kz (A/m)", row=3, col=2)
    fig.update_xaxes(title_text="x (mm)", row=4, col=1)
    fig.update_xaxes(title_text="x (mm)", row=4, col=2)
    fig.update_yaxes(title_text="By (T)", row=4, col=1)
    fig.update_yaxes(title_text="Bx (T)", row=4, col=2)
    fig.update_xaxes(title_text="Time (μs)", row=5, col=1)
    fig.update_xaxes(title_text="Time (μs)", row=5, col=2)
    fig.update_yaxes(title_text="Aeddy m=0, standard a₀ (Wb/m)", row=5, col=1)
    fig.update_yaxes(title_text="Aeddy m=0, raw BEM (Wb/m)", row=5, col=2)
    fig.update_layout(
        title=(
            "Elliptical analytical solution vs Green-function BEM "
            f"(BEM panels={comparison.panel_count}, ω={omega[kf]:.4e} rad/s for spatial plots)"
        ),
        template="plotly_white", height=2050, width=1650,
        hovermode="closest", showlegend=True,
        margin=dict(t=175),
    )
    fig.add_annotation(
        text=(
            "BEM aligned Aeddy = raw BEM Aeddy − solved A_offset; pulse spectrum is applied to both methods.<br>"
            "Time-domain panels are separated because the unaligned raw BEM m=0 trace is nearly zero, "
            "while the aligned and analytical traces are much larger.<br>"
            "Fourier convention: a₀ = mean(A), aₘ = 2 mean(A cos(mψ)) for m ≥ 1. "
            "The legacy GUI used 2/N also for m=0, so its displayed m=0 coefficient is twice the standard a₀. "
            "This is a post-processing normalization difference, not a Green-kernel or current-solver difference."
        ),
        xref="paper", yref="paper", x=0.0, y=1.115, showarrow=False,
        align="left", font=dict(size=11),
    )
    return _write_html(fig, output_file)

def write_result_reports(
    result,
    parameters: dict[str, Any],
    output_dir: str | Path,
    fit_order: int = 20,
) -> dict[str, Path]:
    """Generate all HTML reports and return them in the order used by the GUI tabs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape_tag = str(parameters.get("shape", "result")).lower().replace(" ", "_")
    reports: dict[str, Path] = {}
    reports["Input / Pre-process"] = write_input_plot(result, parameters, out / "pre_plots.html", fit_order)
    reports["Overview"] = write_result_plot(result, out / f"{shape_tag}_results.html")
    reports["Frequency response"] = write_frequency_plot(result, out / "post_plots.html")
    if _result_name(result) in ("CircularResult", "EllipticalResult"):
        if _result_name(result) == "EllipticalResult":
            mode_report_name = "all_ellipse_fourier_orders.html"
        else:
            mode_report_name = "all_circular_fourier_orders.html"
        reports["Fourier modes"] = write_fourier_modes_plot(result, out / mode_report_name)
    # else:
    #     reports["Plate response"] = write_plate_response_plot(result, out / "plate_response.html")
    reports["By(x,t) animation"] = write_by_animation(result, out / "By_x_t_animation.html")
    reports["Multipole fit"] = write_multipole_fit_plot(result, out / "multipole_coefficients.html", fit_order)
    # if _result_name(result) == "EllipticalResult" and getattr(result, "bem_comparison", None) is not None:
    #     reports["Green's-function check"] = write_ellipse_bem_comparison_plot(
    #         result, out / "ellipse_bem_comparison.html"
    #     )
    return reports
