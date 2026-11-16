"""2-D thin-conducting-wall eddy-current boundary-integral solver.

Model
-----
* Invariance along the longitudinal z direction: d/dz = 0.
* The only current component represented is the longitudinal sheet current K_z.
* A boundary is represented by straight panels in the x-y cross-section.
* K_z is piecewise constant on each panel.
* The logarithmic Green kernel is integrated analytically over each source panel.

Phasor/Fourier convention
-------------------------
The forward transform is F(w) = integral f(t) exp(-i*w*t) dt and the
inverse transform is f(t) = (1/2*pi) integral F(w) exp(+i*w*t) dw.
Thus d/dt -> +i*w and E_z = -dA_z/dt = -i*w*A_z (before adding
a longitudinal scalar-potential gradient). This matches the sign convention
used by the legacy GUI's time reconstruction.

Electrical constraints
----------------------
Each current_group is assigned one optional zero-net-longitudinal-current
constraint. Components with the same current_group share one constraint,
which is useful when they are connected by an externally specified return
path. By default, each geometric component is its own current group.

This is a 2-D cross-sectional model. It does NOT calculate longitudinal end
closures, z-dependent current redistribution, or the reduction of eddy
currents caused by finite-length strip segmentation. The current_group option
only imposes a cross-section-integrated current condition; it is not a 3-D
circuit or end-effect model.

Units
-----
Boundary point coordinates and wire coordinates: mm
Panel geometry / observation points: m
Wire current: A
Conductivity: S/m
Film thickness: m
Angular frequency: rad/s
K_z: A/m
A_z: T*m (equivalently Wb/m)
B: T
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence
import csv
import math

import numpy as np

MU0 = 4.0e-7 * np.pi
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class BoundaryComponent:
    """One polyline in the transverse (x-y) plane.

    Parameters
    ----------
    points_mm:
        Ordered vertices, shape (n, 2), in mm.
    closed:
        If True, the last vertex is connected to the first vertex.
        This is a geometric closure in the x-y plane, not a statement about
        electrical connection at the longitudinal ends of the chamber.
    name:
        Display/debug label.
    current_group:
        Components with the same non-None label share one zero-net-current
        constraint. None means a separate group for this component.
    zero_net_current:
        Apply integral(K_z ds) = 0 to this current group. This is the default
        isolated/no-net-current approximation. Set False only when the
        longitudinal voltage/current condition is specified externally; this
        solver then assumes zero additional longitudinal scalar-potential
        gradient for that component.
    """

    points_mm: np.ndarray
    closed: bool = False
    name: str = "boundary"
    current_group: Hashable | None = None
    zero_net_current: bool = True

    def __post_init__(self) -> None:
        pts = np.asarray(self.points_mm, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"{self.name}: points_mm must have shape (N, 2)")
        if not np.all(np.isfinite(pts)):
            raise ValueError(f"{self.name}: boundary coordinates must be finite")
        min_points = 3 if self.closed else 2
        if len(pts) < min_points:
            raise ValueError(f"{self.name}: need at least {min_points} vertices")

        # Closed CSV/polyline files often repeat the first point at the end.
        # Remove that duplicate because the closing panel is added below.
        if self.closed and len(pts) > 3 and np.linalg.norm(pts[-1] - pts[0]) < 1e-12:
            pts = pts[:-1]
        if len(pts) < min_points:
            raise ValueError(f"{self.name}: not enough unique vertices")

        seg = np.diff(pts, axis=0)
        if np.any(np.linalg.norm(seg, axis=1) < 1e-12):
            raise ValueError(f"{self.name}: consecutive duplicate vertices create zero-length panels")
        if not self.closed and np.linalg.norm(pts[-1] - pts[0]) < 1e-12:
            raise ValueError(
                f"{self.name}: open component has identical first/last points; "
                "set closed=True if it is intended to close"
            )
        object.__setattr__(self, "points_mm", pts.copy())


@dataclass
class BoundaryMesh:
    """Straight-panel mesh used by the integral solver."""

    starts_m: np.ndarray
    ends_m: np.ndarray
    midpoints_m: np.ndarray
    lengths_m: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    component_ids: np.ndarray
    component_names: tuple[str, ...]
    panel_current_group: tuple[Hashable, ...]
    constraint_labels: tuple[Hashable, ...]
    constraint_matrix: np.ndarray  # shape (Npanel, Nconstraint)

    @property
    def n_panels(self) -> int:
        return int(len(self.lengths_m))

    @property
    def n_constraints(self) -> int:
        return int(self.constraint_matrix.shape[1])


@dataclass
class FrequencySolution:
    omega: float
    current_density: np.ndarray  # K_z on panels, A/m
    induced_A: np.ndarray        # A_eddy at panel midpoints, T*m
    total_A: np.ndarray          # A_drive + A_eddy, T*m
    A_offsets: np.ndarray        # one offset per constrained current group, T*m
    residual_norm: float


@dataclass
class SpectrumSolution:
    omega: np.ndarray
    drive_A: np.ndarray
    current_density: np.ndarray  # shape (Nomega, Npanel)
    induced_A: np.ndarray        # shape (Nomega, Npanel)
    A_offsets: np.ndarray        # shape (Nomega, Nconstraint)
    log_kernel: np.ndarray       # shape (Npanel, Npanel), units m


def component_from_point_dicts(
    points: Sequence[Mapping[str, float]],
    *,
    closed: bool = True,
    name: str = "boundary",
    current_group: Hashable | None = None,
    zero_net_current: bool = True,
) -> BoundaryComponent:
    """Adapter for the legacy [{'x': mm, 'y': mm}, ...] point format."""
    xy = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    return BoundaryComponent(xy, closed, name, current_group, zero_net_current)


def make_parallel_plate_components(
    width_mm: float,
    gap_mm: float,
    points_per_plate: int,
    *,
    shared_current_group: Hashable | None = None,
    zero_net_current: bool = True,
) -> list[BoundaryComponent]:
    """Create two finite-width horizontal plates in the x-y cross-section.

    By default the upper and lower plates are separate current groups. If the
    physical end connections are represented by one common return circuit,
    pass a shared_current_group label to impose one combined net-current
    constraint. The 2-D model still does not resolve the end connections.
    """
    if width_mm <= 0 or gap_mm <= 0:
        raise ValueError("width_mm and gap_mm must be positive")
    if points_per_plate < 2:
        raise ValueError("points_per_plate must be >= 2")
    x_top = np.linspace(-width_mm / 2.0, width_mm / 2.0, points_per_plate)
    x_bottom = np.linspace(width_mm / 2.0, -width_mm / 2.0, points_per_plate)
    top = np.column_stack((x_top, np.full_like(x_top, gap_mm / 2.0)))
    bottom = np.column_stack((x_bottom, np.full_like(x_bottom, -gap_mm / 2.0)))
    return [
        BoundaryComponent(top, closed=False, name="upper_plate",
                          current_group=shared_current_group,
                          zero_net_current=zero_net_current),
        BoundaryComponent(bottom, closed=False, name="lower_plate",
                          current_group=shared_current_group,
                          zero_net_current=zero_net_current),
    ]


def load_boundary_csv(
    file_path: str | Path,
    *,
    legacy_single_component_closed: bool = True,
) -> list[BoundaryComponent]:
    """Load one or more components from CSV.

    Required columns: x,y (or x_mm,y_mm), in mm.

    Optional columns:
      component       component name; if omitted, all rows form one component
      closed          true/false; must be consistent within a component
      current_group   equal labels share one current constraint; blank = own group
      zero_net_current true/false; must be consistent within a current group

    Legacy two-column x,y files are treated as one closed boundary by default.
    For multi-component files, `closed` should be supplied explicitly; if it is
    omitted, components are treated as open polylines.
    """
    path = Path(file_path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header")
        fields = {s.strip().lower(): s for s in reader.fieldnames}
        xkey = fields.get("x_mm", fields.get("x"))
        ykey = fields.get("y_mm", fields.get("y"))
        if xkey is None or ykey is None:
            raise ValueError("CSV needs x,y or x_mm,y_mm columns")
        comp_key = fields.get("component")
        closed_key = fields.get("closed")
        group_key = fields.get("current_group")
        znet_key = fields.get("zero_net_current")

        rows: dict[str, list[tuple[float, float]]] = {}
        closed_values: dict[str, set[bool]] = {}
        group_values: dict[str, set[str | None]] = {}
        znet_values: dict[str, set[bool]] = {}

        for row_number, row in enumerate(reader, start=2):
            if not row or not (row.get(xkey) or "").strip():
                continue
            try:
                x = float(row[xkey])
                y = float(row[ykey])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid x/y at CSV line {row_number}") from exc

            name = ((row.get(comp_key, "") if comp_key else "") or "component_1").strip()
            if not name:
                name = "component_1"
            rows.setdefault(name, []).append((x, y))

            if closed_key:
                closed_values.setdefault(name, set()).add(_parse_bool(row.get(closed_key), default=False))
            if group_key:
                g = (row.get(group_key) or "").strip()
                group_values.setdefault(name, set()).add(g if g else None)
            if znet_key:
                znet_values.setdefault(name, set()).add(_parse_bool(row.get(znet_key), default=True))

    if not rows:
        raise ValueError("CSV contains no boundary points")

    result: list[BoundaryComponent] = []
    multi = len(rows) > 1
    for name, xy in rows.items():
        if closed_key:
            vals = closed_values.get(name, set())
            if len(vals) != 1:
                raise ValueError(f"Component {name!r}: closed must have one consistent value")
            closed = next(iter(vals))
        else:
            # Preserve the historical two-column x,y convention (one closed
            # loop), but require explicit closure metadata for component-aware
            # files so an open/segmented boundary is never silently closed.
            has_component_column = comp_key is not None
            closed = legacy_single_component_closed if (not has_component_column and not multi) else False

        if group_key:
            vals_g = group_values.get(name, set())
            if len(vals_g) != 1:
                raise ValueError(f"Component {name!r}: current_group must be consistent")
            group = next(iter(vals_g))
        else:
            group = None

        if znet_key:
            vals_z = znet_values.get(name, set())
            if len(vals_z) != 1:
                raise ValueError(f"Component {name!r}: zero_net_current must be consistent")
            znet = next(iter(vals_z))
        else:
            znet = True

        result.append(BoundaryComponent(np.asarray(xy), closed, name, group, znet))

    return result


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "closed"}:
        return True
    if s in {"0", "false", "no", "n", "open"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as Boolean")


def build_boundary_mesh(components: Sequence[BoundaryComponent]) -> BoundaryMesh:
    """Convert polylines into straight panels and build current constraints."""
    if not components:
        raise ValueError("At least one boundary component is required")

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    component_ids: list[np.ndarray] = []
    panel_groups: list[Hashable] = []
    component_names = tuple(c.name for c in components)
    group_zero_net: dict[Hashable, bool] = {}

    for ic, comp in enumerate(components):
        pts = np.asarray(comp.points_mm, dtype=float) * 1e-3
        p0 = pts[:-1]
        p1 = pts[1:]
        if comp.closed:
            p0 = np.vstack((p0, pts[-1]))
            p1 = np.vstack((p1, pts[0]))
        seglen = np.linalg.norm(p1 - p0, axis=1)
        if np.any(seglen <= 1e-15):
            raise ValueError(f"{comp.name}: zero-length panel found")
        starts.append(p0)
        ends.append(p1)
        component_ids.append(np.full(len(seglen), ic, dtype=int))

        group = comp.current_group
        if group is None:
            # Make a unique default group for each geometric component.
            group = ("component", ic)
        try:
            hash(group)
        except TypeError as exc:
            raise ValueError(f"{comp.name}: current_group must be hashable") from exc
        if group in group_zero_net and group_zero_net[group] != bool(comp.zero_net_current):
            raise ValueError(f"Current group {group!r} has inconsistent zero_net_current flags")
        group_zero_net[group] = bool(comp.zero_net_current)
        panel_groups.extend([group] * len(seglen))

    p0 = np.vstack(starts)
    p1 = np.vstack(ends)
    ids = np.concatenate(component_ids)
    vec = p1 - p0
    lengths = np.linalg.norm(vec, axis=1)
    tangents = vec / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    mids = 0.5 * (p0 + p1)

    constrained_labels = tuple(g for g, apply in group_zero_net.items() if apply)
    C = np.zeros((len(lengths), len(constrained_labels)), dtype=float)
    label_to_col = {g: j for j, g in enumerate(constrained_labels)}
    for i, g in enumerate(panel_groups):
        if g in label_to_col:
            C[i, label_to_col[g]] = 1.0

    return BoundaryMesh(
        starts_m=p0,
        ends_m=p1,
        midpoints_m=mids,
        lengths_m=lengths,
        tangents=tangents,
        normals=normals,
        component_ids=ids,
        component_names=component_names,
        panel_current_group=tuple(panel_groups),
        constraint_labels=constrained_labels,
        constraint_matrix=C,
    )


def _log_primitive(q: np.ndarray, abs_v: np.ndarray, reference_length_m: float) -> np.ndarray:
    """Primitive of log(sqrt(q^2+v^2)/r_ref) with respect to q."""
    rho2 = q * q + abs_v * abs_v
    log_r = 0.5 * np.log(np.maximum(rho2, np.finfo(float).tiny)) - math.log(reference_length_m)
    return q * (log_r - 1.0) + abs_v * np.arctan2(q, abs_v)


def build_log_integral_matrix(
    mesh: BoundaryMesh,
    *,
    reference_length_m: float = 1.0,
    target_points_m: np.ndarray | None = None,
) -> np.ndarray:
    """Compute H_ij = integral_panel_j log(|r_i-r'|/r_ref) ds' exactly.

    Default targets are panel midpoints. Exact integration over each straight
    source panel removes the point-kernel diagonal singularity and the need for
    an ad-hoc `log(weight)-1` diagonal replacement.
    """
    if reference_length_m <= 0:
        raise ValueError("reference_length_m must be positive")
    targets = mesh.midpoints_m if target_points_m is None else np.asarray(target_points_m, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("target_points_m must have shape (N, 2)")

    delta = targets[:, None, :] - mesh.midpoints_m[None, :, :]
    u = np.einsum("ijk,jk->ij", delta, mesh.tangents)
    v = np.einsum("ijk,jk->ij", delta, mesh.normals)
    av = np.abs(v)
    half = 0.5 * mesh.lengths_m[None, :]
    q_lo = u - half
    q_hi = u + half
    return (
        _log_primitive(q_hi, av, reference_length_m)
        - _log_primitive(q_lo, av, reference_length_m)
    )


def vector_potential_from_wires(
    points_m: np.ndarray,
    wires: Sequence[Mapping[str, float]],
    *,
    mu0: float = MU0,
    reference_length_m: float = 1.0,
) -> np.ndarray:
    """A_z from ideal infinite z-directed wires; wire x/y inputs are in mm."""
    pts = np.asarray(points_m, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_m must have shape (N, 2)")
    Az = np.zeros(len(pts), dtype=float)
    for iw, wire in enumerate(wires):
        wx = float(wire["x"]) * 1e-3
        wy = float(wire["y"]) * 1e-3
        current = float(wire["I"])
        r = np.linalg.norm(pts - np.array([wx, wy]), axis=1)
        if np.any(r < 1e-14):
            raise ValueError(f"Wire {iw} lies on a boundary collocation point")
        Az -= mu0 * current / _TWO_PI * np.log(r / reference_length_m)
    return Az


def solve_frequency(
    omega: float,
    drive_A: np.ndarray,
    mesh: BoundaryMesh,
    log_kernel: np.ndarray,
    conductivity: float,
    thickness_m: float,
    *,
    mu0: float = MU0,
) -> FrequencySolution:
    """Solve one frequency, returning longitudinal sheet current K_z.

    Derivation (exp(+i*w*t) phasors, matching the legacy GUI):
        K = -i*w*sigma*d * (A_drive + A_eddy - C*a0)
        A_eddy = -(mu0/2pi) H K
        C.T @ diag(panel_length) @ K = 0

    The unknown `u = beta*a0`, with `beta = -i*w*sigma*d`, is used in the augmented system to
    avoid a poorly scaled offset unknown as omega approaches zero.
    """
    drive = np.asarray(drive_A, dtype=complex).reshape(-1)
    H = np.asarray(log_kernel, dtype=float)
    n = mesh.n_panels
    if drive.shape != (n,):
        raise ValueError(f"drive_A must have length {n}")
    if H.shape != (n, n):
        raise ValueError(f"log_kernel must have shape ({n}, {n})")
    if conductivity <= 0 or thickness_m <= 0:
        raise ValueError("conductivity and thickness_m must be positive")

    beta = -1j * float(omega) * conductivity * thickness_m
    alpha = mu0 / _TWO_PI
    if abs(omega) < 1e-30:
        K = np.zeros(n, dtype=complex)
        phi = np.zeros(n, dtype=complex)
        offsets = np.zeros(mesh.n_constraints, dtype=complex)
        return FrequencySolution(float(omega), K, phi, drive.copy(), offsets, 0.0)

    lhs11 = np.eye(n, dtype=complex) + beta * alpha * H
    C = mesh.constraint_matrix.astype(complex, copy=False)
    m = C.shape[1]

    if m:
        # Each column of C selects one electrically-defined current group.
        # B @ K is the integrated longitudinal current for that group.
        B = C.T * mesh.lengths_m[None, :]
        lhs = np.block([
            [lhs11, C],
            [B, np.zeros((m, m), dtype=complex)],
        ])
        rhs = np.concatenate((beta * drive, np.zeros(m, dtype=complex)))
        sol = np.linalg.solve(lhs, rhs)
        K = sol[:n]
        u = sol[n:]
        offsets = u / beta
        constraint_residual = B @ K
    else:
        K = np.linalg.solve(lhs11, beta * drive)
        offsets = np.zeros(0, dtype=complex)
        constraint_residual = np.zeros(0, dtype=complex)

    phi = -alpha * (H @ K)
    total_A = drive + phi

    # Relative algebraic residuals for diagnostic logging/tests.
    eq_res = lhs11 @ K + (C @ (beta * offsets) if m else 0.0) - beta * drive
    denom = max(np.linalg.norm(beta * drive), np.linalg.norm(K), 1e-30)
    residual = float(max(np.linalg.norm(eq_res), np.linalg.norm(constraint_residual)) / denom)
    return FrequencySolution(float(omega), K, phi, total_A, offsets, residual)


def solve_spectrum(
    omega: Sequence[float],
    drive_A: np.ndarray,
    mesh: BoundaryMesh,
    conductivity: float,
    thickness_m: float,
    *,
    reference_length_m: float = 1.0,
    mu0: float = MU0,
) -> SpectrumSolution:
    """Solve K_z and induced A_z over a frequency array."""
    om = np.asarray(omega, dtype=float).reshape(-1)
    if len(om) < 1 or not np.all(np.isfinite(om)):
        raise ValueError("omega must contain at least one finite value")
    H = build_log_integral_matrix(mesh, reference_length_m=reference_length_m)
    K = np.zeros((len(om), mesh.n_panels), dtype=complex)
    phi = np.zeros_like(K)
    offsets = np.zeros((len(om), mesh.n_constraints), dtype=complex)
    for i, w in enumerate(om):
        result = solve_frequency(w, drive_A, mesh, H, conductivity, thickness_m, mu0=mu0)
        K[i] = result.current_density
        phi[i] = result.induced_A
        offsets[i] = result.A_offsets
    return SpectrumSolution(om, np.asarray(drive_A, dtype=complex), K, phi, offsets, H)


def build_B_field_kernel(
    mesh: BoundaryMesh,
    observation_points_m: np.ndarray,
    *,
    mu0: float = MU0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matrices mapping panel K_z to induced (B_x, B_y) at points.

    The integrals of (x-x')/r^2 and (y-y')/r^2 over each straight panel are
    evaluated analytically. Observation points must not lie on a source panel.
    """
    obs = np.asarray(observation_points_m, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("observation_points_m must have shape (N, 2)")

    delta = obs[:, None, :] - mesh.midpoints_m[None, :, :]
    u = np.einsum("ijk,jk->ij", delta, mesh.tangents)
    v = np.einsum("ijk,jk->ij", delta, mesh.normals)
    half = 0.5 * mesh.lengths_m[None, :]
    q1 = u - half
    q2 = u + half
    av = np.abs(v)
    den1 = q1 * q1 + v * v
    den2 = q2 * q2 + v * v

    # A field evaluation directly on the conducting panel has a jump/singular
    # principal-value issue; require an interior/exterior observation point.
    on_panel = (av < 1e-13) & (np.abs(u) < half - 1e-13)
    if np.any(on_panel):
        raise ValueError("Observation point lies on a conducting panel; evaluate off the boundary")

    tiny = np.finfo(float).tiny
    int_t = 0.5 * np.log(np.maximum(den2, tiny) / np.maximum(den1, tiny))
    int_n = np.zeros_like(int_t)
    mask = av > 1e-15
    int_n[mask] = np.sign(v[mask]) * (
        np.arctan2(q2[mask], av[mask]) - np.arctan2(q1[mask], av[mask])
    )

    int_dx = int_t * mesh.tangents[None, :, 0] + int_n * mesh.normals[None, :, 0]
    int_dy = int_t * mesh.tangents[None, :, 1] + int_n * mesh.normals[None, :, 1]
    factor = mu0 / _TWO_PI
    Bx_kernel = -factor * int_dy
    By_kernel = factor * int_dx
    return Bx_kernel, By_kernel


def evaluate_eddy_field(
    mesh: BoundaryMesh,
    current_density: np.ndarray,
    observation_points_m: np.ndarray,
    *,
    mu0: float = MU0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate induced Bx and By from one frequency's panel currents."""
    K = np.asarray(current_density, dtype=complex).reshape(-1)
    if len(K) != mesh.n_panels:
        raise ValueError("current_density length does not match mesh")
    Bx_kernel, By_kernel = build_B_field_kernel(mesh, observation_points_m, mu0=mu0)
    return Bx_kernel @ K, By_kernel @ K


def inverse_fourier(
    omega: Sequence[float],
    spectrum: np.ndarray,
    time_s: Sequence[float],
) -> np.ndarray:
    """Numerically invert a two-sided angular-frequency spectrum.

    Uses f(t) = (1/2pi) integral F(w) exp(+i*w*t) dw and trapezoidal
    integration on the supplied (strictly increasing) omega grid. The first
    axis of `spectrum` must correspond to omega; remaining dimensions are kept.
    Returns the real part and raises if the residual imaginary part is large.
    """
    om = np.asarray(omega, dtype=float).reshape(-1)
    tt = np.asarray(time_s, dtype=float).reshape(-1)
    S = np.asarray(spectrum, dtype=complex)
    if S.ndim < 1 or S.shape[0] != len(om):
        raise ValueError("spectrum first dimension must match omega")
    if len(om) < 2 or np.any(np.diff(om) <= 0):
        raise ValueError("omega must be strictly increasing with at least two points")

    d_om = np.diff(om)
    trap_w = np.empty_like(om)
    trap_w[0] = 0.5 * d_om[0]
    trap_w[-1] = 0.5 * d_om[-1]
    if len(om) > 2:
        trap_w[1:-1] = 0.5 * (d_om[:-1] + d_om[1:])

    flat = S.reshape(len(om), -1)
    phase = np.exp(1j * np.outer(tt, om))
    result = (phase @ (trap_w[:, None] * flat)) / _TWO_PI
    result = result.reshape((len(tt),) + S.shape[1:])
    real_scale = max(float(np.max(np.abs(result.real))), 1e-30)
    imag_ratio = float(np.max(np.abs(result.imag)) / real_scale)
    if imag_ratio > 1e-7:
        # Do not silently discard a sizeable imaginary residual: it usually
        # indicates a one-sided/asymmetric frequency grid or inconsistent data.
        raise ValueError(
            f"Inverse Fourier result is not real (max imaginary/real = {imag_ratio:.3e}); "
            "check that the spectrum has conjugate symmetry"
        )
    return result.real


def half_sine_pulse_spectrum(omega: Sequence[float], pulse_length_s: float) -> np.ndarray:
    """Fourier transform of cos(pi*t/T) for |t| <= T/2, zero otherwise.

    This is the even, centre-at-t=0 half-sine pulse used by the legacy code.
    """
    om = np.asarray(omega, dtype=float)
    T = float(pulse_length_s)
    if T <= 0:
        raise ValueError("pulse_length_s must be positive")
    x = T * om
    denom = np.pi**2 - x**2
    out = np.empty_like(x, dtype=float)
    singular = np.isclose(denom, 0.0, rtol=0.0, atol=1e-10)
    out[singular] = T / 2.0
    out[~singular] = (2.0 * np.pi * T * np.cos(x[~singular] / 2.0)) / denom[~singular]
    return out


def rectangular_pulse_spectrum(omega: Sequence[float], pulse_length_s: float) -> np.ndarray:
    """Fourier transform of a unit rectangular pulse on [-T/2, T/2]."""
    om = np.asarray(omega, dtype=float)
    T = float(pulse_length_s)
    if T <= 0:
        raise ValueError("pulse_length_s must be positive")
    return T * np.sinc(om * T / (2.0 * np.pi))


def skin_depth_m(omega_abs: float | np.ndarray, conductivity: float, mu: float = MU0) -> np.ndarray:
    """Classical skin depth sqrt(2/(mu*sigma*|omega|)); infinity at omega=0."""
    w = np.asarray(omega_abs, dtype=float)
    if conductivity <= 0 or mu <= 0:
        raise ValueError("conductivity and mu must be positive")
    out = np.full_like(w, np.inf, dtype=float)
    mask = w > 0
    out[mask] = np.sqrt(2.0 / (mu * conductivity * w[mask]))
    return out


def _self_test() -> None:
    """Small numerical checks; run `python thin_wall_eddy_solver.py`."""
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    circle = np.column_stack((20.0 * np.cos(theta), 20.0 * np.sin(theta)))
    mesh = build_boundary_mesh([BoundaryComponent(circle, closed=True, name="circle")])
    H = build_log_integral_matrix(mesh)

    # Exact self-panel integral for a straight panel of length L:
    # integral[-L/2,L/2] log(|s|/r_ref) ds = L*(log(L/(2*r_ref))-1).
    expected = mesh.lengths_m[0] * (np.log(mesh.lengths_m[0] / 2.0) - 1.0)
    assert np.isclose(H[0, 0], expected, rtol=1e-12, atol=1e-14)

    wires = [{"x": 0.0, "y": 0.0, "I": 1000.0}]
    drive = vector_potential_from_wires(mesh.midpoints_m, wires)
    result = solve_frequency(2.0 * np.pi * 1.0e5, drive, mesh, H, 1.25e6, 5.0e-6)
    net_current = np.sum(result.current_density * mesh.lengths_m)
    assert abs(net_current) < 1e-9 * max(np.sum(np.abs(result.current_density) * mesh.lengths_m), 1.0)
    assert result.residual_norm < 1e-8

    # Negative frequencies must be complex conjugates for a real drive.
    neg = solve_frequency(-2.0 * np.pi * 1.0e5, drive, mesh, H, 1.25e6, 5.0e-6)
    assert np.allclose(neg.current_density, np.conjugate(result.current_density), rtol=1e-9, atol=1e-12)

    # Two open plates form two constraints by default; a shared group forms one.
    two = make_parallel_plate_components(40.0, 8.0, 40)
    mesh_two = build_boundary_mesh(two)
    assert mesh_two.n_constraints == 2
    two_shared = make_parallel_plate_components(40.0, 8.0, 40, shared_current_group="end_return")
    mesh_shared = build_boundary_mesh(two_shared)
    assert mesh_shared.n_constraints == 1

    print("thin_wall_eddy_solver self-test passed")
    print(f"circle panels: {mesh.n_panels}, max relative residual: {result.residual_norm:.3e}")


if __name__ == "__main__":
    _self_test()
