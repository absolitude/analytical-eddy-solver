# Analytical Eddy Current Solver v0.2.4+grid

A PyQt6 application for analytical thin-conducting-coating eddy-current
calculations in three 2-D chamber geometries:

- **Circular chamber** — independent angular Fourier modes.
- **Elliptical chamber** — coupled cosine and sine Fourier/Galerkin systems.
- **Flat pipe** — two infinite parallel conducting plates, solved using a
  spatial Fourier transform and a two-sheet 2x2 system.

The application shares wire inputs, pulse definitions, material parameters,
frequency/time grids, and a configurable horizontal-axis observation grid
(`nopxa`) across all three geometries. Geometry-specific CSV/NPZ output and
interactive Plotly HTML reports are displayed in tabs inside a separate results
window.

## Model assumptions

- Chamber and external wires are infinitely long and uniform along `z`.
- Only `A_z(x,y,t)` is represented.
- The coating is non-magnetic and thin compared with the relevant skin depth.
- Temporal phasors use `exp(+i*omega*t)`.
- `Kz = -i*omega*sigma*d*A_total`.
- Finite-longitudinal end effects and three-dimensional current closure are not
  represented.

The elliptical solver includes the cosine `m=0` equation to impose zero net
longitudinal current on the closed coating. The flat-pipe solver uses the
applied vector potential sampled separately on both plates; in a 180-degree
symmetric configuration this reduces to the +/- spatial-wavenumber matrix in
the analytical derivation. A finite spatial FFT window is used, so increase
`xmax` and the spatial sample count to check convergence. Flat-pipe runs also
require zero net external wire current so the applied vector potential has a
finite spatial Fourier transform.

See [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md) for the equations and zero-
wavenumber treatment.

## Install (Windows PowerShell)

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Choose Circular, Ellipse, or Flat pipe, configure the material/pulse and wire
positions/currents, set `nopxa` (the number of horizontal-axis observation
points), then press **Calculate**. Results are written to
`analytical_results/` by default. The application creates geometry-specific
CSV/NPZ data, `run_parameters.json`, and several interactive HTML reports.
After calculation, the reports open in a separate window with tabs:

- **Input / Pre-process** — chamber geometry, external wires, applied field,
  pulse spectrum/time waveform, and the static polynomial-fit comparison.
- **Overview** — the six-panel summary of current and magnetic-field results.
- **Frequency response** — real/imaginary field, vector-potential, and current
  spectra.
- **Fourier modes** for circular/elliptical chambers, or **Plate response** for
  the flat-pipe geometry.
- **By(x,t) animation** — an interactive animation of the induced field.
- **Multipole fit** — polynomial coefficients of the induced field versus time.

The HTML files share a local `plotly.min.js` bundle, so they remain interactive
offline when kept together in the output folder. The polynomial fitting order
can be selected in the GUI.

The default horizontal-axis output grid is 101 points. The ranges are
`[-0.9R, +0.9R]` for a circle, `[-0.9a, +0.9a]` for an ellipse, and
`[-0.9*xmax, +0.9*xmax]` for a flat pipe. The output grid is independent of
the flat-pipe internal spatial FFT grid: its default remains a ±200 mm window
and 4096 FFT samples. Flat-pipe fields and plate currents are evaluated on
`nopxa` output points by direct Fourier-series synthesis (not interpolation),
while the internal FFT resolution remains unchanged. Increase `xmax` and the
internal FFT sample count together until the solution is spatially converged.
Flat-pipe runs also save a compressed `flat_pipe_full_data.npz` file containing
the internal FFT grid and output grid.

All three geometries use the same temporal grid `t = linspace(-T, +T, noptd)`.
For point-by-point comparison with the Green solver, use the same pulse duration
`T`, `noptd`, and horizontal-axis coordinates in both applications. In the
Green GUI, the flat-pipe observation range is `±0.9*(a/2)`; set its half-width
`a/2` equal to the Fourier solver's `xmax` to match the ranges.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Current validation status

The test suite covers the circular mode boundary condition, pulse spectrum,
ellipse zero-net-current constraint and near-circle limit, and flat-pipe
symmetric/antisymmetric modes including the exact `Omega=0` limit. These tests
are numerical/algebraic checks, not a full cross-validation against the
Green's-function solver or FEM/CST. GUI execution should be tested on Windows
after installing PyQt6 and PyQt6-WebEngine.

### Ellipse Green-function cross-check (v0.2.3)

For elliptical cases, the GUI includes **Run Green's-function cross-check** (enabled by default). This runs an independent straight-panel boundary-integral calculation using the same wires, conductivity, coating thickness, frequency grid, and pulse spectrum as the analytical calculation. The BEM panel count is independent of the analytical Fourier boundary samples; the default is 160 panels.

The cross-check compares the cosine/sine Fourier coefficients of the induced vector potential and sheet current, as well as the induced on-axis magnetic field. For a like-for-like comparison of the constant Fourier mode, it subtracts the BEM longitudinal potential offset (`A_offsets`) from the BEM induced potential before projection. This offset is the constant needed by the BEM zero-net-current constraint. The resulting `Green's-function check` tab and the files `ellipse_bem_fourier.csv` and `ellipse_bem_diagnostics.csv` are produced only when the option is enabled. The m=0 time-domain comparison separates the offset-aligned analytical/BEM traces from the raw BEM trace so each panel has a useful vertical scale.

The analytical ellipse's applied-potential Fourier decomposition uses the standard normalization: `1/N` for `m=0` and `2/N` for `m>=1`. Increase the BEM panel count to check convergence of higher Fourier modes; the BEM solution uses piecewise-constant current on straight panels, so high-mode differences are expected to be more sensitive to panel resolution than the low modes.


### m=0 Fourier normalization note (v0.2.4)

The standard Fourier-series convention used by the current analytical solver
and BEM cross-check is `a0 = mean(A)` and `am = 2*mean(A*cos(m*psi))`
for `m >= 1`. The legacy GUI's `compute_ellipse_fourier()` applied `2/N`
to every cosine mode, including `m=0`; its displayed zero-mode coefficient
was therefore twice the standard constant coefficient. Divide legacy m=0
Fourier coefficients by two before comparing them with current standard
coefficients. This is a post-processing normalization difference; it does not
change the Green-function kernel or the sheet-current solution.
