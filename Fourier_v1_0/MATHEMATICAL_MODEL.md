# Analytical thin-coating eddy-current models

This document describes the three analytical solvers currently implemented
in the application. They share the same temporal phasor convention and thin-
sheet constitutive law, but their geometrical expansions are different.

## 1. Common assumptions and conventions

- The chamber and applied wires are infinitely long and uniform along the
  accelerator longitudinal coordinate `z`.
- The vector potential has only a longitudinal component:
  `A = (0, 0, Az(x,y,t))`.
- The coating is non-magnetic, has conductivity `sigma`, and thickness `d`.
- The coating is sufficiently thin that the current is treated as uniform
  through its thickness: `Kz = d*Jz`.
- Temporal phasors use `exp(+i*omega*t)`. Thus `Ez = -i*omega*Az` when the
  longitudinal scalar-potential gradient is zero, and
  `Kz = -i*omega*sigma*d*A_total`.
- The applied pulse spectrum is computed with the forward convention
  `F(omega) = integral f(t)*exp(-i*omega*t) dt`; the inverse is
  `f(t) = (1/(2*pi))*integral F(omega)*exp(+i*omega*t) d omega`.
- Define the signed sheet parameter
  `alpha = mu0*sigma*d*omega/2 = d/delta^2`, where
  `delta = sqrt(2/(mu0*sigma*abs(omega)))`.

Integrating `nabla_perp^2 Az = -mu0*Jz` through the sheet, with the normal
pointing from the coating's interior to its exterior, gives

`dAz_out/dn - dAz_in/dn = -mu0*Kz = +i*2*alpha*A_total`.

The sign used by all three implementations follows this relation. In the
user's original draft, the sign in the circular boundary equation differs
from the sign implied by the common constitutive equations; this should be
reconciled in the manuscript.

The thin-sheet approximation still requires `d << delta(omega)` throughout
the frequency range that contributes materially to the pulse response. It
does not model skin-depth variation through the coating thickness, finite
longitudinal ends, or general 3-D current closure.

## 2. Circular chamber

For a circular chamber of radius `R`, expand the applied and induced boundary
potentials in angular Fourier modes. For `m >= 1`, the regular interior and
decaying exterior radial solutions are proportional to `(r/R)^m` and
`(R/r)^m`. The derivative jump of one induced mode `C_m` is

`(dA_out/dr - dA_in/dr) = -(2*m/R)*C_m`.

The sheet condition therefore gives

`C_m = -i*alpha*A_m / (m/R + i*alpha)`.

Cosine and sine modes are independent; the circle has no angular mode
coupling. The constant `m=0` component has no transverse magnetic field. The
current-mode expansion omits this constant/gauge component; equivalently, a
longitudinal electric-potential offset is chosen so the closed coating has no
net longitudinal current. The field plots are unaffected by an additive
constant in `Az`.

## 3. Elliptical chamber

### 3.1 Geometry

Use elliptic coordinates

`x = f*cosh(eta)*cos(psi)`,
`y = f*sinh(eta)*sin(psi)`,
`f = sqrt(a^2-b^2)`,
`eta0 = atanh(b/a)`.

The boundary scale factor is

`h(psi) = f*sqrt(cosh(eta0)^2 - cos(psi)^2)`

or, numerically more robustly,

`h(psi) = sqrt(a^2*sin(psi)^2 + b^2*cos(psi)^2)`.

### 3.2 Boundary Fourier expansion

The code expands the *boundary values* as

`A_app(psi) = A0 + sum_{m=1..M} [A_m^c*cos(m*psi) + A_m^s*sin(m*psi)]`,

`A_eddy(psi) = c0 + sum_{m=1..M} [c_m*cos(m*psi) + s_m*sin(m*psi)]`.

The interior regular harmonic functions are

`cosh(m*eta)*cos(m*psi)` for the cosine sector, and
`sinh(m*eta)*sin(m*psi)` for the sine sector.

The exterior solutions decay as `exp(-m*(eta-eta0))` and are normalized to
match the boundary coefficients. With the boundary coefficients `c_m` and
`s_m` as the unknowns, the magnitude of the derivative jump is represented
by diagonal terms

`D_m^c = 2*m/(1 + exp(-2*m*eta0))`,
`D_m^s = 2*m/(1 - exp(-2*m*eta0))`, for `m >= 1`,

and `D_0^c = 0`.

### 3.3 Galerkin coupling matrices

For a periodic average

`<g> = (1/(2*pi))*integral_0^(2*pi) g(psi) dpsi`,

define

`H^c_nm = 4*<h*cos(n*psi)*cos(m*psi)>`, `n,m = 0..M`,

`H^s_nm = 4*<h*sin(n*psi)*sin(m*psi)>`, `n,m = 1..M`.

The factor of four is convenient for the cosine/sine coefficient
normalization. The `n=0` cosine equation is multiplied by an irrelevant
constant factor; it remains the zero-net-current condition.

For each temporal frequency, the finite-dimensional systems are

`(D^c + i*alpha*H^c)c = -i*alpha*H^c*A_app^c`,

`(D^s + i*alpha*H^s)s = -i*alpha*H^s*A_app^s`.

The cosine `m=0` row enforces

`integral Kz ds = -i*omega*sigma*d*integral h(psi)*A_total(psi) dpsi = 0`.

The sine sector has zero weighted mean automatically because `h(psi)` is
even and the sine basis is odd. The scale factor is computed by periodic
quadrature and the matrices are formed directly from the basis products.

### 3.4 Interior magnetic field

Let `zeta = x + i*y`. The normalized harmonic functions can be written with
Chebyshev polynomials:

`q_m(zeta) = T_m(zeta/f)/cosh(m*eta0)`,
`r_m(zeta) = T_m(zeta/f)/sinh(m*eta0)`.

Then

`A_eddy,in = c0 + sum [c_m*Re(q_m) + s_m*Im(r_m)]`.

The implementation evaluates the derivatives of these normalized functions
with scaled recurrences to avoid overflow near the circular limit. It then
uses `Bx = dAz/dy` and `By = -dAz/dx` on the horizontal observation axis.

The current implementation supports both cosine and sine sectors. The
ellipse must have `a > b`; an exactly circular shape should use the Circular
solver instead.

## 4. Infinite flat pipe (two infinite parallel plates)

### 4.1 Geometry and spatial Fourier transform

The two plates are at `y=+b` and `y=-b`; the transverse coordinate `x` is
unbounded. The spatial transform convention is

`A_tilde(Omega) = integral A(x)*exp(-i*Omega*x) dx`,

with inverse `A(x) = (1/(2*pi))*integral A_tilde(Omega)*exp(+i*Omega*x) dOmega`.

Let `k = abs(Omega)` and `q = exp(-2*b*k)`. The implementation samples the
applied vector potential independently on the upper and lower plates and
uses a finite, centered spatial FFT window `[-xmax, xmax)`. This is a
numerical approximation to the infinite transform; increase `xmax` and the
spatial sample count to establish convergence.

A finite spatial transform of the wire vector potential requires zero net
external wire current. The application checks this condition. This is also
consistent with a physical external-current return path.

### 4.2 Two-sheet 2x2 system

Let `X_t(Omega)` and `X_b(Omega)` denote the self-induced sheet-potential
amplitudes on the upper and lower plates, and let `A_t`, `A_b` be the applied
potentials on those plates. The sheet boundary conditions give

`(k+i*alpha)*X_t + i*alpha*q*X_b = -i*alpha*A_t`,

`i*alpha*q*X_t + (k+i*alpha)*X_b = -i*alpha*A_b`.

This is a generalized two-sheet formulation. If the applied field has the
180-degree rotational symmetry assumed in the draft, then the lower-plate
spectrum is the reflected upper-plate spectrum and the system reduces to the
positive/negative spatial-wavenumber 2x2 system derived there.

Define symmetric and antisymmetric quantities

`A_s=(A_t+A_b)/2`, `A_a=(A_t-A_b)/2`,
`X_s=(X_t+X_b)/2`, `X_a=(X_t-X_b)/2`.

The equations decouple:

`X_s = -i*alpha*A_s / (k+i*alpha*(1+q))`,

`X_a = -i*alpha*A_a / (k+i*alpha*(1-q))`.

The induced potentials on the plates are

`Aeddy_t = (1+q)*X_s + (1-q)*X_a`,
`Aeddy_b = (1+q)*X_s - (1-q)*X_a`.

At the midplane,

`Aeddy_mid = 2*exp(-b*k)*X_s`,
`Bx_eddy_mid = 2*k*exp(-b*k)*X_a`,
`By_eddy_mid = -i*Omega*Aeddy_mid`.

### 4.3 The zero-spatial-wavenumber limit

The antisymmetric amplitude `X_a` may behave as `1/k` as `k -> 0`, but
observable combinations remain finite. For `alpha != 0`,

`X_s -> -A_s/2`,
`k*X_a -> -i*alpha*A_a/(1+2*i*alpha*b)`.

Thus the exact `Omega=0` limits used by the code are

`Aeddy_t -> -A_s - i*2*alpha*b*A_a/(1+2*i*alpha*b)`,

`Aeddy_b -> -A_s + i*2*alpha*b*A_a/(1+2*i*alpha*b)`,

`Aeddy_mid -> -A_s`,
`Bx_eddy_mid -> -i*2*alpha*A_a/(1+2*i*alpha*b)`,
`By_eddy_mid -> 0`.

At temporal `omega=0`, `alpha=0` and the induced response is set to zero.
The solver computes these finite physical combinations directly and never
stores the divergent antisymmetric sheet amplitude at `k=0`.

## 5. Numerical frequency/time processing and output

All solvers use a symmetric, uniformly spaced temporal-frequency grid that
contains zero. The inverse temporal Fourier transform uses trapezoidal
integration including half weights at the two frequency endpoints. Real-time
signals are checked for temporal conjugate symmetry.

The GUI produces CSV files and a common six-panel Plotly HTML report for each
geometry. The HTML report includes boundary current, horizontal/midplane field
profiles, selected time traces, frequency-domain response, and space-time
heatmaps. Flat-pipe runs also write a compressed NPZ archive containing the
complete two-plate frequency-domain and time-domain arrays; selected-frequency
and x-by-time CSV files remain easy to inspect in spreadsheet software.

## 6. Scope and validation status

These are 2-D, longitudinally uniform models. They do not predict finite-
length end effects, current turning/return paths at chamber ends, or the full
eddy-current-loop suppression of longitudinally segmented coatings. The flat
pipe spatial FFT also has finite-window and finite-resolution errors that
must be checked by increasing the numerical window and sample count.

The automated tests check algebraic boundary conditions, the ellipse
zero-current constraint, near-circle consistency, frequency conjugate
symmetry, and the zero-spatial-wavenumber flat-pipe limit. They are not a
replacement for a full comparison against the user's independent analytical
derivations, Green's-function solver, or FEM/CST results.
