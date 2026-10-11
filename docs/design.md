# Design: algorithms and sources

`integrate` provides what `scipy.integrate` provides for quadrature and
initial value problems, in pure Go, with the same algorithms, defaults and
diagnostics. Where an algorithm is a port, it follows its reference line by
line, so that a run here and a run of the reference on the same problem take
the same steps; [verification.md](verification.md) measures how far that
holds.

## Quadrature

### QUADPACK (`Quad`)

Source: R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber, D. K. Kahaner,
*QUADPACK, A Subroutine Package for Automatic Integration*, Springer 1983;
the double-precision Fortran at netlib.org/quadpack.

| Go | Fortran | used for |
|---|---|---|
| `qagsCore` + `qk21` | `dqagse`, `dqk21` | finite range, no weight |
| `qagsCore` + `qk15i` | `dqagie`, `dqk15i` | `[a, ∞)`, `(-∞, b]`, `(-∞, ∞)` via x = a + (1-t)/t |
| `qagpe` | `dqagpe` | break points (`Points`) |
| `qawoe`, `qc25f`, `fourierMoments` | `dqawoe`, `dqc25f` | `cos(ωx)`, `sin(ωx)` on a finite range |
| `qawfe` | `dqawfe` | Fourier integrals on `[a, ∞)` |
| `qawce`, `qc25c` | `dqawce`, `dqc25c` | Cauchy principal value of f(x)/(x-c) |
| `qawse`, `qc25s`, `qmomo` | `dqawse`, `dqc25s`, `dqmomo` | (x-a)^α (b-x)^β · {1, log(x-a), log(b-x), both} |
| `qelg`, `qpsrt`, `qcheb`, `qk15w`, `gtsl` | `dqelg`, `dqpsrt`, `dqcheb`, `dqk15w`, LINPACK `dgtsl` | epsilon algorithm, error ordering, Chebyshev coefficients, weighted 15-point rule, tridiagonal solve |

The translation keeps the Fortran's 1-based arrays, constants (`d1mach`
values), loop bounds and the order in which subintervals are bisected and
extrapolated; the `goto` structure becomes `for`/`break`/`continue` with the
labels named in comments. `QuadResult` carries QUADPACK's `result`, `abserr`,
`neval` and `last`; `ier` becomes a `*QuadError` wrapping one sentinel per
code (1 `ErrLimit`, 2 `ErrRoundoff`, 3 `ErrBadIntegrand`, 4 `ErrExtrapolation`,
5 `ErrDivergent`, 6 `ErrInvalidInput`, 7 `ErrCycles`). For codes other than 6
the estimate is still returned, as SciPy returns it with an
`IntegrationWarning`.

The dispatch is SciPy's `quad`: limits are swapped (and the result negated)
when b < a, the routine is chosen from the limits and the weight, break points
are de-duplicated and restricted to the open interval, a cos/sin weight on
`(-∞, b]` integrates f(-x) on `[-b, ∞)`. SciPy's `wvar` is split into typed
fields (`Omega`, `Alpha`/`Beta`, `C`). `EpsAbs`/`EpsRel` default to 1.49e-8;
a negative value switches a criterion off (SciPy's `epsabs=0`).

Deliberate departures from the netlib text, all three in code paths where the
Fortran is wrong:

* `dqc25s` accumulates `res24 = res12 + cheb24(i)*rg(i)` in one loop of its
  `integr = 2` branch, a known typo; the intended `res24 + ...` is used.
* `dqawfe` with ω = 0 integrates over `(0, ∞)` instead of `(a, ∞)`; `a` is
  used.
* `dqagpe`'s point sorting and `dqawce`'s limit swap are not reproduced
  because `Quad` hands both routines ordered input; `dgtsl`'s singular-matrix
  exits are dropped (the moment systems are never singular).

The Chebyshev moments of QAWO are kept per bisection level as `dqc25f` keeps
them in `chebmo`/`momcom`, including the `MaxP1` bound and the reuse across
QAWF cycles. SciPy 1.18.1's C translation of QUADPACK mishandles `maxp1`
smaller than the number of levels and differs from the Fortran on two other
fixture integrals; the netlib Fortran, compiled with gfortran, settles each
case and this port agrees with it (see verification.md).

### Multiple integrals

`NQuad` nests `Quad` calls exactly as SciPy's `nquad` does (variable 0
innermost, ranges of a variable may depend on the outer ones, per-level
options); `DblQuad` and `TplQuad` are its two- and three-dimensional forms
with SciPy's argument order (`f(y, x)`, `f(z, y, x)`). `AbsErr` is the largest
one-dimensional error estimate, as in SciPy; `NEval` counts integrand calls.

### Fixed rules

`FixedQuad` is the n-point Gauss–Legendre rule (nodes by Newton's method on
the three-term recurrence from Tricomi's initial guesses, cached per n);
`Romberg` is the removed `scipy.integrate.romberg` (Richardson extrapolation of
the trapezoid rule, `DivMax` levels); `Romb`, `Trapezoid`,
`CumulativeTrapezoid` and `Simpson` are the sample rules, `Simpson` with
Cartwright's correction for an even number of samples as SciPy ≥ 1.11. The
`*ND` variants apply them along an axis of an `ndarray.Array`.

## Initial value problems

### Common machinery

`SolveIVP` is SciPy's `solve_ivp` loop: one accepted step at a time, events
located on the step's dense output with Brent's method (SciPy's `brentq`,
`xtol = rtol = 4·eps`), `TEval` filled from the dense output, terminal events
counted per event. The initial step is Hairer–Nørsett–Wanner's
(*Solving ODEs I*, II.4) as `select_initial_step` computes it; error norms are
RMS norms of the error scaled by `atol + rtol·|y|`; finite-difference
Jacobians are SciPy's `num_jac` (per-column step factors adapted from call to
call, a retry for columns lost in roundoff). Evaluations made for those
Jacobians are reported separately (`NFevJac`), because SciPy does not count
them in `nfev`.

Step-size factors use Python's built-in `max`/`min` semantics, which matter
when an estimate is NaN: `max(0.2, nan)` is 0.2, so a step is rejected and
shrunk where `math.Max` would propagate the NaN. A NaN step size ends the
integration with `ErrStepTooSmall` (SciPy loops forever there, for instance
on y₀ = 0 with `atol = 0`, where every error scale is zero).

### Explicit Runge–Kutta: `RK45`, `RK23`, `DOP853`

* RK45: Dormand & Prince, *J. Comput. Appl. Math.* 6 (1980); free 4th-order
  interpolant of Shampine, *Math. Comp.* 46 (1986).
* RK23: Bogacki & Shampine, *Appl. Math. Lett.* 2 (1989), with its cubic
  Hermite interpolant.
* DOP853: Hairer's `dop853.f` (Hairer, Nørsett & Wanner, *Solving ODEs I*,
  II.10): 12 stages, the 5th/3rd-order blended error estimate, the 7th-order
  dense output with 3 extra stages (evaluated only when a dense output is
  needed, as SciPy does).

The tableaux are SciPy's (`rk.py`, `dop853_coefficients.py`), which are the
published ones; `dop853_coefficients.go` is generated from them. Step-size
control is SciPy's: safety 0.9, factors in [0.2, 10], no increase right after a
rejection.

### Radau IIA (order 5)

Hairer & Wanner, *Solving ODEs II*, IV.8, and `radau5.f`, as restructured in
SciPy's `radau.py`: the collocation system is solved by simplified Newton on
the transformed variables (one real and one complex LU of `μ/h·I - J`), at
most 6 iterations with the convergence-rate test; the error estimate is
radau5's, refined once after a rejection; the step controller is Gustafsson's
predictive one; the Jacobian is recomputed only when Newton converges slowly.
Dense LU with partial pivoting (pivot choice as LAPACK's `getrf`, |re|+|im| for
the complex matrix) replaces LAPACK.

### BDF (orders 1–5)

Shampine & Reichelt, "The MATLAB ODE Suite", *SIAM J. Sci. Comput.* 18
(1997) — the NDF variant (κ = −0.185, −1/9, −0.0823, −0.0415) — with the
quasi-constant step size implementation of SciPy's `bdf.py`: a table of
backward differences rescaled by `change_D` whenever the step changes, order
selection after order+1 equal steps, at most 4 Newton iterations.

### LSODA-style switching (`LSODA`, `Odeint`)

ODEPACK's LSODA (Petzold, *SIAM J. Sci. Stat. Comput.* 4 (1983)) pairs Adams
methods with BDF and switches by comparing the step sizes each would take.
This package does not port it. Its `LSODA` pairs the explicit Dormand–Prince
methods (RK45, or DOP853 when the smallest `rtol` is below 1e-6) with the BDF
above:

* non-stiff → stiff: Hairer's stiffness detector (*Solving ODEs II*, IV.2;
  the test inside dopri5/dop853): after each accepted step
  hλ ≈ |h|·‖f(y₁) − f(ỹ)‖ / ‖y₁ − ỹ‖ (ỹ the argument of the last internal
  stage, also at t + h) estimates |h| times the dominant eigenvalue; when it
  exceeds the stability boundary (3.25 for RK45, 6.1 for DOP853) on 15 steps
  without 6 consecutive steps below, the step is limited by stability, not
  accuracy, and BDF takes over with the current step size;
* stiff → non-stiff: every 5 BDF steps the spectral radius ρ of the Jacobian
  BDF holds is estimated by power iteration; when |h|·ρ < 1 on two
  consecutive checks, the explicit method would be stable at BDF's step size
  and takes over.

This keeps the property that matters to an `odeint` user — a stiff problem is
not integrated with a stability-limited explicit method, a non-stiff one not
with BDF — but its counts differ from ODEPACK's: on the stiff fixtures it
uses 1.2–2.2× the right-hand-side evaluations of ODEPACK (Jacobian columns included on both sides). `Odeint` reports the method of the last
step in `MUsed` (1 explicit, 2 BDF) with ODEPACK's meaning.

## Interfaces

* Functions are Go funcs: `func(x float64) float64` for integrands,
  `func(t float64, y, dydt []float64)` for right-hand sides (the derivative is
  written, not returned, so steps allocate nothing), and
  `func(t float64, y []float64, jac []float64)` for row-major Jacobians.
* Results are plain slices; `Solution.YArray`, `ODESolution.EvalArray` and
  `Odeint` return `*ndarray.Array` in SciPy's layouts (`sol.y` is n × len(t),
  `odeint` is len(t) × n).
* Not provided: complex-valued integrands and states, vectorized right-hand
  sides, sparse Jacobians and `jac_sparsity`, banded Jacobians and the
  `ml`/`mu`/`ixpr`/`mxordn`/`mxords`/`hmin`/`tcrit` options of `odeint`,
  `quad_vec`, `cubature`, `tanhsinh`, `solve_bvp`, and the legacy `ode`
  class.
