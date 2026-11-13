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

### LSODA (`LSODA`, `Odeint`): ODEPACK's lsoda

A port of ODEPACK's LSODA (Hindmarsh, *ODEPACK, A Systematized Collection of
ODE Solvers*, IMACS Trans. Sci. Comput. 1 (1983); Petzold, *SIAM J. Sci. Stat.
Comput.* 4 (1983) 136–148) as SciPy 1.18.1 ships it: `scipy/integrate/src/lsoda.c`,
the C translation of `lsoda.f`, `stoda.f`, `cfode.f`, `prja.f`, `solsy.f`,
`intdy.f`, `ewset.f`, `vmnorm.f` and `fnorm.f`. `SolveIVP` drives it as
`_ivp/lsoda.py` does (one step per call, `itask = 5`, not past `t_bound`, the
dense output the Nordsieck polynomial of the step) and `Odeint` as
`_odepackmodule.c` does (`itask = 1`: step past each output time and
interpolate with `intdy`). It replaces the Runge–Kutta/BDF switch this package
had before (Hairer's stiffness detector and a spectral radius estimate), which
nothing else used and which is gone, with the hooks `rk.go` and `bdf.go` had for
it.

* Variable-order Adams–Moulton (orders 1–12) and BDF (orders 1–5) on a Nordsieck
  history, the coefficients of `cfode`, a corrector by functional iteration
  (Adams) or the chord method (BDF), `maxcor = 3`.
* The method switch: every 20 steps (`icount`) the step size each method could
  have taken is computed (`dsm`, `cm1`, `cm2`, the Lipschitz estimate `pdest`
  and `pdlast` for Adams, the norm `pdnorm` of the Jacobian for BDF) and BDF
  takes over when it wins by `ratio = 5`, Adams when BDF's step is not larger
  than 5/`ratio` = 1 times Adams's.
* The error weights `1/(rtol|y| + atol)` and the weighted max-norms `vmnorm` and
  `fnorm` of ODEPACK; the Jacobian is refreshed when `h·el₀` moved by more than
  `ccmax = 0.3`, every `msbp = 20` steps, or when the corrector fails with an old
  one; `prja`'s forward difference increments `max(√eps|y_j|, r₀/ewt_j)`.
* The first step `h₀⁻² = 1/(tol·w₀²) + tol·‖f‖²`, the order and step selection of
  `stoda` (rates `rhup`, `rhsm`, `rhdn`, the 10 % rule, `rmax`), and its
  recoveries after 2, 3 and 10 failed error tests and 10 failed corrections.

The port agrees with `lsoda.c` step for step. Run on the harmonic oscillator at
rtol 1e-6, the Go integrator (amd64, so without fused multiply-adds) and
`lsoda.c` compiled with them off take the same 113 first steps to the last bit
and the same 682 evaluations in all. ODEPACK's step selection hinges on the last
bit (its error estimates are differences of nearly equal numbers): `lsoda.c`
itself takes 647 evaluations with fused multiply-adds and 682 without. So the
evaluation counts are SciPy's up to what the compiler's floating-point
contraction decides. On the three cases of the performance table (`SolveIVP`
LSODA on the Van der Pol problem μ = 1000, `Odeint` on Robertson and on
Lotka–Volterra at 101 times) they are equal on arm64 (1803, 1683 and 1243 calls
of the right-hand side, against 2594, 2222 and 1880 before), and on the other
fixtures within 20 %.

Options, with SciPy's names: `Odeint`'s `HMin`, `MxOrdN`, `MxOrdS`, `TCrit`
(`itask` 4: the integration does not step past a critical time), `Band` and
`BandJac` (`ml`, `mu` and the banded Jacobian: `bnorm`, banded `prja` with the
`jt` 4 and 5 paths, an LU of the band as LAPACK's `dgbtf2` and `dgbtrs`, stored
by rows of the band, `BandJac[(Upper+i-j)*n+j]`), and `IVPOptions`' `MinStep`,
`Band` and `BandJac` (SciPy's `min_step`, `lband`, `uband`; LSODA only). The
band is exact for a banded problem: the solution agrees with the full matrix's
to the tolerance, and on a tridiagonal diffusion problem the evaluation counts
are within 15 % of `lsoda.c`'s (621 against 621 with the analytic band on
arm64), the chaos of the step selection as above. Not ported, because the API
reaches none of it: `itask` 2 and 3, `istate` 3, `ixpr` and `mxhnil` (printing),
and the workspace-size bookkeeping. `lsoda.c` reads `rwork[6]` (hmin) into a
local and never stores it, so SciPy's `hmin` and `min_step` have no effect;
here they are ODEPACK's (the smallest step) and the default 0 is SciPy's
behaviour. `bnorm` is kept as `lsoda.c` has it (it reads the band storage
`ml` rows above where P is, so `pdnorm` of a banded problem is not the norm of
P; it only steers the method switch). Two guards are not ODEPACK's: a step that leaves a non-finite state is
a failure (its norms, written with C's `fmax`, do not see a NaN), and
`SolveIVP` stops when `h` has underflowed to 0 (ODEPACK has no floor on `h` and,
called one step at a time, would not stop). Nothing else changes ODEPACK's
behaviour.

`Odeint` returns ODEPACK's `infodict`: `HU`, `TCur`, `NST`, `NFE` (every call of
the right-hand side, those of finite-difference Jacobians included), `NJE` and
`MUsed` (1 Adams, 2 BDF). `Solution.NFev` of an LSODA solution counts every call
too, as SciPy's does, `NFevJac` is 0, and `NJev` = `NLU`.

#### Accuracy contract

LSODA controls the local error of each step, not the global error, so the
contract is stated in tolerance units. Against a reference solution computed in
300-bit arithmetic (`math/big`; `TestLSODAAccuracyContract`), the global error
`E = max_i |y_i − ref_i| / (atol + rtol·|ref_i|)` at the end of the span is at
most

| problem | tolerances | bound on E |
|---|---|---|
| `y′ = −y/2` on [0, 10] | rtol 1e-6, atol 1e-9 and rtol 1e-9, atol 1e-12 | 20 |
| harmonic oscillator, 5 periods | rtol 1e-6, atol 1e-9 | 1e4 |
| harmonic oscillator, 5 periods | rtol 1e-9, atol 1e-12 | 2e4 |
| `y′ = −1000(y − cos t)`, with and without the Jacobian | rtol 1e-6, atol 1e-9 | 20 |

(measured 1.0 and 0.01; 1.7e3 to 4.7e3; 4.8e3 to 5.5e3; 1e-3 to 2e-2, on arm64
and amd64: the bounds leave room for the architectures that take other steps).
The growth on the oscillator is ODEPACK's: SciPy's own `lsoda` reaches 2.1e3 on
the same problem at rtol 1e-9. On the LSODA runs of the SciPy fixtures
(`testdata/ode.json`) E stays within 20 times SciPy's own error (7 times on
amd64's worst, the Van der Pol limit cycle at rtol 1e-9, whose error is a phase
error), and 20 tolerances at least, with `nfev` within 20 % of SciPy's. Between output times
`Odeint` interpolates with the Nordsieck polynomial (`intdy`) of the order used,
to the accuracy of the method (`TestOdeintManyOutputs`).

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
