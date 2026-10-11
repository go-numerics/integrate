# Verification

Everything here is reproduced by `go test ./...`; the tables are regenerated
with `INTEGRATE_REPORT=1 go test -run VerificationReport .`.

## Oracles

* **SciPy 1.18.1** (NumPy 2.5.4, mpmath 1.4.1, Python 3.12.12), run by
  `testdata/gen/fixtures.py` under `uv run --with scipy --with numpy --with
  mpmath` (nothing installed system-wide). It records, for every integral and
  every ODE run, SciPy's value, error estimate, evaluation counts, diagnostics
  and (for ODEs) the first step times, into `testdata/quad.json` and
  `testdata/ode.json`.
* **References**: closed forms, mpmath at 40 digits (`quad`, `quadosc`), and
  for ODEs without a closed form SciPy's DOP853 at rtol 1e-13 (non-stiff) or
  Radau at rtol 1e-12 (stiff).
* **The netlib Fortran QUADPACK**, compiled with gfortran
  (`testdata/gen/netlib_harness.f`), to settle the cases where SciPy and this
  port disagree.

## Quadrature: 124 integrals against SciPy's QUADPACK

[quad-table.md](quad-table.md) lists them all: smooth integrands,
end-point singularities (x^-0.9, log x/√x), oscillatory and infinite ranges,
break points, the four algebraic-logarithmic weights, Cauchy principal
values, Fourier integrals, divergent integrals, and configurations found by a
branch search over integrand families that drive every diagnostic of every
driver (roundoff, interval too small, extrapolation failure, divergence,
limits, QAWF cycle failures).

* **Same subdivision path**: the number of integrand evaluations equals
  SciPy's on 116 of 124 integrals on arm64 (115 on amd64), and the values
  agree to 1e-12 relative wherever the path is the same.
* **Accuracy**: on every integral with a reference and no diagnostic, the true
  error is below the returned error estimate.
* **The 8 differences**:
  * `qawo_div` (∫ cos(30x)/x, divergent) and `br_qawo_small`
    (∫ cos(x)/|x−0.3|): SciPy 1.18.1 stops early (1355 and 2505 evaluations);
    the netlib Fortran takes 6035 and 2685 evaluations with values
    140.956644439315767 and 67.6282900830324536, which this port reproduces
    exactly (`TestNetlibOracle`).
  * `qawo_maxp1` (∫₀¹ cos(200x)/√x with `maxp1=2`): same counts, but SciPy
    returns 0.0806726 with ier = 0, while the Fortran and this port return
    0.0842501986376902, which mpmath confirms to 3e-16. SciPy's C translation
    mishandles a moment table smaller than the number of bisection levels.
  * `qawc_tiny`, `br_x_m12_rel`, `br_x_m12_abs`, `br_pts_x_m12_50`,
    `br_qawo_div`, `br_qawf_ier4`: tolerances at the roundoff limit or
    divergent integrals, where the path is decided by the last bit of f and of
    each rule. Perturbing |x|^-1.2 by one ulp moves SciPy itself from 777 to
    945 or 735 evaluations; the Fortran, SciPy and this port each take their
    own path (the Fortran gives 3105 evaluations on `qawc_tiny`, SciPy 3075,
    this port 3375). All report the same diagnostic and values within the
    error estimates. Go's `math.Pow` and the fused multiply-adds Go emits on
    arm64 are the bit-level differences.
* Multiple integrals (`dblquad`, `tplquad`, `nquad` on rectangles, a triangle
  and a simplex) agree with SciPy to 1e-13; fixed rules (`trapezoid`,
  `simpson` with odd/even and uneven samples, `cumulative_trapezoid`, `romb`,
  `fixed_quad`, `roots_legendre`) to 1e-14.

## ODEs: 153 runs against `solve_ivp`

[ode-table.md](ode-table.md) lists every run: exponential decay, harmonic
oscillator, Lotka–Volterra, Van der Pol μ = 1, Brusselator, the Arenstorf
orbit (all six methods, rtol 1e-3, 1e-6, 1e-9), and the stiff Van der Pol
μ = 1000, Robertson, HIRES and OREGO (Radau, BDF, LSODA; with and without an
analytic Jacobian where one is given). Errors are scaled by
`atol + rtol·|y_ref|`, so 1 means "at the requested tolerance" (global errors
exceed local tolerances on long or chaotic runs, for SciPy as for us).

* **Same step sequence**: for the ported methods (RK45, RK23, DOP853, Radau,
  BDF), nfev, the number of steps, njev and nlu are all identical to SciPy's
  on 110 of 120 runs on arm64 (109 on amd64). The step times agree to 1e-12 to
  1e-8 relative (the error estimates are differences of nearly equal
  numbers, so the summation order of the stage combinations shows up in the
  last digits of h). The 10 other runs (Radau and BDF on harmonic 1e-9,
  Arenstorf 1e-6, VdP μ = 1000, OREGO) differ by 0.04–1.1 % in nfev, where one
  borderline step decision tipped the other way; their accuracy is the same
  as SciPy's (0.0104 vs 0.0112, 47.5 vs 56.9, ... see the table).
* **Accuracy**: every ported run is as accurate as SciPy's at the same
  tolerances (identical runs give identical end states).
* **Events**: a falling ball (terminal impact, direction −1, plus an apex
  event) with all six methods: event times agree with SciPy to 1e-12 and
  with the exact 2.025769006530... s, and nfev equals SciPy's for the five
  ported methods; a counting terminal event (terminal = 3) stops at the third
  zero of cos t, as SciPy.
* **t_eval, dense output, backwards integration** agree with SciPy to 1e-9
  (RK45 defaults) and 1e-12 (DOP853, rtol 1e-8).
* **LSODA / odeint** are a different algorithm from ODEPACK's (see
  design.md): accuracy is held to that of ODEPACK and of the method actually
  running; on the stiff problems this method uses 1.2–2.2× ODEPACK's
  right-hand-side evaluations. `odeint` on Robertson (rtol 1e-8, atol 1e-12)
  stays within 10× ODEPACK's error at every output time, both end on the
  stiff method.

## Negative control

Each perturbation below was applied to the source, the suite run, and the
change reverted; each one fails the suite:

| perturbation | failing tests |
|---|---|
| first Kronrod weight of the 21-point rule × (1 + 1e-6) | TestQuadAgainstSciPy (51 failures), TestMultiAgainstSciPy, TestNQuadErrors |
| RK45 Butcher entry a₃₂ = 9/40 + 1e-6 | TestTEvalAndDenseAgainstSciPy, TestEventsAgainstSciPy, TestODEAgainstSciPy (e.g. Lotka–Volterra rtol 1e-9: scaled error 3.3e3 vs SciPy 2.9) |
| Radau transformation matrix T₂₁ + 1e-6 | TestEventsAgainstSciPy, TestODEAgainstSciPy |
| BDF κ₁ = −0.186 instead of −0.185 | TestEventsAgainstSciPy, TestODEAgainstSciPy |
| DOP853 first 5th-order error weight × 1.001 | TestTEvalAndDenseAgainstSciPy, TestEventsAgainstSciPy, TestOdeintAgainstSciPy, TestOdeintErrors, TestODEAgainstSciPy |

## Coverage

100 % of statements, with `-race`, on arm64 and amd64 (the CI gate runs on
Linux, macOS and Windows). The quadrature branches that only adversarial
integrands reach are driven by the branch-search fixtures above, checked
against SciPy like the others; the few branches unreachable from the public API
(`qelg` with fewer than three entries, the tails of `dqagse`/`dqawfe`) have
direct unit tests.
