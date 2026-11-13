# Coverage against SciPy and MATLAB

Inventory of what `go-numerics/integrate` (main) implements against
`scipy.integrate` v1.18.1 and the MATLAB integration, ODE, BVP and PDE
functions. Status: **done** (same semantics and options that matter),
**partial** (what is missing is named), **missing**. "Dep" is what a missing
item needs beyond this package.

Sources. SciPy: `scipy/integrate/__init__.py` at tag v1.18.1 (every name of its
autosummary lists appears below, in its order). MATLAB: the MathWorks function
lists could not be read by the tool used (the pages are rendered client-side),
so the MATLAB column is built from the documented function families and solver
names and must be re-checked against the MathWorks reference lists before a
slice is started. Go side: README ("Not provided"), `docs/design.md`,
`go doc -all` and the sources on main.

Row numbers are identifiers, not a count (MATLAB's `quad`/`quadl`/`quadgk` were
folded into rows 1 and M10). SciPy's `args=` and `full_output` conveniences are not counted as gaps where a
Go closure replaces them; the one place `full_output` carries data that is not
returned is named in row 1.

## Totals

Counted by `go run` over the tables below (rows, not names).

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 (sections 1-4) | 22 | 5 | 11 | 38 |
| MATLAB-only (section 5) | 0 | 0 | 10 | 10 |

## 1. Quadrature of functions

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `quad` finite range (QAGS), `points=` (QAGP) | `integral`, `quadgk`, `Waypoints` | `Quad`, `QuadOptions.Points` | partial | `full_output` info arrays (`alist`, `blist`, `rlist`, `elist`, `iord`) not returned; `last` and `neval` are |
| 2 | `quad` infinite ranges (QAGI) | `integral` (Inf limits) | `Quad` | done | |
| 3 | `quad` weights `cos`, `sin` (QAWO, QAWF) | | `WeightCos`, `WeightSin` | done | |
| 4 | `quad` weights `alg`, `alg-loga`, `alg-logb`, `alg-log` (QAWS) | | `WeightAlg...` | done | |
| 5 | `quad` weight `cauchy` (QAWC) | | `WeightCauchy` | done | |
| 6 | `IntegrationWarning` | `warning` identifiers | `*QuadError` and sentinel errors | done | |
| 7 | `dblquad` | `integral2`, `dblquad`, `quad2d` | `DblQuad` | done | |
| 8 | `tplquad` | `integral3`, `triplequad` | `TplQuad` | done | |
| 9 | `nquad` | | `NQuad` | done | |
| 10 | `quad_vec` | `integral` `ArrayValued`, `quadv` | | missing | none (adaptive Gauss-Kronrod / trapezoid on vector integrands; SciPy's `norm`, `cache_size`, `workers`) |
| 11 | `cubature` | `integral2`, `integral3` (adaptive, not the same rule) | | missing | none (Genz-Malik and Gauss-Kronrod product rules, array-valued, N-D, region transforms) |
| 12 | `tanhsinh` | | | missing | none (needs `special` log helpers; vectorized elementwise, error control by successive levels) |
| 13 | `nsum` | `symsum` (Symbolic) | | missing | `tanhsinh` (row 12); Euler-Maclaurin |
| 14 | `fixed_quad` | `quadgk` fixed nodes: none | `FixedQuad`, `GaussLegendre` | done | |
| 15 | `newton_cotes` | | | missing | none (rational weights, closed form) |
| 16 | `lebedev_rule` | | | missing | tables of 32 rules (embed) |
| 17 | `qmc_quad` | | | missing | `go-statistics/stats` QMC engines (Sobol, Halton) — not yet there, see its coverage.md |
| 18 | `romberg` (removed in SciPy 1.15) | | `Romberg` | done | provided although SciPy dropped it |

## 2. Quadrature of samples

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 20 | `trapezoid` | `trapz` | `Trapezoid`, `TrapezoidND` | done | |
| 21 | `cumulative_trapezoid` | `cumtrapz` | `CumulativeTrapezoid`, `CumulativeTrapezoidND` | done | |
| 22 | `simpson` | | `Simpson`, `SimpsonND` | done | |
| 23 | `cumulative_simpson` | | | missing | none |
| 24 | `romb` | | `Romb` | done | |

## 3. Initial value problems

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 25 | `solve_ivp` | `ode45`, `ode23`, `ode113`, `ode15s`, `ode23s`, `ode23t`, `ode23tb` | `SolveIVP`, `IVPOptions` | partial | `vectorized`, `jac_sparsity`, banded Jacobians, complex states; dep for sparse: go-sparse |
| 26 | `RK45` | `ode45` | `RK45` | done | |
| 27 | `RK23` | `ode23` | `RK23` | done | |
| 28 | `DOP853` | | `DOP853` | done | |
| 29 | `Radau` | `ode23s`-like (Rosenbrock, different) | `Radau` | done | |
| 30 | `BDF` | `ode15s` (NDF) | `BDF` | done | |
| 31 | `LSODA` | `ode113` (Adams) in spirit | `LSODA` | partial | ODEPACK's Adams/BDF lsoda as in SciPy 1.18.1's `lsoda.c` (same evaluation counts on the benchmark cases); `min_step` (no effect in SciPy either), `lband`, `uband` missing |
| 32 | `OdeSolver`, `DenseOutput`, `OdeSolution` (low-level step API) | `deval` | `ODESolution.Eval` | partial | the stepper classes (`step()`, `t_old`, `step_size`, `status`) are internal; only the dense solution is public |
| 33 | `odeint` | | `Odeint` | partial | `ml`/`mu` banded Jacobians, `tcrit`, `hmin`, `ixpr`, `mxordn`, `mxords`, `tfirst`-less form |
| 34 | `ode` (VODE, ZVODE, dopri5, dop853, lsoda), `complex_ode` | | | missing | VODE and ZVODE ports (netlib); complex support |
| 35 | `ODEintWarning` | | `OdeintInfo` | done | |
| 36 | events (`solve_ivp(events=)`) | `odeset('Events')` | `Event`, `IVPOptions.Events` | done | |
| 37 | `t_eval`, `dense_output` | `tspan` vector, `deval` | `TEval`, `DenseOutput` | done | |
| 38 | MATLAB-specific `odeset` options: `Mass`, `MStateDependence`, `NonNegative`, `OutputFcn`, `Refine`, `Stats`, `InitialSlope` | `odeset`, `odeget` | | missing | none for `OutputFcn`, `Refine`, `NonNegative`; mass-matrix problems `M(t,y) y' = f` need a new core path |

## 4. Boundary value problems

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 39 | `solve_bvp` | `bvp4c`, `bvp5c`, `bvpinit`, `bvpget`, `bvpset`, `deval` | | missing | none: collocation (Lobatto IIIA, residual control), banded Newton; `go-ndarray` linalg for the sparse-ish Jacobian |

## 5. MATLAB-only families

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| M1 | | `ode15i` (fully implicit `F(t,y,y')=0`), `decic` | | missing | BDF variable-order for DAE form |
| M2 | | `ode23tb`, `ode23t` (trapezoid-rule family) | | missing | stiff solvers, same Newton machinery as `BDF` |
| M3 | | `dde23`, `ddesd`, `ddensd`, `ddeget`, `ddeset` (delay differential equations) | | missing | history interpolation on the dense output |
| M4 | | `pdepe`, `pdeval` (1-D parabolic-elliptic PDEs) | | missing | method of lines over `BDF`; spatial discretisation of Skeel-Berzins |
| M5 | | `odextend` | | missing | continue a stored solution |
| M6 | | `integralN` (N-D, 2023b) | | missing | row 11 |
| M7 | | `integral2` / `integral3` with `Method` (`tiled`, `iterated`), infinite and triangular regions (`DblQuad` takes finite or function limits only through `Quad`) | | missing | transformations for infinite and triangular regions |
| M8 | | `integral` complex waypoints (contour integrals) | | missing | complex integrands |
| M9 | | `ode` object (`ode`, `ode.solve`), `odeEvent` (R2021b) | | missing | object API; in Go, `SolveIVP` options |
| M10 | | `polyint` (polynomial integral), `quadl`, `quad` (legacy adaptive Lobatto / Simpson) | | missing | polynomial type; the legacy quadratures do the job of `Quad` and are not worth porting |

## Proposed slices

1. **Sample rules and weights, no dependency.** Rows 15, 23. One short PR.
2. **Vector and N-D quadrature.** Rows 10, 11 (`quad_vec`, `cubature`), M6,
   M7. Dep: none; `cubature` also gives the cleanest path to `integral2/3`
   infinite regions.
3. **`tanhsinh` and `nsum`.** Rows 12, 13. Dep: `go-numerics/special` log
   helpers (`Log1pExp`, `LogSumExp`, already there).
4. **`solve_bvp`.** Row 39, MATLAB `bvp4c/5c`. Dep: dense LU (have it in
   `linalg.go`); sparse form would use go-sparse.
5. **Legacy ODE classes.** Row 34 (`ode`/VODE/ZVODE, `complex_ode`), 31 and 33
   (the ODEPACK LSODA port is done: `Odeint` evaluation counts match SciPy's).
   Dep: complex arithmetic (stdlib `complex128`); netlib
   Fortran as the oracle, as for QUADPACK.
6. **Solver API.** Expose the stepper, add `OutputFcn`,
   `NonNegative`, `Refine`, mass matrices (rows 32, 38, M5, M9). Dep: none.
7. **Sparse and banded Jacobians.** Rows 25, 33 (`jac_sparsity`, `ml/mu`).
   Dep: go-sparse (or a banded LU of our own).
8. **MATLAB-only solvers.** M1 (`ode15i`), M2 (`ode23t/tb`), M3 (DDEs), M4
   (`pdepe`). Dep: slice 6; M4 needs a spatial discretisation layer.
9. **QMC and Lebedev.** Rows 16, 17. Dep: Sobol/Halton engines, which belong
   in `go-statistics/stats` (not there yet) or go-ndarray `random`.
