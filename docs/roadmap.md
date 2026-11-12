# Roadmap: what other reference libraries do better

This page lists what the quadrature and ODE libraries of Julia, R and GSL do
better than this package. [coverage.md](coverage.md) already lists the gaps
against SciPy and MATLAB, so they are not repeated here.

The goal is to bring these improvements into go-numerics. Every binding then
gets them: Ruby (rbgo), Julia, R and C. Treat the reference libraries as sources
of algorithms, defaults, test problems and edge cases, not only as APIs to
imitate.

## How the inventory was taken

The inventory was taken on 2026-11-11, from source. Each reference was opened
at the path cited:

- **Julia (MIT):** QuadGK, HCubature, FastGaussQuadrature,
  OrdinaryDiffEqTsit5/Verner/Rosenbrock/BDF/SDIRK/FIRK/Core/Default.
- **deSolve 1.40 (GPL):** the R package, including its bundled ODEPACK, VODE
  and radau5 Fortran.
- **R 4.6.2 (GPL):** the `stats` package.
- **GSL 2.8 (GPL-3+):** `integration` and `ode-initval2`.

Absence was checked with `git grep -i` against every Go file and against
design.md and coverage.md. Nothing was implemented.

## Licence rule

R, deSolve and GSL were read only to understand the algorithms. An
implementation must come from one of these:

- the papers;
- the public-domain netlib originals (ODEPACK, VODE, QUADPACK);
- Hairer's codes;
- MIT Julia packages.

It must never come from R, deSolve or GSL code.

## Oracle rule

This follows the fleet rule of 2026-11-11: bit identity is a porting step,
never the goal.

**Acceptance.** Each item is accepted against a stated error bound, asserted
against an extended-precision oracle (double-double, `math/big` or mpmath):

- **Quadrature:** the integral is computed by mpmath `quad` at 50+ digits.
  Two things are asserted. The true error is within the requested tolerance.
  The reported error estimate bounds the true error.
- **ODEs:** a reference solution is computed by mpmath `odefun`, or by a
  high-order integration at extended precision. Published reference solutions
  are also used where they exist, such as the Bari test set. The global error
  at the output times is asserted against the requested tolerance, with a
  stated factor.

**Speed and cost.** Each item is also measured on speed and on its cost in
right-hand-side and integrand calls, against the reference implementation.

**Porting step.** Reproducing the reference's steps and evaluation counts
(OrdinaryDiffEq, ODEPACK built with gfortran, QuadGK) is how a port is
checked. It is not the acceptance test.

**Defaults.** A method or controller that is faster and stays within the
bound may become the default, as D1 proposes. A gain is never kept opt-in
only to preserve parity with SciPy. The SciPy-exact paths stay available as an
option.

The "Oracle" column below names the reference for the porting step. The
bound above applies to every row.

## Ranked by gain / cost

Sizes: S < 300 lines, M 300–1500, L > 1500.

### New algorithms and accuracy/speed

| # | Item | Reference (licence), evidence | What is missing or worse here | Gain for the fleet | Implementation source | Oracle | Size |
|---|---|---|---|---|---|---|---|
| 1 | Tsit5 and Verner 6/7/8/9 with their (lazy) interpolants | OrdinaryDiffEqTsit5 (MIT) `src/tsit_tableaus.jl:45-103`, `interpolants.jl:3`; OrdinaryDiffEqVerner `src/verner_tableaus.jl:798,1068,2986,3497` | Explicit RK stops at RK45/RK23/DOP853 (design.md "Explicit Runge–Kutta") | Julia's default non-stiff methods; fewer RHS calls than RK45 or DOP853 at equal error | Tsitouras, Comput. Math. Appl. 62 (2011); Verner, Numer. Algorithms 53 (2010); OrdinaryDiffEq (MIT), tableaux generated as `dop853_coefficients.go` is | Same naccept, nreject and nf as OrdinaryDiffEq, with its controller set explicitly, on the non-stiff test problems | M |
| 2 | Real ODEPACK LSODA (Adams–Moulton ↔ BDF, Nordsieck history, `DSTODA` switching) | deSolve `src/opkdmain.f:3887` (DLSODA), `opkda1.f:3925` (DSTODA), read only | design.md: "This package does not port it". It uses 1.2–2.2× ODEPACK's RHS evaluations (issue #14). coverage.md lists missing `ml/mu`, `tcrit`, `hmin`, `mxordn`, `mxords` | `odeint` and R `deSolve::lsoda` parity, both counts and options. Closes #14 | ODEPACK, netlib (public domain): `dlsoda`, `dstoda`, `dprja`, `dintdy`, `dsolsy`; Petzold, SIAM J. Sci. Stat. Comput. 4 (1983) | ODEPACK compiled with gfortran: same nst, nfe, nje and mused. This also makes `Odeint` equal SciPy's counts | L |
| 3 | Rosenbrock23 and Rodas5P (linearly implicit) | OrdinaryDiffEqRosenbrock (MIT) `src/rosenbrock_tableaus.jl:6,47`, `algorithms.jl:84-86` | No Rosenbrock method. coverage.md row 29 notes MATLAB `ode23s` as "different" | Best methods for small stiff systems at loose tolerances; MATLAB `ode23s` and Julia default stiff choices | Shampine & Reichelt, SIAM J. Sci. Comput. 18 (1997); Steinebach, BIT 63 (2023); OrdinaryDiffEq (MIT) | Same nsteps, nf and njac as OrdinaryDiffEq on ROBER, HIRES and OREGO (already fixtures here) | M |
| 4 | Adaptive Gauss–Kronrod of any order, QuadGK-style. It has a segment heap, vector/complex/in-place integrands with a user norm, batched calls, a smoother infinite-interval map, and Kronrod/Gauss nodes for any n | QuadGK (MIT) `src/adapt.jl:104-133,163-183`, `api.jl:81,129`, `batch.jl:22,129`, `gausskronrod.jl:225,337,485` | Only fixed `qk21`/`qk15` (`rules.go`). README "Not provided": `quad_vec`, complex integrands | `quad_vec` (coverage row 10), complex integrands, QuadGK parity for Julia | Laurie, Math. Comp. 66 (1997) for the Kronrod nodes; QuadGK (MIT) | Same value and `quadgk_count` evaluation count as QuadGK | M |
| 5 | h-adaptive cubature with the Genz–Malik embedded rule | HCubature.jl (MIT) `src/genz-malik.jl:59-133`, `HCubature.jl:99-113,252` | `cubature` missing (coverage rows 11 and M6). Only nested `NQuad` | Multi-dimensional integrals without nested 1-D cost; Julia `hcubature` parity | Genz & Malik, J. Comput. Appl. Math. 6 (1980); HCubature.jl (MIT) | Same value and evaluation count as `hcubature_count` | M |
| 6 | Gauss rules computed in O(n): Gauss–Legendre by asymptotics for large n, plus Gauss–Jacobi, Laguerre, Hermite, Lobatto and Radau | FastGaussQuadrature (MIT) `src/gausslegendre.jl:61-66,71`, `gaussjacobi.jl:23`, `gausslaguerre.jl:59`, `gausshermite.jl:28`, `gausslobatto.jl:31`, `gaussradau.jl:31`; GSL `integration/fixed.c` types (read only) | `fixed.go` uses Newton on the recurrence, which is O(n²), and has the Legendre weight only | Large-n fixed rules; weighted rules for statistics (Hermite for normal expectations) | Hale & Townsend, SIAM J. Sci. Comput. 35 (2013); Bogaert, SIAM J. Sci. Comput. 36 (2014); FastGaussQuadrature (MIT) | Nodes and weights within a few ulp of FastGaussQuadrature, and of mpmath at 100 digits | M |
| 7 | CQUAD, doubly-adaptive Clenshaw–Curtis. It drops NaN/Inf nodes by downdating the coefficients instead of failing, and has a divergence test | GSL `integration/cquad.c:158-176,233-248,481-483` (read only) | QUADPACK stops with `ErrBadIntegrand` on a NaN | Robust default for integrands with singular points or NaN | The paper: Gonnet, "Increasing the reliability of adaptive quadrature using explicit interpolants", ACM TOMS 37 (2010). GSL's `cquad.c` is Gonnet's own code, under GPL-3 (`cquad.c:3-7`), so it is read only and is not a source. No standalone code is cited, because none was opened and its licence was not checked | True error within the requested tolerance against mpmath `quad`, NaN/Inf nodes handled, on the paper's test battery; calls compared with QUADPACK `Quad` | M |
| 8 | FBDF/QNDF (variable-coefficient BDF) | OrdinaryDiffEqBDF (MIT) `src/algorithms.jl:375,518`, `bdf_caches.jl:613` | Only the quasi-constant-step NDF of SciPy's `bdf.py` | More robust when the step changes often | Shampine & Reichelt (1997) for QNDF; OrdinaryDiffEq (MIT) | OrdinaryDiffEq FBDF nf and nsteps | M |
| 9 | TRBDF2 and KenCarp4 (SDIRK, IMEX splitting) | OrdinaryDiffEqSDIRK (MIT) `src/sdirk_tableaus.jl:1,1154`, `algorithms.jl:207,1019` | MATLAB `ode23tb` missing (coverage M2); no IMEX | Stiff/non-stiff split problems (reaction–diffusion) | Hosea & Shampine, Appl. Numer. Math. 20 (1996); Kennedy & Carpenter, Appl. Numer. Math. 44 (2003) | OrdinaryDiffEq counts | M |
| 10 | Adaptive-order Radau IIA (5, 9 and 13) | OrdinaryDiffEqFIRK (MIT) `src/algorithms.jl:180`, `firk_caches.jl:504` | Radau order 5 only | Accuracy at tight tolerances | Hairer & Wanner, `radau.f` (Hairer's site); OrdinaryDiffEq (MIT) | OrdinaryDiffEq AdaptiveRadau counts | M |
| 11 | QUADPACK `dqng` (10/21/43/87-point, non-adaptive) and `dqag` keys 1–6 (qk15…qk61) | not in the clone (`git grep qng` is empty; only qk21/qk15) | Missing | Cheap smooth integrals; R `integrate` and Fortran users | QUADPACK, netlib (public domain) | `dqng`/`dqag` compiled with gfortran: same value and neval | S |
| 12 | VODE/ZVODE (variable-coefficient Adams/BDF, complex states) | deSolve `src/dvode.f`, `zvode.f` (read only); coverage.md notes `ode`/vode missing | Missing | SciPy legacy `ode` class, R `deSolve::vode`, complex states | VODE, netlib (public domain); Brown, Byrne & Hindmarsh, SIAM J. Sci. Stat. Comput. 10 (1989) | gfortran VODE counts | L |

### Better defaults

| # | Item | Evidence | Gain | Source / oracle | Size |
|---|---|---|---|---|---|
| D1 | Default auto-switching algorithm. Tsit5 or Vern7 is chosen by `reltol`. The stiff method is Rosenbrock23, Rodas5P or FBDF by system size. The stiffness test is NaN-safe (a NaN estimate counts as stiff). The step is rescaled on each switch | OrdinaryDiffEqDefault (MIT) `src/default_alg.jl:116-206`, NaN-safe test at :124-130, choices at :146-164 | A better `LSODA`-style default than today's RK45/DOP853 ↔ BDF pair, without ODEPACK's cost | OrdinaryDiffEq (MIT). Oracle: same switch points and nf as `solve(prob)` in OrdinaryDiffEq. Needs items 1 and 3 | S |
| D2 | PI, PID and predictive step-size controllers | OrdinaryDiffEqCore (MIT) `src/integrators/controllers.jl:744,924,1146` | Fewer rejected steps for every RK method. Today `rk.go:14-16,239-255` uses SciPy's I-controller (safety 0.9, factors 0.2 to 10) | Söderlind, Numer. Algorithms 31 (2002) and ACM TOMS 29 (2003). Acceptance: the global-error bound on the test set, with fewer RHS calls than the I-controller. If both hold, it becomes the default, and SciPy's controller stays available as an option. Porting step: the same accept/reject sequence as OrdinaryDiffEq with the controller set explicitly | S |

### Test problems to import

| # | Suite | Where | Gain | Size |
|---|---|---|---|---|
| T1 | Remaining stiff test set problems: Pollution, E5, plate, ring modulator, with reference solutions | Bari IVP test set (Mazzia & Magherini). ROBER, HIRES and OREGO are already fixtures (`ode_fixtures_test.go`) | Oracle problems for items 2, 3 and 8–10 | S–M |
| T2 | QuadGK's test cases: BigFloat order 40, complex integrands, `quadgk_count` counts, batch against non-batch, `kronrod` checks | QuadGK `test/runtests.jl` | Oracle for item 4 | S |
| T3 | Piessens/Gonnet integrand battery | The QUADPACK book and Gonnet (2010). GSL `integration/tests.c` was used only to locate them; the battery is re-derived from the publications | Oracle for items 7 and 11 | S |

## Not read

These were not read in this inventory. They are unknowns, not absences:

- DiffEqBase, and DiffEqProblemLibrary (not in the depot).
- From deSolve: the `rk_auto.c` tableaux, the R files, `call_lsoda.c` root
  finding and banded Jacobians, and `ddaspk.f`.
- From GSL: the odeiv2 steppers in detail (msadams, msbdf, bsimp, rk4imp),
  `romberg.c`, `glfixed.c` and `lebedev.c`.
- R's `integrate.R` and `integrate.c`.
- From OrdinaryDiffEq: `initdt.jl` and `is_stiff`, the body of the Rodas5P
  tableau, Vern6 and Vern8.
- The FastGaussQuadrature tests.
