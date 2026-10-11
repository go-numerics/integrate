# Performance against SciPy

## Method

`bench/compare.sh` runs the Go benchmarks of `bench_test.go` and the same
problems through SciPy (`bench/scipy_bench.py`, under `uv run --with scipy
--with numpy`) alternately, five rounds, and reports the median of each side.
Each SciPy figure is the median of 7 repeats of a timed loop lasting about
0.2 s; each Go figure is `go test -bench` with `-benchtime 1s`.

SciPy's time includes calling the Python integrand or right-hand side, which
is the price of writing it in Python, not of SciPy's algorithms. The script
also estimates that part: it counts the calls SciPy makes on one run and
multiplies by the cost of one call of the same Python function measured alone.
The last column divides SciPy's time without that part by Go's: what remains
is SciPy's own machinery (QUADPACK in C for `quad`, NumPy-level Python for
`solve_ivp`, ODEPACK in Fortran for `odeint` and `method="LSODA"`).

Host: Apple M4 Max (16 cores), macOS, Go 1.27.2, Python 3.12.12, SciPy
1.18.1, NumPy 2.5.4. The machine is shared with other jobs: `uptime` gave load
averages of 7.4 at the start and 8.2 at the end of the run below (16 cores),
so absolute numbers carry some noise; the ratios are stable across rounds.

## Results (2026-10-10)

| problem | Go median | SciPy median | SciPy / Go | of which Python callbacks | SciPy without callbacks / Go |
|---|---:|---:|---:|---:|---:|
| QuadSmooth: ∫₋₁¹ 1/(1+25x²) | 1.7 µs | 14.3 µs | 8× | 8.9 µs | 3× |
| QuadSingular: ∫₀¹ log x/√x | 4.7 µs | 30.1 µs | 6× | 18.2 µs | 3× |
| QuadInfinite: ∫ e^(−x²) over ℝ | 3.5 µs | 20.8 µs | 6× | 11.5 µs | 3× |
| SolveIVP Lotka–Volterra, RK45, defaults | 18.1 µs | 1.67 ms | 92× | 201.5 µs | 81× |
| SolveIVP Lotka–Volterra, DOP853, rtol 1e-10 | 100.6 µs | 10.01 ms | 100× | 1.41 ms | 86× |
| SolveIVP Van der Pol μ=1000, Radau | 199.9 µs | 37.72 ms | 189× | 992.4 µs | 184× |
| … Radau with analytic Jacobian | 189.9 µs | 34.39 ms | 181× | 940.1 µs | 176× |
| SolveIVP Van der Pol μ=1000, BDF | 236.0 µs | 39.35 ms | 167× | 722.4 µs | 164× |
| … BDF with analytic Jacobian | 223.7 µs | 35.07 ms | 157× | 601.8 µs | 154× |
| SolveIVP Van der Pol μ=1000, LSODA | 177.8 µs | 2.93 ms | 16× | 623.7 µs | 13× |
| odeint Robertson, rtol 1e-8, atol 1e-12 | 256.2 µs | 1.93 ms | 8× | 1.30 ms | 2× |

## Reading the numbers

* **quad**: SciPy's QUADPACK is compiled C, and 55–62 % of its time is the
  Python integrand. The same subdivision (same evaluation count, see
  verification.md) costs 3× less here without the callbacks, 6–8× with them.
* **solve_ivp**: SciPy's RK45, DOP853, Radau and BDF are written in Python on
  NumPy; per step they pay NumPy call overhead on 2-element arrays, which the
  callbacks do not explain (they are 2–14 % of the time). The step sequences
  are the same as here, so the 80–190× is per-step overhead, not a different
  amount of work.
* **LSODA and odeint**: SciPy runs ODEPACK's Fortran; most of `odeint`'s time
  on Robertson is the Python right-hand side. ODEPACK's own work is about 2×
  ours on that problem. The comparison is between two different algorithms
  (see design.md): on Van der Pol μ = 1000 this package's switching method
  uses 2 337 evaluations to ODEPACK's 1 803, on Robertson 2 175 to 1 683, and
  is still faster in wall time because each evaluation is a Go call.
* Allocation: a `Quad` call allocates its subinterval workspace (6 allocations
  for QAGS); an ODE step allocates only what it stores in the `Solution`
  (the state at each output time) plus, for Radau, the interpolant reused as
  the next step's predictor.

To reproduce: `ROUNDS=5 ./bench/compare.sh` (needs `uv` on the PATH).
