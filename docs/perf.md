# Performance against SciPy

## Method

`bench/compare.sh` runs the Go benchmarks of `bench_test.go` and the same
problems through SciPy (`bench/scipy_bench.py`, under `uv run --with
scipy==1.18.1 --with numpy==2.5.4`) alternately, five rounds, and reports
the median of each side. The Go side runs on one thread (`-test.cpu 1`), as
SciPy does. Each SciPy figure is the median of 7 repeats of a timed loop
lasting about 0.2 s; each Go figure is `go test -bench` with `-benchtime 1s`.
`bench/compare.sh name...` restricts a run to some problems.

SciPy's time includes calling the Python integrand or right-hand side, which
is the price of writing it in Python, not of SciPy's algorithms. The script
also estimates that part: it counts the calls SciPy makes on one run and
multiplies by the cost of one call of the same Python function measured alone.
The last column divides SciPy's time without that part by Go's: what remains
is SciPy's own machinery (QUADPACK in C for `quad`, NumPy-level Python for
`solve_ivp`, ODEPACK in Fortran for `odeint` and `method="LSODA"`). The
sample rules (`trapezoid`, `simpson`, `romb`, …) take no callback.

Host: Apple M4 Max (16 cores), macOS, Go 1.27.2, Python 3.12, SciPy 1.18.1,
NumPy 2.5.4. The machine is shared with other jobs: `uptime` gave load
averages of 13.7 at the start and 15.3 at the end of the run below (16
cores), so absolute numbers carry some noise (±20 % between runs).

## Results (2026-10-17, 5 interleaved rounds)

| problem | Go median | SciPy median | SciPy / Go | of which Python callbacks | SciPy without callbacks / Go |
|---|---:|---:|---:|---:|---:|
| QuadSmooth | 1.4 µs | 11.5 µs | 8.5× | 7.0 µs | 3.3× |
| QuadSingular | 3.6 µs | 22.5 µs | 6.3× | 14.2 µs | 2.3× |
| QuadInfinite | 2.9 µs | 15.4 µs | 5.4× | 8.5 µs | 2.4× |
| SolveIVPLotkaRK45 | 15.2 µs | 1.24 ms | 82× | 150.4 µs | 72× |
| SolveIVPLotkaDOP853Tight | 82.1 µs | 7.44 ms | 91× | 1.05 ms | 78× |
| SolveIVPVdP1000Radau | 155.4 µs | 28.36 ms | 182× | 780.7 µs | 177× |
| SolveIVPVdP1000RadauJac | 176.2 µs | 26.28 ms | 149× | 724.7 µs | 145× |
| SolveIVPVdP1000BDF | 194.5 µs | 32.54 ms | 167× | 600.4 µs | 164× |
| SolveIVPVdP1000BDFJac | 194.7 µs | 29.87 ms | 153× | 443.8 µs | 151× |
| SolveIVPVdP1000LSODA | 150.6 µs | 2.22 ms | 15× | 466.7 µs | 12× |
| OdeintRobertson | 194.5 µs | 1.54 ms | 7.9× | 989.7 µs | 2.8× |
| QuadCos | 1.6 µs | 2.6 µs | 1.6× | 732 ns | 1.1× |
| QuadFourier | 7.2 µs | 13.2 µs | 1.8× | 5.9 µs | 1× |
| QuadAlgLog | 1.4 µs | 3.8 µs | 2.7× | 1.3 µs | 1.7× |
| QuadCauchy | 1.8 µs | 12.2 µs | 6.7× | 7.0 µs | 2.8× |
| QuadPoints | 3.3 µs | 32.5 µs | 9.9× | 17.7 µs | 4.5× |
| DblQuad | 19.6 µs | 61.3 µs | 3.1× | 14.7 µs | 2.4× |
| TplQuad | 392.6 µs | 1.44 ms | 3.7× | 494.6 µs | 2.4× |
| NQuad | 368.4 µs | 1.41 ms | 3.8× | 481.0 µs | 2.5× |
| FixedQuad | 27 ns | 6.7 µs | 251× | 574 ns | 230× |
| GaussLegendre50 | 8 ns | 63.9 µs | 7802× | 0 ns | 7802× |
| Trapezoid1e5 | 53.3 µs | 72.3 µs | 1.4× | 0 ns | 1.4× |
| Simpson1e5 | 97.8 µs | 507.3 µs | 5.2× | 0 ns | 5.2× |
| CumulativeTrapezoid1e5 | 245.5 µs | 315.0 µs | 1.3× | 0 ns | 1.3× |
| Romb | 41.6 µs | 64.4 µs | 1.5× | 0 ns | 1.5× |
| TrapezoidNDAxis0 | 43.2 µs | 60.5 µs | 1.4× | 0 ns | 1.4× |
| TrapezoidNDAxis1 | 43.9 µs | 100.2 µs | 2.3× | 0 ns | 2.3× |
| SimpsonNDAxis1 | 47.3 µs | 130.7 µs | 2.8× | 0 ns | 2.8× |
| CumulativeTrapezoidNDAxis0 | 165.1 µs | 384.5 µs | 2.3× | 0 ns | 2.3× |
| SolveIVPLotkaRK23 | 33.6 µs | 2.10 ms | 63× | 186.2 µs | 57× |
| SolveIVPLotkaDense | 49.3 µs | 1.73 ms | 35× | 149.8 µs | 32× |
| SolveIVPLotkaEvents | 17.7 µs | 2.03 ms | 115× | 152.7 µs | 106× |
| OdeintLotka | 65.6 µs | 671.3 µs | 10× | 389.9 µs | 4.3× |

Every operation is faster than SciPy. `GaussLegendre` returns a cached
rule (as `fixed_quad`'s `_cached_roots_legendre` does), while
`roots_legendre` recomputes it, so that row compares a lookup with a
computation. `romberg` has no SciPy 1.18 counterpart (SciPy 1.15 removed
it) and is not compared.

## Before and after this round of work

The same comparison on the code before this work (2026-10-12, 3 rounds,
load 14–18). Rows not listed were unchanged within the noise.

| problem | SciPy / Go before | after |
|---|---:|---:|
| Trapezoid1e5 | **0.67×** | 1.4× |
| TrapezoidNDAxis0 | **0.10×** | 1.4× |
| TrapezoidNDAxis1 | **0.23×** | 2.3× |
| SimpsonNDAxis1 | **0.23×** | 2.8× |
| CumulativeTrapezoidNDAxis0 | **0.17×** | 2.3× |
| Romb | 1.3× | 1.5× |
| CumulativeTrapezoid1e5 | 1.6× | 1.3× (noise: same code path) |

What changed:

* **The sample rules sum as NumPy sums.** `trapezoid`, `simpson` and
  `romb` end in `numpy.add.reduce`, which on a contiguous array is a
  pairwise sum: blocks of 128 terms with eight running accumulators,
  combined pairwise. Our rules used one running sum, which is
  latency-bound and rounds differently. They now form every term as SciPy
  forms it (products rounded before they are added, so Go never fuses them
  into an FMA on arm64) and sum with the same pairwise order (`sum.go`).
  `cumulative_trapezoid` is `cumsum`, a running sum, and stays one.
* **The ND rules follow both of NumPy's orders.** NumPy reduces the last
  (contiguous) axis pairwise, but any other axis row after row, each output
  being a running sum from 0.0. The ND rules do the same, and the row order
  lets every lane advance in the same pass. They read a contiguous float64
  array in place; they used to fetch every element through `At`, which
  dominated their time.
* **Results now equal SciPy's to the last bit.** `exact_test.go` pins 96
  results of the 1-D rules and 54 results of the ND rules (every axis of
  three 3-D arrays) to what SciPy 1.18.1 returns on inputs both sides form
  exactly (`bench/exact.py` generates them). The previous code differed
  from SciPy in 35 and 11 of them, by up to 9·10⁻¹⁴ relative, and differed
  between arm64 and amd64; both architectures now return SciPy's bits.

## Reading the numbers

* **quad**: SciPy's QUADPACK is compiled C, and 55–62 % of its time is the
  Python integrand. The same subdivision (same evaluation count, see
  verification.md) costs 2–3× less here without the callbacks, 5–10× with
  them. The weighted routines (QAWO, QAWF) spend proportionally more in
  Chebyshev moments and less in callbacks, so the gap there is smaller
  (1.6–1.8×; about 1× once SciPy is credited with free callbacks).
* **solve_ivp**: SciPy's RK45, DOP853, Radau and BDF are written in Python on
  NumPy; per step they pay NumPy call overhead on 2-element arrays, which the
  callbacks do not explain (they are 2–14 % of the time). The step sequences
  are the same as here, so the 60–180× is per-step overhead, not a different
  amount of work.
* **LSODA and odeint**: SciPy runs ODEPACK's Fortran; most of `odeint`'s time
  on Robertson is the Python right-hand side. ODEPACK's own work is about 3×
  ours on that problem. The comparison is between two different algorithms
  (see design.md): on Van der Pol μ = 1000 this package's switching method
  uses 2 337 evaluations to ODEPACK's 1 803, on Robertson 2 175 to 1 683, and
  is still faster in wall time because each evaluation is a Go call.
* **Sample rules**: NumPy's loops are compiled and vectorised; ours are
  scalar Go but make one pass where NumPy makes several (one per
  temporary array), which is why they are 1.3–5× faster.
* Allocation: a `Quad` call allocates its subinterval workspace (6 allocations
  for QAGS); an ODE step allocates only what it stores in the `Solution`
  (the state at each output time) plus, for Radau, the interpolant reused as
  the next step's predictor.

To reproduce: `ROUNDS=5 ./bench/compare.sh` (needs `uv` on the PATH).
