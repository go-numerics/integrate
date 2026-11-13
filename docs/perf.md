# Performance against SciPy 1.18.1

The comparison runs under [refbench](https://github.com/go-scikit/refbench)
v0.1.2 (`rbtest.Perf`): one core (GOMAXPROCS=1; BLAS threads pinned to 1),
interleaved rounds (Go, SciPy, Go, SciPy, ...), the 1-minute load read before
every round and the measurement refused at a load of 8 or more. Target: no
slower than SciPy (ratio <= 1); a row above 1 is an open defect.

- The cases are in [`bench/perf_test.go`](../bench/perf_test.go): one or more
  per public function, against the `scipy.integrate` function it ports (SciPy
  1.18.1 and NumPy 2.5.4, pinned with their hashes in `bench/refs.txt`).
- Functions with no SciPy call to compare are named in
  [perf-exempt.md](perf-exempt.md) with the reason.
- The table, with the allocation count of every case, is
  [perf-table.md](perf-table.md), generated from [perf.json](perf.json) (amd64).
  Its timings are **deferred**: no amd64 host has been idle yet, so only the
  allocation counts are recorded (exact under any load).
- **Timings** are in [perf-table-arm64.md](perf-table-arm64.md) (darwin/arm64,
  16 CPUs, Go 1.27.4, one core, 2026-11-13, after the port of ODEPACK's lsoda,
  docs/design.md): every one of the 42 rows is faster than SciPy 1.18.1, with no
  row above ratio 1. The slowest ratios are `Trapezoid` 1e5 (0.71),
  `TrapezoidND` axis 0 (0.63), `CumulativeTrapezoid` (0.58) and `Romb` (0.57),
  where SciPy's compiled code is the competitor; the `(libm)` rows, with no
  Python callback on the SciPy side, are 0.08 to 0.21. `SolveIVP` LSODA is 0.07
  and `Odeint` 0.12 (Robertson) and 0.12 (Lotka, 101 times); the Adams/BDF port
  runs more steps than the Runge–Kutta/BDF switch it replaced (those two rows
  were 0.06, 0.13 and 0.08), at the same calls of the right-hand side as
  SciPy's. The 1-minute load over the rounds was 5.01 to 6.73 (gate 8), so the
  run is valid.
- The previous table (own harness, host load 14 to 18) is kept as
  [perf-legacy.md](perf-legacy.md).

```
cd bench
go test -run TestPerf .                          CI mode: allocations, table, inventory
go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
go test -run TestDataParity -perf.parity .       same bytes on both sides
go test -run TestWorkParity -perf.parity .       same number of integrand calls
go test -run TestRefSmoke -perf.smoke .          every SciPy expression runs once
```

CI runs the first line: the allocation count of every case must equal
`perf.json` (an increase is a regression, a decrease must be recorded), the table
must be what `perf.json` generates, and every public function must have a case or
an exemption.

## What makes a row comparable

- **Same data.** The sample rules take 100 000 samples (Romb 65 537, the ND
  rules a 1000 x 100 grid) built by one formula on both sides with no
  transcendental function and every product rounded before it is added.
  `TestDataParity` hashes (SHA-256) the arrays on both sides and passes.
- **Same work.** An adaptive integrator does the same work when it calls the
  integrand as often. `TestWorkParity` wraps the Python callback in a counter,
  counts the Go integrand the same way, and compares. Equal for every `Quad`
  row (including the weighted routines and the singular, infinite and
  break-point cases), `DblQuad`, `TplQuad`, `NQuad`, and `SolveIVP` with RK45,
  RK23, DOP853 (also with `t_eval`, dense output and an event) and BDF with a
  Jacobian. Within 2 % for Radau (285 steps against 283 without a Jacobian,
  282 against 285 with one: the stiff Newton iteration is sensitive to the
  rounding of the LU solves) and within 1 % for BDF without a Jacobian (one
  call). Equal too for `SolveIVP` LSODA (1803 calls) and `Odeint` Robertson
  (1683) and Lotka (1243) on arm64: LSODA is ODEPACK's lsoda, ported from
  SciPy's `lsoda.c` (docs/design.md), where it was 2594, 2222 and 1880 calls
  with the Runge–Kutta/BDF switch the port replaced and pinned as different by
  design. Their step selection hinges on the last bit, so the equality is that
  of this machine's compiler (the C source itself takes 647 or 682 evaluations
  on another problem with and without fused multiply-adds), and no row is
  pinned.
- **Callbacks.** SciPy calls its integrand from Python. The rows say what they
  do about it:
  - `Quad` rows named `(libm)` integrate a function of the C library (`log`,
    `sin`, `exp`) passed as a `scipy.LowLevelCallable` built with `ctypes`: no
    Python runs inside QUADPACK. The twin rows named `(Python)` integrate the
    same function written in Python (`log`, `sin 0..100`); the difference is
    the cost of the callback. The other `Quad` rows are Python callbacks.
  - `FixedQuad` passes `numpy.cos`, the vectorised callback `fixed_quad`
    offers (one call on the array of nodes); the Go side calls `math.Cos` at
    each node.
  - `DblQuad`, `TplQuad`, `NQuad`, `SolveIVP` and `Odeint` have no compiled or
    vectorised callback in SciPy (a `ctypes` function must have the signature
    `double(double)` or `double(int, double*)`; `solve_ivp`'s `vectorized`
    flag only concerns finite-difference Jacobians), so they call a Python
    function; the equal counts above show both sides call it as often.
- `GaussLegendre` compares Go's cached rule (a `sync.Map` lookup) with SciPy's
  `_cached_roots_legendre` (a cached lookup), the function
  `fixed_quad` itself uses; the cold computation (SciPy's `roots_legendre`) is
  not reachable from the public API once a rule is cached.

## Allocation counts

The counts of the table are per call, recorded on linux/amd64 (the `perf` job's
lane). Three rows depend on the architecture: `SolveIVP` Radau, Radau with a
Jacobian and BDF store one state per step, and a stiff problem takes a few
steps more or fewer when arm64 fuses a multiply-add (Radau 353 on amd64, 350 on
arm64; Radau with a Jacobian 349 and 345; BDF 596 and 595). Each architecture
has its own exact record: `perf.json` / `perf-table.md` (amd64, from CI) and
`perf-arm64.json` / `perf-table-arm64.md` (arm64, measured locally), selected
by `GOARCH` in `bench/perf_test.go`; the `perf` job runs on both. Timings, when
recorded, go in the file of the machine that took them. Where this work changed them (see the pull
request): a plain `Quad` allocates nothing (its six lists now come from a pool,
and a triple quadrature went from 2 780 allocations to 2), a Radau step no
longer allocates its interpolant (1 395 allocations to 350 on the Van der Pol
problem), and a finite-difference Jacobian no longer allocates its work vectors
(`BDF` 798 to 595, `LSODA` 781 to 608). The ODE rows still allocate what they
store (the state at every step or output time): with ODEPACK's lsoda `Odeint`
allocates 44 (Robertson) and 68 (Lotka) times per call, no longer 106 and 99,
interpolating in place. The stored states of `SolveIVP` come out of chunks of
4096 floats (a handful of allocations per call, not one per step): `SolveIVP`
LSODA 37, was 764 with one state allocated per accepted step (727 steps), BDF
60 and 62, Radau 63 and 65, RK45 38. 32-bit lanes are not part of the
`perf` job; a count that differed there would be recorded per `GOARCH`.
