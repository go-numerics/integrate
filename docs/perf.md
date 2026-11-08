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
  [perf-table.md](perf-table.md), generated from [perf.json](perf.json). Its
  timings are **deferred**: the host was at a 1-minute load of 10 to 24 when this
  was written, so only the allocation counts are recorded (exact under any load).
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
  call). By design different, so pinned in the test (Go, SciPy): `SolveIVP`
  LSODA 2594 against 1803 calls, `Odeint` Robertson 2222 against 1683 and
  Lotka 1880 against 1243: this package's stiff/non-stiff switching is not
  ODEPACK's (docs/design.md). Those three ratios carry that difference.
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

The counts of the table are per call. Where this work changed them (see the pull
request): a plain `Quad` allocates nothing (its six lists now come from a pool,
and a triple quadrature went from 2 780 allocations to 2), a Radau step no
longer allocates its interpolant (1 395 allocations to 350 on the Van der Pol
problem), and a finite-difference Jacobian no longer allocates its work vectors
(`BDF` 798 to 595, `LSODA` 781 to 608). The ODE rows still allocate what they
store (the state at every step or output time). 32-bit lanes are not part of the
`perf` job; a count that differed there would be recorded per `GOARCH`.
