# LSODA's linear algebra on go-simd/floats kernels, and the Jacobian heuristics

Issue #26, the default path (`Exact` keeps the loops of lsoda.c, and the results of
every system shorter than `simdMin` = 32 are bit for bit what they were: the
corpus counts of `docs/lsoda-pow.md` do not move).

## Kernels

go-simd/floats v0.21.0 (`Axpy`, `Dot`, `Scale`; the kernel of the dense LU
factorisation's rank-1 update, of the two triangular solves, of the banded LU's
column updates and of the Nordsieck rescale). arm64 (Apple M4 Max), one core,
`go test -bench 'LU|DiffusionN' -cpu 1`, four interleaved rounds of the old and
the new test binary, the median kept. **The 1-minute load was 24 to 28 (the
rule is under 8): the ratios are indicative, the absolute times are not
recorded** (they go in `perf-arm64.json` with the refbench record once the load
allows it).

| kernel (size) | scalar loop | kernel | speedup |
|---|---:|---:|---:|
| dense LU solve, n = 64 | 5.75 us | 2.02 us | 2.9 x |
| dense LU solve, n = 256 | 55.4 us | 14.3 us | 3.9 x |
| dense LU factor, n = 64 | 35.3 us | 19.8 us | 1.8 x |
| dense LU factor, n = 256 | 2.17 ms | 1.16 ms | 1.9 x |
| banded LU factor+solve, kl = ku = 32 | 112.7 us | 64.7 us | 1.7 x |
| banded LU factor+solve, kl = ku = 100 | 2.68 ms | 0.94 ms | 2.8 x |
| `SolveIVP` LSODA, dense diffusion n = 200 | 2.21 ms | 1.68 ms | 1.3 x |
| LU, n = 3, 8, 20; banded kl = ku = 1, 8 | | | 1.0 x (the same code) |

Where the kernel does not pay, it is not called. The crossover, measured one
kernel at a time (`Dot`, `Axpy`, `Scale`, `Add` against the plain loop): the call
into assembly costs about 3 ns, the scalar loop wins up to 20 elements (a dot of
3 entries is 2.1 ns against 9.1 ns), they meet around 20 to 32, and the kernel
is 4 to 13 times faster at 256 (a dot 495 ns against 38 ns, an axpy 120 against
35, a scale 83 against 22). `simdMin` = 32 is that crossover, and the corpus
(n = 2 to 20, and the banded diffusion of 20) stays on the scalar loops; the
kernels serve the systems the corpus does not have (large dense and wide-band
problems).

Not done, and why: the weighted norms `vmnorm`, `fnorm` and `bnorm` have no
kernel in go-simd (`MaxAbs` is unweighted; `Mul` into a scratch plus `MaxAbs`
is 4.4 times faster at 256, 46 ns against 205 ns, but it is O(n) against the
O(n^2) solve and the O(n^3) factorisation it sits beside, so under 1 % of a step at
the sizes where it pays, and it needs a scratch vector); the Pascal-triangle
update `yh[i] += yh[i+n]` adds a slice to a shifted copy of itself, and
`floats.Add` does not document overlapping operands, so it stays a loop.

## Jacobian reuse and step heuristics

Tried on the corpus of `docs/lsoda-pow.md` (the same runs; error is the scaled
error against the fixtures' reference end state, in tolerances), changing LSODA's
two Jacobian-reuse constants, `ccmax` (the relative change of `h*el0` that
forces a new Jacobian, 0.3) and `msbp` (the longest run of steps with one
Jacobian, 20):

| ccmax | msbp | sum nfev | sum njev | nfev + njev | median error | geometric mean error | largest error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 (current) | 20 | 45752 | 1677 | 47429 | 18.8 | 83.7 | 6.39e+07 |
| 0.3 | 30 | 46248 | 1686 | 47934 | 18.8 | 81.9 | 6.39e+07 |
| 0.3 | 50 | 46234 | 1683 | 47917 | 18.8 | 81.9 | 6.39e+07 |
| 0.5 | 20 | 48289 | 1665 | 49954 | 22.7 | 79.7 | 6.4e+07 |
| 0.5 | 50 | 48246 | 1598 | 49844 | 16.9 | 73.0 | 6.38e+07 |
| 1.0 | 50 | 59882 | 2037 | 61919 | 36.3 | 171 | 6.32e+07 |
| 0.2 | 20 | 44684 | 1793 | 46477 | 24.5 | 93.5 | 6.44e+07 |

No setting cuts nfev + njev at an equal or better error: the one that cuts it
(ccmax 0.2, 2.0 % fewer) has a median error 30 % higher and a worse geometric
mean; the ones that save Jacobians (ccmax 0.5) pay more in evaluations, because
a stale Jacobian costs corrector iterations and step failures. The defaults stay.
The errors of the stiff rows are dominated by the reference (SciPy's own end
state, which differs from the exact solution by more than the tolerance), and the
300-bit references of `lsoda_test.go` cover only a few problems, so a change that
moved the error by less than the spread of the table could not be told from
noise; none of these did better.
