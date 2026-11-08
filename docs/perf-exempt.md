# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/perf_test.go` or is named here
with its reason. The method and the table are in [perf.md](perf.md).

- `Romberg`: `scipy.integrate.romberg` was removed in SciPy 1.15; SciPy 1.18.1 has no counterpart to compare. The function is the pre-1.15 `romberg` (trapezoid refinement with Richardson extrapolation) and is covered by the tests.
- `Const`: returns the constant function `x -> v`; SciPy writes it as `lambda x: v`, a Python construct and not a call.
- `Const2`: as `Const`, for two variables.
- `ConstRange`: as `Const`; SciPy writes the range as the list `[lo, hi]`.
- `Solution.YArray`: a layout conversion of the stored states (SciPy's `sol.y` is an attribute, already an array).
- `Solution.TArray`: a copy of the stored times (SciPy's `sol.t` is an attribute).
- `Solution.Component`: one component's history (SciPy's `sol.y[i]` is an indexing expression).
- `ODESolution.TMin`: a field read (SciPy's `sol.sol.t_min` is an attribute).
- `ODESolution.TMax`: a field read (SciPy's `sol.sol.t_max` is an attribute).
