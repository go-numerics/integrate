# Times the problems of bench_test.go with SciPy (the reference this library
# is measured against). Run with:
#   uv run --with scipy --with numpy python bench/scipy_bench.py [name...]
# Prints "name ns/op callbacks_ns" lines: the median over 7 repeats of the
# mean time of one call, and an estimate of the part of it spent calling the
# Python right-hand side or integrand (the number of calls SciPy makes,
# counted on one run, times the cost of one call measured alone). That part is
# Python overhead whatever the algorithm; the difference is SciPy's own work.
import math
import sys
import timeit

import numpy as np
from scipy import integrate


def runge(x):
    return 1 / (1 + 25 * x * x)


def logsqrt(x):
    return math.log(x) / math.sqrt(x)


def gauss(x):
    return math.exp(-x * x)


def lotka(t, y):
    return [1.5 * y[0] - y[0] * y[1], -3 * y[1] + y[0] * y[1]]


MU = 1000.0


def vdp(t, y):
    return [y[1], MU * (1 - y[0] ** 2) * y[1] - y[0]]


def vdp_jac(t, y):
    return [[0, 1], [-2 * MU * y[0] * y[1] - 1, MU * (1 - y[0] ** 2)]]


def robertson(t, y):
    return [-0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2]


TS = np.array([0, 0.4, 4, 40, 400, 4000, 40000])
Y2 = np.array([1.0, 2.0])
Y3 = np.array([0.9, 1e-5, 0.1])

# name: (run(f), f, one-call statement)
CASES = {
    "QuadSmooth": (lambda f: integrate.quad(f, -1, 1), runge, "runge(0.3)"),
    "QuadSingular": (lambda f: integrate.quad(f, 0, 1), logsqrt, "logsqrt(0.3)"),
    "QuadInfinite": (lambda f: integrate.quad(f, -math.inf, math.inf), gauss, "gauss(0.3)"),
    "SolveIVPLotkaRK45": (lambda f: integrate.solve_ivp(f, (0, 15), [10, 5]), lotka, "lotka(0.0, Y2)"),
    "SolveIVPLotkaDOP853Tight": (lambda f: integrate.solve_ivp(f, (0, 15), [10, 5], method="DOP853", rtol=1e-10, atol=1e-12),
                                 lotka, "lotka(0.0, Y2)"),
    "SolveIVPVdP1000Radau": (lambda f: integrate.solve_ivp(f, (0, 3000), [2, 0], method="Radau"), vdp, "vdp(0.0, Y2)"),
    "SolveIVPVdP1000RadauJac": (lambda f: integrate.solve_ivp(f, (0, 3000), [2, 0], method="Radau", jac=vdp_jac), vdp, "vdp(0.0, Y2)"),
    "SolveIVPVdP1000BDF": (lambda f: integrate.solve_ivp(f, (0, 3000), [2, 0], method="BDF"), vdp, "vdp(0.0, Y2)"),
    "SolveIVPVdP1000BDFJac": (lambda f: integrate.solve_ivp(f, (0, 3000), [2, 0], method="BDF", jac=vdp_jac), vdp, "vdp(0.0, Y2)"),
    "SolveIVPVdP1000LSODA": (lambda f: integrate.solve_ivp(f, (0, 3000), [2, 0], method="LSODA"), vdp, "vdp(0.0, Y2)"),
    "OdeintRobertson": (lambda f: integrate.odeint(f, [1, 0, 0], TS, rtol=1e-8, atol=1e-12, tfirst=True), robertson,
                        "robertson(0.0, Y3)"),
}


def count_calls(run, f):
    n = [0]

    def g(*a):
        n[0] += 1
        return f(*a)
    run(g)
    return n[0]


def main():
    only = set(sys.argv[1:])
    for name, (run, f, stmt) in CASES.items():
        if only and name not in only:
            continue
        t = timeit.Timer(lambda: run(f))
        n, _ = t.autorange()
        runs = sorted(x / n for x in t.repeat(repeat=7, number=n))
        calls = count_calls(run, f)
        tc = timeit.Timer(stmt, globals=globals())
        m, _ = tc.autorange()
        per = sorted(x / m for x in tc.repeat(repeat=5, number=m))[2]
        print(f"{name} {runs[3] * 1e9:.0f} {per * calls * 1e9:.0f}")


if __name__ == "__main__":
    main()
