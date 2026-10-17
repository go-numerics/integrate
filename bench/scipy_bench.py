# Times the problems of bench_test.go with SciPy (the reference this library
# is measured against). Run with:
#   uv run --with scipy==1.18.1 --with numpy==2.5.4 python bench/scipy_bench.py [name...]
# Prints "name ns/op callbacks_ns" lines: the median over 7 repeats of the
# mean time of one call, and an estimate of the part of it spent calling the
# Python right-hand side or integrand (the number of calls SciPy makes,
# counted on one run, times the cost of one call measured alone). That part is
# Python overhead whatever the algorithm; the difference is SciPy's own work.
import math
import sys
import timeit

import numpy as np
from scipy import integrate, special


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


# The rest of the public operations (bench_test.go poses the same problems).
def expm(x):
    return math.exp(-x)


def inv1p(x):
    return 1 / (1 + x)


def lorentz(x):
    return 1 / (1 + x * x)


def sqrtabs(x):
    return math.sqrt(abs(x - 0.3))


def expxy(y, x):
    return math.exp(-x * y)


def gauss3(z, y, x):
    return math.exp(-(x * x + y * y + z * z))


def gauss3n(x, y, z):
    return math.exp(-(x * x + y * y + z * z))


def lotka_event(t, y):
    return y[0] - 5


TE = np.linspace(0, 15, 1001)
TS_LOTKA = np.linspace(0, 15, 101)

CASES.update({
    "QuadCos": (lambda f: integrate.quad(f, 0, 10, weight="cos", wvar=20), expm, "expm(0.3)"),
    "QuadFourier": (lambda f: integrate.quad(f, 0, math.inf, weight="sin", wvar=1), expm, "expm(0.3)"),
    "QuadAlgLog": (lambda f: integrate.quad(f, 0, 1, weight="alg-loga", wvar=(-0.5, 0)), inv1p, "inv1p(0.3)"),
    "QuadCauchy": (lambda f: integrate.quad(f, -1, 2, weight="cauchy", wvar=0.5), lorentz, "lorentz(0.3)"),
    "QuadPoints": (lambda f: integrate.quad(f, 0, 1, points=[0.3]), sqrtabs, "sqrtabs(0.5)"),
    "DblQuad": (lambda f: integrate.dblquad(f, 0, 1, 0, lambda x: x), expxy, "expxy(0.3, 0.5)"),
    "TplQuad": (lambda f: integrate.tplquad(f, 0, 1, 0, 1, 0, 1), gauss3, "gauss3(0.3, 0.5, 0.7)"),
    "NQuad": (lambda f: integrate.nquad(f, [[0, 1], [0, 1], [0, 1]]), gauss3n, "gauss3n(0.3, 0.5, 0.7)"),
    "FixedQuad": (lambda f: integrate.fixed_quad(f, 0, 1, n=5), np.cos, "np.cos(np.ones(5))"),
    "SolveIVPLotkaRK23": (lambda f: integrate.solve_ivp(f, (0, 15), [10, 5], method="RK23"), lotka, "lotka(0.0, Y2)"),
    "SolveIVPLotkaDense": (lambda f: integrate.solve_ivp(f, (0, 15), [10, 5], t_eval=TE, dense_output=True),
                           lotka, "lotka(0.0, Y2)"),
    "SolveIVPLotkaEvents": (lambda f: integrate.solve_ivp(f, (0, 15), [10, 5], events=lotka_event),
                            lotka, "lotka(0.0, Y2)"),
    "OdeintLotka": (lambda f: integrate.odeint(f, [10, 5], TS_LOTKA, tfirst=True), lotka, "lotka(0.0, Y2)"),
})


def samples(n):
    t = np.arange(n) / (n - 1)
    x = 10 * t * t
    return np.sin(x), x


Y5, X5 = samples(100000)
YR, _ = samples(2 ** 16 + 1)
G = np.sin(np.arange(1000 * 100) * 0.001).reshape(1000, 100)

# Operations on sample arrays: no Python callback.
ARRAYS = {
    "GaussLegendre50": lambda: special.roots_legendre(50),
    "Trapezoid1e5": lambda: integrate.trapezoid(Y5, X5),
    "Simpson1e5": lambda: integrate.simpson(Y5, x=X5),
    "CumulativeTrapezoid1e5": lambda: integrate.cumulative_trapezoid(Y5, X5, initial=0),
    "Romb": lambda: integrate.romb(YR, dx=0.001),
    "TrapezoidNDAxis0": lambda: integrate.trapezoid(G, dx=0.1, axis=0),
    "TrapezoidNDAxis1": lambda: integrate.trapezoid(G, dx=0.1, axis=1),
    "SimpsonNDAxis1": lambda: integrate.simpson(G, dx=0.1, axis=1),
    "CumulativeTrapezoidNDAxis0": lambda: integrate.cumulative_trapezoid(G, dx=0.1, axis=0, initial=0),
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
    for name, run in ARRAYS.items():
        if only and name not in only:
            continue
        t = timeit.Timer(run)
        n, _ = t.autorange()
        runs = sorted(x / n for x in t.repeat(repeat=7, number=n))
        print(f"{name} {runs[3] * 1e9:.0f} 0")


if __name__ == "__main__":
    main()
