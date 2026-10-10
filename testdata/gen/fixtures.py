# Fixture generator for github.com/go-numerics/integrate.
#
# Run from the repository root, with nothing installed system-wide:
#
#   uv run --with scipy --with numpy --with mpmath python testdata/gen/fixtures.py
#
# It writes testdata/quad.json and testdata/ode.json. SciPy is used as an
# oracle (its QUADPACK and solve_ivp results, evaluation counts and step
# sequences) and mpmath / closed forms / very tight integrations as the
# reference values the errors are measured against. The integrand and problem
# names must match the Go tables in fixtures_test.go.

import json
import math
import os
import sys
import warnings

import mpmath as mp
import numpy as np
import scipy
from scipy import integrate

mp.mp.dps = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def f64(x):
    return float(x)


def jf(x):
    """A float for JSON, with the infinities spelled as strings (JSON has no
    infinity)."""
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


# ---------------------------------------------------------------- quadrature
# name: (python integrand, a, b, quad kwargs, reference value)
QUAD = {}


def q(name, fn, a, b, ref, **kw):
    QUAD[name] = (fn, a, b, kw, ref)


inf = math.inf
q("exp", math.exp, 0.0, 1.0, mp.e - 1)
q("sin", math.sin, 0.0, math.pi, 2)
q("runge", lambda x: 1 / (1 + 25 * x * x), -1.0, 1.0, mp.mpf(2) / 5 * mp.atan(5))
q("log_over_sqrt", lambda x: math.log(x) / math.sqrt(x), 0.0, 1.0, -4)
q("inv_sqrt", lambda x: 1 / math.sqrt(x), 0.0, 1.0, 2)
q("x_pow_m09", lambda x: x ** -0.9, 0.0, 1.0, 10)
q("log", math.log, 0.0, 1.0, -1)
q("exp_cos50", lambda x: math.exp(-x) * math.cos(50 * x), 0.0, 2 * math.pi,
  mp.quad(lambda x: mp.exp(-x) * mp.cos(50 * x), mp.linspace(0, 2 * mp.pi, 60)))
q("tight_runge", lambda x: 1 / (1 + 25 * x * x), -1.0, 1.0, mp.mpf(2) / 5 * mp.atan(5),
  epsabs=0.0, epsrel=1e-13, limit=200)
q("gauss_inf", lambda x: math.exp(-x * x), -inf, inf, mp.sqrt(mp.pi))
q("lorentz_half", lambda x: 1 / (1 + x * x), 0.0, inf, mp.pi / 2)
q("exp_left", math.exp, -inf, 0.0, 1)
q("x2_exp", lambda x: x * x * math.exp(-x), 0.0, inf, 2)
q("lorentz_shift", lambda x: 1 / ((x - 1) ** 2 + 1), -inf, inf, mp.pi)
q("kink_points", lambda x: abs(x - 1 / 3), 0.0, 1.0, mp.mpf(5) / 18, points=[1 / 3])
q("step_points", lambda x: 1.0 if x < 0.5 else 2.0, 0.0, 1.0, mp.mpf(3) / 2, points=[0.5])
q("step_nopoints", lambda x: 1.0 if x < 0.5 else 2.0, 0.0, 1.0, mp.mpf(3) / 2)
q("sqrt_kink_points", lambda x: math.sqrt(abs(x - 0.25)), 0.0, 1.0,
  mp.mpf(2) / 3 * (mp.mpf("0.25") ** 1.5 + mp.mpf("0.75") ** 1.5), points=[0.25, 0.7, 0.25])
q("cos_weight", math.exp, 0.0, 1.0,
  (mp.e * (mp.cos(100) + 100 * mp.sin(100)) - 1) / (1 + 10000), weight="cos", wvar=100.0)
q("sin_weight_log1p", math.log1p, 0.0, 1.0,
  mp.quad(lambda x: mp.log(1 + x) * mp.sin(10 * mp.pi * x), mp.linspace(0, 1, 21)), weight="sin", wvar=10 * math.pi)
q("cos_weight_small", lambda x: x * x, 0.0, 1.0,
  mp.quad(lambda x: x * x * mp.cos(1.5 * x), [0, 1]), weight="cos", wvar=1.5)
q("fourier_cos", lambda x: math.exp(-x), 0.0, inf, mp.mpf(1) / 5, weight="cos", wvar=2.0)
q("fourier_sin", lambda x: 1 / (1 + x * x), 0.0, inf,
  mp.quadosc(lambda x: mp.sin(x) / (1 + x * x), [0, mp.inf], omega=1), weight="sin", wvar=1.0)
q("fourier_sinc", lambda x: 1 / x, 1.0, inf, mp.pi / 2 - mp.si(1), weight="sin", wvar=1.0)
q("fourier_left", lambda x: math.exp(x), -inf, 0.0, mp.mpf(1) / 5, weight="cos", wvar=2.0)
q("fourier_left_sin", lambda x: math.exp(x), -inf, 0.0, mp.mpf(-2) / 5, weight="sin", wvar=2.0)
q("alg_chebyshev", lambda x: 1.0, -1.0, 1.0, mp.pi, weight="alg", wvar=(-0.5, -0.5))
q("alg_cos", math.cos, 0.0, 1.0,
  mp.quad(lambda x: mp.cos(x) * x ** -0.5 * (1 - x) ** 0.3, [0, 1]), weight="alg", wvar=(-0.5, 0.3))
q("alg_loga", lambda x: 1.0, 0.0, 1.0, -1, weight="alg-loga", wvar=(0.0, 0.0))
q("alg_logb", lambda x: 1.0, 0.0, 1.0,
  mp.quad(lambda x: mp.sqrt(x) * mp.log(1 - x), [0, 1]), weight="alg-logb", wvar=(0.5, 0.0))
q("alg_log", lambda x: 1.0, 0.0, 1.0, 2 - mp.pi ** 2 / 6, weight="alg-log", wvar=(0.0, 0.0))
q("alg_log_exp", math.exp, 0.0, 2.0,
  mp.quad(lambda x: mp.exp(x) * x ** 0.25 * (2 - x) ** -0.4 * mp.log(x) * mp.log(2 - x), [0, 1, 2]),
  weight="alg-log", wvar=(0.25, -0.4))
q("cauchy_one", lambda x: 1.0, -1.0, 2.0, mp.log(2), weight="cauchy", wvar=0.0)
q("cauchy_exp", math.exp, -1.0, 1.0, 2 * mp.shi(1), weight="cauchy", wvar=0.0)
q("cauchy_far", math.exp, 0.0, 1.0, mp.quad(lambda x: mp.exp(x) / (x - 3), [0, 1]), weight="cauchy", wvar=3.0)
q("divergent", lambda x: 1 / x, 0.0, 1.0, mp.inf)
q("sin_inv", lambda x: math.sin(1 / x), 0.0, 1.0,
  mp.sin(1) - mp.ci(1))  # ∫_0^1 sin(1/x) dx = sin 1 - Ci(1)
q("limit_small", lambda x: math.sin(1 / x), 0.0, 1.0, mp.sin(1) - mp.ci(1), limit=5)

# Diagnostics and the adaptive paths of the weighted routines (references are
# SciPy's own values: these runs check that the same subdivision happens).
q("inv_x_lim200", lambda x: 1 / x, 0.0, 1.0, mp.inf, limit=200)
q("x_pow_m12", lambda x: x ** -1.2 if x > 0 else 0.0, 0.0, 1.0, mp.inf, limit=200)
q("noise", lambda x: math.sin(1e7 * x) * 1e-3 + 1, 0.0, 1.0, mp.inf)
q("tiny_exp", math.exp, 0.0, 1.0, mp.e - 1, epsabs=0.0, epsrel=1.2e-14)
q("tiny_sqrt", math.sqrt, 0.0, 1.0, mp.mpf(2) / 3, epsabs=0.0, epsrel=1.2e-14, limit=200)
q("tiny_log", lambda x: math.log(x), 0.0, 1.0, -1, epsabs=0.0, epsrel=1.2e-14, limit=200)
q("inf_div", lambda x: 1 / (1 + abs(x)), 0.0, inf, mp.inf, limit=200)
q("inf_osc", math.sin, 0.0, inf, mp.inf)
q("inf_slow", lambda x: 1 / (1 + x) ** 1.05, 0.0, inf, 20)
q("pts_div", lambda x: 1 / abs(x - 0.5) if x != 0.5 else 0.0, 0.0, 1.0, mp.inf, points=[0.5], limit=200)
q("pts_log_tiny", lambda x: math.log(abs(x - 0.3)) if x != 0.3 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.log(abs(x - mp.mpf("0.3"))), [0, mp.mpf("0.3"), 1]), points=[0.3], epsabs=0.0, epsrel=1.2e-14, limit=200)
q("pts_sqrt_tiny", lambda x: math.sqrt(abs(x - 0.3)), 0.0, 1.0,
  mp.mpf(2) / 3 * (mp.mpf("0.3") ** 1.5 + mp.mpf("0.7") ** 1.5), points=[0.3, 0.6], epsabs=0.0, epsrel=1.2e-14, limit=200)
q("pts_lim", lambda x: math.sin(1 / x) if x > 0 else 0.0, 0.0, 1.0, mp.inf, points=[0.5], limit=10)
q("pts_x_m12", lambda x: abs(x - 0.3) ** -1.2 if x != 0.3 else 0.0, 0.0, 1.0, mp.inf, points=[0.3], limit=500)
q("qawo_long", lambda x: math.exp(-x / 30) / (1 + x), 0.0, 200.0,
  mp.quad(lambda x: mp.exp(-x / 30) / (1 + x) * mp.cos(10 * x), mp.linspace(0, 200, 400)), weight="cos", wvar=10.0)
q("qawo_long_sin", lambda x: 1 / (1 + x * x), 0.0, 50.0,
  mp.quad(lambda x: mp.sin(x) / (1 + x * x), mp.linspace(0, 50, 60)), weight="sin", wvar=1.0)
q("qawo_log", lambda x: math.log(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.log(x) * mp.cos(50 * x), mp.linspace(0, 1, 30)), weight="cos", wvar=50.0)
q("qawo_rsqrt", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.cos(200 * x) / mp.sqrt(x), mp.linspace(0, 1, 80)), weight="cos", wvar=200.0)
q("qawo_rsqrt_negsin", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.sin(-200 * x) / mp.sqrt(x), mp.linspace(0, 1, 80)), weight="sin", wvar=-200.0)
q("qawo_lim", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0, mp.inf, weight="cos", wvar=200.0, limit=4)
q("qawo_tiny", math.sqrt, 0.0, 1.0,
  mp.quad(lambda x: mp.sqrt(x) * mp.cos(20 * x), mp.linspace(0, 1, 10)), weight="cos", wvar=20.0, epsabs=0.0, epsrel=1.2e-14)
q("qawo_maxp1", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.cos(200 * x) / mp.sqrt(x), mp.linspace(0, 1, 80)), weight="cos", wvar=200.0, maxp1=2)
q("qawf_limlst", lambda x: 1 / math.sqrt(x), 1.0, inf, mp.inf, weight="cos", wvar=1.0, limlst=3)
q("qawf_rsqrt", lambda x: 1 / math.sqrt(x), 1.0, inf,
  mp.quadosc(lambda x: mp.cos(x) / mp.sqrt(x), [1, mp.inf], omega=1), weight="cos", wvar=1.0)
q("qawf_shift", lambda x: 1 / math.sqrt(x - 1) if x > 1 else 0.0, 1.0, inf,
  mp.sqrt(mp.pi / 6) * (mp.cos(3) + mp.sin(3)), weight="sin", wvar=3.0)
q("qawf_omega0", lambda x: math.exp(-x), 0.0, inf, 1, weight="cos", wvar=0.0)
q("qawf_omega0_sin", lambda x: math.exp(-x), 0.0, inf, 0, weight="sin", wvar=0.0)
q("qawf_slow_omega", lambda x: math.exp(-x / 5), 0.0, inf, mp.mpf(1) / 5 / (mp.mpf(1) / 25 + mp.mpf("0.09")),
  weight="cos", wvar=0.3)
q("qawc_sqrt", lambda x: math.sqrt(x + 1), -1.0, 1.0,
  mp.quad(lambda x: (mp.sqrt(x + 1) - mp.sqrt(mp.mpf("1.2"))) / (x - mp.mpf("0.2")), [-1, mp.mpf("0.2"), 1])
  + mp.sqrt(mp.mpf("1.2")) * mp.log(mp.mpf("0.8") / mp.mpf("1.2")), weight="cauchy", wvar=0.2)
q("qawc_peak", lambda x: 1 / (1 + 100 * (x - 0.6) ** 2), -1.0, 1.0, mp.inf, weight="cauchy", wvar=0.1)
q("qawc_lim", lambda x: math.sqrt(x + 1), -1.0, 1.0, mp.inf, weight="cauchy", wvar=0.2, limit=3)
q("qawc_flip", lambda x: math.sqrt(x + 1), 1.0, -1.0, mp.inf, weight="cauchy", wvar=0.2)
q("qawc_tiny", lambda x: math.sqrt(x + 1), -1.0, 1.0, mp.inf, weight="cauchy", wvar=0.2, epsabs=0.0, epsrel=1.2e-14, limit=200)
q("qaws_cos30", lambda x: math.cos(30 * x), 0.0, 1.0,
  mp.quad(lambda x: mp.cos(30 * x) / mp.sqrt(x), mp.linspace(0, 1, 20)), weight="alg", wvar=(-0.5, 0.0))
q("qaws_lim", lambda x: math.cos(30 * x), 0.0, 1.0, mp.inf, weight="alg", wvar=(-0.5, 0.0), limit=3)
q("qaws_lim2", lambda x: math.cos(30 * x), 0.0, 1.0, mp.inf, weight="alg", wvar=(-0.5, 0.0), limit=2)
q("qaws_logb", lambda x: math.sin(20 * x), 0.0, 2.0,
  mp.quad(lambda x: mp.sin(20 * x) * x ** 0.3 * (2 - x) ** -0.5 * mp.log(2 - x), mp.linspace(0, 2, 30)),
  weight="alg-logb", wvar=(0.3, -0.5))
q("qaws_loga", lambda x: math.sin(20 * x), 0.0, 2.0,
  mp.quad(lambda x: mp.sin(20 * x) * x ** 0.3 * (2 - x) ** -0.5 * mp.log(x), mp.linspace(0, 2, 30)),
  weight="alg-loga", wvar=(0.3, -0.5))
q("qaws_tiny", lambda x: math.cos(30 * x), 0.0, 1.0, mp.inf, weight="alg", wvar=(-0.5, 0.0), epsabs=0.0, epsrel=1.2e-14, limit=200)
q("qaws_rough", lambda x: abs(x - 0.37) ** 0.5, 0.0, 1.0, mp.inf, weight="alg", wvar=(0.5, 0.5), epsabs=0.0, epsrel=1.2e-14, limit=200)

q("x_m2", lambda x: 1 / x ** 2 if x > 0 else 0.0, 0.0, 1.0, mp.inf, limit=200)
q("log_div", lambda x: (1 / x) * (1 / abs(math.log(x))) ** 1.01 if 0 < x < 1 else 0.0, 0.0, 1.0, mp.inf, limit=200)
q("inf_x_m09", lambda x: (x + 1) ** -0.9, 0.0, inf, mp.inf, limit=500)
q("inf_osc_slow", lambda x: math.sin(x) / (1 + x), 0.0, inf, mp.inf, limit=500)
q("inf_fresnel", lambda x: math.cos(x * x), -inf, inf, mp.sqrt(mp.pi / 2), limit=200)
q("qawo_div", lambda x: 1 / x if x > 0 else 0.0, 0.0, 1.0, mp.inf, weight="cos", wvar=30.0, limit=200)
q("qawo_x_m0999", lambda x: x ** -0.999 if x > 0 else 0.0, 0.0, 1.0, mp.inf, weight="cos", wvar=30.0, limit=200)
q("qawo_noise", lambda x: math.exp(x) + 1e-12 * math.sin(1e6 * x), 0.0, 1.0, mp.inf, weight="sin", wvar=30.0,
  epsabs=0.0, epsrel=1.2e-14, limit=200)
q("qawo_round", lambda x: round(math.exp(x) * 1e10) / 1e10, 0.0, 1.0, mp.inf, weight="cos", wvar=30.0,
  epsabs=0.0, epsrel=1e-13, limit=200)
q("qawo_small_w", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.cos(3 * x) / mp.sqrt(x), [0, 1]), weight="cos", wvar=3.0, limit=200)
q("qawo_small_w_log", lambda x: math.log(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.cos(x) * mp.log(x), [0, 1]), weight="cos", wvar=1.0, limit=200)
q("qawo_mid_w", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0,
  mp.quad(lambda x: mp.cos(6 * x) / mp.sqrt(x), [0, 1]), weight="cos", wvar=6.0, limit=200)
q("qawf_tight", lambda x: math.exp(-x), 0.0, inf, mp.mpf(1) / 101, weight="cos", wvar=10.0, epsabs=1e-15)
q("qawf_tight_lorentz", lambda x: 1 / (1 + x * x), 0.0, inf, mp.pi / 2 * mp.exp(-10), weight="cos", wvar=10.0, epsabs=1e-15)
q("qawf_loose", lambda x: math.exp(-x), 0.0, inf, mp.mpf(1) / 101, weight="cos", wvar=10.0, epsabs=1e-3)
q("qawf_linear", lambda x: x, 1.0, inf, mp.inf, weight="cos", wvar=2.0)
q("qawc_tough", lambda x: 1 / math.sqrt(1 - x * x) if abs(x) < 1 else 0.0, -1.0, 1.0, 0, weight="cauchy", wvar=0.5, limit=200)
q("qawc_rough", lambda x: abs(x - 0.7) ** 0.1, -1.0, 1.0, mp.inf, weight="cauchy", wvar=0.2, limit=500, epsabs=0.0, epsrel=1e-13)
q("qaws_alglog_cos", lambda x: math.cos(30 * x), 0.0, 1.0,
  mp.quad(lambda x: mp.cos(30 * x) * x ** -0.5 * (1 - x) ** 0.2 * mp.log(x) * mp.log(1 - x), mp.linspace(0, 1, 20)),
  weight="alg-log", wvar=(-0.5, 0.2))
q("qaws_inner_sing", lambda x: abs(x - 0.37) ** -0.5 if x != 0.37 else 0.0, 0.0, 1.0, mp.inf, weight="alg", wvar=(0.5, 0.5), limit=200)


# Configurations found by a branch search over integrand families: each one
# drives a rarely taken path of the QUADPACK control flow.


def powabs(c, p, noise=0.0):
    def f(x):
        if x == c:
            return 0.0
        return abs(x - c) ** p * (1 + noise * math.sin(1e6 * x))
    return f


q("br_x_m12_rel", powabs(0, -1.2), 0.0, 1.0, mp.inf, epsabs=0.0, epsrel=1.2e-14, limit=50)
q("br_x_m12_abs", powabs(0, -1.2), 0.0, 1.0, mp.inf, epsabs=1e-14, epsrel=0.0, limit=50)
q("br_sqrt_noise", powabs(0, 0.5, 1e-6), 0.0, 1.0, mp.inf, epsabs=0.0, epsrel=1e-12, limit=50)
q("br_round_lim1", lambda x: round(math.exp(x) * 1e10) / 1e10 + (1.0 if x > 0 else 0.0), 0.0, 1.0, mp.inf,
  epsabs=1e-14, epsrel=0.0, limit=1)
q("br_lim1", powabs(0, -1.2), 0.0, 1.0, mp.inf, limit=1)
q("br_sin_ier2", math.sin, 0.0, 2 * math.pi, 0, epsabs=1e-300, epsrel=1.2e-14)
q("br_pts_x_m12_12", powabs(0, -1.2), 0.0, 1.0, mp.inf, points=[0.3, 0.5], epsabs=0.0, epsrel=1.2e-14, limit=12)
q("br_pts_x_m12_50", powabs(0, -1.2), 0.0, 1.0, mp.inf, points=[0.3, 0.5], epsabs=0.0, epsrel=1.2e-14, limit=50)
q("br_pts_x_m12_noise", powabs(0, -1.2, 1e-13), 0.0, 1.0, mp.inf, points=[0.3, 0.5], epsabs=0.0, epsrel=1.2e-14, limit=50)
q("br_pts_p01_noise", powabs(0, 0.1, 1e-6), 0.0, 1.0, mp.inf, points=[0.3, 0.5], epsabs=0.0, epsrel=1e-12, limit=50)
q("br_pts_p01_loose", powabs(0, 0.1), 0.0, 1.0, mp.mpf(1) / mp.mpf("1.1"), points=[0.3, 0.5], epsabs=0.0, epsrel=1e-3, limit=12)
q("br_pts_lim3", powabs(0, -1.2), 0.0, 1.0, mp.inf, points=[0.3, 0.5], limit=3)
q("br_pts_sin_ier2", math.sin, 0.0, 2 * math.pi, 0, points=[math.pi], epsabs=1e-300, epsrel=1.2e-14)
q("br_qawo_ier2", powabs(0, -1.0), 0.0, 1.0, mp.inf, weight="sin", wvar=1.0, epsabs=1e-14, epsrel=0.0, limit=1)
q("br_qawo_lim1", powabs(0, -1.2), 0.0, 1.0, mp.inf, weight="cos", wvar=1.0, limit=1)
q("br_qawo_small", powabs(0.3, -1.0), 0.0, 1.0, mp.inf, weight="cos", wvar=1.0, limit=200)
q("br_qawo_jup", powabs(0, -1.2), 0.0, 1.0, mp.inf, weight="cos", wvar=1.0, epsabs=0.0, epsrel=1e-12, limit=12)
q("br_qawo_div", powabs(0, -1.2), 0.0, 1.0, mp.inf, weight="cos", wvar=1.0, epsabs=0.0, epsrel=1.2e-14, limit=50)
q("br_qawo_x_m105", powabs(0, -1.05), 0.0, 1.0, mp.inf, weight="cos", wvar=1.0)
q("br_qawo_ext", lambda x: math.sin(1 / x) / x if x > 0 else 0.0, 0.0, 1.0, mp.inf, weight="cos", wvar=3.0,
  epsabs=0.0, epsrel=1e-13, limit=50)
q("br_qawf_limexp", lambda x: (1 + abs(x)) ** -0.5, 0.0, inf, mp.inf, weight="cos", wvar=1.0, epsabs=1e-15, limit=1, limlst=50)
q("br_qawf_ier7", lambda x: (1 + abs(x)) ** -0.9, 0.0, inf, mp.inf, weight="cos", wvar=10.0, epsabs=1e-3, limit=1, limlst=10)
q("br_qawf_ier7b", lambda x: math.sin(x) * (1 + abs(x)) ** -0.5, 0.0, inf, mp.inf, weight="sin", wvar=0.3, epsabs=1e-3, limit=1, limlst=50)
q("br_qawf_ier4", lambda x: (1 + abs(x)) ** -0.5, 0.0, inf, mp.inf, weight="cos", wvar=0.3, epsabs=1e-15, limit=1, limlst=50)
q("br_qawf_drl", lambda x: (1 + abs(x)) ** -0.5, 0.0, inf, mp.inf, weight="cos", wvar=0.3, epsabs=1e-8, limit=1, limlst=10)
q("br_qawc_lim1", lambda x: math.sqrt(x + 1), -1.0, 1.0, mp.inf, weight="cauchy", wvar=0.2, limit=1)


def run_quad():
    out = []
    for name, (fn, a, b, kw, ref) in QUAD.items():
        neval = [0]

        def g(x, fn=fn):
            neval[0] += 1
            return fn(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = integrate.quad(g, a, b, full_output=1, **kw)
        val, err, info = r[0], r[1], r[2]
        ier = 0
        if len(r) > 3:
            msg = r[3]
            ier = {"maximum number of subdivisions": 1, "roundoff error is detected": 2,
                   "Extremely bad integrand": 3, "does not converge": 4,
                   "divergent": 5, "maximum number of cycles": 1, "Bad integrand behavior": 7,
                   "extrapolation table constructed": 4}
            code = -1
            for k, v in ier.items():
                if k in msg:
                    code = v
                    break
            ier = code
        rec = {"name": name, "a": jf(a), "b": jf(b), "value": val, "abserr": err,
               "neval": int(info.get("neval", neval[0])) if isinstance(info, dict) else neval[0],
               "calls": neval[0], "last": int(info.get("last", 0)) if isinstance(info, dict) else 0,
               "ier": ier, "ref": None if ref == mp.inf else f64(ref)}
        out.append(rec)
    return out


def run_multi():
    out = []
    r = integrate.dblquad(lambda y, x: x * y * y, 0, 2, 0, 1)
    out.append({"name": "dbl_rect", "value": r[0], "abserr": r[1], "ref": 2 / 3})
    r = integrate.dblquad(lambda y, x: math.exp(-x * y), 0, 1, lambda x: 0, lambda x: x)
    ref = mp.quad(lambda x: mp.quad(lambda y: mp.exp(-x * y), [0, x]), [0, 1])
    out.append({"name": "dbl_tri", "value": r[0], "abserr": r[1], "ref": f64(ref)})
    r = integrate.tplquad(lambda z, y, x: x * y * z, 1, 2, 2, 3, 0, 1)
    out.append({"name": "tpl_box", "value": r[0], "abserr": r[1], "ref": 1.875})
    r = integrate.tplquad(lambda z, y, x: 1.0, 0, 1, 0, lambda x: 1 - x, 0, lambda x, y: 1 - x - y)
    out.append({"name": "tpl_simplex", "value": r[0], "abserr": r[1], "ref": 1 / 6})
    r = integrate.nquad(lambda x0, x1, x2, x3: x0 * x1 * x2 * x3, [[0, 1]] * 4)
    out.append({"name": "nquad_4", "value": r[0], "abserr": r[1], "ref": 1 / 16})
    return out


def run_fixed():
    out = {}
    x = np.linspace(0, 2, 9)
    xu = np.array([0.0, 0.1, 0.35, 0.5, 0.9, 1.3, 1.31, 2.0])
    y = np.exp(x)
    yu = np.exp(xu)
    out["trapezoid"] = integrate.trapezoid(y, x)
    out["trapezoid_dx"] = integrate.trapezoid(y, dx=0.25)
    out["simpson_odd"] = integrate.simpson(y, x=x)
    out["simpson_odd_dx"] = integrate.simpson(y, dx=0.25)
    out["simpson_even"] = integrate.simpson(yu, x=xu)
    out["simpson_even_dx"] = integrate.simpson(y[:8], dx=0.25)
    out["simpson_two"] = integrate.simpson(y[:2], x=x[:2])
    out["cumtrapz"] = integrate.cumulative_trapezoid(yu, xu).tolist()
    out["cumtrapz0"] = integrate.cumulative_trapezoid(yu, xu, initial=0).tolist()
    out["romb"] = integrate.romb(np.exp(np.linspace(0, 1, 17)), dx=1 / 16)
    out["fixed_quad5"] = integrate.fixed_quad(np.cos, 0, 1, n=5)[0]
    out["fixed_quad20"] = integrate.fixed_quad(np.cos, 0, 1, n=20)[0]
    xs, ws = scipy.special.roots_legendre(7)
    out["legendre7_x"] = xs.tolist()
    out["legendre7_w"] = ws.tolist()
    return out


# ---------------------------------------------------------------- ODEs
MU1 = 1.0
MU1000 = 1000.0


def decay(t, y):
    return -0.5 * y


def harmonic(t, y):
    return [y[1], -y[0]]


def lotka(t, y):
    return [1.5 * y[0] - y[0] * y[1], -3 * y[1] + y[0] * y[1]]


def vdp(mu):
    def f(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]
    return f


def vdp_jac(mu):
    def j(t, y):
        return [[0, 1], [-2 * mu * y[0] * y[1] - 1, mu * (1 - y[0] ** 2)]]
    return j


def robertson(t, y):
    return [-0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2]


def robertson_jac(t, y):
    return [[-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0, 6e7 * y[1], 0]]


def brusselator(t, y):
    a, b = 1.0, 3.0
    return [a + y[0] ** 2 * y[1] - (b + 1) * y[0], b * y[0] - y[0] ** 2 * y[1]]


ARENSTORF_MU = 0.012277471


def arenstorf(t, y):
    mu = ARENSTORF_MU
    mup = 1 - mu
    d1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mup) ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3],
            y[0] + 2 * y[3] - mup * (y[0] + mu) / d1 - mu * (y[0] - mup) / d2,
            y[1] - 2 * y[2] - mup * y[1] / d1 - mu * y[1] / d2]


def hires(t, y):
    return [-1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007,
            1.71 * y[0] - 8.75 * y[1],
            -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
            8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
            -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
            -280 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
            280 * y[5] * y[7] - 1.81 * y[6],
            -280 * y[5] * y[7] + 1.81 * y[6]]


def orego(t, y):
    s, w, qq = 77.27, 0.161, 8.375e-6
    return [s * (y[1] - y[0] * y[1] + y[0] - qq * y[0] ** 2),
            (-y[1] - y[0] * y[1] + y[2]) / s,
            w * (y[0] - y[2])]


ARENSTORF_T = 17.0652165601579625588917206249
ODE = {
    # name: (f, jac, span, y0, exact(t)->y or None, stiff)
    "decay": (decay, None, (0, 10), [2.0, 4.0, 8.0], lambda t: [2 * math.exp(-0.5 * t), 4 * math.exp(-0.5 * t), 8 * math.exp(-0.5 * t)], False),
    "harmonic": (harmonic, None, (0, 10 * math.pi), [1.0, 0.0], lambda t: [math.cos(t), -math.sin(t)], False),
    "lotka": (lotka, None, (0, 15), [10.0, 5.0], None, False),
    "vdp1": (vdp(MU1), vdp_jac(MU1), (0, 20), [2.0, 0.0], None, False),
    "brusselator": (brusselator, None, (0, 20), [1.5, 3.0], None, False),
    "arenstorf": (arenstorf, None, (0, ARENSTORF_T), [0.994, 0.0, 0.0, -2.00158510637908252240537862224], None, False),
    "vdp1000": (vdp(MU1000), vdp_jac(MU1000), (0, 3000), [2.0, 0.0], None, True),
    "robertson": (robertson, robertson_jac, (0, 1e5), [1.0, 0.0, 0.0], None, True),
    "hires": (hires, None, (0, 321.8122), [1, 0, 0, 0, 0, 0, 0, 0.0057], None, True),
    "orego": (orego, None, (0, 360), [1.0, 2.0, 3.0], None, True),
}

NONSTIFF = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
STIFF = ["Radau", "BDF", "LSODA"]


def reference(name):
    f, jac, span, y0, exact, stiff = ODE[name]
    if exact is not None:
        return exact(span[1])
    if stiff:
        r = integrate.solve_ivp(f, span, y0, method="Radau", rtol=1e-12, atol=1e-14 if name != "robertson" else 1e-16, jac=jac)
    else:
        r = integrate.solve_ivp(f, span, y0, method="DOP853", rtol=1e-13, atol=1e-15)
    assert r.success, name
    return r.y[:, -1].tolist()


def run_ode():
    out = []
    for name, (f, jac, span, y0, exact, stiff) in ODE.items():
        ref = reference(name)
        methods = STIFF if stiff else NONSTIFF
        tols = [(1e-3, 1e-6), (1e-6, 1e-9)]
        if not stiff:
            tols.append((1e-9, 1e-12))
        if name == "robertson":
            tols = [(1e-4, 1e-8), (1e-7, 1e-10)]
        for m in methods:
            for rtol, atol in tols:
                for usejac in ([False, True] if jac is not None and m in ("Radau", "BDF", "LSODA") else [False]):
                    kw = {}
                    if usejac:
                        kw["jac"] = jac
                    r = integrate.solve_ivp(f, span, y0, method=m, rtol=rtol, atol=atol, **kw)
                    rec = {"problem": name, "method": m, "rtol": rtol, "atol": atol, "jac": usejac,
                           "status": int(r.status), "nfev": int(r.nfev), "njev": int(r.njev), "nlu": int(r.nlu),
                           "nsteps": int(len(r.t) - 1), "y_end": r.y[:, -1].tolist(), "ref": ref,
                           "t_first": r.t[:6].tolist()}
                    out.append(rec)
    return out


def run_ode_features():
    out = {}
    # t_eval + dense output on Lotka-Volterra (RK45 defaults).
    te = np.linspace(0, 15, 31)
    r = integrate.solve_ivp(lotka, (0, 15), [10, 5], t_eval=te, dense_output=True)
    out["teval_y"] = r.y.T.tolist()
    out["teval_sol"] = r.sol(np.array([0.3, 7.77, 14.9])).T.tolist()
    # Backwards integration with t_eval.
    r = integrate.solve_ivp(decay, (10, 0), [1.0], t_eval=[10, 7.5, 5, 2.5, 0], method="DOP853", rtol=1e-8, atol=1e-10)
    out["backward_y"] = r.y[0].tolist()
    out["backward_nfev"] = int(r.nfev)

    # Events: a falling ball (terminal) with an apex event (direction -1).
    def ball(t, y):
        return [y[1], -9.81]

    def hit(t, y):
        return y[0]
    hit.terminal = True
    hit.direction = -1

    def apex(t, y):
        return y[1]
    for m in ["RK45", "DOP853", "Radau", "BDF", "LSODA", "RK23"]:
        r = integrate.solve_ivp(ball, (0, 10), [10.0, 5.0], events=[hit, apex], method=m, rtol=1e-6, atol=1e-9)
        out["ball_" + m] = {"t_end": float(r.t[-1]), "t_events": [list(map(float, e)) for e in r.t_events],
                            "y_hit": r.y_events[0].tolist(), "status": int(r.status), "nfev": int(r.nfev)}

    # Counting event with terminal = 3 on the harmonic oscillator.
    def zero(t, y):
        return y[0]
    zero.terminal = 3
    r = integrate.solve_ivp(harmonic, (0, 20), [1, 0], events=zero, rtol=1e-8, atol=1e-10)
    out["harmonic_zero3"] = {"t_events": list(map(float, r.t_events[0])), "t_end": float(r.t[-1]), "status": int(r.status)}

    # odeint (LSODA in ODEPACK) on Robertson and Lotka-Volterra.
    ts = np.array([0, 0.4, 4, 40, 400, 4000, 40000])
    y, info = integrate.odeint(robertson, [1, 0, 0], ts, full_output=True, tfirst=True, rtol=1e-8, atol=1e-12)
    out["odeint_robertson"] = {"t": ts.tolist(), "y": y.tolist(), "nfe": int(info["nfe"][-1]), "nje": int(info["nje"][-1]),
                               "nst": int(info["nst"][-1]), "mused": info["mused"].tolist()}
    ts = np.linspace(0, 15, 16)
    y, info = integrate.odeint(lotka, [10, 5], ts, full_output=True, tfirst=True)
    out["odeint_lotka"] = {"t": ts.tolist(), "y": y.tolist(), "nfe": int(info["nfe"][-1]), "nst": int(info["nst"][-1]),
                           "mused": info["mused"].tolist()}
    ref = integrate.solve_ivp(lotka, (0, 15), [10, 5], method="DOP853", rtol=1e-13, atol=1e-15, t_eval=ts)
    out["odeint_lotka_ref"] = ref.y.T.tolist()
    ref = integrate.solve_ivp(robertson, (0, 40000), [1, 0, 0], method="Radau", rtol=1e-12, atol=1e-16,
                              jac=robertson_jac, t_eval=[0, 0.4, 4, 40, 400, 4000, 40000])
    out["odeint_robertson_ref"] = ref.y.T.tolist()
    return out


def main():
    meta = {"scipy": scipy.__version__, "numpy": np.__version__, "mpmath": mp.__version__,
            "python": sys.version.split()[0]}
    with open(os.path.join(OUT, "quad.json"), "w") as fh:
        json.dump({"meta": meta, "quad": run_quad(), "multi": run_multi(), "fixed": run_fixed()}, fh, indent=1)
    with open(os.path.join(OUT, "ode.json"), "w") as fh:
        json.dump({"meta": meta, "ode": run_ode(), "features": run_ode_features()}, fh, indent=1)


if __name__ == "__main__":
    main()
