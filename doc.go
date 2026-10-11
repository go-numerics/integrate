// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Package integrate is numerical integration and ODE solving in pure Go
// (CGO_ENABLED=0): the functionality of scipy.integrate, with the same
// algorithms, tolerances and diagnostics, so that a notebook written against
// SciPy finds the same answers here.
//
// Quadrature:
//
//   - Quad: QUADPACK's adaptive Gauss–Kronrod drivers, ported from the
//     netlib Fortran: QAGS (21-point rule with Wynn's epsilon
//     extrapolation), QAGI (infinite ranges), QAGP (break points), QAWO and
//     QAWF (cos/sin weights, Fourier integrals), QAWC (Cauchy principal
//     values) and QAWS (algebraic-logarithmic end-point singularities).
//     QUADPACK's ier diagnostics are returned as *QuadError values that wrap
//     ErrLimit, ErrRoundoff, ErrBadIntegrand, ErrExtrapolation, ErrDivergent,
//     ErrInvalidInput and ErrCycles.
//   - DblQuad, TplQuad and NQuad: iterated adaptive quadrature over regions
//     whose inner limits may depend on the outer variables.
//   - FixedQuad (Gauss–Legendre), Romberg, Romb, Simpson, Trapezoid and
//     CumulativeTrapezoid on functions or samples, with ndarray variants
//     integrating along an axis (TrapezoidND, SimpsonND,
//     CumulativeTrapezoidND).
//
// Initial value problems:
//
//   - SolveIVP with RK45 (Dormand–Prince), RK23 (Bogacki–Shampine), DOP853
//     (Hairer), Radau (implicit Radau IIA, order 5), BDF (variable order 1–5,
//     NDF variant) and LSODA (automatic switching between Runge–Kutta and
//     BDF). Events with direction and terminal counts, output at chosen times
//     (TEval), continuous dense output, scalar or per-component tolerances,
//     MaxStep, FirstStep and optional Jacobians (finite differences
//     otherwise).
//   - Odeint: SciPy's classic interface over the switching method.
//
// Right-hand sides are func(t float64, y, dydt []float64), writing the
// derivative into dydt so that a step allocates nothing; SciPy's
// fun(t, y) -> dydt maps one to one. Each exported function documents the
// SciPy call it corresponds to. docs/design.md describes the algorithms and
// their sources, docs/verification.md how they were checked against SciPy,
// the netlib Fortran and mpmath, docs/perf.md the timings.
package integrate
