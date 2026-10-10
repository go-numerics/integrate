// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Default tolerances and limits of Quad, the same as SciPy's quad.
const (
	DefaultEpsAbs = 1.49e-8
	DefaultEpsRel = 1.49e-8
	DefaultLimit  = 50
	DefaultMaxP1  = 50
	DefaultLimLst = 50
)

// Weight selects a weight function w(x) for Quad, which then integrates
// f(x)·w(x) with a QUADPACK routine specialised for that weight. It is SciPy's
// quad(..., weight=...) argument; the SciPy wvar parameter is spread over the
// typed fields of QuadOptions (Omega, Alpha and Beta, C).
type Weight int

const (
	// WeightNone integrates f itself (QAGS, QAGI, or QAGP with Points).
	WeightNone Weight = iota
	// WeightCos is cos(Omega·x): QAWO on a finite range, QAWF on
	// [a, +inf) or (-inf, b]. SciPy weight="cos", wvar=omega.
	WeightCos
	// WeightSin is sin(Omega·x). SciPy weight="sin", wvar=omega.
	WeightSin
	// WeightAlg is (x-a)^Alpha·(b-x)^Beta (QAWS). SciPy weight="alg",
	// wvar=(alpha, beta).
	WeightAlg
	// WeightAlgLogA is (x-a)^Alpha·(b-x)^Beta·log(x-a). SciPy "alg-loga".
	WeightAlgLogA
	// WeightAlgLogB is (x-a)^Alpha·(b-x)^Beta·log(b-x). SciPy "alg-logb".
	WeightAlgLogB
	// WeightAlgLog is (x-a)^Alpha·(b-x)^Beta·log(x-a)·log(b-x). SciPy
	// "alg-log".
	WeightAlgLog
	// WeightCauchy is 1/(x-C): the Cauchy principal value (QAWC). SciPy
	// weight="cauchy", wvar=c.
	WeightCauchy
)

// QuadOptions tunes Quad. The zero value (or a nil pointer) gives SciPy's
// defaults.
type QuadOptions struct {
	// EpsAbs and EpsRel are the absolute and relative error targets: Quad
	// tries to make |Value - I| ≤ max(EpsAbs, EpsRel·|I|). Zero selects
	// 1.49e-8. A negative value switches that criterion off, which is
	// SciPy's epsabs=0 (or epsrel=0).
	EpsAbs, EpsRel float64
	// Limit bounds the number of subintervals (default 50).
	Limit int
	// Points are interior break points where the integrand is known to be
	// difficult (discontinuities, kinks, integrable singularities). Only
	// for finite ranges without a weight (QAGP). Points outside (a, b) are
	// ignored, as SciPy does.
	Points []float64
	// Weight selects a weight function; see Weight.
	Weight Weight
	// Omega is the angular frequency of WeightCos and WeightSin.
	Omega float64
	// Alpha and Beta are the exponents of the algebraic weights (> -1).
	Alpha, Beta float64
	// C is the pole of WeightCauchy, strictly inside (a, b).
	C float64
	// MaxP1 bounds the number of Chebyshev moment sets kept by QAWO
	// (default 50).
	MaxP1 int
	// LimLst bounds the number of cycles of QAWF (default 50, at least 3).
	LimLst int
}

// QuadResult is the outcome of a quadrature.
type QuadResult struct {
	// Value is the estimate of the integral.
	Value float64
	// AbsErr is the estimated absolute error.
	AbsErr float64
	// NEval is the number of integrand evaluations.
	NEval int
	// NIntervals is the number of subintervals the adaptive routine used
	// (QUADPACK's last); for QAWF it is the number of cycles.
	NIntervals int
	// CycleIer holds the QUADPACK ier of each QAWF cycle (SciPy's
	// infodict["ierlst"]); nil for other routines.
	CycleIer []int
}

// Sentinel errors, one per QUADPACK diagnostic. A *QuadError wraps one of
// them, so errors.Is(err, integrate.ErrRoundoff) works.
var (
	// ErrLimit: the maximum number of subdivisions (or QAWF cycles) was
	// reached (ier 1).
	ErrLimit = errors.New("integrate: maximum number of subdivisions reached")
	// ErrRoundoff: roundoff error prevents the requested tolerance from
	// being achieved (ier 2).
	ErrRoundoff = errors.New("integrate: roundoff error detected")
	// ErrBadIntegrand: extremely bad integrand behaviour at some points
	// (ier 3).
	ErrBadIntegrand = errors.New("integrate: extremely bad integrand behaviour")
	// ErrExtrapolation: the extrapolation does not converge because of
	// roundoff (ier 4).
	ErrExtrapolation = errors.New("integrate: extrapolation does not converge")
	// ErrDivergent: the integral is probably divergent or slowly
	// convergent (ier 5).
	ErrDivergent = errors.New("integrate: integral is probably divergent")
	// ErrInvalidInput: the input is invalid (ier 6); no estimate is
	// returned.
	ErrInvalidInput = errors.New("integrate: invalid input")
	// ErrCycles: abnormal termination in one or more QAWF cycles (ier 7).
	ErrCycles = errors.New("integrate: bad integrand behaviour in one or more cycles")
)

var ierErrors = [8]error{nil, ErrLimit, ErrRoundoff, ErrBadIntegrand, ErrExtrapolation, ErrDivergent, ErrInvalidInput, ErrCycles}

// QuadError reports a QUADPACK diagnostic. For ier 1–5 and 7 the QuadResult
// returned alongside still carries the best estimate available, as SciPy
// returns it with an IntegrationWarning.
type QuadError struct {
	// Ier is QUADPACK's error code.
	Ier int
	// Detail says which routine or which argument is concerned.
	Detail string
}

func (e *QuadError) Error() string {
	if e.Detail == "" {
		return ierErrors[e.Ier].Error()
	}
	return ierErrors[e.Ier].Error() + ": " + e.Detail
}

// Unwrap returns the sentinel error of the diagnostic.
func (e *QuadError) Unwrap() error { return ierErrors[e.Ier] }

func invalid(format string, args ...any) error {
	return &QuadError{Ier: 6, Detail: fmt.Sprintf(format, args...)}
}

func orDefault(v, def float64) float64 {
	switch {
	case v == 0:
		return def
	case v < 0:
		return 0
	}
	return v
}

func intOr(v, def int) int {
	if v == 0 {
		return def
	}
	return v
}

// Quad computes the definite integral of f from a to b, where a and b may be
// infinite. It is SciPy's scipy.integrate.quad:
//
//	quad(f, a, b)                                   → Quad(f, a, b, nil)
//	quad(f, a, b, epsabs=0, epsrel=1e-12, limit=200) → Quad(f, a, b, &QuadOptions{EpsAbs: -1, EpsRel: 1e-12, Limit: 200})
//	quad(f, a, b, points=[c])                       → Quad(f, a, b, &QuadOptions{Points: []float64{c}})
//	quad(f, a, b, weight="cos", wvar=w)             → Quad(f, a, b, &QuadOptions{Weight: WeightCos, Omega: w})
//	quad(f, a, b, weight="alg", wvar=(al, be))      → Quad(f, a, b, &QuadOptions{Weight: WeightAlg, Alpha: al, Beta: be})
//	quad(f, a, b, weight="cauchy", wvar=c)          → Quad(f, a, b, &QuadOptions{Weight: WeightCauchy, C: c})
//
// The routine is chosen as SciPy chooses it: QAGS on finite ranges, QAGI on
// infinite ones, QAGP with break points, QAWO/QAWF/QAWS/QAWC with a weight.
// When QUADPACK reports a problem the result is returned together with a
// *QuadError; for invalid input (ier 6) the result is zero.
func Quad(f func(float64) float64, a, b float64, opts *QuadOptions) (QuadResult, error) {
	var o QuadOptions
	if opts != nil {
		o = *opts
	}
	epsabs := orDefault(o.EpsAbs, DefaultEpsAbs)
	epsrel := orDefault(o.EpsRel, DefaultEpsRel)
	limit := intOr(o.Limit, DefaultLimit)
	if limit < 1 {
		return QuadResult{}, invalid("Limit must be at least 1")
	}
	if a == b {
		return QuadResult{}, nil
	}
	flip := b < a
	if flip {
		a, b = b, a
	}
	var r qout
	var cycles []int
	var err error
	if o.Weight == WeightNone {
		r, err = quadPlain(f, a, b, epsabs, epsrel, limit, o.Points)
	} else {
		r, cycles, err = quadWeighted(f, a, b, epsabs, epsrel, limit, &o)
	}
	if err != nil {
		return QuadResult{}, err
	}
	if r.ier == 6 {
		return QuadResult{}, invalid("tolerances: with EpsAbs off, EpsRel must exceed max(50·eps, 5e-29)")
	}
	if flip {
		r.result = -r.result
	}
	res := QuadResult{Value: r.result, AbsErr: r.abserr, NEval: r.neval, NIntervals: r.last, CycleIer: cycles}
	if r.ier != 0 {
		return res, &QuadError{Ier: r.ier}
	}
	return res, nil
}

func quadPlain(f func(float64) float64, a, b, epsabs, epsrel float64, limit int, points []float64) (qout, error) {
	ainf, binf := math.IsInf(a, -1), math.IsInf(b, 1)
	if points != nil {
		if ainf || binf {
			return qout{}, invalid("break points cannot be used on an infinite range")
		}
		pts := uniqueInside(points, a, b)
		if len(pts) >= limit {
			return qout{}, invalid("number of break points (%d) must be less than Limit (%d)", len(pts), limit)
		}
		return qagpe(f, a, b, pts, epsabs, epsrel, limit), nil
	}
	switch {
	case ainf && binf:
		return qagie(f, 0, 2, epsabs, epsrel, limit), nil
	case binf:
		return qagie(f, a, 1, epsabs, epsrel, limit), nil
	case ainf:
		return qagie(f, b, -1, epsabs, epsrel, limit), nil
	}
	return qagse(f, a, b, epsabs, epsrel, limit), nil
}

// uniqueInside returns the sorted distinct points strictly inside (a, b).
func uniqueInside(points []float64, a, b float64) []float64 {
	s := append([]float64(nil), points...)
	sort.Float64s(s)
	out := s[:0]
	for i, p := range s {
		if p > a && p < b && (i == 0 || p != s[i-1]) {
			out = append(out, p)
		}
	}
	return out
}

func quadWeighted(f func(float64) float64, a, b, epsabs, epsrel float64, limit int, o *QuadOptions) (qout, []int, error) {
	if o.Points != nil {
		return qout{}, nil, invalid("break points cannot be combined with a weight")
	}
	ainf, binf := math.IsInf(a, -1), math.IsInf(b, 1)
	switch o.Weight {
	case WeightCos, WeightSin:
		integr := 1
		if o.Weight == WeightSin {
			integr = 2
		}
		maxp1 := intOr(o.MaxP1, DefaultMaxP1)
		if maxp1 < 1 {
			return qout{}, nil, invalid("MaxP1 must be at least 1")
		}
		limlst := intOr(o.LimLst, DefaultLimLst)
		switch {
		case ainf && binf:
			return qout{}, nil, invalid("cannot integrate with a cos/sin weight from -inf to +inf")
		case binf:
			r, cyc := qawfe(f, a, o.Omega, integr, epsabs, limlst, limit, maxp1)
			if r.ier == 6 {
				return qout{}, nil, invalid("Fourier integrals need EpsAbs > 0 and LimLst ≥ 3")
			}
			return r, cyc, nil
		case ainf:
			// ∫_{-∞}^{b} f(x) w(ωx) dx = ∫_{-b}^{∞} f(-y) w(-ωy) dy.
			g := func(y float64) float64 { return f(-y) }
			if integr == 2 {
				g = func(y float64) float64 { return -f(-y) }
			}
			r, cyc := qawfe(g, -b, o.Omega, integr, epsabs, limlst, limit, maxp1)
			if r.ier == 6 {
				return qout{}, nil, invalid("Fourier integrals need EpsAbs > 0 and LimLst ≥ 3")
			}
			return r, cyc, nil
		}
		return qawoe(f, a, b, o.Omega, integr, epsabs, epsrel, limit, 1, newMomentCache(maxp1)), nil, nil
	case WeightAlg, WeightAlgLogA, WeightAlgLogB, WeightAlgLog:
		if ainf || binf {
			return qout{}, nil, invalid("cannot integrate with an algebraic weight over an infinite range")
		}
		if o.Alpha <= -1 || o.Beta <= -1 {
			return qout{}, nil, invalid("Alpha and Beta must both be > -1")
		}
		if limit < 2 {
			return qout{}, nil, invalid("Limit must be at least 2 for algebraic weights")
		}
		integr := int(o.Weight-WeightAlg) + 1
		return qawse(f, a, b, o.Alpha, o.Beta, integr, epsabs, epsrel, limit), nil, nil
	case WeightCauchy:
		if ainf || binf {
			return qout{}, nil, invalid("cannot integrate with the Cauchy weight over an infinite range")
		}
		if o.C == a || o.C == b {
			return qout{}, nil, invalid("the Cauchy pole C must not equal a or b")
		}
		return qawce(f, a, b, o.C, epsabs, epsrel, limit), nil, nil
	}
	return qout{}, nil, invalid("unknown weight %d", o.Weight)
}
