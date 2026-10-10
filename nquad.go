// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "errors"

// Range gives the integration limits of one variable of NQuad as a function
// of the outer variables: outer[0] is the variable integrated just outside
// this one, outer[1] the next, and so on (the arguments SciPy's nquad passes
// to a callable range).
type Range func(outer []float64) (lo, hi float64)

// ConstRange is a Range with fixed limits.
func ConstRange(lo, hi float64) Range {
	return func([]float64) (float64, float64) { return lo, hi }
}

// Const returns the function x ↦ v, for constant limits of DblQuad and
// TplQuad.
func Const(v float64) func(float64) float64 {
	return func(float64) float64 { return v }
}

// Const2 returns the function (x, y) ↦ v, for constant inner limits of
// TplQuad.
func Const2(v float64) func(float64, float64) float64 {
	return func(float64, float64) float64 { return v }
}

// NQuadOptions tunes NQuad.
type NQuadOptions struct {
	// Opts gives the QuadOptions of each level, innermost first (index i
	// applies to x[i]). A single entry applies to every level; nil uses
	// the defaults. Points of a level are interpreted in that level's
	// variable.
	Opts []*QuadOptions
}

// NQuad integrates f over a hyper-rectangle or a more general region whose
// inner limits depend on the outer variables. ranges[i] gives the limits of
// x[i]; x[0] is integrated innermost. It is SciPy's
// scipy.integrate.nquad(func, ranges, opts=opts):
//
//	nquad(f, [[0, 1], [0, 2]])              → NQuad(f, []Range{ConstRange(0, 1), ConstRange(0, 2)}, nil)
//	nquad(f, [lambda x1: (0, x1), [0, 1]])  → NQuad(f, []Range{func(o []float64) (float64, float64) { return 0, o[0] }, ConstRange(0, 1)}, nil)
//
// f receives x with len(ranges) entries (SciPy passes them as separate
// arguments). AbsErr is the largest error estimate of any one-dimensional
// quadrature, as in SciPy; NEval counts the evaluations of f. The first
// QUADPACK diagnostic met, if any, is returned with the result.
func NQuad(f func(x []float64) float64, ranges []Range, opts *NQuadOptions) (QuadResult, error) {
	n := len(ranges)
	if n == 0 {
		return QuadResult{}, invalid("NQuad needs at least one range")
	}
	var o NQuadOptions
	if opts != nil {
		o = *opts
	}
	if len(o.Opts) > 1 && len(o.Opts) != n {
		return QuadResult{}, invalid("NQuad: %d option sets for %d ranges", len(o.Opts), n)
	}
	levelOpts := func(i int) *QuadOptions {
		switch len(o.Opts) {
		case 0:
			return nil
		case 1:
			return o.Opts[0]
		}
		return o.Opts[i]
	}
	x := make([]float64, n)
	var res QuadResult
	var firstErr error
	var integrate func(level int) float64
	integrate = func(level int) float64 {
		lo, hi := ranges[level](x[level+1:])
		g := func(t float64) float64 {
			x[level] = t
			if level == 0 {
				res.NEval++
				return f(x)
			}
			return integrate(level - 1)
		}
		r, err := Quad(g, lo, hi, levelOpts(level))
		if err != nil && firstErr == nil {
			firstErr = err
		}
		res.AbsErr = max(res.AbsErr, r.AbsErr)
		return r.Value
	}
	res.Value = integrate(n - 1)
	var qe *QuadError
	if errors.As(firstErr, &qe) && qe.Ier == 6 {
		return QuadResult{}, firstErr
	}
	return res, firstErr
}

// DblQuad computes the double integral of f(y, x) for x from a to b and y
// from gfun(x) to hfun(x). It is SciPy's dblquad(func, a, b, gfun, hfun,
// epsabs=..., epsrel=...), with constant limits written Const(v):
//
//	dblquad(f, 0, 1, 0, 2)                → DblQuad(f, 0, 1, Const(0), Const(2), nil)
//	dblquad(f, 0, 1, lambda x: 0, lambda x: x) → DblQuad(f, 0, 1, Const(0), func(x float64) float64 { return x }, nil)
//
// opts (tolerances, Limit) apply to both levels, as in SciPy.
func DblQuad(f func(y, x float64) float64, a, b float64, gfun, hfun func(x float64) float64, opts *QuadOptions) (QuadResult, error) {
	return NQuad(func(v []float64) float64 { return f(v[0], v[1]) }, []Range{
		func(o []float64) (float64, float64) { return gfun(o[0]), hfun(o[0]) },
		ConstRange(a, b),
	}, &NQuadOptions{Opts: []*QuadOptions{opts}})
}

// TplQuad computes the triple integral of f(z, y, x) for x in [a, b], y in
// [gfun(x), hfun(x)] and z in [qfun(x, y), rfun(x, y)]. It is SciPy's
// tplquad(func, a, b, gfun, hfun, qfun, rfun).
func TplQuad(f func(z, y, x float64) float64, a, b float64, gfun, hfun func(x float64) float64, qfun, rfun func(x, y float64) float64, opts *QuadOptions) (QuadResult, error) {
	return NQuad(func(v []float64) float64 { return f(v[0], v[1], v[2]) }, []Range{
		func(o []float64) (float64, float64) { return qfun(o[1], o[0]), rfun(o[1], o[0]) },
		func(o []float64) (float64, float64) { return gfun(o[0]), hfun(o[0]) },
		ConstRange(a, b),
	}, &NQuadOptions{Opts: []*QuadOptions{opts}})
}
