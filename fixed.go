// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// ErrSamples reports sample arrays that do not fit a fixed rule.
var ErrSamples = errors.New("integrate: invalid samples")

func samplesErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrSamples}, args...)...)
}

var glCache sync.Map // n → [2][]float64

// GaussLegendre returns the n nodes (ascending) and weights of the
// Gauss–Legendre rule on [-1, 1] (SciPy's roots_legendre). Nodes are found by
// Newton's method on the three-term recurrence, started from Tricomi's
// asymptotic approximation; results are cached and must not be modified.
func GaussLegendre(n int) (x, w []float64) {
	if n < 1 {
		return nil, nil
	}
	if v, ok := glCache.Load(n); ok {
		p := v.([2][]float64)
		return p[0], p[1]
	}
	x = make([]float64, n)
	w = make([]float64, n)
	nf := float64(n)
	for i := 0; i < (n+1)/2; i++ {
		// Tricomi's initial guess for the (i+1)-th largest root.
		theta := math.Pi * (float64(i) + 0.75) / (nf + 0.5)
		z := (1 - (nf-1)/(8*nf*nf*nf)) * math.Cos(theta)
		var dp float64
		for it := 0; it < 100; it++ {
			p0, p1 := 1.0, z
			for k := 2; k <= n; k++ {
				kf := float64(k)
				p0, p1 = p1, ((2*kf-1)*z*p1-(kf-1)*p0)/kf
			}
			if n == 1 {
				p0 = 1
			}
			dp = nf * (z*p1 - p0) / (z*z - 1)
			dz := p1 / dp
			z -= dz
			if math.Abs(dz) <= 1e-16*math.Max(1, math.Abs(z)) {
				break
			}
		}
		// Recompute the derivative at the converged root.
		p0, p1 := 1.0, z
		for k := 2; k <= n; k++ {
			kf := float64(k)
			p0, p1 = p1, ((2*kf-1)*z*p1-(kf-1)*p0)/kf
		}
		if n == 1 {
			p0 = 1
		}
		dp = nf * (z*p1 - p0) / (z*z - 1)
		wi := 2 / ((1 - z*z) * dp * dp)
		x[n-1-i] = z
		x[i] = -z
		w[i] = wi
		w[n-1-i] = wi
	}
	if n%2 == 1 {
		x[n/2] = 0
	}
	glCache.Store(n, [2][]float64{x, w})
	return x, w
}

// FixedQuad integrates f over [a, b] with the n-point Gauss–Legendre rule. It
// is SciPy's fixed_quad(func, a, b, n=5) (whose second return value is
// always None).
func FixedQuad(f func(float64) float64, a, b float64, n int) (float64, error) {
	if n < 1 {
		return 0, samplesErr("FixedQuad needs n ≥ 1, got %d", n)
	}
	x, w := GaussLegendre(n)
	s := 0.0
	for i := range x {
		s += w[i] * f((b-a)*(x[i]+1)/2+a)
	}
	return (b - a) / 2 * s, nil
}

// RombergOptions tunes Romberg; zero values select SciPy's defaults.
type RombergOptions struct {
	Tol    float64 // absolute tolerance (default 1.48e-8)
	RTol   float64 // relative tolerance (default 1.48e-8)
	DivMax int     // maximum order of extrapolation (default 10)
}

// ErrDivMax reports that Romberg reached DivMax before converging.
var ErrDivMax = errors.New("integrate: Romberg: DivMax exceeded")

// Romberg integrates f over [a, b] by Richardson extrapolation of the
// trapezoid rule on 1, 2, 4, ... 2^DivMax intervals. It is the
// scipy.integrate.romberg of SciPy before 1.15, which removed it. If DivMax is
// reached first the last estimate is returned with ErrDivMax.
func Romberg(f func(float64) float64, a, b float64, opts *RombergOptions) (float64, error) {
	var o RombergOptions
	if opts != nil {
		o = *opts
	}
	tol := orDefault(o.Tol, 1.48e-8)
	rtol := orDefault(o.RTol, 1.48e-8)
	divmax := o.DivMax
	if divmax == 0 {
		divmax = 10
	}
	intrange := b - a
	ordsum := 0.5 * (f(a) + f(b))
	lastRow := []float64{intrange * ordsum}
	n := 1
	for i := 1; i <= divmax; i++ {
		n *= 2
		// Sum f at the midpoints of the previous n/2 panels.
		num := n / 2
		h := intrange / float64(num)
		lox := a + 0.5*h
		for k := 0; k < num; k++ {
			ordsum += f(lox + h*float64(k))
		}
		row := make([]float64, i+1)
		row[0] = intrange * ordsum / float64(n)
		for k := 0; k < i; k++ {
			tmp := math.Pow(4, float64(k+1))
			row[k+1] = (tmp*row[k] - lastRow[k]) / (tmp - 1)
		}
		result := row[i]
		err := math.Abs(result - lastRow[i-1])
		if err < tol || err < rtol*math.Abs(result) {
			return result, nil
		}
		lastRow = row
	}
	return lastRow[divmax], fmt.Errorf("%w (%d); latest difference larger than tolerance", ErrDivMax, divmax)
}

// Romb integrates 2^k + 1 equally spaced samples by Romberg extrapolation.
// It is SciPy's romb(y, dx).
func Romb(y []float64, dx float64) (float64, error) {
	nsamps := len(y)
	nint := nsamps - 1
	k := 0
	for n := 1; n < nint; n <<= 1 {
		k++
	}
	if nint < 1 || 1<<k != nint {
		return 0, samplesErr("Romb needs 2^k + 1 samples, got %d", nsamps)
	}
	// SciPy's table R[(i, j)], one row per level.
	R := make([][]float64, k+1)
	for i := range R {
		R[i] = make([]float64, i+1)
	}
	h := float64(nint) * dx
	R[0][0] = (y[0] + y[nint]) / 2 * h
	start, stop, step := nint, nint, nint
	for i := 1; i <= k; i++ {
		start >>= 1
		// y[start:stop:step].sum(), summed as NumPy sums.
		st := step
		s := npSum((stop-start+st-1)/st, func(lo int, dst []float64) {
			for q := range dst {
				dst[q] = y[start+(lo+q)*st]
			}
		})
		R[i][0] = 0.5 * (R[i-1][0] + float64(h*s))
		step >>= 1
		for j := 1; j <= i; j++ {
			prev := R[i][j-1]
			R[i][j] = prev + (prev-R[i-1][j-1])/(math.Ldexp(1, 2*j)-1) // (1 << 2j) - 1, exact; an int shift overflows on 32-bit targets
		}
		h /= 2
	}
	return R[k][k], nil
}

// checkXY validates a sample vector against optional abscissae.
func checkXY(y, x []float64, need int, name string) error {
	return checkLen(len(y), x, need, name)
}

// checkLen is checkXY for n samples.
func checkLen(n int, x []float64, need int, name string) error {
	if n < need {
		return samplesErr("%s needs at least %d samples, got %d", name, need, n)
	}
	if x != nil && len(x) != n {
		return samplesErr("%s: x has %d entries, y has %d", name, len(x), n)
	}
	return nil
}

func spacing(x []float64, dx float64, i int) float64 {
	if x == nil {
		return dx
	}
	return x[i+1] - x[i]
}

// Trapezoid integrates samples y with the composite trapezoidal rule, at
// abscissae x (or spacing dx when x is nil). It is SciPy's
// trapezoid(y, x=None, dx=1.0).
func Trapezoid(y, x []float64, dx float64) (float64, error) {
	if err := checkXY(y, x, 0, "Trapezoid"); err != nil {
		return 0, err
	}
	if len(y) < 2 {
		return 0, nil
	}
	// SciPy: add.reduce(d * (y[1:] + y[:-1]) / 2.0).
	return npSum(len(y)-1, func(lo int, dst []float64) { trapezoidTerms(y, x, dx, lo, dst) }), nil
}

// trapezoidTerms sets dst[k] to SciPy's term d·(y[i+1] + y[i])/2 for
// i = lo+k.
func trapezoidTerms(y, x []float64, dx float64, lo int, dst []float64) {
	y0, y1 := y[lo:lo+len(dst)], y[lo+1:lo+1+len(dst)]
	if x == nil {
		for k := range dst {
			dst[k] = float64(dx*(y1[k]+y0[k])) / 2
		}
		return
	}
	x0, x1 := x[lo:lo+len(dst)], x[lo+1:lo+1+len(dst)]
	for k := range dst {
		dst[k] = float64((x1[k]-x0[k])*(y1[k]+y0[k])) / 2
	}
}

// trapezoidLane is Trapezoid for a validated lane, with its terms formed in
// scratch (at least len(y) long) instead of in npSum's blocks.
func trapezoidLane(y, x []float64, dx float64, scratch []float64) float64 {
	if len(y) < 2 {
		return 0
	}
	t := scratch[:len(y)-1]
	trapezoidTerms(y, x, dx, 0, t)
	return 0.0 + pairwiseSlice(t)
}

// CumulativeTrapezoid returns the running trapezoidal integral of y. Without
// initial the result has len(y)-1 entries; with initial a leading 0 is
// prepended (SciPy's initial=0). It is SciPy's
// cumulative_trapezoid(y, x=None, dx=1.0, initial=None).
func CumulativeTrapezoid(y, x []float64, dx float64, initial bool) ([]float64, error) {
	if err := checkXY(y, x, 1, "CumulativeTrapezoid"); err != nil {
		return nil, err
	}
	// SciPy: cumsum(d * (y[1:] + y[:-1]) / 2.0), a running sum.
	n := len(y) - 1
	off := 0
	if initial {
		off = 1 // out[0] = 0
	}
	out := make([]float64, off+n)
	s := 0.0
	for i := range n {
		t := float64(spacing(x, dx, i)*(y[i+1]+y[i])) / 2
		if i == 0 {
			s = t // cumsum starts from the first term, not from 0.0
		} else {
			s += t
		}
		out[off+i] = s
	}
	return out, nil
}

// div0 is n/d, or 0 where d is 0 (SciPy's apply_where guard against
// repeated abscissae).
func div0(n, d float64) float64 {
	if d == 0 {
		return 0
	}
	return n / d
}

// simpsonTerms sets dst[k] to SciPy's _basic_simpson term of the panel
// starting at index i = 2(lo+k): y[i] + 4y[i+1] + y[i+2] without abscissae,
// the non-uniform panel formula with them.
func simpsonTerms(y, x []float64, lo int, dst []float64) {
	if x == nil {
		for k := range dst {
			i := 2 * (lo + k)
			dst[k] = y[i] + float64(4*y[i+1]) + y[i+2]
		}
		return
	}
	for k := range dst {
		i := 2 * (lo + k)
		h0 := x[i+1] - x[i]
		h1 := x[i+2] - x[i+1]
		hsum := h0 + h1
		hprod := h0 * h1
		h0divh1 := div0(h0, h1)
		dst[k] = hsum / 6 * (float64(y[i]*(2-div0(1, h0divh1))) +
			float64(y[i+1]*float64(hsum*div0(hsum, hprod))) + float64(y[i+2]*(2-h0divh1)))
	}
}

// simpsonCore is Simpson for validated samples; sum(p) returns the sum (as
// numpy.add.reduce) of the terms of the first p panels.
func simpsonCore(y, x []float64, dx float64, sum func(panels int) float64) float64 {
	n := len(y)
	if n == 2 {
		return 0 + float64(0.5*spacing(x, dx, 0))*(y[1]+y[0])
	}
	stop := n - 2
	if n%2 == 0 {
		stop = n - 3
	}
	s := sum((stop + 1) / 2)
	if x == nil {
		s = float64(dx/3) * s // dx / 3.0 * sum(...)
	}
	if n%2 == 1 {
		return s
	}
	alpha, beta, eta := simpsonEnd(x, dx, n)
	// The correction is summed before it is added, as NumPy evaluates it.
	corr := float64(alpha*y[n-1]) + float64(beta*y[n-2]) - float64(eta*y[n-3])
	return s + corr + 0
}

// simpsonLane is Simpson for a validated lane, with its terms formed in
// scratch (at least len(y)/2 long).
func simpsonLane(y, x []float64, dx float64, scratch []float64) float64 {
	return simpsonCore(y, x, dx, func(p int) float64 {
		t := scratch[:p]
		simpsonTerms(y, x, 0, t)
		return 0.0 + pairwiseSlice(t)
	})
}

// simpsonEnd returns the coefficients of Cartwright's correction for the
// last interval of n (even) samples, with the products rounded before they
// are summed, as NumPy rounds them.
func simpsonEnd(x []float64, dx float64, n int) (alpha, beta, eta float64) {
	h0, h1 := dx, dx
	if x != nil {
		h0 = x[n-2] - x[n-3]
		h1 = x[n-1] - x[n-2]
	}
	alpha = div0(float64(2*h1*h1)+float64(3*h0*h1), 6*(h1+h0))
	beta = div0(float64(h1*h1)+float64(3*h0*h1), 6*h0)
	eta = div0(h1*h1*h1, 6*h0*(h0+h1))
	return alpha, beta, eta
}

// Simpson integrates samples y with the composite Simpson rule at abscissae x
// (or spacing dx when x is nil). For an even number of samples the last
// interval uses Cartwright's correction, as SciPy ≥ 1.11 does. It is SciPy's
// simpson(y, x=None, dx=1.0).
func Simpson(y, x []float64, dx float64) (float64, error) {
	if err := checkXY(y, x, 2, "Simpson"); err != nil {
		return 0, err
	}
	return simpsonCore(y, x, dx, func(p int) float64 {
		return npSum(p, func(lo int, dst []float64) { simpsonTerms(y, x, lo, dst) })
	}), nil
}
