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
	R := make(map[[2]int]float64)
	h := float64(nint) * dx
	R[[2]int{0, 0}] = (y[0] + y[nint]) / 2 * h
	start, stop, step := nint, nint, nint
	for i := 1; i <= k; i++ {
		start >>= 1
		s := 0.0
		for j := start; j < stop; j += step {
			s += y[j]
		}
		R[[2]int{i, 0}] = 0.5 * (R[[2]int{i - 1, 0}] + h*s)
		step >>= 1
		for j := 1; j <= i; j++ {
			prev := R[[2]int{i, j - 1}]
			R[[2]int{i, j}] = prev + (prev-R[[2]int{i - 1, j - 1}])/(float64(int(1)<<(2*j))-1)
		}
		h /= 2
	}
	return R[[2]int{k, k}], nil
}

// checkXY validates a sample vector against optional abscissae.
func checkXY(y, x []float64, need int, name string) error {
	if len(y) < need {
		return samplesErr("%s needs at least %d samples, got %d", name, need, len(y))
	}
	if x != nil && len(x) != len(y) {
		return samplesErr("%s: x has %d entries, y has %d", name, len(x), len(y))
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
	s := 0.0
	for i := 0; i+1 < len(y); i++ {
		s += spacing(x, dx, i) * (y[i] + y[i+1]) / 2
	}
	return s, nil
}

// CumulativeTrapezoid returns the running trapezoidal integral of y. Without
// initial the result has len(y)-1 entries; with initial a leading 0 is
// prepended (SciPy's initial=0). It is SciPy's
// cumulative_trapezoid(y, x=None, dx=1.0, initial=None).
func CumulativeTrapezoid(y, x []float64, dx float64, initial bool) ([]float64, error) {
	if err := checkXY(y, x, 1, "CumulativeTrapezoid"); err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(y))
	if initial {
		out = append(out, 0)
	}
	s := 0.0
	for i := 0; i+1 < len(y); i++ {
		s += spacing(x, dx, i) * (y[i] + y[i+1]) / 2
		out = append(out, s)
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

// basicSimpson applies Simpson's rule on the panels starting at the even
// indices 0, 2, ..., stop-2 (SciPy's _basic_simpson with start 0).
func basicSimpson(y, x []float64, dx float64, stop int) float64 {
	s := 0.0
	if x == nil {
		for i := 0; i < stop; i += 2 {
			s += y[i] + 4*y[i+1] + y[i+2]
		}
		return s * dx / 3
	}
	for i := 0; i < stop; i += 2 {
		h0 := x[i+1] - x[i]
		h1 := x[i+2] - x[i+1]
		hsum := h0 + h1
		hprod := h0 * h1
		h0divh1 := div0(h0, h1)
		s += hsum / 6 * (y[i]*(2-div0(1, h0divh1)) + y[i+1]*(hsum*div0(hsum, hprod)) + y[i+2]*(2-h0divh1))
	}
	return s
}

// Simpson integrates samples y with the composite Simpson rule at abscissae x
// (or spacing dx when x is nil). For an even number of samples the last
// interval uses Cartwright's correction, as SciPy ≥ 1.11 does. It is SciPy's
// simpson(y, x=None, dx=1.0).
func Simpson(y, x []float64, dx float64) (float64, error) {
	if err := checkXY(y, x, 2, "Simpson"); err != nil {
		return 0, err
	}
	n := len(y)
	if n%2 == 1 {
		return basicSimpson(y, x, dx, n-2), nil
	}
	if n == 2 {
		return 0.5 * spacing(x, dx, 0) * (y[0] + y[1]), nil
	}
	result := basicSimpson(y, x, dx, n-3)
	h0, h1 := dx, dx
	if x != nil {
		h0 = x[n-2] - x[n-3]
		h1 = x[n-1] - x[n-2]
	}
	alpha := div0(2*h1*h1+3*h0*h1, 6*(h1+h0))
	beta := div0(h1*h1+3*h0*h1, 6*h0)
	eta := div0(h1*h1*h1, 6*h0*(h0+h1))
	return result + alpha*y[n-1] + beta*y[n-2] - eta*y[n-3], nil
}
