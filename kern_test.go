// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"math"
	"testing"
)

// residual is max_i |(A x - b)_i| for the dense n×n row-major a.
func residual(a, x, b []float64, n int) float64 {
	r := 0.0
	for i := 0; i < n; i++ {
		s := -b[i]
		for j := 0; j < n; j++ {
			s += a[i*n+j] * x[j]
		}
		r = math.Max(r, math.Abs(s))
	}
	return r
}

// The dense LU at the sizes around simdMin (the scalar loops below, the
// go-simd kernels from it) solves A x = b to the roundoff of the system.
func TestLUKernelSizes(t *testing.T) {
	for _, n := range []int{simdMin - 1, simdMin, simdMin + 1, 100} {
		m := kernMatrix(n)
		lu := newLUReal(n)
		lu.factor(m)
		b := kernVec(n)
		x := append([]float64(nil), b...)
		lu.solve(x)
		if r := residual(m, x, b, n); r > 1e-12 {
			t.Errorf("n=%d: residual %.3g", n, r)
		}
	}
}

// The banded LU, with the band wide enough for the kernels in the forward
// run (kl), the backward run (kl+ku) or both, solves the same system as the
// dense LU of the same matrix.
func TestLUBandKernelSizes(t *testing.T) {
	const n = 150
	for _, w := range [][2]int{{3, 3}, {3, 40}, {40, 0}, {40, 40}, {33, 5}} {
		kl, ku := w[0], w[1]
		band := newLUBand(n, kl, ku)
		kernBand(band)
		dense := make([]float64, n*n)
		for j := 0; j < n; j++ {
			for i := max(0, j-ku); i <= min(n-1, j+kl); i++ {
				dense[i*n+j] = *band.at(i, j)
			}
		}
		if !band.factor() {
			t.Fatalf("kl=%d ku=%d: zero pivot", kl, ku)
		}
		b := kernVec(n)
		x := append([]float64(nil), b...)
		band.solve(x)
		if r := residual(dense, x, b, n); r > 1e-12 {
			t.Errorf("kl=%d ku=%d: residual %.3g", kl, ku, r)
		}
	}
}

// LSODA on n decoupled decays y_i' = -λ_i y_i (stiff for the large λ_i,
// n ≥ simdMin so the rescale, the dense and the banded solves all take the
// kernels), against the exact exp(-λ_i t).
func TestLsodaKernelSizes(t *testing.T) {
	const n = 80
	lam := make([]float64, n)
	y0 := make([]float64, n)
	for i := range lam {
		lam[i] = 1 + 120*float64(i)
		y0[i] = 1
	}
	f := func(_ float64, y, d []float64) {
		for i := range y {
			d[i] = -lam[i] * y[i]
		}
	}
	for _, band := range []*Bandwidth{nil, {Lower: 33, Upper: 34}} {
		sol, err := SolveIVP(f, [2]float64{0, 1}, y0, &IVPOptions{Method: LSODA, RTol: 1e-8, ATol: 1e-12, Band: band})
		if err != nil {
			t.Fatal(err)
		}
		y := sol.Y[len(sol.Y)-1]
		for i := range y {
			if e := math.Abs(y[i] - math.Exp(-lam[i])); e > 1e-6 {
				t.Errorf("band %v: y[%d] = %g, exact %g", band, i, y[i], math.Exp(-lam[i]))
			}
		}
	}
}
