// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"fmt"
	"math"
	"testing"
)

// Benchmarks of the linear algebra kernels at the sizes where the choice of
// scalar loop or vector kernel matters (docs/perf.md): run with
// go test -bench 'LU|DiffusionN' -cpu 1 and compare two builds with benchstat.

var kernSizes = []int{3, 8, 20, 64, 256}

func kernMatrix(n int) []float64 {
	m := make([]float64, n*n)
	for i := range m {
		m[i] = math.Sin(float64(i)) * 0.1
	}
	for i := 0; i < n; i++ {
		m[i*n+i] += 2
	}
	return m
}

func kernVec(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1 + 0.37*float64(i%7) - 0.11*float64(i%3)
	}
	return v
}

func BenchmarkLUSolve(b *testing.B) {
	for _, n := range kernSizes {
		lu := newLUReal(n)
		lu.factor(kernMatrix(n))
		rhs, x := kernVec(n), make([]float64, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			for b.Loop() {
				copy(x, rhs)
				lu.solve(x)
			}
		})
	}
}

func BenchmarkLUFactor(b *testing.B) {
	for _, n := range kernSizes {
		lu := newLUReal(n)
		m := kernMatrix(n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			for b.Loop() {
				lu.factor(m)
			}
		})
	}
}

// kernBand fills a diagonally dominant band matrix of n rows.
func kernBand(lu *luBand) {
	for j := 0; j < lu.n; j++ {
		for i := max(0, j-lu.ku); i <= min(lu.n-1, j+lu.kl); i++ {
			*lu.at(i, j) = math.Sin(float64(i*7+j)) * 0.1
			if i == j {
				*lu.at(i, j) += 2
			}
		}
	}
}

func BenchmarkLUBand(b *testing.B) {
	for _, w := range []int{1, 8, 32, 100} {
		n := 4 * w
		if n < 64 {
			n = 64
		}
		lu := newLUBand(n, w, w)
		rhs, x := kernVec(n), make([]float64, n)
		b.Run(fmt.Sprintf("factor+solve/w%d", w), func(b *testing.B) {
			for b.Loop() {
				kernBand(lu)
				lu.factor()
				copy(x, rhs)
				lu.solve(x)
			}
		})
	}
}

// the whole integrator on a dense stiff problem of n = 200 (finite-difference
// Jacobian, so P is factored from n evaluations of f): the nearest-neighbour
// diffusion, which only the dense kernels see.
func BenchmarkDiffusionN200(b *testing.B) {
	const n = 200
	y0 := make([]float64, n)
	y0[n/2] = 1
	f := func(_ float64, y, d []float64) {
		for i := range y {
			var l, r float64
			if i > 0 {
				l = y[i-1]
			}
			if i < n-1 {
				r = y[i+1]
			}
			d[i] = diffK * (l - 2*y[i] + r)
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := SolveIVP(f, [2]float64{0, 0.2}, y0, &IVPOptions{Method: LSODA, RTol: 1e-6, ATol: 1e-9}); err != nil {
			b.Fatal(err)
		}
	}
}
