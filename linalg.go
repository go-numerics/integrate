// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Dense LU factorisations with partial pivoting for the Newton iterations of
// the implicit solvers. Matrices are n×n, row-major. Pivots are chosen as
// LAPACK's getrf chooses them (largest |a| for real matrices, largest
// |re|+|im| for complex ones), so the factors agree with the ones a LAPACK
// based solver computes. A zero pivot is left in place: the solve then yields
// non-finite values, which the Newton iteration rejects and answers with a
// smaller step, as it does when LAPACK reports a singular factor.

type luReal struct {
	n   int
	a   []float64
	piv []int
}

func newLUReal(n int) *luReal {
	return &luReal{n: n, a: make([]float64, n*n), piv: make([]int, n)}
}

// factor overwrites the receiver with the LU factors of m.
func (lu *luReal) factor(m []float64) {
	n := lu.n
	a := lu.a
	copy(a, m)
	for k := 0; k < n; k++ {
		p := k
		big := math.Abs(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := math.Abs(a[i*n+k]); v > big {
				big, p = v, i
			}
		}
		lu.piv[k] = p
		if p != k {
			for j := 0; j < n; j++ {
				a[k*n+j], a[p*n+j] = a[p*n+j], a[k*n+j]
			}
		}
		d := a[k*n+k]
		if d == 0 {
			continue
		}
		for i := k + 1; i < n; i++ {
			l := a[i*n+k] / d
			a[i*n+k] = l
			if l == 0 {
				continue
			}
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= l * a[k*n+j]
			}
		}
	}
}

// solve overwrites b with the solution of A x = b.
func (lu *luReal) solve(b []float64) {
	n := lu.n
	a := lu.a
	for k := 0; k < n; k++ {
		if p := lu.piv[k]; p != k {
			b[k], b[p] = b[p], b[k]
		}
	}
	for i := 1; i < n; i++ {
		s := b[i]
		for j := 0; j < i; j++ {
			s -= a[i*n+j] * b[j]
		}
		b[i] = s
	}
	for i := n - 1; i >= 0; i-- {
		s := b[i]
		for j := i + 1; j < n; j++ {
			s -= a[i*n+j] * b[j]
		}
		b[i] = s / a[i*n+i]
	}
}

type luComplex struct {
	n   int
	a   []complex128
	piv []int
}

func newLUComplex(n int) *luComplex {
	return &luComplex{n: n, a: make([]complex128, n*n), piv: make([]int, n)}
}

func cabs1(z complex128) float64 { return math.Abs(real(z)) + math.Abs(imag(z)) }

func (lu *luComplex) factor(m []complex128) {
	n := lu.n
	a := lu.a
	copy(a, m)
	for k := 0; k < n; k++ {
		p := k
		big := cabs1(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cabs1(a[i*n+k]); v > big {
				big, p = v, i
			}
		}
		lu.piv[k] = p
		if p != k {
			for j := 0; j < n; j++ {
				a[k*n+j], a[p*n+j] = a[p*n+j], a[k*n+j]
			}
		}
		d := a[k*n+k]
		if d == 0 {
			continue
		}
		for i := k + 1; i < n; i++ {
			l := a[i*n+k] / d
			a[i*n+k] = l
			if l == 0 {
				continue
			}
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= l * a[k*n+j]
			}
		}
	}
}

func (lu *luComplex) solve(b []complex128) {
	n := lu.n
	a := lu.a
	for k := 0; k < n; k++ {
		if p := lu.piv[k]; p != k {
			b[k], b[p] = b[p], b[k]
		}
	}
	for i := 1; i < n; i++ {
		s := b[i]
		for j := 0; j < i; j++ {
			s -= a[i*n+j] * b[j]
		}
		b[i] = s
	}
	for i := n - 1; i >= 0; i-- {
		s := b[i]
		for j := i + 1; j < n; j++ {
			s -= a[i*n+j] * b[j]
		}
		b[i] = s / a[i*n+i]
	}
}

// allFinite reports whether every entry of v is finite.
func allFinite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}
