// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// luBand is the LU decomposition of a banded matrix with partial pivoting, as
// LAPACK's dgbtf2 and dgbtrs compute it, in LAPACK's band storage: column
// major with leading dimension 2*kl+ku+1, element (i, j) at ab[kl+ku+i-j+j*ld]
// (the first kl rows are the room for the fill-in of the pivoting).
type luBand struct {
	n, kl, ku, ld int
	ab            []float64
	piv           []int
}

func newLUBand(n, kl, ku int) *luBand {
	ld := 2*kl + ku + 1
	return &luBand{n: n, kl: kl, ku: ku, ld: ld, ab: make([]float64, ld*n), piv: make([]int, n)}
}

func (lu *luBand) at(i, j int) *float64 { return &lu.ab[lu.kl+lu.ku+i-j+j*lu.ld] }

// factor overwrites ab, which holds the matrix, with its factors; it reports
// false when a pivot is exactly zero (dgbtrf's info > 0).
func (lu *luBand) factor() bool {
	n, kl, ku := lu.n, lu.kl, lu.ku
	ok := true
	ju := -1
	for j := 0; j < n; j++ {
		km := min(kl, n-1-j)
		p := j
		big := math.Abs(*lu.at(j, j))
		for i := j + 1; i <= j+km; i++ {
			if v := math.Abs(*lu.at(i, j)); v > big {
				big, p = v, i
			}
		}
		lu.piv[j] = p
		if *lu.at(p, j) == 0 {
			ok = false
			continue
		}
		ju = max(ju, min(j+ku+p-j, n-1))
		if p != j {
			for c := j; c <= ju; c++ {
				a, b := lu.at(p, c), lu.at(j, c)
				*a, *b = *b, *a
			}
		}
		r := 1 / *lu.at(j, j)
		for i := j + 1; i <= j+km; i++ {
			*lu.at(i, j) *= r
		}
		for c := j + 1; c <= ju; c++ {
			f := *lu.at(j, c)
			if f == 0 {
				continue
			}
			for i := j + 1; i <= j+km; i++ {
				*lu.at(i, c) -= *lu.at(i, j) * f
			}
		}
	}
	return ok
}

// solve overwrites b with the solution of A x = b.
func (lu *luBand) solve(b []float64) {
	n, kl, ku := lu.n, lu.kl, lu.ku
	for j := 0; j < n-1; j++ {
		km := min(kl, n-1-j)
		if p := lu.piv[j]; p != j {
			b[p], b[j] = b[j], b[p]
		}
		for i := j + 1; i <= j+km; i++ {
			b[i] -= b[j] * *lu.at(i, j)
		}
	}
	for j := n - 1; j >= 0; j-- {
		b[j] /= *lu.at(j, j)
		for i := max(0, j-kl-ku); i < j; i++ {
			b[i] -= b[j] * *lu.at(i, j)
		}
	}
}
