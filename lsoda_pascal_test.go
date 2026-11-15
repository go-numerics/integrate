// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"math"
	"math/rand"
	"testing"
)

// pascalRef is lsoda.c's loop: ascending, in place, overlapping operands.
func pascalRef(yh []float64, n, nq int, sign float64) {
	i1 := nq * n
	for jb := 0; jb < nq; jb++ {
		i1 -= n
		for i := i1; i < nq*n; i++ {
			yh[i] += sign * yh[i+n]
		}
	}
}

// stodaPascal (column-wise kernels from simdMin) is bit-identical to the
// scalar loop, in both directions, at sizes around simdMin and every order.
func TestStodaPascalBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 5, simdMin - 1, simdMin, simdMin + 1, 37, 100, 257} {
		for nq := 1; nq <= 12; nq++ {
			for _, sign := range []float64{1, -1} {
				yh := make([]float64, (nq+1)*n)
				for i := range yh {
					yh[i] = rng.NormFloat64() * math.Pow(10, float64(rng.Intn(9)-4))
				}
				want := append([]float64(nil), yh...)
				pascalRef(want, n, nq, sign)
				s := &lsodaSolver{odeCore: &odeCore{n: n}, yh: yh, nq: nq, nqnyh: nq * n}
				s.stodaPascal(sign)
				for i := range yh {
					if math.Float64bits(yh[i]) != math.Float64bits(want[i]) {
						t.Fatalf("n=%d nq=%d sign=%v: yh[%d] = %v, want %v", n, nq, sign, i, yh[i], want[i])
					}
				}
			}
		}
	}
}
