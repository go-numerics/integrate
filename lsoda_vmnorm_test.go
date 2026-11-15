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

// vmnorm from simdMin entries up (floats.MaxAbsMul) is bit-identical to the
// fmax loop of vmnorm.f, NaN, Inf and signed zeros included.
func TestVmnormBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	special := []float64{0, math.Copysign(0, -1), math.NaN(), math.Inf(1), math.Inf(-1), 1e-310, 1e300}
	for _, n := range []int{1, simdMin - 1, simdMin, simdMin + 1, 100, 257} {
		for trial := 0; trial < 200; trial++ {
			v, w := make([]float64, n), make([]float64, n)
			for i := range v {
				v[i], w[i] = rng.NormFloat64(), math.Exp(rng.NormFloat64())
				if rng.Intn(8) == 0 {
					v[i] = special[rng.Intn(len(special))]
				}
				if rng.Intn(16) == 0 {
					w[i] = special[rng.Intn(len(special))]
				}
			}
			want := 0.0
			for i, x := range v {
				want = cmax(want, math.Abs(x)*w[i])
			}
			if got := vmnorm(v, w); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("n=%d: %v, want %v", n, got, want)
			}
		}
	}
}
