// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"math"
	"testing"
)

// TestDDPow: the power is correctly rounded where that is checkable (squares
// and square roots are correctly rounded by the hardware), within an ulp of
// math.Pow elsewhere, and delegates the special cases.
func TestDDPow(t *testing.T) {
	seed := uint64(3)
	next := func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed>>11) / (1 << 53)
	}
	for i := 0; i < 2000; i++ {
		x := math.Exp(40*next() - 20)
		if g := ddPow(x, 2); g != x*x {
			t.Fatalf("ddPow(%v, 2) = %v, want %v", x, g, x*x)
		}
		if g := ddPow(x, 0.5); g != math.Sqrt(x) {
			t.Fatalf("ddPow(%v, 1/2) = %v, want %v", x, g, math.Sqrt(x))
		}
		y := 1 / float64(2+int(10*next()))
		g, w := ddPow(x, y), math.Pow(x, y)
		if math.Abs(g-w) > 4*math.Abs(math.Nextafter(w, math.Inf(1))-w) { // math.Pow is off by up to a few ulp
			t.Fatalf("ddPow(%v, %v) = %v, math.Pow %v", x, y, g, w)
		}
	}
	for _, c := range [][2]float64{{0, 0.5}, {-2, 3}, {1, 7}, {5, 0}, {math.Inf(1), 2}, {2, math.NaN()}, {2, math.Inf(1)}, {2, 5000}, {math.NaN(), 2}} {
		g, w := ddPow(c[0], c[1]), math.Pow(c[0], c[1])
		if g != w && !(math.IsNaN(g) && math.IsNaN(w)) {
			t.Errorf("ddPow(%v, %v) = %v, math.Pow %v", c[0], c[1], g, w)
		}
	}
}

// TestPowRecip: the cheap power agrees with the double-double one on every
// exponent lsoda uses (a few hundred thousand cases), and delegates the rest.
func TestPowRecip(t *testing.T) {
	seed := uint64(11)
	next := func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed>>11) / (1 << 53)
	}
	bad := 0
	for k := 1; k <= 16; k++ {
		e := 1 / float64(k)
		for i := 0; i < 20000; i++ {
			x := math.Exp(60*next() - 40)
			if g, w := powRecip(x, e), ddPow(x, e); g != w {
				bad++
				if bad < 5 {
					t.Errorf("powRecip(%v, 1/%d) = %v, ddPow %v", x, k, g, w)
				}
			}
		}
	}
	for _, c := range [][2]float64{{0, 0.5}, {-1, 0.25}, {math.Inf(1), 0.5}, {2, 2}, {2, 0}, {3, 0.4}, {3, 1.0 / 17}, {0.5, math.NaN()}} {
		if g, w := powRecip(c[0], c[1]), math.Pow(c[0], c[1]); g != w && !(math.IsNaN(g) && math.IsNaN(w)) && math.Abs(g-w) > 1e-15*math.Abs(w) {
			t.Errorf("powRecip(%v, %v) = %v, math.Pow %v", c[0], c[1], g, w)
		}
	}
}

func BenchmarkPow(b *testing.B) {
	x := 0.37
	b.Run("math.Pow", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sinkF = math.Pow(x+float64(i&7)*0.01, 1.0/3)
		}
	})
	b.Run("powRecip", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sinkF = powRecip(x+float64(i&7)*0.01, 1.0/3)
		}
	})
	b.Run("ddPow", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sinkF = ddPow(x+float64(i&7)*0.01, 1.0/3)
		}
	})
}

var sinkF float64

// TestLsodaExactBanded: the banded diffusion problem on the Exact path (lsoda.c's
// own bnorm, the correctly rounded power) takes the evaluations lsoda.c takes
// with and without fused multiply-adds (harness of the pinned lsoda.c, itask 5,
// the same y0 and f): jt 4: 582 and 582; jt 5: 564 and 567, Jacobians 9 in all.
// Before the correctly rounded power, Go's math.Pow gave 642 and 622 evaluations
// here against lsoda.c's 621 and 622 (arm64) and 667 and 706 against 582 and 567
// (amd64): the last bit of pow, not the port, decides the step sequence. The
// default path now has the same power (powRecip) and a bnorm that reads P: it
// takes the same counts here, and is held to the same solution to 50 tolerances.
func TestLsodaExactBanded(t *testing.T) {
	band := &Bandwidth{Lower: 1, Upper: 1}
	for _, c := range []struct {
		name string
		bj   BandJacFunc
		want int
	}{{"analytic band (jt 4)", diffusionBand, 582}, {"finite-difference band (jt 5)", nil, 566}} {
		run := func(exact bool) *Solution {
			sol, err := SolveIVP(diffusionF, [2]float64{0, 1}, diffusionY0(), &IVPOptions{Method: LSODA, RTol: 1e-6, ATol: 1e-9, Band: band, BandJac: c.bj, Exact: exact})
			if err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			return sol
		}
		ex, def := run(true), run(false)
		t.Logf("%s: Exact %d evaluations %d Jacobians %d steps, default %d, %d, %d; lsoda.c %d", c.name,
			ex.NFev, ex.NJev, ex.NSteps, def.NFev, def.NJev, def.NSteps, c.want)
		if d := math.Abs(float64(ex.NFev-c.want)) / float64(c.want); d > 0.035 { // 0 to 1 % on arm64 and amd64; 2.5 % on ppc64le and s390x (other fused multiply-adds)
			t.Errorf("%s: Exact takes %d evaluations, lsoda.c %d", c.name, ex.NFev, c.want)
		}
		if ex.NJev < 8 || ex.NJev > 10 {
			t.Errorf("%s: Exact takes %d Jacobians, lsoda.c 9", c.name, ex.NJev)
		}
		a, b := ex.Y[len(ex.Y)-1], def.Y[len(def.Y)-1]
		for i := range a {
			if math.Abs(a[i]-b[i]) > 50*(1e-9+1e-6*math.Abs(a[i])) { // 50 tolerances
				// (the two paths take different steps; the global error is a multiple of the tolerance)
				t.Errorf("%s: y[%d] = %v (default) against %v (Exact)", c.name, i, b[i], a[i])
			}
		}
	}
}

// TestLsodaExactDense: the Exact path on the benchmark cases gives the same
// solution as the default (the powers differ in the last bit only).
func TestLsodaExactDense(t *testing.T) {
	d, err := SolveIVP(vdpF(1000), [2]float64{0, 3000}, []float64{2, 0}, &IVPOptions{Method: LSODA})
	if err != nil {
		t.Fatal(err)
	}
	e, err := SolveIVP(vdpF(1000), [2]float64{0, 3000}, []float64{2, 0}, &IVPOptions{Method: LSODA, Exact: true})
	if err != nil {
		t.Fatal(err)
	}
	if relErr(d.Y[len(d.Y)-1], e.Y[len(e.Y)-1], 1e-3, 1e-6) > 1e3 {
		t.Errorf("default %v, Exact %v", d.Y[len(d.Y)-1], e.Y[len(e.Y)-1])
	}
	if _, err := SolveIVP(vdpF(1000), [2]float64{0, 1}, []float64{2, 0}, &IVPOptions{Method: BDF, Exact: true}); err == nil {
		t.Error("BDF accepted Exact")
	}
}
