// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The benchmark set of docs/perf.md; bench/scipy_bench.py times the same
// problems with SciPy, and bench/compare.sh interleaves the two.

func BenchmarkQuadSmooth(b *testing.B) {
	f := func(x float64) float64 { return 1 / (1 + 25*x*x) }
	for b.Loop() {
		if _, err := Quad(f, -1, 1, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuadSingular(b *testing.B) {
	f := func(x float64) float64 { return math.Log(x) / math.Sqrt(x) }
	for b.Loop() {
		if _, err := Quad(f, 0, 1, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuadInfinite(b *testing.B) {
	f := func(x float64) float64 { return math.Exp(-x * x) }
	for b.Loop() {
		if _, err := Quad(f, math.Inf(-1), math.Inf(1), nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveIVPLotkaRK45(b *testing.B) {
	for b.Loop() {
		if _, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveIVPLotkaDOP853Tight(b *testing.B) {
	o := &IVPOptions{Method: DOP853, RTol: 1e-10, ATol: 1e-12}
	for b.Loop() {
		if _, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, o); err != nil {
			b.Fatal(err)
		}
	}
}

func benchVdP(b *testing.B, m Method, jac bool) {
	o := &IVPOptions{Method: m}
	if jac {
		o.Jac = vdpJ(1000)
	}
	f := vdpF(1000)
	for b.Loop() {
		if _, err := SolveIVP(f, [2]float64{0, 3000}, []float64{2, 0}, o); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveIVPVdP1000Radau(b *testing.B)    { benchVdP(b, Radau, false) }
func BenchmarkSolveIVPVdP1000RadauJac(b *testing.B) { benchVdP(b, Radau, true) }
func BenchmarkSolveIVPVdP1000BDF(b *testing.B)      { benchVdP(b, BDF, false) }
func BenchmarkSolveIVPVdP1000BDFJac(b *testing.B)   { benchVdP(b, BDF, true) }
func BenchmarkSolveIVPVdP1000LSODA(b *testing.B)    { benchVdP(b, LSODA, false) }

func BenchmarkOdeintRobertson(b *testing.B) {
	ts := []float64{0, 0.4, 4, 40, 400, 4000, 40000}
	o := &OdeintOptions{RTol: 1e-8, ATol: 1e-12}
	for b.Loop() {
		if _, _, err := Odeint(robertsonF, []float64{1, 0, 0}, ts, o); err != nil {
			b.Fatal(err)
		}
	}
}

// The rest of the public operations, each on a problem that
// bench/scipy_bench.py poses to SciPy in the same form.

func benchQuad(b *testing.B, f func(float64) float64, a, c float64, o *QuadOptions) {
	for b.Loop() {
		if _, err := Quad(f, a, c, o); err != nil {
			b.Fatal(err)
		}
	}
}

func expm(x float64) float64 { return math.Exp(-x) }

func BenchmarkQuadCos(b *testing.B) {
	benchQuad(b, expm, 0, 10, &QuadOptions{Weight: WeightCos, Omega: 20})
}

func BenchmarkQuadFourier(b *testing.B) {
	benchQuad(b, expm, 0, math.Inf(1), &QuadOptions{Weight: WeightSin, Omega: 1})
}

func BenchmarkQuadAlgLog(b *testing.B) {
	benchQuad(b, func(x float64) float64 { return 1 / (1 + x) }, 0, 1,
		&QuadOptions{Weight: WeightAlgLogA, Alpha: -0.5})
}

func BenchmarkQuadCauchy(b *testing.B) {
	benchQuad(b, func(x float64) float64 { return 1 / (1 + x*x) }, -1, 2,
		&QuadOptions{Weight: WeightCauchy, C: 0.5})
}

func BenchmarkQuadPoints(b *testing.B) {
	benchQuad(b, func(x float64) float64 { return math.Sqrt(math.Abs(x - 0.3)) }, 0, 1,
		&QuadOptions{Points: []float64{0.3}})
}

func BenchmarkDblQuad(b *testing.B) {
	f := func(y, x float64) float64 { return math.Exp(-x * y) }
	for b.Loop() {
		if _, err := DblQuad(f, 0, 1, Const(0), func(x float64) float64 { return x }, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTplQuad(b *testing.B) {
	f := func(z, y, x float64) float64 { return math.Exp(-(x*x + y*y + z*z)) }
	for b.Loop() {
		if _, err := TplQuad(f, 0, 1, Const(0), Const(1), Const2(0), Const2(1), nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNQuad(b *testing.B) {
	f := func(x []float64) float64 { return math.Exp(-(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])) }
	r := []Range{ConstRange(0, 1), ConstRange(0, 1), ConstRange(0, 1)}
	for b.Loop() {
		if _, err := NQuad(f, r, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFixedQuad(b *testing.B) {
	for b.Loop() {
		if _, err := FixedQuad(math.Cos, 0, 1, 5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGaussLegendre50(b *testing.B) {
	for b.Loop() {
		GaussLegendre(50)
	}
}

// samples returns n samples of sin on a non-uniform grid in [0, 10].
func samples(n int) (y, x []float64) {
	x = make([]float64, n)
	y = make([]float64, n)
	for i := range x {
		t := float64(i) / float64(n-1)
		x[i] = 10 * t * t
		y[i] = math.Sin(x[i])
	}
	return y, x
}

func BenchmarkTrapezoid1e5(b *testing.B) {
	y, x := samples(100000)
	for b.Loop() {
		if _, err := Trapezoid(y, x, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSimpson1e5(b *testing.B) {
	y, x := samples(100000)
	for b.Loop() {
		if _, err := Simpson(y, x, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCumulativeTrapezoid1e5(b *testing.B) {
	y, x := samples(100000)
	for b.Loop() {
		if _, err := CumulativeTrapezoid(y, x, 0, true); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRomb(b *testing.B) {
	y, _ := samples(1<<16 + 1)
	for b.Loop() {
		if _, err := Romb(y, 0.001); err != nil {
			b.Fatal(err)
		}
	}
}

// grid returns a 1000×100 array of samples.
func grid() *ndarray.Array {
	v := make([]float64, 1000*100)
	for i := range v {
		v[i] = math.Sin(float64(i) * 0.001)
	}
	a, _ := ndarray.FromData(v, 1000, 100)
	return a
}

func BenchmarkTrapezoidNDAxis0(b *testing.B) {
	g := grid()
	for b.Loop() {
		if _, err := TrapezoidND(g, nil, 0.1, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrapezoidNDAxis1(b *testing.B) {
	g := grid()
	for b.Loop() {
		if _, err := TrapezoidND(g, nil, 0.1, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSimpsonNDAxis1(b *testing.B) {
	g := grid()
	for b.Loop() {
		if _, err := SimpsonND(g, nil, 0.1, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCumulativeTrapezoidNDAxis0(b *testing.B) {
	g := grid()
	for b.Loop() {
		if _, err := CumulativeTrapezoidND(g, nil, 0.1, 0, true); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveIVPLotkaRK23(b *testing.B) {
	o := &IVPOptions{Method: RK23}
	for b.Loop() {
		if _, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, o); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveIVPLotkaDense(b *testing.B) {
	te := make([]float64, 1001)
	for i := range te {
		te[i] = 15 * float64(i) / 1000
	}
	o := &IVPOptions{TEval: te, DenseOutput: true}
	for b.Loop() {
		if _, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, o); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveIVPLotkaEvents(b *testing.B) {
	o := &IVPOptions{Events: []Event{{F: func(_ float64, y []float64) float64 { return y[0] - 5 }}}}
	for b.Loop() {
		if _, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, o); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOdeintLotka(b *testing.B) {
	ts := make([]float64, 101)
	for i := range ts {
		ts[i] = 15 * float64(i) / 100
	}
	for b.Loop() {
		if _, _, err := Odeint(lotkaF, []float64{10, 5}, ts, nil); err != nil {
			b.Fatal(err)
		}
	}
}
