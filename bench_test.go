// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"math"
	"testing"
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
