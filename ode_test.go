// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"errors"
	"math"
	"testing"
)

var allMethods = []Method{RK45, RK23, DOP853, Radau, BDF, LSODA}

func TestSolveIVPInvalid(t *testing.T) {
	if s := Method(99).String(); s != "Method(99)" {
		t.Error(s)
	}
	f := decayF
	y0 := []float64{1}
	span := [2]float64{0, 1}
	for name, c := range map[string]struct {
		span [2]float64
		o    *IVPOptions
	}{
		"method":      {span, &IVPOptions{Method: Method(99)}},
		"nan span":    {[2]float64{0, math.NaN()}, nil},
		"inf span":    {[2]float64{math.Inf(-1), 0}, nil},
		"teval range": {span, &IVPOptions{TEval: []float64{0.5, 2}}},
		"teval order": {span, &IVPOptions{TEval: []float64{0.5, 0.2}}},
		"teval back":  {[2]float64{1, 0}, &IVPOptions{TEval: []float64{0.2, 0.5}}},
		"rtolvec":     {span, &IVPOptions{RTolVec: []float64{1, 2}}},
		"atolvec":     {span, &IVPOptions{ATolVec: []float64{-1}}},
		"maxstep":     {span, &IVPOptions{MaxStep: -1}},
		"firststep":   {span, &IVPOptions{FirstStep: 2}},
		"firstneg":    {span, &IVPOptions{FirstStep: -1}},
	} {
		if sol, err := SolveIVP(f, c.span, y0, c.o); !errors.Is(err, ErrODEInput) || sol != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	// Valid per-component tolerances.
	sol, err := SolveIVP(f, span, []float64{1, 2}, &IVPOptions{RTolVec: []float64{1e-6, 1e-8}, ATolVec: []float64{1e-9, 1e-9}})
	if err != nil || math.Abs(sol.Y[len(sol.Y)-1][1]-2*math.Exp(-0.5)) > 1e-7 {
		t.Errorf("vector tolerances: %v", err)
	}
}

func TestSolveIVPFailure(t *testing.T) {
	blowup := func(_ float64, y, d []float64) { d[0] = y[0] * y[0] }
	for _, m := range allMethods {
		sol, err := SolveIVP(blowup, [2]float64{0, 2}, []float64{1}, &IVPOptions{Method: m})
		if !errors.Is(err, ErrStepTooSmall) || sol.Success || sol.Status != -1 || sol.Message != tooSmallStep {
			t.Errorf("%v: %v %+v", m, err, sol)
		}
		if tl := sol.T[len(sol.T)-1]; tl > 1.01 || tl < 0.99 {
			t.Errorf("%v: stopped at %g", m, tl)
		}
	}
	// Stiff first, then a blow-up the BDF side of the switch meets.
	stiffBlow := func(tt float64, y, d []float64) {
		d[0] = -1000 * (y[0] - math.Cos(tt))
		d[1] = y[1] * y[1]
	}
	sol, err := SolveIVP(stiffBlow, [2]float64{0, 2}, []float64{1, 1}, &IVPOptions{Method: LSODA})
	if !errors.Is(err, ErrStepTooSmall) || sol.Success {
		t.Errorf("LSODA stiff blow-up: %v", err)
	}
	// So stiff that BDF never hands back, and undefined past t = 0.5: the
	// failure is BDF's.
	stiffNaN := func(tt float64, y, d []float64) {
		d[0] = -1e15 * (y[0] - math.Cos(tt))
		d[1] = 1
		if tt > 0.5 {
			d[1] = math.NaN()
		}
	}
	sol, err = SolveIVP(stiffNaN, [2]float64{0, 1}, []float64{1, 0}, &IVPOptions{Method: LSODA})
	if !errors.Is(err, ErrStepTooSmall) || sol.Success || sol.T[len(sol.T)-1] > 0.5 {
		t.Errorf("LSODA stiff NaN: %v at t=%g", err, sol.T[len(sol.T)-1])
	}
}

func TestSolveIVPEdgeCases(t *testing.T) {
	// An empty system and an empty span.
	for _, m := range allMethods {
		sol, err := SolveIVP(decayF, [2]float64{0, 1}, nil, &IVPOptions{Method: m, DenseOutput: true})
		if err != nil || len(sol.T) != 2 || sol.T[1] != 1 {
			t.Errorf("%v empty: %v %v", m, err, sol.T)
		}
		if y := sol.Sol.Eval(0.5); len(y) != 0 {
			t.Errorf("%v empty dense: %v", m, y)
		}
		sol, err = SolveIVP(decayF, [2]float64{1, 1}, []float64{3}, &IVPOptions{Method: m, DenseOutput: true})
		if err != nil || sol.Y[len(sol.Y)-1][0] != 3 || sol.Sol.Eval(1)[0] != 3 {
			t.Errorf("%v zero span: %v", m, err)
		}
	}
	// A zero right-hand side: the initial step falls back to its floor and
	// DOP853's error estimate is exactly zero.
	zero := func(_ float64, _, d []float64) { clear(d) }
	for _, m := range allMethods {
		sol, err := SolveIVP(zero, [2]float64{0, 10}, []float64{1, 2}, &IVPOptions{Method: m})
		if err != nil || sol.Y[len(sol.Y)-1][1] != 2 {
			t.Errorf("%v zero rhs: %v", m, err)
		}
	}
	// Step size clamps: a tiny first step (below the spacing at t0) and a
	// small MaxStep.
	for _, m := range allMethods {
		sol, err := SolveIVP(decayF, [2]float64{1, 2}, []float64{1}, &IVPOptions{Method: m, FirstStep: 1e-300})
		if err != nil || math.Abs(sol.Y[len(sol.Y)-1][0]-math.Exp(-0.5)) > 1e-3 {
			t.Errorf("%v tiny first step: %v", m, err)
		}
		sol, err = SolveIVP(decayF, [2]float64{0, 1}, []float64{1}, &IVPOptions{Method: m, MaxStep: 0.05})
		if err != nil || sol.NSteps < 20 {
			t.Errorf("%v max step: %v %d", m, err, sol.NSteps)
		}
	}
	// Zero state with no absolute tolerance: the finite-difference step
	// starts at zero and grows until it is non-zero.
	for _, m := range []Method{Radau, BDF} {
		sol, err := SolveIVP(decayF, [2]float64{0, 1}, []float64{0}, &IVPOptions{Method: m, ATol: -1})
		if err != nil && !errors.Is(err, ErrStepTooSmall) {
			t.Errorf("%v zero state: %v", m, err)
		}
		_ = sol
	}
	// A right-hand side that is NaN outside its domain: Newton must reject
	// the iterate and retry with a smaller step.
	sqrtF := func(_ float64, y, d []float64) { d[0] = -2 * math.Sqrt(y[0]) }
	for _, m := range []Method{Radau, BDF} {
		_, err := SolveIVP(sqrtF, [2]float64{0, 3}, []float64{1}, &IVPOptions{Method: m, RTol: 1e-8, ATol: 1e-12})
		if err != nil && !errors.Is(err, ErrStepTooSmall) {
			t.Errorf("%v NaN rhs: %v", m, err)
		}
	}
	// A discontinuous right-hand side forces repeated rejections.
	jump := func(tt float64, y, d []float64) {
		d[0] = -y[0]
		if math.Mod(tt, 1) > 0.5 {
			d[0] = 1e4
		}
	}
	for _, m := range allMethods {
		if _, err := SolveIVP(jump, [2]float64{0, 3}, []float64{1}, &IVPOptions{Method: m, RTol: 1e-8, ATol: 1e-10}); err != nil {
			t.Errorf("%v discontinuous: %v", m, err)
		}
	}
}

func TestDenseOutputDirections(t *testing.T) {
	exact := func(tt float64) float64 { return math.Exp(-0.5 * (tt - 5)) }
	for _, m := range allMethods {
		for _, span := range [][2]float64{{5, 10}, {5, 0}} {
			sol, err := SolveIVP(decayF, span, []float64{1}, &IVPOptions{Method: m, DenseOutput: true, RTol: 1e-8, ATol: 1e-10})
			if err != nil {
				t.Fatal(err)
			}
			if len(sol.Sol.ts) != len(sol.T) {
				t.Errorf("%v %v: %d breakpoints for %d times", m, span, len(sol.Sol.ts), len(sol.T))
			}
			for _, tt := range append([]float64{span[0], span[1]}, sol.T...) {
				if e := math.Abs(sol.Sol.Eval(tt)[0] - exact(tt)); e > 1e-6*math.Max(1, exact(tt)) {
					t.Errorf("%v %v: dense error %g at %g", m, span, e, tt)
				}
			}
		}
	}
}

func TestEventsBackwardAndOrdering(t *testing.T) {
	// Integrating backwards from t=10 to 0: two terminal events in the same
	// step must stop at the one met first (the larger time).
	lin := func(_ float64, _, d []float64) { d[0] = 1 }
	events := []Event{
		{F: func(_ float64, y []float64) float64 { return y[0] - 4.0 }, Terminal: 1},
		{F: func(_ float64, y []float64) float64 { return y[0] - 4.5 }, Terminal: 1},
	}
	sol, err := SolveIVP(lin, [2]float64{10, 0}, []float64{10}, &IVPOptions{Events: events, MaxStep: 100})
	if err != nil || sol.Status != 1 {
		t.Fatalf("%v %v", err, sol.Status)
	}
	if math.Abs(sol.T[len(sol.T)-1]-4.5) > 1e-12 || len(sol.TEvents[0]) != 0 || len(sol.TEvents[1]) != 1 {
		t.Errorf("stopped at %g, events %v", sol.T[len(sol.T)-1], sol.TEvents)
	}
}

func TestBrentq(t *testing.T) {
	if r := brentq(func(x float64) float64 { return x }, 0, 1, 1e-12, 1e-12, 100); r != 0 {
		t.Errorf("root at xa: %g", r)
	}
	if r := brentq(func(x float64) float64 { return x - 1 }, 0, 1, 1e-12, 1e-12, 100); r != 1 {
		t.Errorf("root at xb: %g", r)
	}
	step := func(x float64) float64 {
		if x < 0.3 {
			return -1
		}
		return 1
	}
	if r := brentq(step, 0, 1, 1e-12, 1e-12, 100); math.Abs(r-0.3) > 1e-11 {
		t.Errorf("step: %g", r)
	}
	cubic := func(x float64) float64 { return x*x*x - 2*x - 5 }
	if r := brentq(cubic, 2, 3, 4*dblEps, 4*dblEps, 100); math.Abs(cubic(r)) > 1e-13 {
		t.Errorf("cubic: %g", r)
	}
	if r := brentq(cubic, 2, 3, 0, 0, 2); r == 2 || r == 3 {
		t.Errorf("iteration cap: %g", r)
	}
	for _, c := range []struct {
		f    func(float64) float64
		root float64
	}{
		{func(x float64) float64 { return math.Pow(x, 9) - 1e-9 }, 0.1},
		{func(x float64) float64 { return math.Exp(10*x) - 2 }, math.Ln2 / 10},
		{func(x float64) float64 { return math.Atan(1000 * (x - 0.123)) }, 0.123},
		{func(x float64) float64 { return (x - 0.4) * (x - 0.4) * (x - 0.4) }, 0.4},
		{func(x float64) float64 { return math.Log(x + 1e-9) }, 1 - 1e-9},
		{func(x float64) float64 { return 1/(x+1e-6) - 5 }, 0.2 - 1e-6},
	} {
		if r := brentq(c.f, 0, 1, 4*dblEps, 4*dblEps, 100); math.Abs(r-c.root) > 1e-9 {
			t.Errorf("root %g, want %g", r, c.root)
		}
	}
	flat := func(x float64) float64 { return math.Tanh(50 * (x - 0.7)) }
	if r := brentq(flat, -10, 1, 1e-14, 1e-14, 200); math.Abs(r-0.7) > 1e-12 {
		t.Errorf("flat: %g", r)
	}
}

func TestLinalgEdges(t *testing.T) {
	lu := newLUReal(2)
	lu.factor([]float64{0, 0, 0, 1})
	b := []float64{1, 1}
	lu.solve(b)
	if allFinite(b) {
		t.Errorf("singular real solve: %v", b)
	}
	lu.factor([]float64{2, 0, 0, 4})
	b = []float64{2, 4}
	lu.solve(b)
	if b[0] != 1 || b[1] != 1 {
		t.Errorf("diagonal: %v", b)
	}
	luc := newLUComplex(2)
	luc.factor([]complex128{0, 0, 0, 1})
	luc.factor([]complex128{1i, 0, 0, 2})
	bc := []complex128{1i, 2}
	luc.solve(bc)
	if bc[0] != 1 || bc[1] != 1 {
		t.Errorf("complex diagonal: %v", bc)
	}
	luc.factor([]complex128{1, 2, 3i, 4})
	bc = []complex128{3, 4 + 3i}
	luc.solve(bc)
	if math.Abs(real(bc[0])-1) > 1e-15 || math.Abs(real(bc[1])-1) > 1e-15 {
		t.Errorf("complex pivoted: %v", bc)
	}
	if spectralRadius(make([]float64, 4), 2) != 0 {
		t.Error("spectral radius of zero")
	}
	if r := spectralRadius([]float64{-3, 0, 0, 1}, 2); math.Abs(r-3) > 1e-6 {
		t.Errorf("spectral radius %g", r)
	}
}
