// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

func feature(t *testing.T, fx *odeFixture, key string, v any) {
	t.Helper()
	if err := json.Unmarshal(fx.Features[key], v); err != nil {
		t.Fatalf("%s: %v", key, err)
	}
}

func closeVec(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d entries, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol*math.Max(1, math.Abs(want[i])) {
			t.Errorf("%s[%d] = %.17g, want %.17g", name, i, got[i], want[i])
		}
	}
}

func TestTEvalAndDenseAgainstSciPy(t *testing.T) {
	fx := loadODEFixture(t)
	te := make([]float64, 31)
	for i := range te {
		te[i] = 0.5 * float64(i)
	}
	sol, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, &IVPOptions{TEval: te, DenseOutput: true})
	if err != nil {
		t.Fatal(err)
	}
	var want [][]float64
	feature(t, fx, "teval_y", &want)
	if len(sol.T) != len(te) {
		t.Fatalf("%d output times, want %d", len(sol.T), len(te))
	}
	for k := range want {
		closeVec(t, "teval", sol.Y[k], want[k], 1e-9)
	}
	feature(t, fx, "teval_sol", &want)
	for k, tk := range []float64{0.3, 7.77, 14.9} {
		closeVec(t, "sol", sol.Sol.Eval(tk), want[k], 1e-9)
	}
	ya := sol.YArray()
	if sh := ya.Shape(); sh[0] != 2 || sh[1] != 31 || ya.At(1, 0) != 5 {
		t.Errorf("YArray shape %v", sh)
	}
	if sol.TArray().At(30) != 15 {
		t.Error("TArray")
	}
	if c := sol.Component(0); c[0] != 10 || len(c) != 31 {
		t.Error("Component")
	}
	ea := sol.Sol.EvalArray([]float64{0.3, 7.77})
	if math.Abs(ea.At(0, 1)-sol.Sol.Eval(7.77)[0]) != 0 {
		t.Error("EvalArray")
	}
	if sol.Sol.TMin() != 0 || sol.Sol.TMax() != 15 {
		t.Errorf("dense span [%g, %g]", sol.Sol.TMin(), sol.Sol.TMax())
	}

	// Backwards with t_eval.
	sol, err = SolveIVP(decayF, [2]float64{10, 0}, []float64{1}, &IVPOptions{Method: DOP853, RTol: 1e-8, ATol: 1e-10,
		TEval: []float64{10, 7.5, 5, 2.5, 0}})
	if err != nil {
		t.Fatal(err)
	}
	var back []float64
	feature(t, fx, "backward_y", &back)
	got := sol.Component(0)
	closeVec(t, "backward", got, back, 1e-12)
	var nfev int
	feature(t, fx, "backward_nfev", &nfev)
	if sol.NFev != nfev {
		t.Errorf("backward nfev %d, SciPy %d", sol.NFev, nfev)
	}
}

func TestEventsAgainstSciPy(t *testing.T) {
	fx := loadODEFixture(t)
	ball := func(_ float64, y, d []float64) { d[0], d[1] = y[1], -9.81 }
	events := []Event{
		{F: func(_ float64, y []float64) float64 { return y[0] }, Terminal: 1, Direction: -1},
		{F: func(_ float64, y []float64) float64 { return y[1] }},
	}
	for _, m := range []Method{RK45, DOP853, Radau, BDF, LSODA, RK23} {
		var want struct {
			TEnd    float64     `json:"t_end"`
			TEvents [][]float64 `json:"t_events"`
			YHit    [][]float64 `json:"y_hit"`
			Status  int         `json:"status"`
			NFev    int         `json:"nfev"`
		}
		feature(t, fx, "ball_"+m.String(), &want)
		sol, err := SolveIVP(ball, [2]float64{0, 10}, []float64{10, 5}, &IVPOptions{Method: m, RTol: 1e-6, ATol: 1e-9, Events: events})
		if err != nil {
			t.Fatal(err)
		}
		if sol.Status != 1 || sol.Message != "A termination event occurred." {
			t.Errorf("%v: status %d %q", m, sol.Status, sol.Message)
		}
		exact := 5/9.81 + math.Sqrt(25/(9.81*9.81)+20/9.81)
		tol := 1e-12
		if m == BDF || m == LSODA {
			tol = 1e-5 // solution accuracy, not root-finding, limits these
		}
		if math.Abs(sol.T[len(sol.T)-1]-exact) > tol*10 || math.Abs(sol.TEvents[0][0]-exact) > tol*10 {
			t.Errorf("%v: hit at %.17g, exact %.17g", m, sol.TEvents[0][0], exact)
		}
		if m != LSODA {
			if math.Abs(sol.TEvents[0][0]-want.TEvents[0][0]) > 1e-12 || math.Abs(sol.TEvents[1][0]-want.TEvents[1][0]) > 1e-12 {
				t.Errorf("%v: events %v, SciPy %v", m, sol.TEvents, want.TEvents)
			}
			if sol.NFev != want.NFev {
				t.Errorf("%v: nfev %d, SciPy %d", m, sol.NFev, want.NFev)
			}
		}
		if math.Abs(sol.TEvents[1][0]-5/9.81) > 1e-9 {
			t.Errorf("%v: apex at %.17g", m, sol.TEvents[1][0])
		}
		if math.Abs(sol.YEvents[0][0][0]) > 1e-9 {
			t.Errorf("%v: height at the hit %g", m, sol.YEvents[0][0][0])
		}
	}

	var z struct {
		TEvents []float64 `json:"t_events"`
		TEnd    float64   `json:"t_end"`
	}
	feature(t, fx, "harmonic_zero3", &z)
	sol, err := SolveIVP(harmonicF, [2]float64{0, 20}, []float64{1, 0}, &IVPOptions{RTol: 1e-8, ATol: 1e-10,
		Events: []Event{{F: func(_ float64, y []float64) float64 { return y[0] }, Terminal: 3}}})
	if err != nil {
		t.Fatal(err)
	}
	closeVec(t, "zero3", sol.TEvents[0], z.TEvents, 1e-12)
	if sol.T[len(sol.T)-1] != z.TEvents[2] {
		t.Errorf("end %.17g, want %.17g", sol.T[len(sol.T)-1], z.TEvents[2])
	}
}

func TestOdeintAgainstSciPy(t *testing.T) {
	fx := loadODEFixture(t)
	var rob struct {
		T     []float64   `json:"t"`
		Y     [][]float64 `json:"y"`
		NFE   int         `json:"nfe"`
		MUsed []int       `json:"mused"`
	}
	feature(t, fx, "odeint_robertson", &rob)
	var ref [][]float64
	feature(t, fx, "odeint_robertson_ref", &ref)
	y, info, err := Odeint(robertsonF, []float64{1, 0, 0}, rob.T, &OdeintOptions{RTol: 1e-8, ATol: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	if sh := y.Shape(); sh[0] != 7 || sh[1] != 3 {
		t.Fatalf("shape %v", sh)
	}
	for k := range rob.T {
		for i := 0; i < 3; i++ {
			got := y.At(k, i)
			// Accuracy relative to the reference, compared with
			// ODEPACK's at the same tolerances.
			e := math.Abs(got - ref[k][i])
			eo := math.Abs(rob.Y[k][i] - ref[k][i])
			if e > math.Max(10*eo, 1e-12+1e-6*math.Abs(ref[k][i])) {
				t.Errorf("t=%g y%d: %.10g (err %.3g), ODEPACK %.10g (err %.3g)", rob.T[k], i, got, e, rob.Y[k][i], eo)
			}
		}
	}
	if info.MUsed[len(info.MUsed)-1] != 2 {
		t.Errorf("Robertson should end on the stiff method, MUsed %v", info.MUsed)
	}
	t.Logf("Robertson odeint: nfe %d (ODEPACK %d), MUsed %v (ODEPACK %v)", info.NFE[len(info.NFE)-1], rob.NFE, info.MUsed, rob.MUsed)

	var lv struct {
		T []float64   `json:"t"`
		Y [][]float64 `json:"y"`
	}
	feature(t, fx, "odeint_lotka", &lv)
	feature(t, fx, "odeint_lotka_ref", &ref)
	y, info, err = Odeint(lotkaF, []float64{10, 5}, lv.T, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range lv.T {
		for i := 0; i < 2; i++ {
			e := math.Abs(y.At(k, i) - ref[k][i])
			eo := math.Abs(lv.Y[k][i] - ref[k][i])
			if e > math.Max(10*eo, 1e-9) {
				t.Errorf("Lotka t=%g y%d err %.3g, ODEPACK %.3g", lv.T[k], i, e, eo)
			}
		}
	}
	if info.MUsed[0] != 1 {
		t.Errorf("Lotka–Volterra is not stiff, MUsed %v", info.MUsed)
	}
}

func TestOdeintErrors(t *testing.T) {
	if _, _, err := Odeint(lotkaF, []float64{1, 1}, nil, nil); !errors.Is(err, ErrODEInput) {
		t.Errorf("empty t: %v", err)
	}
	if _, _, err := Odeint(lotkaF, []float64{1, 1}, []float64{0, 2, 1}, nil); !errors.Is(err, ErrODEInput) {
		t.Errorf("non-monotonic t: %v", err)
	}
	y, info, err := Odeint(lotkaF, []float64{1, 1}, []float64{3}, nil)
	if err != nil || y.At(0, 1) != 1 || len(info.NFE) != 0 {
		t.Errorf("single time: %v", err)
	}
	if _, _, err := Odeint(lotkaF, []float64{1, 1}, []float64{0, 1}, &OdeintOptions{ATolVec: []float64{1}}); !errors.Is(err, ErrODEInput) {
		t.Errorf("bad ATolVec: %v", err)
	}
	y, info, err = Odeint(lotkaF, []float64{10, 5}, []float64{0, 15}, &OdeintOptions{MxStep: 3})
	if !errors.Is(err, ErrExcessWork) || !math.IsNaN(y.At(1, 0)) || info.Message == "" {
		t.Errorf("MxStep: %v", err)
	}
	// Backwards, with a Jacobian, H0 and HMax.
	y, _, err = Odeint(vdpF(1), []float64{2, 0}, []float64{1, 0.5, 0}, &OdeintOptions{Jac: vdpJ(1), H0: 1e-3, HMax: 0.1, RTol: 1e-10, ATol: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	back, _ := SolveIVP(vdpF(1), [2]float64{1, 0}, []float64{2, 0}, &IVPOptions{Method: DOP853, RTol: 1e-12, ATol: 1e-14})
	if e := math.Abs(y.At(2, 0) - back.Y[len(back.Y)-1][0]); e > 1e-8 {
		t.Errorf("backward odeint error %g", e)
	}
	// A large MxStep so that the step-size floor, not the step budget, stops
	// it on every architecture (s390x reached 500 steps first).
	_, _, err = Odeint(func(_ float64, y, d []float64) { d[0] = 1 / (1 - y[0]) }, []float64{0}, []float64{0, 1}, &OdeintOptions{MxStep: 1000000})
	if !errors.Is(err, ErrStepTooSmall) {
		t.Errorf("blow-up: %v", err)
	}
}
