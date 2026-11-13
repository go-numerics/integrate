// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"math"
	"testing"
)

// TestOdeintTCrit: with critical times the integration never steps past them:
// a right-hand side that jumps at t = 1/2 is integrated to the jump, and on
// from it, with the exact solution of each piece.
func TestOdeintTCrit(t *testing.T) {
	jump := func(tt float64, y, d []float64) {
		d[0] = -y[0]
		if tt > 0.5 {
			d[0] = -5 * y[0]
		}
	}
	ts := []float64{0}
	for i := 1; i <= 60; i++ {
		ts = append(ts, 0.01*float64(i)) // several outputs per step: intdy between steps
	}
	ts = append(ts, 0.7, 1.5, 2)
	y, info, err := Odeint(jump, []float64{1}, ts, &OdeintOptions{RTol: 1e-10, ATol: 1e-12, TCrit: []float64{0.5, 1.2}})
	if err != nil {
		t.Fatal(err)
	}
	for k, tk := range ts {
		want := math.Exp(-tk)
		if tk > 0.5 {
			want = math.Exp(-0.5) * math.Exp(-5*(tk-0.5))
		}
		if e := math.Abs(y.At(k, 0) - want); e > 1e-8 {
			t.Errorf("t = %v: error %g", tk, e)
		}
	}
	if info.TCur[49] > 0.5 { // the step to the output time 0.5 ends at tcrit, not beyond
		t.Errorf("TCur at t = 0.5: %v", info.TCur[49])
	}
	// a long step cut to end at tcrit = tout
	y, info, err = Odeint(decayF, []float64{1}, []float64{0, 5}, &OdeintOptions{TCrit: []float64{5}})
	if err != nil || info.TCur[0] != 5 || math.Abs(y.At(1, 0)-math.Exp(-2.5)) > 1e-4 {
		t.Errorf("tcrit = tout: %v %v %v", err, info.TCur, y.At(1, 0))
	}
}

// TestLsodaItask4: the stop conditions of itask 4 and a first call whose tcrit
// lies before tout.
func TestLsodaItask4(t *testing.T) {
	y := make([]float64, 2)
	s := lsodaTestSolver(t, lotkaF, nil, []float64{10, 5}, 10)
	s.tcrit = 1
	if _, st := s.call(y, 20, 4); st != -3 {
		t.Errorf("tcrit before tout at the start: istate %d", st)
	}
	s = lsodaTestSolver(t, lotkaF, nil, []float64{10, 5}, 10)
	s.tcrit = 1
	if tt, st := s.call(y, 0.0001, 4); st != 2 || tt != 0.0001 {
		t.Fatalf("first call: %v %d", tt, st)
	}
	s.tcrit = s.tn - 1e-3 // already past
	if _, st := s.call(y, 0.5, 4); st != -3 {
		t.Errorf("tn past tcrit: istate %d", st)
	}
	s.tcrit = s.tn + 1e-18 // reached, to within roundoff, before a step
	if tt, st := s.call(y, s.tcrit, 4); st != 2 || tt != s.tcrit {
		t.Errorf("tn at tcrit: %v %d", tt, st)
	}
	// a step cut to end at tcrit = tout lands just below it and is a hit
	s.tcrit = s.tn + s.h
	want := s.tcrit
	if tt, st := s.call(y, want, 4); st != 2 || tt != want || s.tn >= want {
		t.Errorf("step to tcrit: %v %d, tn %v", tt, st, s.tn)
	}
}

// TestLsodaHmin: the step cannot go below hmin: an error test that fails at
// h <= hmin ends the step (kflag -1, istate -4).
func TestLsodaHmin(t *testing.T) {
	y := make([]float64, 2)
	s := lsodaTestSolver(t, lotkaF, nil, []float64{10, 5}, 10)
	if _, st := s.call(y, 10, 5); st != 2 {
		t.Fatalf("first step: istate %d", st)
	}
	told := s.tn
	s.fun = func(tt float64, _, d []float64) { d[0], d[1] = 1e3/(tt-told), 0 } // fails whatever h
	s.hmin = 1
	if _, st := s.call(y, 10, 5); st != -4 || s.kflag != -1 {
		t.Errorf("hmin: istate %d kflag %d", st, s.kflag)
	}
}

// TestLsodaBandEdges: a banded P that is singular, and a finite-difference
// increment of 1 when f is 0.
func TestLsodaBandEdges(t *testing.T) {
	core, err := newCore(decayF, 0, []float64{1, 1}, 1, &IVPOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ident := func(_ float64, _, jac []float64) { jac[0], jac[1] = 1, 1 } // (0+0+1)×2
	s, err := newLsodaConfig(core, lsodaConfig{band: &Bandwidth{}, bandJac: ident})
	if err != nil {
		t.Fatal(err)
	}
	s.miter, s.h, s.el0 = 4, 1, 1
	s.ewt[0], s.ewt[1] = 1, 1
	s.prja([]float64{1, 1})
	if s.ierpj != 1 {
		t.Errorf("singular band: ierpj %d", s.ierpj)
	}
	zero := func(_ float64, _, d []float64) { clear(d) }
	core, _ = newCore(zero, 0, []float64{1, 1}, 1, &IVPOptions{})
	s, _ = newLsodaConfig(core, lsodaConfig{band: &Bandwidth{}})
	s.miter, s.h, s.el0 = 5, 1, 1
	s.ewt[0], s.ewt[1] = 1, 1
	s.prja([]float64{1, 1})
	if s.ierpj != 0 || s.nfe != 1 || *s.lub.at(0, 0) == 0 {
		t.Errorf("zero f: ierpj %d nfe %d", s.ierpj, s.nfe)
	}
}
