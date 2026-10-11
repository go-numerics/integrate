// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func TestQuadInvalidInput(t *testing.T) {
	inf := math.Inf(1)
	for _, c := range []struct {
		name string
		a, b float64
		o    *QuadOptions
	}{
		{"limit", 0, 1, &QuadOptions{Limit: -1}},
		{"qags tol", 0, 1, &QuadOptions{EpsAbs: -1, EpsRel: 1e-20}},
		{"qagi tol", 0, inf, &QuadOptions{EpsAbs: -1, EpsRel: 1e-20}},
		{"qagp tol", 0, 1, &QuadOptions{EpsAbs: -1, EpsRel: 1e-20, Points: []float64{0.5}}},
		{"points inf", 0, inf, &QuadOptions{Points: []float64{0.5}}},
		{"points limit", 0, 1, &QuadOptions{Points: []float64{0.1, 0.2}, Limit: 2}},
		{"points+weight", 0, 1, &QuadOptions{Points: []float64{0.5}, Weight: WeightCos}},
		{"maxp1", 0, 1, &QuadOptions{Weight: WeightCos, MaxP1: -1}},
		{"cos both inf", math.Inf(-1), inf, &QuadOptions{Weight: WeightCos}},
		{"qawf epsabs", 0, inf, &QuadOptions{Weight: WeightCos, Omega: 1, EpsAbs: -1}},
		{"qawf left limlst", math.Inf(-1), 0, &QuadOptions{Weight: WeightSin, Omega: 1, LimLst: 2}},
		{"qawo tol", 0, 1, &QuadOptions{Weight: WeightSin, Omega: 1, EpsAbs: -1, EpsRel: 1e-20}},
		{"alg inf", 0, inf, &QuadOptions{Weight: WeightAlg}},
		{"alg alpha", 0, 1, &QuadOptions{Weight: WeightAlg, Alpha: -1}},
		{"alg limit", 0, 1, &QuadOptions{Weight: WeightAlg, Limit: 1}},
		{"alg tol", 0, 1, &QuadOptions{Weight: WeightAlgLog, EpsAbs: -1, EpsRel: 1e-20}},
		{"cauchy inf", 0, inf, &QuadOptions{Weight: WeightCauchy}},
		{"cauchy end", 0, 1, &QuadOptions{Weight: WeightCauchy, C: 1}},
		{"cauchy tol", 0, 1, &QuadOptions{Weight: WeightCauchy, C: 0.5, EpsAbs: -1, EpsRel: 1e-20}},
		{"weight", 0, 1, &QuadOptions{Weight: Weight(99)}},
	} {
		r, err := Quad(math.Exp, c.a, c.b, c.o)
		var qe *QuadError
		if !errors.Is(err, ErrInvalidInput) || !errors.As(err, &qe) || qe.Ier != 6 || r.Value != 0 || r.NEval != 0 {
			t.Errorf("%s: %v, %+v", c.name, err, r)
		}
		if !strings.HasPrefix(err.Error(), "integrate: invalid input") {
			t.Errorf("%s: message %q", c.name, err)
		}
	}
	if e := (&QuadError{Ier: 2}).Error(); e != ErrRoundoff.Error() {
		t.Errorf("bare QuadError message %q", e)
	}
	if r, err := Quad(math.Exp, 2, 2, nil); err != nil || r.Value != 0 {
		t.Error("empty interval")
	}
}

func TestQuadFlipAndRoutines(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	for _, o := range []*QuadOptions{nil, {Points: []float64{0.5, 0.5, 7}}, {Weight: WeightCos, Omega: 3},
		{Weight: WeightAlg}, {Weight: WeightCauchy, C: 0.25}} {
		r1, _ := Quad(f, 0, 1, o)
		r2, _ := Quad(f, 1, 0, o)
		if r1.Value != -r2.Value {
			t.Errorf("%+v: %g vs %g", o, r1.Value, r2.Value)
		}
	}
	r, err := Quad(func(x float64) float64 { return math.Exp(x) }, 0, math.Inf(-1), nil)
	if err != nil || math.Abs(r.Value+1) > 1e-12 {
		t.Errorf("reversed infinite range: %v %v", r, err)
	}
}

func TestQAWFTail(t *testing.T) {
	for _, c := range []struct {
		abserr, result, ps, errsum float64
		ier                        int
		wantErr                    float64
		useSum                     bool
	}{
		{1, 2, 2, 5, 0, 1 + 10, false},
		{1, 0, 2, 5, 1, 11, true},
		{1, 0, 0, 20, 1, 11, false},
		{1, 2, 2, 0.1, 1, 11, true},
		{1e-9, 2, 2, 100, 1, 1e-9 + 10 + 0.5, false},
		{1e-9, 2, 2, 100, 7, 1e-9 + 10, false},
	} {
		e, u := qawfeTail(c.abserr, 1, c.result, c.ps, c.errsum, 0.5, c.ier)
		if math.Abs(e-c.wantErr) > 1e-12 || u != c.useSum {
			t.Errorf("%+v: (%g, %v)", c, e, u)
		}
	}
}

func TestExtrapTail(t *testing.T) {
	w := newWorkspace(3)
	w.rlist[1], w.rlist[2] = 1, 2
	// result 0: the larger extrapolation error selects the sum.
	r, e, ier := extrapTail(w, 2, 0, 5, 1, 1, 0, 1, 0, 3, 1, false)
	if r != 3 || e != 1 || ier != 3 {
		t.Errorf("useSum: %g %g %d", r, e, ier)
	}
	// area 0 with a small error keeps the extrapolated result unchecked.
	r, e, ier = extrapTail(w, 2, 0, 0.5, 0, 1, 0, 1, 1, 0, 1, false)
	if r != 0 || e != 0.5 || ier != 1 {
		t.Errorf("area 0: %g %g %d", r, e, ier)
	}
	var et epsTable
	n := 2
	et.tab[1], et.tab[2] = 1, 2
	if res, abserr := et.qelg(&n); res != 2 || abserr != oflow {
		t.Errorf("qelg n<3: %g %g", res, abserr)
	}
}

func TestFixedRules(t *testing.T) {
	if x, w := GaussLegendre(0); x != nil || w != nil {
		t.Error("n=0")
	}
	x, w := GaussLegendre(1)
	if x[0] != 0 || w[0] != 2 {
		t.Errorf("n=1: %v %v", x, w)
	}
	x2, _ := GaussLegendre(1)
	if &x2[0] != &x[0] {
		t.Error("cache")
	}
	for _, n := range []int{2, 3, 64, 101} {
		x, w := GaussLegendre(n)
		s, m := 0.0, 0.0
		for i := range x {
			s += w[i]
			m += w[i] * x[i] * x[i]
		}
		if math.Abs(s-2) > 1e-14 || math.Abs(m-2.0/3) > 1e-14 {
			t.Errorf("n=%d: Σw=%g Σwx²=%g", n, s, m)
		}
	}
	if _, err := FixedQuad(math.Sin, 0, 1, 0); !errors.Is(err, ErrSamples) {
		t.Error("FixedQuad n=0")
	}
	v, err := Romberg(math.Exp, 0, 1, nil)
	if err != nil || math.Abs(v-(math.E-1)) > 1e-12 {
		t.Errorf("Romberg: %v %v", v, err)
	}
	v, err = Romberg(math.Sqrt, 0, 1, &RombergOptions{Tol: 1e-14, RTol: 1e-14, DivMax: 4})
	if !errors.Is(err, ErrDivMax) || math.Abs(v-2.0/3) > 1e-2 {
		t.Errorf("Romberg divmax: %v %v", v, err)
	}
	if _, err := Romb([]float64{1, 2, 3, 4}, 1); !errors.Is(err, ErrSamples) {
		t.Error("Romb 4 samples")
	}
	if v, err := Romb([]float64{1, 3}, 2); err != nil || v != 4 {
		t.Errorf("Romb 2 samples: %v %v", v, err)
	}
	if _, err := Trapezoid([]float64{1, 2}, []float64{1}, 0); !errors.Is(err, ErrSamples) {
		t.Error("Trapezoid x length")
	}
	if _, err := CumulativeTrapezoid(nil, nil, 1, false); !errors.Is(err, ErrSamples) {
		t.Error("CumulativeTrapezoid empty")
	}
	if _, err := Simpson([]float64{1}, nil, 1); !errors.Is(err, ErrSamples) {
		t.Error("Simpson 1 sample")
	}
	// Repeated abscissae: SciPy's zero-division guards.
	v, err = Simpson([]float64{1, 1, 1, 1}, []float64{0, 0, 1, 2}, 0)
	if err != nil || math.IsNaN(v) {
		t.Errorf("Simpson repeated x: %v %v", v, err)
	}
	if v, _ := Simpson([]float64{1, 1, 1}, []float64{0, 1, 1}, 0); math.IsNaN(v) {
		t.Error("Simpson zero h1")
	}
}

func TestAxisRules(t *testing.T) {
	// y[i][j] = (i+1)·x_j² on x = 0, 0.5, 1, 1.5, 2.
	x := []float64{0, 0.5, 1, 1.5, 2}
	data := make([]float64, 0, 15)
	for i := 0; i < 3; i++ {
		for _, v := range x {
			data = append(data, float64(i+1)*v*v)
		}
	}
	y, _ := ndarray.FromData(data, 3, 5)
	s, err := SimpsonND(y, x, 0, -1)
	if err != nil || s.Shape()[0] != 3 {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(s.At(i)-float64(i+1)*8/3) > 1e-14 {
			t.Errorf("Simpson row %d: %g", i, s.At(i))
		}
	}
	tr, err := TrapezoidND(y, nil, 1, 0)
	if err != nil || tr.Shape()[0] != 5 || tr.At(4) != 16 {
		t.Errorf("Trapezoid axis 0: %v %v", tr, err)
	}
	ct, err := CumulativeTrapezoidND(y, x, 0, 1, true)
	if err != nil || ct.Shape()[1] != 5 || ct.At(0, 0) != 0 {
		t.Fatalf("cumulative: %v", err)
	}
	ct, _ = CumulativeTrapezoidND(y, x, 0, 1, false)
	if ct.Shape()[1] != 4 {
		t.Errorf("cumulative shape %v", ct.Shape())
	}
	if _, err := TrapezoidND(y, nil, 1, 2); !errors.Is(err, ErrSamples) {
		t.Error("axis range")
	}
	if _, err := CumulativeTrapezoidND(y, nil, 1, -3, false); !errors.Is(err, ErrSamples) {
		t.Error("cumulative axis range")
	}
	one, _ := ndarray.FromData([]float64{1, 2, 3}, 3, 1)
	if _, err := CumulativeTrapezoidND(one, nil, 1, 1, false); !errors.Is(err, ErrSamples) {
		t.Error("cumulative needs 2 samples")
	}
	if _, err := SimpsonND(one, nil, 1, 1); !errors.Is(err, ErrSamples) {
		t.Error("simpson lane error")
	}
	v1, _ := ndarray.FromData([]float64{0, 1, 4}, 3)
	r, err := TrapezoidND(v1, nil, 1, 0)
	if err != nil || len(r.Shape()) != 0 || r.At() != 3 {
		t.Errorf("0-d result: %v %v", r, err)
	}
}

func TestNQuadErrors(t *testing.T) {
	if _, err := NQuad(func([]float64) float64 { return 1 }, nil, nil); !errors.Is(err, ErrInvalidInput) {
		t.Error("no ranges")
	}
	r := ConstRange(0, 1)
	if _, err := NQuad(func([]float64) float64 { return 1 }, []Range{r, r, r}, &NQuadOptions{Opts: []*QuadOptions{nil, nil}}); !errors.Is(err, ErrInvalidInput) {
		t.Error("option count")
	}
	res, err := NQuad(func(x []float64) float64 { return x[0] + x[1] }, []Range{r, r},
		&NQuadOptions{Opts: []*QuadOptions{{EpsRel: 1e-10}, {Points: []float64{0.5}}}})
	if err != nil || math.Abs(res.Value-1) > 1e-14 || res.NEval == 0 {
		t.Errorf("per-level options: %v %v", res, err)
	}
	if _, err := NQuad(func(x []float64) float64 { return 1 }, []Range{r}, &NQuadOptions{Opts: []*QuadOptions{{Weight: Weight(42)}}}); !errors.Is(err, ErrInvalidInput) {
		t.Error("invalid inner options")
	}
	res, err = DblQuad(func(y, x float64) float64 { return math.Sqrt(y) * x }, 0, 1, Const(0), Const(1), &QuadOptions{Limit: 1})
	if !errors.Is(err, ErrLimit) || res.Value == 0 {
		t.Errorf("inner diagnostic: %v %v", res, err)
	}
}
