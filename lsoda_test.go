// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"errors"
	"math"
	"math/big"
	"testing"
)

func TestCmaxCmin(t *testing.T) {
	nan := math.NaN()
	for _, c := range []struct{ a, b, max, min float64 }{
		{1, 2, 2, 1}, {2, 1, 2, 1}, {nan, 3, 3, 3}, {3, nan, 3, 3},
	} {
		if g := cmax(c.a, c.b); g != c.max {
			t.Errorf("cmax(%v, %v) = %v, want %v", c.a, c.b, g, c.max)
		}
		if g := cmin(c.a, c.b); g != c.min {
			t.Errorf("cmin(%v, %v) = %v, want %v", c.a, c.b, g, c.min)
		}
	}
}

// TestCfode: the method coefficients of the low orders, known in closed form:
// Adams-Moulton order 1 (backward Euler) l = (1, 1), order 2 (trapezoid)
// l = (1/2, 1, 1/2); BDF order 1 l = (1, 1), order 2 l = (2/3, 1, 1/3), order 3
// l = (6/11, 1, 6/11, 1/11).
func TestCfode(t *testing.T) {
	s := newLsoda(&odeCore{n: 1, maxStep: math.Inf(1)})
	s.cfode(1)
	for i, w := range []float64{1, 1} {
		if s.elco[i] != w {
			t.Errorf("Adams order 1: l[%d] = %v, want %v", i, s.elco[i], w)
		}
	}
	for i, w := range []float64{0.5, 1, 0.5} {
		if math.Abs(s.elco[i+13]-w) > 1e-15 {
			t.Errorf("Adams order 2: l[%d] = %v, want %v", i, s.elco[i+13], w)
		}
	}
	s.cfode(2)
	for i, w := range []float64{1, 1} {
		if s.elco[i] != w {
			t.Errorf("BDF order 1: l[%d] = %v, want %v", i, s.elco[i], w)
		}
	}
	for i, w := range []float64{2.0 / 3, 1, 1.0 / 3} {
		if math.Abs(s.elco[i+13]-w) > 1e-15 {
			t.Errorf("BDF order 2: l[%d] = %v, want %v", i, s.elco[i+13], w)
		}
	}
	for i, w := range []float64{6.0 / 11, 1, 6.0 / 11, 1.0 / 11} {
		if math.Abs(s.elco[i+26]-w) > 1e-15 {
			t.Errorf("BDF order 3: l[%d] = %v, want %v", i, s.elco[i+26], w)
		}
	}
}

// bigPrec is the precision of the reference solutions: far above float64's, so
// that the reference carries no error of its own at the scale of the solvers'.
const bigPrec = 300

func bigF(x float64) *big.Float { return new(big.Float).SetPrec(bigPrec).SetFloat64(x) }

// bigExpSeries returns exp(x) by its Taylor series (x of any sign; |x| below a
// few hundred, where the terms' cancellation is covered by bigPrec).
func bigExpSeries(x *big.Float) *big.Float {
	sum := bigF(1)
	term := bigF(1)
	for k := 1; k < 2000; k++ {
		term.Mul(term, x)
		term.Quo(term, bigF(float64(k)))
		sum.Add(sum, term)
		if term.Sign() == 0 || term.MantExp(nil)-sum.MantExp(nil) < -bigPrec-10 {
			break
		}
	}
	return sum
}

// bigSinCos returns sin x and cos x by their Taylor series.
func bigSinCos(x *big.Float) (sin, cos *big.Float) {
	sin, cos = bigF(0), bigF(1)
	term := bigF(1)
	for k := 1; k < 2000; k++ {
		term.Mul(term, x)
		term.Quo(term, bigF(float64(k)))
		switch k % 4 {
		case 1:
			sin.Add(sin, term)
		case 2:
			cos.Sub(cos, term)
		case 3:
			sin.Sub(sin, term)
		case 0:
			cos.Add(cos, term)
		}
		if term.Sign() == 0 || term.MantExp(nil) < -bigPrec-10 {
			break
		}
	}
	return sin, cos
}

func bigToF(x *big.Float) float64 { f, _ := x.Float64(); return f }

// TestLSODAAccuracyContract is the accuracy contract of docs/design.md: against
// reference solutions computed in 300-bit arithmetic (math/big) the global error
// of LSODA, in tolerance units E = max_i |y_i - ref_i| / (atol + rtol |ref_i|),
// stays within the bound stated for each problem. ODEPACK controls the local
// error only, so the global error of a long integration is a multiple of the
// tolerance that grows with the number of steps (SciPy's own ODEPACK: 2.1e3 on
// the harmonic problem at rtol 1e-9, here the same order); the bounds are the
// ones the tests below measured, with room for the rounding differences of
// architectures that fuse multiply-adds.
func TestLSODAAccuracyContract(t *testing.T) {
	tb := 10 * math.Pi
	sinT, cosT := bigSinCos(bigF(tb))
	lam := 1000.0
	// y' = -lam (y - cos t), y(0) = 1: y = (lam^2 cos t + lam sin t)/(1+lam^2) + C e^{-lam t}
	sinS, cosS := bigSinCos(bigF(3))
	l2 := bigF(lam * lam)
	stiffRef := new(big.Float).SetPrec(bigPrec).Mul(l2, cosS)
	stiffRef.Add(stiffRef, new(big.Float).SetPrec(bigPrec).Mul(bigF(lam), sinS))
	stiffRef.Quo(stiffRef, new(big.Float).SetPrec(bigPrec).Add(bigF(1), l2))
	// (the transient C e^{-3000} is 10^-1303, far below float64's range)

	cases := []struct {
		name  string
		f     ODEFunc
		jac   JacFunc
		y0    []float64
		tf    float64
		ref   []float64
		rtol  float64
		atol  float64
		bound float64
	}{
		{"decay", decayF, nil, []float64{1}, 10, []float64{bigToF(bigExpSeries(bigF(-5)))}, 1e-6, 1e-9, 20},
		{"decay", decayF, nil, []float64{1}, 10, []float64{bigToF(bigExpSeries(bigF(-5)))}, 1e-9, 1e-12, 20},
		{"harmonic 5 periods", harmonicF, nil, []float64{1, 0}, tb, []float64{bigToF(cosT), -bigToF(sinT)}, 1e-6, 1e-9, 1e4},
		{"harmonic 5 periods", harmonicF, nil, []float64{1, 0}, tb, []float64{bigToF(cosT), -bigToF(sinT)}, 1e-9, 1e-12, 2e4},
		{"stiff linear", func(tt float64, y, d []float64) { d[0] = -lam * (y[0] - math.Cos(tt)) },
			func(_ float64, _ []float64, j []float64) { j[0] = -lam }, []float64{1}, 3, []float64{bigToF(stiffRef)}, 1e-6, 1e-9, 20},
		{"stiff linear, finite-difference Jacobian", func(tt float64, y, d []float64) { d[0] = -lam * (y[0] - math.Cos(tt)) },
			nil, []float64{1}, 3, []float64{bigToF(stiffRef)}, 1e-6, 1e-9, 20},
	}
	for _, c := range cases {
		sol, err := SolveIVP(c.f, [2]float64{0, c.tf}, c.y0, &IVPOptions{Method: LSODA, RTol: c.rtol, ATol: c.atol, Jac: c.jac})
		if err != nil {
			t.Errorf("%s rtol %g: %v", c.name, c.rtol, err)
			continue
		}
		y := sol.Y[len(sol.Y)-1]
		E := relErr(y, c.ref, c.rtol, c.atol)
		t.Logf("%s rtol %g: E = %.3g (bound %g), %d evaluations, %d steps", c.name, c.rtol, E, c.bound, sol.NFev, sol.NSteps)
		if !(E <= c.bound) {
			t.Errorf("%s rtol %g: E = %.3g over the bound %g", c.name, c.rtol, E, c.bound)
		}
	}
}

// TestOdeintManyOutputs: output times closer than a step are interpolated from
// the Nordsieck array of the step that passed them (ODEPACK's intdy), to the
// accuracy of the method, and the counts do not depend on how many are asked.
func TestOdeintManyOutputs(t *testing.T) {
	ts := make([]float64, 1501)
	for i := range ts {
		ts[i] = 15 * float64(i) / 1500
	}
	y, info, err := Odeint(lotkaF, []float64{10, 5}, ts, &OdeintOptions{RTol: 1e-9, ATol: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	ref, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, &IVPOptions{Method: DOP853, RTol: 1e-13, ATol: 1e-15, DenseOutput: true})
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]float64, 2)
	worst := 0.0
	for k, tk := range ts {
		ref.Sol.EvalInto(tk, buf)
		for i := range buf {
			worst = math.Max(worst, math.Abs(y.At(k, i)-buf[i])/(1e-12+1e-9*math.Abs(buf[i])))
		}
	}
	if worst > 1e3 {
		t.Errorf("worst interpolated error %.3g tolerances", worst)
	}
	if info.TCur[len(info.TCur)-1] < 15 || info.NST[len(info.NST)-1] > 1500 {
		t.Errorf("TCur %v NST %v", info.TCur[len(info.TCur)-1], info.NST[len(info.NST)-1])
	}
}

// lsodaTestSolver builds a solver on the problem f with y0 over [0, tf].
func lsodaTestSolver(t *testing.T, f ODEFunc, jac JacFunc, y0 []float64, tf float64) *lsodaSolver {
	t.Helper()
	core, err := newCore(f, 0, y0, tf, &IVPOptions{Jac: jac, RTol: 1e-6, ATol: 1e-9})
	if err != nil {
		t.Fatal(err)
	}
	return newLsoda(core)
}

// TestLsodaPrja: P = I - h el0 J is factored; an exactly singular P is flagged
// (ierpj), and the finite-difference increment falls back to 1 when f is 0.
func TestLsodaPrja(t *testing.T) {
	identity := func(_ float64, _, j []float64) { j[0] = 1 }
	s := lsodaTestSolver(t, decayF, identity, []float64{1}, 1)
	s.miter, s.h, s.el0 = 1, 1, 1
	s.ewt[0] = 1
	s.prja([]float64{1})
	if s.ierpj != 1 || s.nje != 1 || s.jcur != 1 {
		t.Errorf("singular P: ierpj %d nje %d jcur %d", s.ierpj, s.nje, s.jcur)
	}
	// the corrector refuses a step whose P is singular
	s.ipup, s.miter = 1, 1
	if st, _, _ := s.stodaCorrector(1); st != corrFailed || s.ierpj != 1 {
		t.Errorf("corrector with singular P: status %v", st)
	}
	zero := func(_ float64, _, d []float64) { d[0] = 0 }
	s = lsodaTestSolver(t, zero, nil, []float64{1}, 1)
	s.miter, s.h, s.el0 = 2, 1, 1
	s.ewt[0] = 1
	s.prja([]float64{1})
	if s.ierpj != 0 || s.nfe != 1 || s.lu.a[0] != 1 {
		t.Errorf("zero f: ierpj %d nfe %d P %v", s.ierpj, s.nfe, s.lu.a)
	}
}

// TestLsodaStodaFailure: the tenth corrector failure of a step ends it (kflag -2).
func TestLsodaStodaFailure(t *testing.T) {
	s := lsodaTestSolver(t, decayF, nil, []float64{1}, 1)
	s.mxncf, s.h, s.nq, s.l, s.rmax, s.nqnyh = 10, 0.1, 1, 2, 10, 1
	ncf := 8
	if !s.stodaFailure(&ncf, 0) || ncf != 9 || s.rmax != 2 || s.icf != 2 || s.h != 0.025 {
		t.Errorf("failure %d: h %v rmax %v", ncf, s.h, s.rmax)
	}
	if s.stodaFailure(&ncf, 0) || s.kflag != -2 {
		t.Errorf("the tenth failure: kflag %d", s.kflag)
	}
}

// TestLsodaStopConditions: the failures lsoda() reports as istate -2, -3 and -6.
func TestLsodaStopConditions(t *testing.T) {
	y := make([]float64, 1)
	// a zero error weight at the start (y = 0, atol = 0), and later
	core, _ := newCore(decayF, 0, []float64{0}, 1, &IVPOptions{RTol: 1e-6, ATol: -1})
	if _, st := newLsoda(core).call(y, 1, 5); st != -3 {
		t.Errorf("zero weight at the start: istate %d", st)
	}
	s := lsodaTestSolver(t, decayF, nil, []float64{1}, 1)
	if _, st := s.call(y, 1, 5); st != 2 {
		t.Fatalf("first step: istate %d", st)
	}
	s.atol[0], s.yh[0] = 0, 0
	if _, st := s.call(y, 1, 5); st != -6 {
		t.Errorf("zero weight later: istate %d", st)
	}
	// a tolerance below the machine precision (the API clamps rtol to 100 eps,
	// so the weight is set directly)
	s = lsodaTestSolver(t, decayF, nil, []float64{1}, 1)
	s.rtol[0], s.atol[0] = 1e-17, 0
	if _, st := s.call(y, 1, 5); st != -3 {
		t.Errorf("too much accuracy at the start: istate %d", st)
	}
	s = lsodaTestSolver(t, decayF, nil, []float64{1}, 1)
	if _, st := s.call(y, 1, 5); st != 2 {
		t.Fatalf("first step: istate %d", st)
	}
	s.rtol[0], s.atol[0] = 1e-17, 0
	if _, st := s.call(y, 1, 5); st != -2 {
		t.Errorf("too much accuracy later: istate %d", st)
	}
	// a first step beyond tcrit is cut to it
	core, _ = newCore(decayF, 0, []float64{1}, 1, &IVPOptions{})
	core.firstStep = 10
	s = newLsoda(core)
	if _, st := s.call(y, 1, 5); st != 2 || s.tn > 1 {
		t.Errorf("first step beyond tcrit: istate %d tn %v", st, s.tn)
	}
	// every message of a failure exists
	for _, st := range []int{-1, -2, -3, -4, -5, -6, -9} {
		if lsodaMessages[st] == "" {
			t.Errorf("no message for istate %d", st)
		}
	}
}

// TestLsodaRefusals: ODEPACK's own refusals reach the API as errors.
func TestLsodaRefusals(t *testing.T) {
	// a first output time closer to t0 than 2 uround |t|: illegal input
	t0 := 1e20
	_, info, err := Odeint(decayF, []float64{1}, []float64{t0, math.Nextafter(t0, math.Inf(1))}, nil)
	if !errors.Is(err, ErrODEInput) || info.Message != lsodaMessages[-3] {
		t.Errorf("close output times: %v", err)
	}
	// a state at zero with no absolute tolerance
	sol, err := SolveIVP(decayF, [2]float64{0, 1}, []float64{0}, &IVPOptions{Method: LSODA, ATol: -1})
	if !errors.Is(err, ErrStepTooSmall) || sol.Success || sol.Message != lsodaMessages[-3] {
		t.Errorf("zero weight: %v %v", err, sol)
	}
	// h underflows to 0 (a blow-up in finite time with a huge budget): stopped
	sol, err = SolveIVP(func(_ float64, y, d []float64) { d[0] = y[0] * y[0] }, [2]float64{0, 2}, []float64{1}, &IVPOptions{Method: LSODA})
	if !errors.Is(err, ErrStepTooSmall) || sol.Success {
		t.Errorf("blow-up: %v", err)
	}
	// a maximum step below the first step
	sol, err = SolveIVP(decayF, [2]float64{0, 1e-4}, []float64{1}, &IVPOptions{Method: LSODA, MaxStep: 1e-6})
	if err != nil || sol.NSteps < 100 {
		t.Errorf("max step: %v %d steps", err, sol.NSteps)
	}
}

// TestLsodaStepFailures: ten error test failures in a row end the step with
// istate -4, ten corrector failures with -5 (the second by bending the
// solver's constants: no ordinary right-hand side keeps failing as h shrinks).
func TestLsodaStepFailures(t *testing.T) {
	y := make([]float64, 2)
	s := lsodaTestSolver(t, lotkaF, nil, []float64{10, 5}, 10)
	if _, st := s.call(y, 10, 5); st != 2 {
		t.Fatalf("first step: istate %d", st)
	}
	// f = 1e3/(t - t0): h f is 1e3 whatever h, so that no step is small enough
	told := s.tn
	s.fun = func(tt float64, _, d []float64) { d[0], d[1] = 1e3/(tt-told), 0 }
	if _, st := s.call(y, 10, 5); st != -4 || s.kflag != -1 {
		t.Errorf("error test: istate %d kflag %d", st, s.kflag)
	}
	s = lsodaTestSolver(t, lotkaF, nil, []float64{10, 5}, 10)
	if _, st := s.call(y, 10, 5); st != 2 {
		t.Fatalf("first step: istate %d", st)
	}
	s.maxcor, s.mxncf = 1, 1 // Adams needs two iterations
	if _, st := s.call(y, 10, 5); st != -5 || s.kflag != -2 {
		t.Errorf("corrector: istate %d kflag %d", st, s.kflag)
	}
}

// TestLsodaTcritAfterSwitch: a step shortened to reach tcrit right after a
// method switch (jstart -1) is rescaled when stoda completes the switch.
func TestLsodaTcritAfterSwitch(t *testing.T) {
	s := lsodaTestSolver(t, vdpF(1000), nil, []float64{2, 0}, 3000)
	y := make([]float64, 2)
	for i := 0; s.jstart != -1; i++ {
		if _, st := s.call(y, 3000, 5); st != 2 || i > 200 {
			t.Fatalf("no switch within 200 steps: istate %d", st)
		}
	}
	s.tcrit = s.tn + s.h/2
	if tt, st := s.call(y, s.tcrit, 5); st != 2 || tt != s.tcrit {
		t.Errorf("step to tcrit: istate %d t %v, want %v", st, tt, s.tcrit)
	}
}

// TestLsodaSwitchesBack: a problem stiff only for a while makes ODEPACK switch to
// BDF and back to Adams (MUsed 2 then 1).
func TestLsodaSwitchesBack(t *testing.T) {
	// y' = -lam(t) (y - cos t), lam large on [0.1, 0.5] only
	f := func(tt float64, y, d []float64) {
		lam := 1.0
		if tt > 0.1 && tt < 0.5 {
			lam = 1e4
		}
		d[0] = -lam * (y[0] - math.Cos(tt))
	}
	ts := make([]float64, 400)
	for i := range ts {
		ts[i] = 3 * float64(i) / 399
	}
	_, info, err := Odeint(f, []float64{0}, ts, &OdeintOptions{RTol: 1e-8, ATol: 1e-10, MxStep: 100000})
	if err != nil {
		t.Fatal(err)
	}
	stiff, back := false, false
	for _, m := range info.MUsed {
		if m == 2 {
			stiff = true
		} else if stiff {
			back = true
		}
	}
	if !stiff || !back {
		t.Errorf("MUsed %v: no switch to BDF and back", info.MUsed)
	}
}

// diffusion is y_i' = k (y_{i-1} - 2 y_i + y_{i+1}), the stiff tridiagonal problem
// whose counts lsoda.c gave (SciPy 1.18.1's, built here, harness run with jt 4
// and 5): 582 evaluations and 9 Jacobians with the analytic band, 564 and 9
// with the band by finite differences, on arm64 (fused multiply-adds; 582 and
// 567 without).
const diffK, diffN = 400.0, 20

func diffusionF(_ float64, y, d []float64) {
	for i := range y {
		var l, r float64
		if i > 0 {
			l = y[i-1]
		}
		if i < len(y)-1 {
			r = y[i+1]
		}
		d[i] = diffK * (l - 2*y[i] + r)
	}
}

// diffusionBand is the Jacobian in band form: jac[(mu+i-j)*n+j], mu = 1.
func diffusionBand(_ float64, y, jac []float64) {
	n := len(y)
	for j := 0; j < n; j++ {
		jac[1*n+j] = -2 * diffK
		if j > 0 {
			jac[0*n+j] = diffK // (j-1, j)
		}
		if j < n-1 {
			jac[2*n+j] = diffK // (j+1, j)
		}
	}
}

func diffusionY0() []float64 {
	y0 := make([]float64, diffN)
	for i := range y0 {
		y0[i] = 1 / (1 + (float64(i)-10)*(float64(i)-10)/4) // no libm: the same bits in the C harness
	}
	return y0
}

func TestLsodaBanded(t *testing.T) {
	band := &Bandwidth{Lower: 1, Upper: 1}
	full, err := SolveIVP(diffusionF, [2]float64{0, 1}, diffusionY0(), &IVPOptions{Method: LSODA, RTol: 1e-6, ATol: 1e-9})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name       string
		bj         BandJacFunc
		nfe, njeSc int
	}{{"analytic band (jt 4)", diffusionBand, 582, 9}, {"finite-difference band (jt 5)", nil, 564, 9}} {
		sol, err := SolveIVP(diffusionF, [2]float64{0, 1}, diffusionY0(), &IVPOptions{Method: LSODA, RTol: 1e-6, ATol: 1e-9, Band: band, BandJac: c.bj})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		t.Logf("%s: %d evaluations (SciPy %d), %d Jacobians (SciPy %d), %d steps; full matrix: %d, %d, %d", c.name,
			sol.NFev, c.nfe, sol.NJev, c.njeSc, sol.NSteps, full.NFev, full.NJev, full.NSteps)
		if d := math.Abs(float64(sol.NFev-c.nfe)) / float64(c.nfe); d > 0.035 { // 1 % on arm64 and amd64 (566, 567 against 564); ppc64le and s390x, which fuse multiply-adds elsewhere, take 580
			t.Errorf("%s: nfev %d, SciPy %d", c.name, sol.NFev, c.nfe)
		}
		if sol.NJev != c.njeSc {
			t.Errorf("%s: njev %d, SciPy %d", c.name, sol.NJev, c.njeSc)
		}
		// the band is the whole Jacobian: the same solution as the full matrix, to the tolerance
		a, b := sol.Y[len(sol.Y)-1], full.Y[len(full.Y)-1]
		for i := range a {
			if math.Abs(a[i]-b[i]) > 1e-4*(1e-6+1e-3*math.Abs(b[i]))/1e-3 && math.Abs(a[i]-b[i]) > 1e-7 {
				t.Errorf("%s: y[%d] = %v, full matrix %v", c.name, i, a[i], b[i])
			}
		}
	}
	// the same through Odeint, with a dense output between steps
	y, info, err := Odeint(diffusionF, diffusionY0(), []float64{0, 0.25, 0.5, 1}, &OdeintOptions{RTol: 1e-6, ATol: 1e-9, Band: band, BandJac: diffusionBand})
	if err != nil || info.NJE[len(info.NJE)-1] == 0 {
		t.Fatalf("Odeint banded: %v", err)
	}
	if e := math.Abs(y.At(3, 10) - full.Y[len(full.Y)-1][10]); e > 1e-7 {
		t.Errorf("Odeint banded: y10 differs from the full matrix by %g", e)
	}
}

// TestLuBand: the banded LU solves what the dense LU solves, on a random
// banded matrix with pivoting (a small diagonal), and flags a singular one.
func TestLuBand(t *testing.T) {
	n, kl, ku := 9, 2, 3
	lub := newLUBand(n, kl, ku)
	dense := make([]float64, n*n)
	seed := uint64(7)
	next := func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed>>11)/(1<<53) - 0.5
	}
	for j := 0; j < n; j++ {
		for i := max(0, j-ku); i <= min(n-1, j+kl); i++ {
			v := next()
			if (i+2*j)%4 == 0 && i != j {
				v = 0 // a zero in the band: the factorization skips its update
			}
			if i == j {
				v *= 0.01 // force row interchanges
			}
			dense[i*n+j] = v
			*lub.at(i, j) = v
		}
	}
	b := make([]float64, n)
	for i := range b {
		b[i] = next()
	}
	want := append([]float64(nil), b...)
	lud := newLUReal(n)
	lud.factor(dense)
	lud.solve(want)
	if !lub.factor() {
		t.Fatal("banded LU flagged a regular matrix")
	}
	lub.solve(b)
	for i := range b {
		if math.Abs(b[i]-want[i]) > 1e-9*(1+math.Abs(want[i])) {
			t.Errorf("x[%d] = %v, dense %v", i, b[i], want[i])
		}
	}
	zero := newLUBand(3, 1, 1)
	if zero.factor() {
		t.Error("a zero matrix is singular")
	}
}

// TestLsodaOptions: hmin, mxordn, mxords and the refusals of bad options.
func TestLsodaOptions(t *testing.T) {
	ts := []float64{0, 1, 2}
	for _, o := range []*OdeintOptions{
		{HMin: -1}, {MxOrdN: -1}, {MxOrdS: -1}, {BandJac: diffusionBand},
		{Band: &Bandwidth{Lower: 5, Upper: 0}}, {Band: &Bandwidth{Lower: 0, Upper: -1}},
		{Band: &Bandwidth{Lower: 0, Upper: 0}, Jac: func(float64, []float64, []float64) {}},
	} {
		if _, _, err := Odeint(lotkaF, []float64{10, 5}, ts, o); !errors.Is(err, ErrODEInput) {
			t.Errorf("%+v: %v", *o, err)
		}
	}
	if _, err := SolveIVP(lotkaF, [2]float64{0, 1}, []float64{10, 5}, &IVPOptions{Method: BDF, MinStep: 1e-3}); !errors.Is(err, ErrODEInput) {
		t.Errorf("BDF with MinStep: %v", err)
	}
	// low maximum orders: the method still integrates, with the orders bounded
	for _, o := range []*OdeintOptions{{MxOrdN: 2, MxOrdS: 2, RTol: 1e-8, ATol: 1e-10}, {MxOrdN: 20, MxOrdS: 9}} {
		sv, err := newLsodaConfig(&odeCore{n: 1, maxStep: math.Inf(1)}, lsodaConfig{mxordn: o.MxOrdN, mxords: o.MxOrdS})
		if err != nil || sv.mxordn > 12 || sv.mxords > 5 || (o.MxOrdN == 2 && sv.mxordn != 2) {
			t.Errorf("orders %d %d: %v %d %d", o.MxOrdN, o.MxOrdS, err, sv.mxordn, sv.mxords)
		}
	}
	stiff := func(tt float64, y, d []float64) { d[0] = -1e4 * (y[0] - math.Cos(tt)) }
	for _, mx := range [][2]int{{2, 2}, {3, 1}, {1, 3}} {
		ts := make([]float64, 30)
		for i := range ts {
			ts[i] = 3 * float64(i) / 29
		}
		y, info, err := Odeint(stiff, []float64{0}, ts, &OdeintOptions{RTol: 1e-8, ATol: 1e-10, MxOrdN: mx[0], MxOrdS: mx[1], MxStep: 100000})
		if err != nil {
			t.Fatalf("orders %v: %v", mx, err)
		}
		if e := math.Abs(y.At(29, 0) - (1e8*math.Cos(3)+1e4*math.Sin(3))/(1+1e8)); e > 1e-6 {
			t.Errorf("orders %v: error %g", mx, e)
		}
		_ = info
	}
}
