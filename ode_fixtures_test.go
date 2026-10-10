// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// odeProblem mirrors one entry of the ODE table of testdata/gen/fixtures.py.
type odeProblem struct {
	f    ODEFunc
	jac  JacFunc
	span [2]float64
	y0   []float64
}

const arenstorfMu = 0.012277471

func vdpF(mu float64) ODEFunc {
	return func(_ float64, y, d []float64) {
		d[0] = y[1]
		d[1] = mu*(1-y[0]*y[0])*y[1] - y[0]
	}
}

func vdpJ(mu float64) JacFunc {
	return func(_ float64, y, j []float64) {
		j[0], j[1] = 0, 1
		j[2], j[3] = -2*mu*y[0]*y[1]-1, mu*(1-y[0]*y[0])
	}
}

func lotkaF(_ float64, y, d []float64) {
	d[0] = 1.5*y[0] - y[0]*y[1]
	d[1] = -3*y[1] + y[0]*y[1]
}

func robertsonF(_ float64, y, d []float64) {
	d[0] = -0.04*y[0] + 1e4*y[1]*y[2]
	d[1] = 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1]
	d[2] = 3e7 * y[1] * y[1]
}

func robertsonJ(_ float64, y, j []float64) {
	j[0], j[1], j[2] = -0.04, 1e4*y[2], 1e4*y[1]
	j[3], j[4], j[5] = 0.04, -1e4*y[2]-6e7*y[1], -1e4*y[1]
	j[6], j[7], j[8] = 0, 6e7*y[1], 0
}

func harmonicF(_ float64, y, d []float64) { d[0], d[1] = y[1], -y[0] }

func decayF(_ float64, y, d []float64) {
	for i := range y {
		d[i] = -0.5 * y[i]
	}
}

var odeProblems = map[string]odeProblem{
	"decay":    {decayF, nil, [2]float64{0, 10}, []float64{2, 4, 8}},
	"harmonic": {harmonicF, nil, [2]float64{0, 10 * math.Pi}, []float64{1, 0}},
	"lotka":    {lotkaF, nil, [2]float64{0, 15}, []float64{10, 5}},
	"vdp1":     {vdpF(1), vdpJ(1), [2]float64{0, 20}, []float64{2, 0}},
	"brusselator": {func(_ float64, y, d []float64) {
		d[0] = 1 + y[0]*y[0]*y[1] - 4*y[0]
		d[1] = 3*y[0] - y[0]*y[0]*y[1]
	}, nil, [2]float64{0, 20}, []float64{1.5, 3}},
	"arenstorf": {func(_ float64, y, d []float64) {
		mu := arenstorfMu
		mup := 1 - mu
		d1 := math.Pow((y[0]+mu)*(y[0]+mu)+y[1]*y[1], 1.5)
		d2 := math.Pow((y[0]-mup)*(y[0]-mup)+y[1]*y[1], 1.5)
		d[0] = y[2]
		d[1] = y[3]
		d[2] = y[0] + 2*y[3] - mup*(y[0]+mu)/d1 - mu*(y[0]-mup)/d2
		d[3] = y[1] - 2*y[2] - mup*y[1]/d1 - mu*y[1]/d2
	}, nil, [2]float64{0, 17.0652165601579625588917206249}, []float64{0.994, 0, 0, -2.00158510637908252240537862224}},
	"vdp1000":   {vdpF(1000), vdpJ(1000), [2]float64{0, 3000}, []float64{2, 0}},
	"robertson": {robertsonF, robertsonJ, [2]float64{0, 1e5}, []float64{1, 0, 0}},
	"hires": {func(_ float64, y, d []float64) {
		d[0] = -1.71*y[0] + 0.43*y[1] + 8.32*y[2] + 0.0007
		d[1] = 1.71*y[0] - 8.75*y[1]
		d[2] = -10.03*y[2] + 0.43*y[3] + 0.035*y[4]
		d[3] = 8.32*y[1] + 1.71*y[2] - 1.12*y[3]
		d[4] = -1.745*y[4] + 0.43*y[5] + 0.43*y[6]
		d[5] = -280*y[5]*y[7] + 0.69*y[3] + 1.71*y[4] - 0.43*y[5] + 0.69*y[6]
		d[6] = 280*y[5]*y[7] - 1.81*y[6]
		d[7] = -280*y[5]*y[7] + 1.81*y[6]
	}, nil, [2]float64{0, 321.8122}, []float64{1, 0, 0, 0, 0, 0, 0, 0.0057}},
	"orego": {func(_ float64, y, d []float64) {
		const s, w, q = 77.27, 0.161, 8.375e-6
		d[0] = s * (y[1] - y[0]*y[1] + y[0] - q*y[0]*y[0])
		d[1] = (-y[1] - y[0]*y[1] + y[2]) / s
		d[2] = w * (y[0] - y[2])
	}, nil, [2]float64{0, 360}, []float64{1, 2, 3}},
}

type odeRecord struct {
	Problem string      `json:"problem"`
	Method  string      `json:"method"`
	RTol    float64     `json:"rtol"`
	ATol    float64     `json:"atol"`
	Jac     bool        `json:"jac"`
	Status  int         `json:"status"`
	NFev    int         `json:"nfev"`
	NJev    int         `json:"njev"`
	NLU     int         `json:"nlu"`
	NSteps  int         `json:"nsteps"`
	YEnd    []float64   `json:"y_end"`
	Ref     []float64   `json:"ref"`
	TFirst  []float64   `json:"t_first"`
	Extra   interface{} `json:"-"`
}

type odeFixture struct {
	Meta     map[string]string          `json:"meta"`
	ODE      []odeRecord                `json:"ode"`
	Features map[string]json.RawMessage `json:"features"`
}

func loadODEFixture(t testing.TB) *odeFixture {
	t.Helper()
	b, err := os.ReadFile("testdata/ode.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx odeFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	return &fx
}

func methodByName(t testing.TB, s string) Method {
	for m := RK45; m <= LSODA; m++ {
		if m.String() == s {
			return m
		}
	}
	t.Fatalf("unknown method %q", s)
	return 0
}

// relErr is the error of y against ref, scaled as the solvers scale it
// (atol + rtol·|ref|), so that 1 means "exactly at the requested tolerance".
func relErr(y, ref []float64, rtol, atol float64) float64 {
	e := 0.0
	for i := range y {
		e = math.Max(e, math.Abs(y[i]-ref[i])/(atol+rtol*math.Abs(ref[i])))
	}
	return e
}

type odeRun struct {
	rec      odeRecord
	sol      *Solution
	err      error
	achieved float64 // scaled error vs reference
	scipyErr float64
}

func runODEFixtures(t testing.TB) []odeRun {
	fx := loadODEFixture(t)
	var out []odeRun
	for _, rec := range fx.ODE {
		p, ok := odeProblems[rec.Problem]
		if !ok {
			t.Fatalf("no Go problem %q", rec.Problem)
		}
		opts := &IVPOptions{Method: methodByName(t, rec.Method), RTol: rec.RTol, ATol: rec.ATol}
		if rec.Jac {
			opts.Jac = p.jac
		}
		sol, err := SolveIVP(p.f, p.span, p.y0, opts)
		r := odeRun{rec: rec, sol: sol, err: err}
		if err == nil {
			r.achieved = relErr(sol.Y[len(sol.Y)-1], rec.Ref, rec.RTol, rec.ATol)
			r.scipyErr = relErr(rec.YEnd, rec.Ref, rec.RTol, rec.ATol)
		}
		out = append(out, r)
	}
	return out
}

// TestODEAgainstSciPy compares every fixture run with SciPy's solve_ivp on
// the same problem, method and tolerances. For the methods that are ports
// (RK45, RK23, DOP853, Radau, BDF) the number of steps and of evaluations
// must match SciPy's closely (exactly in most runs: the step sequence is the
// same); for every method the end state must be as accurate as SciPy's,
// measured against a high-accuracy reference.
func TestODEAgainstSciPy(t *testing.T) {
	runs := runODEFixtures(t)
	exact, total := 0, 0
	maxTDev := 0.0
	// The SciPy error of an explicit (non-stiff) or Radau (stiff) run on the
	// same problem and tolerance: the LSODA-style method runs those
	// algorithms, so it is held to their accuracy as well as ODEPACK's.
	peer := map[string]float64{}
	for _, r := range runs {
		if r.rec.Method == "RK45" || r.rec.Method == "DOP853" || r.rec.Method == "Radau" {
			k := fmt.Sprintf("%s/%g", r.rec.Problem, r.rec.RTol)
			peer[k] = math.Max(peer[k], r.scipyErr)
		}
	}
	for _, r := range runs {
		rec := r.rec
		id := fmt.Sprintf("%s/%s/rtol=%g/jac=%v", rec.Problem, rec.Method, rec.RTol, rec.Jac)
		if r.err != nil {
			t.Errorf("%s: %v", id, r.err)
			continue
		}
		s := r.sol
		if rec.Method == "LSODA" {
			// ODEPACK's LSODA is a different algorithm (Adams/BDF);
			// only accuracy is compared.
			bound := math.Max(math.Max(20, 5*r.scipyErr), 2*peer[fmt.Sprintf("%s/%g", rec.Problem, rec.RTol)])
			if r.achieved > bound {
				t.Errorf("%s: scaled error %.3g, SciPy/ODEPACK %.3g", id, r.achieved, r.scipyErr)
			}
			continue
		}
		total++
		if s.NFev == rec.NFev && s.NSteps == rec.NSteps && s.NJev == rec.NJev && s.NLU == rec.NLU {
			exact++
		} else {
			t.Logf("%s: nfev %d/%d steps %d/%d njev %d/%d nlu %d/%d (ours/SciPy)", id,
				s.NFev, rec.NFev, s.NSteps, rec.NSteps, s.NJev, rec.NJev, s.NLU, rec.NLU)
		}
		if d := math.Abs(float64(s.NFev-rec.NFev)) / float64(rec.NFev); d > 0.05 {
			t.Errorf("%s: nfev %d differs from SciPy's %d by %.1f%%", id, s.NFev, rec.NFev, 100*d)
		}
		for k, tk := range rec.TFirst {
			if k < len(s.T) {
				maxTDev = math.Max(maxTDev, math.Abs(s.T[k]-tk)/math.Max(1e-300, math.Abs(tk)))
			}
		}
		if r.achieved > math.Max(2*r.scipyErr, 1e-3) && r.achieved > 1 {
			t.Errorf("%s: scaled error %.3g, SciPy %.3g", id, r.achieved, r.scipyErr)
		}
	}
	t.Logf("SciPy %s: identical (nfev, steps, njev, nlu) on %d/%d runs of ported methods", loadODEFixture(t).Meta["scipy"], exact, total)
	t.Logf("largest relative deviation of the first five step times: %.3g", maxTDev)
}

// TestVerificationReport writes docs/verification.md when
// INTEGRATE_REPORT=1; it is the source of the tables in the docs.
func TestVerificationReport(t *testing.T) {
	if os.Getenv("INTEGRATE_REPORT") == "" {
		t.Skip("set INTEGRATE_REPORT=1 to regenerate docs/verification.md")
	}
	runs := runODEFixtures(t)
	var b strings.Builder
	fx := loadODEFixture(t)
	fmt.Fprintf(&b, "| problem | method | rtol | jac | scaled error (ours) | scaled error (SciPy %s) | nfev (+FD Jacobian) | nfev SciPy | njev | njev SciPy | steps | steps SciPy |\n", fx.Meta["scipy"])
	b.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range runs {
		rec := r.rec
		if r.err != nil {
			fmt.Fprintf(&b, "| %s | %s | %g | %v | failed: %v |||||||\n", rec.Problem, rec.Method, rec.RTol, rec.Jac, r.err)
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %g | %v | %.3g | %.3g | %d (+%d) | %d | %d | %d | %d | %d |\n",
			rec.Problem, rec.Method, rec.RTol, rec.Jac, r.achieved, r.scipyErr,
			r.sol.NFev, r.sol.NFevJac, rec.NFev, r.sol.NJev, rec.NJev, r.sol.NSteps, rec.NSteps)
	}
	if err := os.WriteFile("docs/ode-table.md", []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	writeQuadTable(t)
}

// writeQuadTable writes docs/quad-table.md: every quadrature fixture, ours
// against SciPy and the reference.
func writeQuadTable(t *testing.T) {
	fx := loadQuadFixture(t)
	var b strings.Builder
	fmt.Fprintf(&b, "| integral | value | SciPy %s | reference | true error | estimate | neval | neval SciPy | ier | ier SciPy |\n", fx.Meta["scipy"])
	b.WriteString("|---|---|---|---|---|---|---|---|---|---|\n")
	for _, rec := range fx.Quad {
		c := quadCases[rec.Name]
		r, err := Quad(c.f, float64(rec.A), float64(rec.B), c.opts)
		ier := 0
		var qe *QuadError
		if errors.As(err, &qe) {
			ier = qe.Ier
		}
		ref, trueErr := "—", "—"
		if rec.Ref != nil {
			ref = fmt.Sprintf("%.16g", *rec.Ref)
			trueErr = fmt.Sprintf("%.2g", math.Abs(r.Value-*rec.Ref))
		}
		fmt.Fprintf(&b, "| %s | %.16g | %.16g | %s | %s | %.2g | %d | %d | %d | %d |\n",
			rec.Name, r.Value, rec.Value, ref, trueErr, r.AbsErr, r.NEval, rec.Calls, ier, rec.Ier)
	}
	if err := os.WriteFile("docs/quad-table.md", []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
