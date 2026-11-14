// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// powCase is one LSODA run of the corpus: the fixtures of testdata/ode.json
// (SciPy's counts and the reference end state), the three benchmark cases and
// the banded diffusion problem (lsoda.c's counts).
type powCase struct {
	name       string
	run        func(f ODEFunc) (nfev, njev int, y []float64)
	f          ODEFunc
	ref        []float64
	rtol, atol float64
	sciNfev    int
	sciNjev    int // -1: not known
}

func powCorpus(t *testing.T) []powCase {
	var cs []powCase
	fx := loadODEFixture(t)
	for _, rec := range fx.ODE {
		if rec.Method != "LSODA" {
			continue
		}
		p := odeProblems[rec.Problem]
		opts := &IVPOptions{Method: LSODA, RTol: rec.RTol, ATol: rec.ATol}
		if rec.Jac {
			opts.Jac = p.jac
		}
		cs = append(cs, powCase{
			name: fmt.Sprintf("%s rtol %g jac %v", rec.Problem, rec.RTol, rec.Jac), f: p.f, ref: rec.Ref, rtol: rec.RTol, atol: rec.ATol,
			sciNfev: rec.NFev, sciNjev: rec.NJev,
			run: func(f ODEFunc) (int, int, []float64) {
				s, err := SolveIVP(f, p.span, p.y0, opts)
				if err != nil {
					return 0, 0, nil
				}
				return s.NFev, s.NJev, s.Y[len(s.Y)-1]
			}})
	}
	cs = append(cs, powCase{name: "SolveIVP VdP mu=1000", f: vdpF(1000), sciNfev: 1803, sciNjev: -1,
		run: func(f ODEFunc) (int, int, []float64) {
			s, _ := SolveIVP(f, [2]float64{0, 3000}, []float64{2, 0}, &IVPOptions{Method: LSODA})
			return s.NFev, s.NJev, nil
		}})
	cs = append(cs, powCase{name: "Odeint Robertson", f: robertsonF, sciNfev: 1683, sciNjev: -1,
		run: func(f ODEFunc) (int, int, []float64) {
			_, info, _ := Odeint(f, []float64{1, 0, 0}, []float64{0, 0.4, 4, 40, 400, 4000, 40000}, &OdeintOptions{RTol: 1e-8, ATol: 1e-12})
			return info.NFE[len(info.NFE)-1], info.NJE[len(info.NJE)-1], nil
		}})
	ts := make([]float64, 101)
	for i := range ts {
		ts[i] = 15 * float64(i) / 100
	}
	cs = append(cs, powCase{name: "Odeint Lotka 101 times", f: lotkaF, sciNfev: 1243, sciNjev: -1,
		run: func(f ODEFunc) (int, int, []float64) {
			_, info, _ := Odeint(f, []float64{10, 5}, ts, nil)
			return info.NFE[len(info.NFE)-1], info.NJE[len(info.NJE)-1], nil
		}})
	band := &Bandwidth{Lower: 1, Upper: 1}
	for _, b := range []struct {
		name string
		bj   BandJacFunc
		nfe  int
	}{{"diffusion, analytic band", diffusionBand, 582}, {"diffusion, FD band", nil, 564}} {
		cs = append(cs, powCase{name: b.name, f: diffusionF, sciNfev: b.nfe, sciNjev: 9,
			run: func(f ODEFunc) (int, int, []float64) {
				s, _ := SolveIVP(f, [2]float64{0, 1}, diffusionY0(), &IVPOptions{Method: LSODA, RTol: 1e-6, ATol: 1e-9, Band: band, BandJac: b.bj})
				return s.NFev, s.NJev, nil
			}})
	}
	return cs
}

type powResult struct {
	nfev, njev int
	err        float64
	cheap      time.Duration // one run, the right-hand side as it is
}

func powMeasure(c powCase, mathPow bool) powResult {
	lsodaMathPow = mathPow
	defer func() { lsodaMathPow = false }()
	var r powResult
	var y []float64
	r.nfev, r.njev, y = c.run(c.f)
	if c.ref != nil && y != nil {
		r.err = relErr(y, c.ref, c.rtol, c.atol)
	}
	reps := 1
	for {
		start := time.Now()
		for i := 0; i < reps; i++ {
			c.run(c.f)
		}
		if d := time.Since(start); d > 200*time.Millisecond {
			r.cheap = d / time.Duration(reps)
			return r
		}
		reps *= 4
	}
}

// TestLsodaPowReport writes docs/lsoda-pow.md when INTEGRATE_POW=1: the whole
// corpus of LSODA runs with math.Pow and with powRecip as the default power.
func TestLsodaPowReport(t *testing.T) {
	if os.Getenv("INTEGRATE_POW") == "" {
		t.Skip("set INTEGRATE_POW=1 to regenerate docs/lsoda-pow.md (times only at a 1-minute load under 8)")
	}
	cs := powCorpus(t)
	var b strings.Builder
	b.WriteString("| case | SciPy nfev | nfev math.Pow | nfev powRecip | njev math.Pow | njev powRecip | error math.Pow | error powRecip | us math.Pow | us powRecip |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	var sumSci, sumM, sumR, njM, njR int
	var tM, tR time.Duration
	var worstM, worstR float64
	var eM, eR []float64
	for _, c := range cs {
		// interleave the two modes, two rounds, keep the faster of each
		m, r := powMeasure(c, true), powMeasure(c, false)
		m2, r2 := powMeasure(c, true), powMeasure(c, false)
		m.cheap, r.cheap = min(m.cheap, m2.cheap), min(r.cheap, r2.cheap)
		sumSci += c.sciNfev
		sumM += m.nfev
		sumR += r.nfev
		njM += m.njev
		njR += r.njev
		tM += m.cheap
		tR += r.cheap
		worstM = math.Max(worstM, float64(m.nfev-c.sciNfev)/float64(c.sciNfev))
		worstR = math.Max(worstR, float64(r.nfev-c.sciNfev)/float64(c.sciNfev))
		errM, errR := "", ""
		if c.ref != nil {
			errM, errR = fmt.Sprintf("%.3g", m.err), fmt.Sprintf("%.3g", r.err)
			eM, eR = append(eM, m.err), append(eR, r.err)
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %d | %d | %s | %s | %.1f | %.1f |\n", c.name, c.sciNfev, m.nfev, r.nfev, m.njev, r.njev, errM, errR,
			float64(m.cheap)/1e3, float64(r.cheap)/1e3)
	}
	sort.Float64s(eM)
	sort.Float64s(eR)
	fmt.Fprintf(&b, "| **total** | %d | %d | %d | %d | %d | | | %.1f | %.1f |\n", sumSci, sumM, sumR, njM, njR, float64(tM)/1e3, float64(tR)/1e3)
	fmt.Fprintf(&b, "\nWorst excess over SciPy's nfev: math.Pow %+.1f %%, powRecip %+.1f %%. Median error (tolerances): math.Pow %.3g, powRecip %.3g; largest %.3g and %.3g.\n",
		100*worstM, 100*worstR, eM[len(eM)/2], eR[len(eR)/2], eM[len(eM)-1], eR[len(eR)-1])
	// the estimated wall time with a right-hand side of 1 us and 10 us
	for _, us := range []float64{1, 10} {
		fmt.Fprintf(&b, "Estimated total time with a %g us right-hand side: math.Pow %.1f ms, powRecip %.1f ms.\n", us,
			(float64(tM)/1e3+us*float64(sumM))/1e3, (float64(tR)/1e3+us*float64(sumR))/1e3)
	}
	if err := os.WriteFile("docs/lsoda-pow.md", []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
