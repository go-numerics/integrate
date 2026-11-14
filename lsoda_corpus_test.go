// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "testing"

// TestLsodaCorpusCounts: the default path's evaluation counts over the corpus
// of docs/lsoda-pow.md (the 33 LSODA fixture runs, the three benchmark cases and
// the banded problem) against SciPy's. Case by case the count cannot be held to
// SciPy's: the step selection hinges on the last bit, and a case that lands on
// a knife-edge takes other steps. What holds is the sum and a bound on the worst
// case. Measured on arm64 with powRecip: the sum is 0.3 % under SciPy's and the
// worst case 5.2 % over (hires, rtol 1e-6); with math.Pow, 0.6 % over and 12.9 %.
// The bounds leave room for the architectures that fuse multiply-adds elsewhere.
func TestLsodaMathPowSwitch(t *testing.T) {
	run := func() int {
		s, err := SolveIVP(lotkaF, [2]float64{0, 15}, []float64{10, 5}, &IVPOptions{Method: LSODA, RTol: 1e-6, ATol: 1e-9})
		if err != nil {
			t.Fatal(err)
		}
		return s.NFev
	}
	a := run()
	lsodaMathPow = true
	b := run()
	lsodaMathPow = false
	if a <= 0 || b <= 0 || run() != a {
		t.Errorf("nfev %d with powRecip, %d with math.Pow", a, b)
	}
}

func TestLsodaCorpusCounts(t *testing.T) {
	var sum, sci int
	worst, worstName := -1.0, ""
	for _, c := range powCorpus(t) {
		nfev, _, _ := c.run(c.f)
		sum += nfev
		sci += c.sciNfev
		if d := float64(nfev-c.sciNfev) / float64(c.sciNfev); d > worst {
			worst, worstName = d, c.name
		}
	}
	t.Logf("corpus: %d evaluations, SciPy %d (%+.2f %%); worst case %s %+.1f %%", sum, sci, 100*float64(sum-sci)/float64(sci), worstName, 100*worst)
	if float64(sum) > 1.03*float64(sci) {
		t.Errorf("the corpus takes %d evaluations, SciPy %d", sum, sci)
	}
	if worst > 0.2 {
		t.Errorf("%s takes %.1f %% more evaluations than SciPy", worstName, 100*worst)
	}
}
