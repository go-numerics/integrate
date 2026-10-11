// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// The LSODA-style method: an explicit Dormand–Prince pair while the problem
// is non-stiff, BDF while it is stiff, with automatic switching in both
// directions. ODEPACK's LSODA (Petzold 1983) pairs Adams–Moulton with BDF and
// switches by comparing the step sizes the two would take; here the
// non-stiff side is RK45 (DOP853 for rtol < 1e-6) and
//
//   - non-stiff → stiff uses Hairer's stiffness detector (Solving ODEs II,
//     IV.2; the test built into dopri5/dop853): after each accepted step
//     hλ ≈ |h|·‖f(y₁) - f(ỹ)‖/‖y₁ - ỹ‖ estimates |h| times the dominant
//     eigenvalue; when it exceeds the stability boundary on 15 steps without
//     6 consecutive steps below it, the step size is limited by stability
//     rather than accuracy and BDF takes over;
//   - stiff → non-stiff estimates the spectral radius ρ of the Jacobian BDF
//     holds by power iteration every few steps; when |h|·ρ < 1 on two
//     consecutive checks the explicit method would be stable at BDF's step
//     size, and it takes over.

const (
	stiffDetectCount   = 15 // Hairer's iasti threshold
	nonstiffResetCount = 6  // Hairer's nonsti reset
	backCheckEvery     = 5  // BDF steps between spectral-radius checks
	backConfirm        = 2  // consecutive favourable checks before switching
)

type switchSolver struct {
	*odeCore
	nonstiff  Method
	rk        *rkSolver
	bdf       *bdfSolver
	stiff     bool
	iasti     int
	nonsti    int
	bdfSteps  int
	backOK    int
	switches  int
	lastStiff bool // method that produced the last step, for dense output
	ftmp      []float64
}

func newSwitching(core *odeCore) *switchSolver {
	s := &switchSolver{odeCore: core, nonstiff: RK45, ftmp: make([]float64, core.n)}
	if minRTol(core.rtol) < 1e-6 {
		s.nonstiff = DOP853
	}
	s.rk = newRK(core, s.nonstiff, nil, 0)
	return s
}

// spectralRadius estimates the largest |eigenvalue| of the n×n matrix J by
// power iteration from a fixed start vector.
func spectralRadius(J []float64, n int) float64 {
	v := make([]float64, n)
	w := make([]float64, n)
	for i := range v {
		v[i] = 1 / math.Sqrt(float64(n)) * (1 + float64(i)/float64(n))
	}
	rho := 0.0
	for it := 0; it < 20; it++ {
		nrm := 0.0
		for i := 0; i < n; i++ {
			s := 0.0
			for j := 0; j < n; j++ {
				s += J[i*n+j] * v[j]
			}
			w[i] = s
			nrm += s * s
		}
		nrm = math.Sqrt(nrm)
		if nrm == 0 {
			return 0
		}
		rho = nrm
		for i := range v {
			v[i] = w[i] / nrm
		}
	}
	return rho
}

func (s *switchSolver) stepImpl() (bool, string) {
	if !s.stiff {
		ok, msg := s.rk.stepImpl()
		if !ok {
			return false, msg
		}
		s.lastStiff = false
		if s.rk.hLambda > s.rk.tab.stiffBound {
			s.nonsti = 0
			s.iasti++
			if s.iasti >= stiffDetectCount {
				s.bdf = newBDF(s.odeCore, s.rk.f, s.rk.hAbs)
				s.stiff = true
				s.switches++
				s.bdfSteps, s.backOK = 0, 0
			}
		} else {
			s.nonsti++
			if s.nonsti == nonstiffResetCount {
				s.iasti = 0
			}
		}
		return true, ""
	}
	ok, msg := s.bdf.stepImpl()
	if !ok {
		return false, msg
	}
	s.lastStiff = true
	s.bdfSteps++
	if s.bdfSteps%backCheckEvery != 0 {
		return true, ""
	}
	if s.bdf.hAbs*spectralRadius(s.bdf.J, s.n) < 1 {
		s.backOK++
	} else {
		s.backOK = 0
	}
	if s.backOK >= backConfirm {
		s.eval(s.t, s.y, s.ftmp)
		s.rk = newRK(s.odeCore, s.nonstiff, s.ftmp, s.bdf.hAbs)
		s.stiff = false
		s.switches++
		s.iasti, s.nonsti = 0, 0
	}
	return true, ""
}

func (s *switchSolver) denseImpl() denseOutput {
	if s.lastStiff {
		return s.bdf.denseImpl()
	}
	return s.rk.denseImpl()
}
