// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Bandwidth is the band of a banded Jacobian: SciPy's lband and uband for
// solve_ivp's LSODA, ml and mu for odeint. Element (i, j) of the Jacobian is
// in the band when j-Upper <= i <= j+Lower.
type Bandwidth struct{ Lower, Upper int }

// BandJacFunc evaluates a banded Jacobian into jac, a (Lower+Upper+1)×n
// row-major matrix: jac[(Upper+i-j)*n+j] = ∂f_i/∂y_j for the i of the band of
// column j (SciPy's Dfun of a banded problem, stored by rows of the band).
// jac is zeroed before the call.
type BandJacFunc func(t float64, y []float64, jac []float64)

// lsodaConfig is the options of lsoda that the API exposes beyond tolerances.
type lsodaConfig struct {
	hmin           float64 // the smallest step (ODEPACK's hmin, rwork[6]); 0 for none
	mxordn, mxords int     // the largest Adams and BDF orders; 0 for 12 and 5
	band           *Bandwidth
	bandJac        BandJacFunc
	// exact is the faithful lsoda.c path: its bnorm (which reads the band
	// ml rows above P: a SciPy bug the default path does not have) and the
	// correctly rounded power in place of Go's math.Pow, so that the step
	// sequence is the C library's, bit for bit, where lsoda.c is built without
	// fused multiply-adds.
	exact bool
}

// configure applies cfg (ODEPACK's checks of its optional inputs, as errors).
func (s *lsodaSolver) configure(cfg lsodaConfig) error {
	n := s.n
	s.exact = cfg.exact
	if cfg.hmin < 0 {
		return odeInput("hmin must be non-negative")
	}
	s.hmin = cfg.hmin
	if cfg.mxordn < 0 || cfg.mxords < 0 {
		return odeInput("mxordn and mxords must be non-negative")
	}
	if cfg.mxordn > 0 {
		s.mxordn = min(cfg.mxordn, lsodaMaxOrdAdams)
	}
	if cfg.mxords > 0 {
		s.mxords = min(cfg.mxords, lsodaMaxOrdBDF)
	}
	if cfg.band == nil {
		if cfg.bandJac != nil {
			return odeInput("a banded Jacobian needs its bandwidth")
		}
		return nil
	}
	if s.jac != nil {
		return odeInput("with a bandwidth the Jacobian is BandJac, not Jac")
	}
	ml, mu := cfg.band.Lower, cfg.band.Upper
	if ml < 0 || mu < 0 || (n > 0 && (ml >= n || mu >= n)) {
		return odeInput("the bandwidths %d and %d are not in [0, n) for n = %d", ml, mu, n)
	}
	s.ml, s.mu = ml, mu
	s.bandJac = cfg.bandJac
	s.jtyp = 5
	if cfg.bandJac != nil {
		s.jtyp = 4
	}
	return nil
}

// allocMatrix allocates the matrix P: n×n, or the band.
func (s *lsodaSolver) allocMatrix() {
	if s.jtyp >= 4 {
		s.lub = newLUBand(s.n, s.ml, s.mu)
		s.bandBuf = make([]float64, (s.ml+s.mu+1)*s.n)
		return
	}
	s.lu = newLUReal(s.n)
}

// bnorm is the norm of the banded matrix a (ODEPACK's bnorm.f, consistent with
// vmnorm): max_i w_i Σ_j |a_ij| / w_j over the band, with a in the band storage
// of lsoda.c, where the diagonal is at row mu of the storage it is given (the
// storage of P has its first ml rows for the fill-in of the factorization, so
// lsoda.c, like ODEPACK, reads the rows above the ones P is in; kept as in
// lsoda.c: pdnorm only steers the method switch).
func bnorm(n int, a []float64, nra, ml, mu int, w []float64) float64 {
	an := 0.0
	for row := 0; row < n; row++ {
		sum := 0.0
		diag := row + mu
		first := max(row-ml, 0)
		last := min(row+mu, n-1)
		for col := first; col <= last; col++ {
			sum += math.Abs(a[diag-col+col*nra]) / w[col]
		}
		an = cmax(an, sum*w[row])
	}
	return an
}

// prjaBand is prja for a banded Jacobian (miter 4: the user's, 5: mband calls
// of f for the columns a band apart): P = I - h el0 J in the storage of the
// LU, its norm pdnorm, the factorization.
func (s *lsodaSolver) prjaBand(y []float64, hl0 float64) {
	n, ml, mu := s.n, s.ml, s.mu
	mband := ml + mu + 1
	meband := mband + ml
	ab := s.lub.ab
	clear(ab)
	if s.miter == 4 {
		jb := s.bandBuf
		clear(jb)
		s.bandJac(s.tn, y, jb)
		con := -hl0
		for r := 0; r < mband; r++ {
			for j := 0; j < n; j++ {
				ab[(ml+r)+j*meband] = jb[r*n+j] * con
			}
		}
	} else {
		mba := min(mband, n)
		srur := math.Sqrt(lsodaUround)
		fac := vmnorm(s.savf, s.ewt)
		r0 := 1000 * math.Abs(s.h) * lsodaUround * float64(n) * fac
		if r0 == 0 {
			r0 = 1
		}
		for j := 0; j < mba; j++ {
			for i := j; i < n; i += mband {
				y[i] += cmax(srur*math.Abs(y[i]), r0/s.ewt[i])
			}
			s.fun(s.tn, y, s.acor)
			for jj := j; jj < n; jj += mband {
				y[jj] = s.yh[jj]
				r := cmax(srur*math.Abs(y[jj]), r0/s.ewt[jj])
				fac = -hl0 / r
				i1 := max(jj-mu, 0)
				i2 := min(jj+ml, n-1)
				for i := i1; i <= i2; i++ {
					ab[(ml+mu+i-jj)+jj*meband] = (s.acor[i] - s.savf[i]) * fac
				}
			}
		}
		s.nfe += mba
	}
	if s.exact {
		s.pdnorm = bnorm(n, ab, meband, ml, mu, s.ewt) / math.Abs(hl0)
	} else {
		// the norm of the band P is in: rows kv = ml+mu of the storage hold the
		// diagonal, ml rows below where lsoda.c's bnorm reads it
		s.pdnorm = bnorm(n, ab[ml:], meband, ml, mu, s.ewt) / math.Abs(hl0)
	}
	for i := 0; i < n; i++ {
		ab[(ml+mu)+i*meband]++
	}
	if !s.lub.factor() {
		s.ierpj = 1
	}
}

// hitCrit: the step reached tcrit to within 100 uround (ihit).
func (s *lsodaSolver) hitCrit() bool {
	hmx := math.Abs(s.tn) + math.Abs(s.h)
	return math.Abs(s.tn-s.tcrit) <= 100*lsodaUround*hmx
}

// cutStep shortens h when the next step would pass tcrit, to reach it (jstart
// -2 tells stoda).
func (s *lsodaSolver) cutStep() {
	tnext := s.tn + s.h*(1+4*lsodaUround)
	if (tnext-s.tcrit)*s.h > 0 {
		s.h = (s.tcrit - s.tn) * (1 - 4*lsodaUround)
		if s.jstart >= 0 {
			s.jstart = -2
		}
	}
}

// call is lsoda(): it advances the solution from s.t toward tout and returns
// the time reached with istate (2 success, a negative ODEPACK code on
// failure) and the solution in y. itask 1 steps until tout is passed and
// interpolates there (Odeint); itask 4 does so without stepping past tcrit
// (Odeint with TCrit); itask 5 takes one step that does not pass tcrit
// (SolveIVP, as _ivp/lsoda.py does; the time returned is tcrit when the step
// reached it to within 100 uround).
func (s *lsodaSolver) call(y []float64, tout float64, itask int) (float64, int) {
	n := s.n
	first := !s.started
	if first {
		if st := s.start(tout, itask); st < 0 {
			return s.t, st
		}
	} else {
		// continuation: stop conditions before taking a step
		s.nslast = s.nst
		switch itask {
		case 1:
			if (s.tn-tout)*s.h >= 0 {
				s.intdy0(tout, y)
				return tout, 2
			}
		case 4:
			if (s.tn-s.tcrit)*s.h > 0 || (s.tcrit-tout)*s.h < 0 {
				return s.t, -3
			}
			if (s.tn-tout)*s.h >= 0 {
				s.intdy0(tout, y)
				return tout, 2
			}
			if s.hitCrit() {
				copy(y, s.yh[:n])
				return s.tcrit, 2
			}
			s.cutStep()
		case 5:
			if !s.hitCrit() {
				s.cutStep()
			}
		}
	}

	for {
		if !first {
			if s.nst-s.nslast >= s.mxstep {
				copy(y, s.yh[:n])
				return s.tn, -1
			}
			if !s.ewset() {
				copy(y, s.yh[:n])
				return s.tn, -6
			}
		}
		first = false

		// too much accuracy requested for the machine precision
		if tolsf := lsodaUround * vmnorm(s.yh[:n], s.ewt); tolsf > 0.01 {
			if s.nst == 0 {
				return s.t, -3
			}
			copy(y, s.yh[:n])
			return s.tn, -2
		}

		s.stoda()
		if s.kflag == -1 || s.kflag == -2 {
			copy(y, s.yh[:n])
			return s.tn, s.kflag - 3
		}
		// Not ODEPACK's: its norms (fmax) do not see a NaN, and a step that
		// accepted one goes on with it. A non-finite state is a failure here,
		// as in the other solvers.
		if !allFinite(s.yh[:n]) {
			copy(y, s.yh[:n])
			return s.tn, -9
		}

		// block f: a successful step. After a method switch tsw and maxord are
		// reset and jstart = -1 makes stoda complete the switch.
		s.started = true
		if s.meth != s.mused {
			s.tsw = s.tn
			s.maxord = s.mxordn
			if s.meth == 2 {
				s.maxord = s.mxords
			}
			s.jstart = -1
		}
		switch itask {
		case 1:
			if (s.tn-tout)*s.h < 0 {
				continue
			}
			s.intdy0(tout, y)
			return tout, 2
		case 4:
			if (s.tn-tout)*s.h >= 0 {
				s.intdy0(tout, y)
				return tout, 2
			}
			if s.hitCrit() {
				copy(y, s.yh[:n])
				return s.tcrit, 2
			}
			s.cutStep()
			continue
		}
		copy(y, s.yh[:n])
		if s.hitCrit() {
			return s.tcrit, 2
		}
		return s.tn, 2
	}
}
