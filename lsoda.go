// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// ODEPACK's LSODA (Hindmarsh, ODEPACK, A Systematized Collection of ODE
// Solvers, IMACS Trans. Sci. Comput. 1 (1983); Petzold, SIAM J. Sci. Stat.
// Comput. 4 (1983) 136-148) as SciPy 1.18.1 ships it: scipy/integrate/src/
// lsoda.c (the C translation of ODEPACK's lsoda.f, stoda.f, cfode.f, prja.f,
// solsy.f, intdy.f, ewset.f, vmnorm.f and fnorm.f) driven the way
// scipy/integrate/_ode.py (class lsoda) and _ivp/lsoda.py (itask = 5, one step
// not past t_bound) drive it for SolveIVP, and _odepackmodule.c (itask = 1,
// the interpolation at each output time) for Odeint.
//
// The names are ODEPACK's (the common block ls0001/lsa001 became the fields of
// lsodaSolver) so that the source reads against lsoda.c line by line. What is
// not ported, because nothing in the package's API reaches it:
//
//   - the diagonal approximation (miter = 3, which lsoda() rejects anyway);
//   - itask = 2 and 3 (the API has no one-step or relaxed mode), istate = 3
//     (change of parameters) and ixpr/mxhnil (printing only);
//   - the workspace-size bookkeeping (insufr, insufi, lenrw): Go slices have
//     no fixed length.
//
// Where this differs from SciPy: lsoda.c reads rwork[6] (hmin) into a local and
// never stores it in the common block, so SciPy's hmin/min_step have no effect;
// here hmin is ODEPACK's, the lower bound of the step (default 0, which is
// SciPy's behaviour).
//
// Arrays keep their Fortran layout (the Nordsieck array yh is column-major,
// yh[i+j*n] being the j-th scaled derivative of component i) with 0-based
// indices; elco[i+nq*13] and tesco[k+nq*3] are elco(i+1, nq+1) and
// tesco(k+1, nq+1) of the Fortran.

const (
	lsodaMaxOrdAdams = 12 // mord(1)
	lsodaMaxOrdBDF   = 5  // mord(2)
	lsodaUround      = dblEps
)

// ODEPACK's istate failure codes, with SciPy's messages (_ode.py lsoda.messages).
var lsodaMessages = map[int]string{
	-1: "Excess work done on this call (perhaps wrong Dfun type).",
	-2: "Excess accuracy requested (tolerances too small).",
	-3: "Illegal input detected (internal error).",
	-4: "Repeated error test failures (internal error).",
	-5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
	-6: "Error weight became zero during problem.",
	// not ODEPACK's: see call
	-9: "The state became non-finite (the right-hand side returned NaN or Inf).",
}

// cmax and cmin are C's fmax and fmin: a NaN argument loses (math.Max and
// math.Min propagate it). lsoda.c is written with them, and its handling of a
// non-finite state (it is not seen by the weighted norms) depends on it.
func cmax(a, b float64) float64 {
	switch {
	case a != a:
		return b
	case b != b:
		return a
	case a > b:
		return a
	}
	return b
}

func cmin(a, b float64) float64 {
	switch {
	case a != a:
		return b
	case b != b:
		return a
	case a < b:
		return a
	}
	return b
}

// vmnorm is the weighted max-norm max_i |v_i| w_i (vmnorm.f).
func vmnorm(v, w []float64) float64 {
	vm := 0.0
	for i, x := range v {
		vm = cmax(vm, math.Abs(x)*w[i])
	}
	return vm
}

// fnorm is the norm of the n×n row-major matrix a consistent with vmnorm
// (fnorm.f): max_i w_i Σ_j |a_ij| / w_j.
func fnorm(n int, a, w []float64) float64 {
	an := 0.0
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += math.Abs(a[i*n+j]) / w[j]
		}
		an = cmax(an, sum*w[i])
	}
	return an
}

// lsodaSolver is ODEPACK's LSODA state (the common blocks ls0001 and lsa001
// and the work arrays of lsoda.c) over an odeCore.
type lsodaSolver struct {
	*odeCore

	// double precision common block
	conit, crate                 float64
	el                           [13]float64
	elco                         [156]float64
	hold, rmax                   float64
	tesco                        [36]float64
	ccmax, el0, h, hmin, hmxi    float64
	hu, rc, tn, tsw              float64
	pdest, pdlast, ratio, pdnorm float64
	cm1                          [12]float64
	cm2                          [5]float64

	// integer common block
	nslast, ialth, ipup, lmax, nqnyh, nslp, icf, ierpj, jcur, jstart int
	kflag, l, meth, miter, maxord, maxcor, msbp, mxncf, nq, nst      int
	nfe, nje, nqu, mxstep, icount, irflag, jtyp, mused               int
	mxordn, mxords                                                   int

	started bool

	// work arrays: yh (13 columns, the largest order 12 plus one), the inverted
	// error weights ewt, savf, acor, and the scratch y of stoda.
	yh, ewt, savf, acor, y1 []float64
	lu                      *luReal
	lub                     *luBand // the banded P of jt 4 and 5, instead of lu
	bandJac                 BandJacFunc
	bandBuf                 []float64 // the user's banded Jacobian, (ml+mu+1)×n
	ml, mu                  int
	tcrit                   float64 // itask 4 and 5: the time not to step past

	// the interpolant of the last step (SolveIVP), refilled by denseImpl(false)
	denseBuf *lsodaDense
}

// newLsoda builds the solver; nothing is evaluated until the first step.
func newLsoda(core *odeCore) *lsodaSolver {
	s := newLsodaBase(core)
	s.allocMatrix()
	return s
}

// newLsodaConfig builds the solver with options (bandwidths, hmin, mxordn,
// mxords); the options are checked here.
func newLsodaConfig(core *odeCore, cfg lsodaConfig) (*lsodaSolver, error) {
	s := newLsodaBase(core)
	if err := s.configure(cfg); err != nil {
		return nil, err
	}
	s.allocMatrix()
	return s, nil
}

func newLsodaBase(core *odeCore) *lsodaSolver {
	n := core.n
	s := &lsodaSolver{odeCore: core}
	s.yh = make([]float64, (lsodaMaxOrdAdams+1)*n)
	s.ewt = make([]float64, n)
	s.savf = make([]float64, n)
	s.acor = make([]float64, n)
	s.y1 = make([]float64, n)
	s.tcrit = core.tBound
	s.mxstep = 500
	s.mxordn, s.mxords = lsodaMaxOrdAdams, lsodaMaxOrdBDF
	if !math.IsInf(core.maxStep, 1) {
		s.hmxi = 1 / core.maxStep
	}
	s.jtyp = 2
	if core.jac != nil {
		s.jtyp = 1
	}
	return s
}

// cfode sets the method coefficients elco and the test constants tesco of
// every order of method meth (1 Adams-Moulton up to order 12, 2 BDF up to
// order 5): cfode.f.
func (s *lsodaSolver) cfode(meth int) {
	elco, tesco := &s.elco, &s.tesco
	var pc [12]float64
	if meth == 1 {
		elco[0] = 1
		elco[1] = 1
		tesco[0] = 0
		tesco[1] = 2
		tesco[3] = 1
		tesco[35] = 0
		pc[0] = 1
		rqfac := 1.0
		for nq := 1; nq < 12; nq++ {
			// The pc array holds the coefficients of p(x) = (x+1)(x+2)...(x+nq-1).
			rq1fac := rqfac
			rqfac /= float64(nq + 1)
			pc[nq] = 0
			for ib := 0; ib <= nq-1; ib++ {
				pc[nq-ib] = pc[nq-ib-1] + float64(nq)*pc[nq-ib]
			}
			pc[0] *= float64(nq)
			// the integrals, from -1 to 0, of p(x) and x p(x)
			pint := pc[0]
			xpin := pc[0] * 0.5
			tsign := 1.0
			for i := 1; i <= nq; i++ {
				tsign = -tsign
				pint += tsign * pc[i] / float64(i+1)
				xpin += tsign * pc[i] / float64(i+2)
			}
			elco[nq*13] = pint * rq1fac
			elco[1+nq*13] = 1
			for i := 1; i <= nq; i++ {
				elco[(i+1)+nq*13] = rq1fac * pc[i] / float64(i+1)
			}
			agamq := rqfac * xpin
			ragq := 1 / agamq
			tesco[1+3*nq] = ragq
			if nq < 11 {
				tesco[3*(nq+1)] = ragq * rqfac / float64(nq+2)
			}
			tesco[2+3*(nq-1)] = ragq
		}
		return
	}
	// BDF: pc holds the coefficients of p(x) = (x+1)(x+2)...(x+nq).
	pc[0] = 1
	rq1fac := 1.0
	for nq := 0; nq < 5; nq++ {
		pc[nq+1] = 0
		for ib := 0; ib <= nq; ib++ {
			pc[nq-ib+1] = pc[nq-ib] + float64(nq+1)*pc[nq-ib+1]
		}
		pc[0] *= float64(nq + 1)
		for i := 0; i <= nq+1; i++ {
			elco[i+nq*13] = pc[i] / pc[1]
		}
		elco[1+nq*13] = 1
		tesco[3*nq] = rq1fac
		tesco[1+3*nq] = float64(nq+2) / elco[nq*13]
		tesco[2+3*nq] = float64(nq+3) / elco[nq*13]
		rq1fac /= float64(nq + 1)
	}
}

// prja computes and factors P = I - h el0 J (prja.f, jt = 1 or 2): J from the
// user's Jacobian or by forward differences with ODEPACK's increments, the
// norm pdnorm of J, then the LU decomposition. y holds the predicted values on
// entry and is restored; acor is the work vector; savf is f at the predicted y.
func (s *lsodaSolver) prja(y []float64) {
	n := s.n
	s.nje++
	s.ierpj = 0
	s.jcur = 1
	hl0 := s.h * s.el0
	if s.miter >= 4 {
		s.prjaBand(y, hl0)
		return
	}
	P := s.lu.a
	if s.miter == 1 {
		clear(P)
		s.jac(s.tn, y, P)
		con := -hl0
		for i := range P {
			P[i] *= con
		}
	} else {
		// miter = 2: n calls of f approximate J
		fac := vmnorm(s.savf, s.ewt)
		r0 := 1000 * math.Abs(s.h) * lsodaUround * float64(n) * fac
		if r0 == 0 {
			r0 = 1
		}
		srur := math.Sqrt(lsodaUround)
		for j := 0; j < n; j++ {
			yj := y[j]
			r := cmax(srur*math.Abs(yj), r0/s.ewt[j])
			y[j] += r
			fac = -hl0 / r
			s.fun(s.tn, y, s.acor)
			for i := 0; i < n; i++ {
				P[i*n+j] = (s.acor[i] - s.savf[i]) * fac
			}
			y[j] = yj
		}
		s.nfe += n
	}
	s.pdnorm = fnorm(n, P, s.ewt) / math.Abs(hl0)
	for i := 0; i < n; i++ {
		P[i*n+i]++
	}
	s.lu.factor(P)
	// dgetrf's info > 0: an exactly zero pivot.
	for i := 0; i < n; i++ {
		if P[i*n+i] == 0 {
			s.ierpj = 1
			return
		}
	}
}

// stodaReset is stoda's label 150: the el vector and the constants that depend
// on the order, set when the order changes and at the start.
func (s *lsodaSolver) stodaReset() {
	for i := 0; i < s.l; i++ {
		s.el[i] = s.elco[i+(s.nq-1)*13]
	}
	s.nqnyh = s.nq * s.n
	s.rc = s.rc * s.el[0] / s.el0
	s.el0 = s.el[0]
	s.conit = 0.5 / float64(s.nq+2)
}

var lsodaSM1 = [12]float64{0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15, 0.10, 0.075, 0.050, 0.025}

// stodaAdjust rescales the step by *rh (labels 170-180): bounded by rmax and
// hmax, for Adams by the stability region, and the yh array rescaled.
func (s *lsodaSolver) stodaAdjust(rh *float64) {
	n := s.n
	*rh = cmin(*rh, s.rmax)
	*rh = *rh / cmax(1, math.Abs(s.h)*s.hmxi*(*rh))
	// For Adams the new step is also limited by the stability region; irflag
	// records that it was, so that roundoff trouble can be blamed on it.
	if s.meth != 2 {
		s.irflag = 0
		pdh := cmax(math.Abs(s.h)*s.pdlast, 0.000001)
		if (*rh)*pdh*1.00001 >= lsodaSM1[s.nq-1] {
			*rh = lsodaSM1[s.nq-1] / pdh
			s.irflag = 1
		}
	}
	r := 1.0
	for j := 1; j < s.l; j++ {
		r *= *rh
		col := s.yh[j*n : (j+1)*n]
		for i := range col {
			col[i] *= r
		}
	}
	s.h *= *rh
	s.rc *= *rh
	s.ialth = s.l
}

// stodaPredict computes the predicted values by multiplying yh by the Pascal
// triangle matrix (label 200); it flags a Jacobian update when rc moved by
// more than ccmax or msbp steps went by.
func (s *lsodaSolver) stodaPredict() {
	if math.Abs(s.rc-1) > s.ccmax {
		s.ipup = s.miter
	}
	if s.nst >= s.nslp+s.msbp {
		s.ipup = s.miter
	}
	s.tn += s.h
	s.stodaPascal(+1)
}

// stodaPascal adds (sign +1, prediction) or subtracts (-1, retraction to the
// values before the prediction) the Pascal triangle product to yh.
func (s *lsodaSolver) stodaPascal(sign float64) {
	n := s.n
	yh := s.yh
	i1 := s.nqnyh
	for jb := 0; jb < s.nq; jb++ {
		i1 -= n
		for i := i1; i < s.nqnyh; i++ {
			yh[i] += sign * yh[i+n]
		}
	}
}

type lsodaCorr int

const (
	corrConverged lsodaCorr = iota
	corrFailed
	corrRetry
)

// stodaCorrector is stoda's corrector loop (labels 220-450): up to maxcor
// iterations, functional (miter 0) or chord with the matrix P, convergence
// tested on the weighted max-norm of the corrections. It returns the status,
// the number m of iterations and the norm del of the last correction.
func (s *lsodaSolver) stodaCorrector(pnorm float64) (lsodaCorr, int, float64) {
	n := s.n
	yh, y, savf, acor, ewt := s.yh, s.y1, s.savf, s.acor, s.ewt
	m := 0
	rate, del, delp := 0.0, 0.0, 0.0
	copy(y, yh[:n])
	s.fun(s.tn, y, savf)
	s.nfe++
	if s.ipup > 0 {
		// P = I - h el(1) J is reevaluated and factored before the iteration.
		s.prja(y)
		s.ipup = 0
		s.rc = 1
		s.nslp = s.nst
		s.crate = 0.7
		if s.ierpj != 0 {
			return corrFailed, m, del
		}
	}
	clear(acor)
	for {
		if s.miter == 0 {
			// functional iteration: y from the last evaluation of f
			for i := 0; i < n; i++ {
				savf[i] = s.h*savf[i] - yh[i+n]
				y[i] = savf[i] - acor[i]
			}
			del = vmnorm(y, ewt)
			for i := 0; i < n; i++ {
				y[i] = yh[i] + s.el[0]*savf[i]
				acor[i] = savf[i]
			}
		} else {
			// chord: the corrector error is the right-hand side of the system with P
			for i := 0; i < n; i++ {
				y[i] = s.h*savf[i] - (yh[i+n] + acor[i])
			}
			if s.lub != nil {
				s.lub.solve(y)
			} else {
				s.lu.solve(y)
			}
			del = vmnorm(y, ewt)
			for i := 0; i < n; i++ {
				acor[i] += y[i]
				y[i] = yh[i] + s.el[0]*acor[i]
			}
		}
		// Convergence test. A change of the iterates of the size of roundoff
		// converges with no new rate estimate. Otherwise at least two
		// iterations are forced for Adams, to estimate a local Lipschitz
		// constant; on convergence pdest is the local maximum of it and pdlast
		// the latest non-zero one.
		if del <= 100*pnorm*lsodaUround {
			break
		}
		if m != 0 || s.meth != 1 {
			if m != 0 {
				rm := 1024.0
				if del <= 1024*delp {
					rm = del / delp
				}
				rate = cmax(rate, rm)
				s.crate = cmax(0.2*s.crate, rm)
			}
			dcon := del * cmin(1, 1.5*s.crate) / (s.tesco[1+(s.nq-1)*3] * s.conit)
			if dcon <= 1 {
				s.pdest = cmax(s.pdest, rate/math.Abs(s.h*s.el[0]))
				if s.pdest != 0 {
					s.pdlast = s.pdest
				}
				break
			}
		}
		m++
		if m == s.maxcor || (m >= 2 && del > 2*delp) {
			// too many iterations, or diverging: retry once with a fresh Jacobian
			if s.miter != 0 && s.jcur != 1 {
				s.icf = 1
				s.ipup = s.miter
				return corrRetry, m, del
			}
			return corrFailed, m, del
		}
		delp = del
		s.fun(s.tn, y, savf)
		s.nfe++
	}
	s.jcur = 0
	return corrConverged, m, del
}

// stodaFailure handles a corrector failure (label 430): the step is retried
// with h/4 and a fresh Jacobian, unless 10 failures occurred (kflag -2).
// It reports whether to retry.
func (s *lsodaSolver) stodaFailure(ncf *int, told float64) bool {
	s.icf = 2
	*ncf++
	s.rmax = 2
	s.tn = told
	s.stodaPascal(-1)
	if math.Abs(s.h) <= s.hmin*1.00001 || *ncf == s.mxncf {
		s.kflag = -2
		return false
	}
	rh := 0.25
	s.ipup = s.miter
	rh = cmax(rh, s.hmin/math.Abs(s.h))
	s.stodaAdjust(&rh)
	return true
}

// stoda performs one step of the integration (stoda.f), with the control flow
// of lsoda.c: the corrector loop is stodaCorrector, "go to 690/700" is the end
// of the step loop. jstart: 0 first call, 1 continue, -1 a method or
// parameter change, -2 a step change only (tcrit).
func (s *lsodaSolver) stoda() {
	n := s.n
	yh, y, savf, acor, ewt := s.yh, s.y1, s.savf, s.acor, s.ewt
	sm1 := &lsodaSM1
	var rh, dsm, exup, exdn, exsm, rhup, rhdn, rhsm, dup, ddn float64
	var dm1, dm2, exm1, rh1, rh2, rh1it, pdh float64
	var newq, nqm1, nqm2, ncf int
	told := s.tn
	resetRmax := false

	s.kflag = 0
	s.ierpj = 0
	s.jcur = 0
	s.icf = 0

	switch s.jstart {
	case 0:
		// the first call: initialize and set up both methods' coefficients
		s.lmax = s.maxord + 1
		s.nq = 1
		s.l = 2
		s.ialth = 2
		s.rmax = 10000
		s.rc = 0
		s.el0 = 1
		s.crate = 0.7
		s.hold = s.h
		s.nslp = 0
		s.ipup = s.miter
		s.icount = 20
		s.irflag = 0
		s.pdest = 0
		s.pdlast = 0
		s.ratio = 5
		s.cfode(2)
		for i := 0; i < 5; i++ {
			s.cm2[i] = s.tesco[1+i*3] * s.elco[i+1+i*13]
		}
		s.cfode(1)
		for i := 0; i < 12; i++ {
			s.cm1[i] = s.tesco[1+i*3] * s.elco[i+1+i*13]
		}
		s.stodaReset()
	case -1:
		// a method switch was just made (or parameters changed)
		// (lsoda.c also sets ialth = 2 when it is 1; stodaAdjust leaves it at l,
		// at least 2, before any switch, so that cannot arise here.)
		s.ipup = s.miter
		s.lmax = s.maxord + 1
		s.cfode(s.meth)
		s.ialth = s.l
		s.stodaReset()
		if s.h != s.hold {
			rh = s.h / s.hold
			s.h = s.hold
			rh = cmax(rh, s.hmin/math.Abs(s.h))
			s.stodaAdjust(&rh)
		}
	case -2:
		// h was changed (to reach tcrit)
		if s.h != s.hold {
			rh = s.h / s.hold
			s.h = s.hold
			s.stodaAdjust(&rh)
		}
	}

	s.stodaPredict()
	pnorm := vmnorm(yh[:n], ewt)

step:
	for {
		st, m, del := s.stodaCorrector(pnorm)
		switch st {
		case corrRetry:
			continue
		case corrFailed:
			if !s.stodaFailure(&ncf, told) {
				s.hold = s.h
				s.jstart = 1
				return
			}
			s.stodaPredict()
			pnorm = vmnorm(yh[:n], ewt)
			continue
		}

		// The corrector converged: the local error test (label 450).
		s.jcur = 0
		if m == 0 {
			dsm = del / s.tesco[1+(s.nq-1)*3]
		} else {
			dsm = vmnorm(acor, ewt) / s.tesco[1+(s.nq-1)*3]
		}

		if dsm > 1 {
			// The error test failed (label 500): restore tn and yh, and retry
			// with the best step for this or one lower order; after two
			// failures h falls by at least 0.2, after three the first
			// derivative is recomputed and the order set to 1.
			s.kflag--
			s.tn = told
			s.stodaPascal(-1)
			s.rmax = 2
			if math.Abs(s.h) <= s.hmin*1.00001 {
				s.kflag = -1
				s.hold = s.h
				s.jstart = 1
				return
			}
			if s.kflag <= -3 {
				if s.kflag == -10 {
					s.kflag = -1
					s.hold = s.h
					s.jstart = 1
					return
				}
				rh = cmax(s.hmin/math.Abs(s.h), 0.1)
				s.h *= rh
				copy(y, yh[:n])
				s.fun(s.tn, y, savf)
				s.nfe++
				for i := 0; i < n; i++ {
					yh[i+n] = s.h * savf[i]
				}
				s.ipup = s.miter
				s.ialth = 5
				if s.nq != 1 {
					s.nq = 1
					s.l = 2
					s.stodaReset()
				}
				s.stodaPredict()
				pnorm = vmnorm(yh[:n], ewt)
				continue
			}
			rhup = 0
		} else {
			// The step succeeded: update yh, and every 20 steps (icount)
			// consider switching methods; otherwise consider changing h (and
			// the order by one) when ialth reaches 1.
			s.kflag = 0
			resetRmax = true
			s.nst++
			s.hu = s.h
			s.nqu = s.nq
			s.mused = s.meth
			for j := 0; j < s.l; j++ {
				for i := 0; i < n; i++ {
					yh[i+j*n] += s.el[j] * acor[i]
				}
			}
			s.icount--
			switch {
			case s.icount >= 0:
				// no switch test
			case s.meth == 1:
				// Adams: consider BDF. Above order 5 the problem is assumed
				// non-stiff. If the Lipschitz constant and the error estimate
				// are polluted by roundoff, switch only when the last step was
				// restricted for stability (irflag), doubling the step;
				// otherwise compare the steps Adams and BDF could have taken
				// (BDF must win by the factor ratio = 5).
				condB := dsm > 100*pnorm*lsodaUround && s.pdest != 0
				if !(s.nq > 5 || (!condB && s.irflag == 0)) {
					// If nq > mxords the order mxords is considered on switching.
					// (nqm2 is set below on every path.)
					nqm2 = s.nq
					switchBDF := true
					if !condB {
						rh2 = 2
						nqm2 = min(s.nq, s.mxords)
					} else {
						exsm = 1 / float64(s.l)
						rh1 = 1 / (1.2*math.Pow(dsm, exsm) + 0.0000012)
						rh1it = 2 * rh1
						pdh = s.pdlast * math.Abs(s.h)
						if pdh*rh1 > 0.00001 {
							rh1it = sm1[s.nq-1] / pdh
						}
						rh1 = cmin(rh1, rh1it)
						if s.nq > s.mxords {
							nqm2 = s.mxords
							lm2 := s.mxords + 1
							exm2 := 1 / float64(lm2)
							dm2 = vmnorm(yh[lm2*n:(lm2+1)*n], ewt) / s.cm2[s.mxords-1]
							rh2 = 1 / (1.2*math.Pow(dm2, exm2) + 0.0000012)
						} else {
							dm2 = dsm * (s.cm1[s.nq-1] / s.cm2[s.nq-1])
							rh2 = 1 / (1.2*math.Pow(dm2, exsm) + 0.0000012)
							nqm2 = s.nq
						}
						switchBDF = rh2 >= s.ratio*rh1
					}
					if switchBDF {
						rh = rh2
						s.icount = 20
						s.meth = 2
						s.miter = s.jtyp
						s.pdlast = 0
						s.nq = nqm2
						s.l = s.nq + 1
						rh = cmax(rh, s.hmin/math.Abs(s.h))
						s.stodaAdjust(&rh)
						break step
					}
				}
			default:
				// BDF: consider Adams. If nq > mxordn the order mxordn is
				// considered on switching. The Adams step must be at least
				// 5/ratio = 1 times the BDF step, and not so small that
				// roundoff pollutes it.
				// (the order is not above mxords here, but may be above mxordn.)
				exsm = 1 / float64(s.l)
				if s.mxordn >= s.nq {
					dm1 = dsm * (s.cm2[s.nq-1] / s.cm1[s.nq-1])
					rh1 = 1 / (1.2*math.Pow(dm1, exsm) + 0.0000012)
					nqm1 = s.nq
					exm1 = exsm
				} else {
					nqm1 = s.mxordn
					lm1 := s.mxordn + 1
					exm1 = 1 / float64(lm1)
					dm1 = vmnorm(yh[lm1*n:(lm1+1)*n], ewt) / s.cm1[s.mxordn-1]
					rh1 = 1 / (1.2*math.Pow(dm1, exm1) + 0.0000012)
				}
				rh1it = 2 * rh1
				pdh = s.pdnorm * math.Abs(s.h)
				if pdh*rh1 > 0.00001 {
					rh1it = sm1[nqm1-1] / pdh
				}
				rh1 = cmin(rh1, rh1it)
				rh2 = 1 / (1.2*math.Pow(dsm, exsm) + 0.0000012)
				if rh1*s.ratio >= 5*rh2 {
					dm1 = math.Pow(cmax(0.001, rh1), exm1) * dm1
					if dm1 > 1000*lsodaUround*pnorm {
						rh = rh1
						s.icount = 20
						s.meth = 1
						s.miter = 0
						s.pdlast = 0
						s.nq = nqm1
						s.l = s.nq + 1
						rh = cmax(rh, s.hmin/math.Abs(s.h))
						s.stodaAdjust(&rh)
						break step
					}
				}
			}

			// No method switch: the usual step and order selection (488).
			s.ialth--
			if s.ialth == 0 {
				rhup = 0
				if s.l != s.lmax {
					for i := 0; i < n; i++ {
						savf[i] = acor[i] - yh[i+(s.lmax-1)*n]
					}
					dup = vmnorm(savf, ewt) / s.tesco[2+(s.nq-1)*3]
					exup = 1 / (float64(s.l) + 1)
					rhup = 1 / (1.4*math.Pow(dup, exup) + 0.0000014)
				}
			} else {
				// acor is kept for a possible order increase on the next step
				if !(s.ialth > 1 || s.l == s.lmax) {
					copy(yh[(s.lmax-1)*n:s.lmax*n], acor)
				}
				// label 700, without the rmax reset
				r := 1 / s.tesco[1+(s.nqu-1)*3]
				for i := range acor {
					acor[i] *= r
				}
				s.hold = s.h
				s.jstart = 1
				return
			}
		}

		// Label 540: the step ratios for the order nq-1, nq and nq+1.
		exsm = 1 / float64(s.l)
		rhsm = 1 / (1.2*math.Pow(dsm, exsm) + 0.0000012)
		rhdn = 0
		if s.nq != 1 {
			ddn = vmnorm(yh[(s.l-1)*n:s.l*n], ewt) / s.tesco[(s.nq-1)*3]
			exdn = 1 / float64(s.nq)
			rhdn = 1 / (1.3*math.Pow(ddn, exdn) + 0.0000013)
		}
		if s.meth == 1 {
			pdh = cmax(math.Abs(s.h)*s.pdlast, 0.000001)
			if s.l < s.lmax {
				rhup = cmin(rhup, sm1[s.l-1]/pdh)
			}
			rhsm = cmin(rhsm, sm1[s.nq-1]/pdh)
			if s.nq > 1 {
				rhdn = cmin(rhdn, sm1[s.nq-2]/pdh)
			}
			s.pdest = 0
		}
		lower := func() {
			newq = s.nq - 1
			rh = rhdn
			if s.kflag < 0 && rh > 1 {
				rh = 1
			}
		}
		switch {
		case rhsm >= rhup && rhsm >= rhdn:
			newq = s.nq
			rh = rhsm
		case rhsm >= rhup:
			lower()
		case rhup > rhdn:
			newq = s.l
			rh = rhup
			if rh < 1.1 {
				s.ialth = 3
				resetRmax = false
				break step
			}
			// the order rises: the new column of yh from acor
			r := s.el[s.l-1] / float64(s.l)
			for i := 0; i < n; i++ {
				yh[i+newq*n] = acor[i] * r
			}
			s.nq = newq
			s.l = s.nq + 1
			s.stodaReset()
			rh = cmax(rh, s.hmin/math.Abs(s.h))
			s.stodaAdjust(&rh)
			resetRmax = true
			break step
		default:
			lower()
		}
		// A change of 10% or less is not worth it, unless Adams is held by its
		// stability region (label 620).
		bypass := s.meth == 1 && rh*pdh*1.00001 >= sm1[newq-1]
		if !bypass && s.kflag == 0 && rh < 1.1 {
			s.ialth = 3
			resetRmax = false
			break step
		}
		if s.kflag <= -2 {
			rh = cmin(rh, 0.2)
		}
		if newq != s.nq {
			s.nq = newq
			s.l = s.nq + 1
			s.stodaReset()
		}
		rh = cmax(rh, s.hmin/math.Abs(s.h))
		s.stodaAdjust(&rh)
		if s.kflag == 0 {
			resetRmax = true
			break step
		}
		// after an error test failure the step is retried (label 200)
		s.stodaPredict()
		pnorm = vmnorm(yh[:n], ewt)
	}

	// label 690: rmax is reset after a successful step; label 700: acor scaled
	// to the error estimate.
	if resetRmax {
		s.rmax = 10
	}
	r := 1 / s.tesco[1+(s.nqu-1)*3]
	for i := range acor {
		acor[i] *= r
	}
	s.hold = s.h
	s.jstart = 1
}

// intdy0 writes the interpolated value at t (the 0th derivative of the
// interpolating polynomial of the Nordsieck array, intdy.f with k = 0) into
// dky. t lies in the last step by construction.
func (s *lsodaSolver) intdy0(t float64, dky []float64) {
	n := s.n
	sx := (t - s.tn) / s.h
	copy(dky, s.yh[(s.l-1)*n:s.l*n])
	for jb := 1; jb <= s.nq; jb++ {
		col := s.yh[(s.nq-jb)*n : (s.nq-jb+1)*n]
		for i := range dky {
			dky[i] = col[i] + sx*dky[i]
		}
	}
}

// ewset sets the inverted error weights: ewt = 1/(rtol |y| + atol). It reports
// false for a zero weight.
func (s *lsodaSolver) ewset() bool {
	for i := 0; i < s.n; i++ {
		w := s.rtol[i]*math.Abs(s.yh[i]) + s.atol[i]
		if w <= 0 {
			return false
		}
		s.ewt[i] = 1 / w
	}
	return true
}

// lsodaStart is block c of lsoda(), the first call: the initial f, the weights
// and the initial step h0 (the user's, or from
// h0^-2 = 1/(tol w0^2) + tol ||f||^2), loaded into yh. It returns istate.
func (s *lsodaSolver) start(tout float64, itask int) int {
	n := s.n
	t := s.t
	s.tn = t
	s.tsw = t
	s.maxord = s.mxordn
	h0 := s.firstStep * s.direction
	if itask == 4 || itask == 5 {
		tcrit := s.tcrit
		if (tcrit-tout)*(tout-t) < 0 {
			return -3
		}
		if h0 != 0 && (t+h0-tcrit)*h0 > 0 {
			h0 = tcrit - t
		}
	}
	s.jstart = 0
	s.nst = 0
	s.nje = 0
	s.nslast = 0
	s.hu = 0
	s.nqu = 0
	s.mused = 0
	s.miter = 0
	s.ccmax = 0.3
	s.maxcor = 3
	s.msbp = 20
	s.mxncf = 10
	s.meth = 1

	f0 := s.yh[n : 2*n]
	s.fun(t, s.y, f0)
	s.nfe = 1
	copy(s.yh[:n], s.y)
	s.nq = 1
	s.h = 1
	if !s.ewset() {
		return -3
	}

	if h0 == 0 {
		tdist := math.Abs(tout - t)
		w0 := cmax(math.Abs(t), math.Abs(tout))
		if tdist < 2*lsodaUround*w0 {
			return -3
		}
		// tol is max(rtol_i), at least 100 uround (the rtols are clamped to
		// that already) and at most 1e-3.
		tol := s.rtol[0]
		for _, r := range s.rtol {
			tol = cmax(tol, r)
		}
		tol = cmax(tol, 100*lsodaUround)
		tol = cmin(tol, 0.001)
		sum := vmnorm(f0, s.ewt)
		sum = 1/(tol*w0*w0) + tol*sum*sum
		h0 = 1 / math.Sqrt(sum)
		h0 = cmin(h0, tdist)
		h0 = math.Copysign(h0, tout-t)
	}
	if rh := math.Abs(h0) * s.hmxi; rh > 1 {
		h0 /= rh
	}
	s.h = h0
	for i := range f0 {
		f0[i] *= h0
	}
	return 0
}

// stepImpl is _ivp/lsoda.py's _step_impl: one lsoda() step with itask 5.
func (s *lsodaSolver) stepImpl() (bool, string) {
	// Not ODEPACK's: with hmin = 0 (see above) it lets h shrink until it
	// underflows to 0 and goes on with steps that do not advance t, as every
	// call is one step and mxstep never counts them. A first step below the
	// spacing at t is fine (h grows by up to 10^4 a step); h = 0 is the end.
	if s.started && s.h == 0 {
		return false, tooSmallStep
	}
	t, istate := s.call(s.y, s.tBound, 5)
	s.nfev = s.nfe
	s.njev = s.nje
	s.nlu = s.nje
	if istate < 0 {
		return false, lsodaMessages[istate]
	}
	s.t = t
	return true, ""
}

// lsodaDense is LsodaDenseOutput: the Nordsieck polynomial of the last step,
// sum_j yh_j x^j with x = (t - tNew)/h, for the orders 0..order.
type lsodaDense struct {
	n, order int
	tNew, h  float64
	yh       []float64 // column-major, n×(order+1)
}

func (d *lsodaDense) dim() int { return d.n }

func (d *lsodaDense) eval(t float64, y []float64) {
	n := d.n
	x := (t - d.tNew) / d.h
	copy(y, d.yh[d.order*n:(d.order+1)*n])
	for j := d.order - 1; j >= 0; j-- {
		col := d.yh[j*n : (j+1)*n]
		for i := range y {
			y[i] = col[i] + x*y[i]
		}
	}
}

// denseImpl is _ivp/lsoda.py's _dense_output_impl: the Nordsieck array of the
// last step, of the order used (nqu), whose columns stoda left scaled by the
// step it proposes next (h): the columns above the next order were not
// rescaled when the order falls and are brought to the same scale here.
func (s *lsodaSolver) denseImpl(fresh bool) denseOutput {
	n := s.n
	order := s.nqu
	d := &lsodaDense{n: n}
	if !fresh {
		if s.denseBuf == nil {
			s.denseBuf = d
		}
		d = s.denseBuf
	}
	if size := (order + 1) * n; cap(d.yh) < size {
		d.yh = make([]float64, size)
	} else {
		d.yh = d.yh[:size]
	}
	d.order, d.tNew, d.h = order, s.t, s.h
	copy(d.yh, s.yh[:(order+1)*n])
	if s.nq < order {
		ratio := s.h / s.hu
		r := 1.0
		for j := 1; j <= order; j++ {
			r *= ratio
			if j > s.nq {
				col := d.yh[j*n : (j+1)*n]
				for i := range col {
					col[i] *= r
				}
			}
		}
	}
	return d
}
