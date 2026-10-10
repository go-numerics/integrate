// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Implicit Radau IIA of order 5 (Hairer & Wanner, Solving ODEs II, IV.8;
// SciPy's Radau class): three-stage collocation solved by a simplified
// Newton iteration on the transformed system (one real and one complex LU),
// with the embedded error estimate of Hairer's radau5.

var (
	radauC  = [3]float64{0.15505102572168222, 0.6449489742783178, 1.0}
	radauE  = [3]float64{-10.048809399827414, 1.382142733160748, -0.3333333333333333}
	radauMu = 3.637834252744496
	// radauMuC is the complex eigenvalue of the inverse Radau matrix.
	radauMuC = complex(2.6810828736277523, -3.050430199247411)
	radauT   = [3][3]float64{
		{0.09443876248897524, -0.1412552950209542, 0.03002919410514742},
		{0.2502131229653333, 0.20412935229379994, -0.3829421127572619},
		{1.0, 1.0, 0.0},
	}
	radauTI = [3][3]float64{
		{4.178718591551904, 0.32768282076106237, 0.5233764454994495},
		{-4.178718591551904, -0.32768282076106237, 0.47662355450055044},
		{0.5028726349457868, -2.571926949855605, 0.5960392048282249},
	}
	radauP = [3][3]float64{
		{10.048809399827414, -25.62959144707664, 15.580782047249224},
		{-1.382142733160748, 10.296258113743303, -8.914115380582556},
		{0.3333333333333333, -2.6666666666666665, 3.3333333333333335},
	}
)

const (
	radauNewtonMaxIter = 6
	radauMinFactor     = 0.2
	radauMaxFactor     = 10
)

type radauSolver struct {
	*odeCore
	f          []float64
	hAbs       float64
	hAbsOld    float64 // 0 means "none"
	errNormOld float64 // < 0 means "none"
	newtonTol  float64
	J          []float64
	currentJac bool
	lu         *luReal
	luc        *luComplex
	luValid    bool
	Z          [3][]float64
	sol        *radauDense
	yOld       []float64
	mReal      []float64
	mCplx      []complex128
	// Newton scratch.
	F    [3][]float64
	W    [3][]float64
	dW   [3][]float64
	Z0   [3][]float64
	fr   []float64
	fc   []complex128
	ytmp []float64
	// Step scratch.
	scale, errv, ze []float64
}

// minRTol is the most stringent relative tolerance (1e-3 for an empty
// system).
func minRTol(rtol []float64) float64 {
	r := 1e-3
	for i, v := range rtol {
		if i == 0 || v < r {
			r = v
		}
	}
	return r
}

// newtonTolerance is SciPy's max(10·eps/rtol, min(0.03, √rtol)); with
// per-component tolerances the most stringent one is used.
func newtonTolerance(rtol []float64) float64 {
	r := minRTol(rtol)
	return math.Max(10*dblEps/r, math.Min(0.03, math.Sqrt(r)))
}

func newRadau(core *odeCore) *radauSolver {
	n := core.n
	s := &radauSolver{odeCore: core, errNormOld: -1}
	s.f = make([]float64, n)
	core.eval(core.t, core.y, s.f)
	s.hAbs = core.initialStep(s.f, 3)
	s.newtonTol = newtonTolerance(core.rtol)
	s.J = make([]float64, n*n)
	core.jacobian(core.t, core.y, s.f, s.J)
	s.currentJac = true
	s.lu = newLUReal(n)
	s.luc = newLUComplex(n)
	s.mReal = make([]float64, n*n)
	s.mCplx = make([]complex128, n*n)
	for k := 0; k < 3; k++ {
		s.Z[k] = make([]float64, n)
		s.F[k] = make([]float64, n)
		s.W[k] = make([]float64, n)
		s.dW[k] = make([]float64, n)
		s.Z0[k] = make([]float64, n)
	}
	s.fr = make([]float64, n)
	s.fc = make([]complex128, n)
	s.ytmp = make([]float64, n)
	s.yOld = make([]float64, n)
	s.scale = make([]float64, n)
	s.errv = make([]float64, n)
	s.ze = make([]float64, n)
	return s
}

// factorize forms and factors MU_REAL/h·I - J and MU_COMPLEX/h·I - J.
func (s *radauSolver) factorize(h float64) {
	n := s.n
	mr := radauMu / h
	mc := radauMuC / complex(h, 0)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := -s.J[i*n+j]
			s.mReal[i*n+j] = v
			s.mCplx[i*n+j] = complex(v, 0)
		}
		s.mReal[i*n+i] += mr
		s.mCplx[i*n+i] += mc
	}
	s.lu.factor(s.mReal)
	s.nlu++
	s.luc.factor(s.mCplx)
	s.nlu++
	s.luValid = true
}

// collocation is SciPy's solve_collocation_system. It leaves the solution in
// s.Z and returns whether Newton converged, the iteration count and the last
// convergence rate (NaN when undefined).
func (s *radauSolver) collocation(t, h float64, scale []float64) (bool, int, float64) {
	n := s.n
	mReal := radauMu / h
	mCplx := radauMuC / complex(h, 0)
	// W = TI·Z0, Z = Z0.
	for i := 0; i < n; i++ {
		for k := 0; k < 3; k++ {
			s.W[k][i] = radauTI[k][0]*s.Z0[0][i] + radauTI[k][1]*s.Z0[1][i] + radauTI[k][2]*s.Z0[2][i]
			s.Z[k][i] = s.Z0[k][i]
		}
	}
	dWNormOld := -1.0
	rate := math.NaN()
	converged := false
	iters := 0
	for k := 0; k < radauNewtonMaxIter; k++ {
		iters = k + 1
		finite := true
		for st := 0; st < 3; st++ {
			for i := 0; i < n; i++ {
				s.ytmp[i] = s.y[i] + s.Z[st][i]
			}
			s.eval(t+h*radauC[st], s.ytmp, s.F[st])
			finite = finite && allFinite(s.F[st])
		}
		if !finite {
			break
		}
		for i := 0; i < n; i++ {
			s.fr[i] = s.F[0][i]*radauTI[0][0] + s.F[1][i]*radauTI[0][1] + s.F[2][i]*radauTI[0][2] - mReal*s.W[0][i]
			re := s.F[0][i]*radauTI[1][0] + s.F[1][i]*radauTI[1][1] + s.F[2][i]*radauTI[1][2]
			im := s.F[0][i]*radauTI[2][0] + s.F[1][i]*radauTI[2][1] + s.F[2][i]*radauTI[2][2]
			s.fc[i] = complex(re, im) - mCplx*complex(s.W[1][i], s.W[2][i])
		}
		s.lu.solve(s.fr)
		s.luc.solve(s.fc)
		ss := 0.0
		for i := 0; i < n; i++ {
			s.dW[0][i] = s.fr[i]
			s.dW[1][i] = real(s.fc[i])
			s.dW[2][i] = imag(s.fc[i])
			for st := 0; st < 3; st++ {
				v := s.dW[st][i] / scale[i]
				ss += v * v
			}
		}
		dWNorm := math.Sqrt(ss) / math.Sqrt(float64(3*n))
		if dWNormOld >= 0 {
			rate = dWNorm / dWNormOld
		}
		if !math.IsNaN(rate) && (rate >= 1 || math.Pow(rate, float64(radauNewtonMaxIter-k))/(1-rate)*dWNorm > s.newtonTol) {
			break
		}
		for i := 0; i < n; i++ {
			for st := 0; st < 3; st++ {
				s.W[st][i] += s.dW[st][i]
			}
			for st := 0; st < 3; st++ {
				s.Z[st][i] = radauT[st][0]*s.W[0][i] + radauT[st][1]*s.W[1][i] + radauT[st][2]*s.W[2][i]
			}
		}
		if dWNorm == 0 || (!math.IsNaN(rate) && rate/(1-rate)*dWNorm < s.newtonTol) {
			converged = true
			break
		}
		dWNormOld = dWNorm
	}
	return converged, iters, rate
}

// predictFactor is SciPy's predict_factor (Gustafsson's predictive
// controller); hAbsOld == 0 or errNormOld < 0 stand for "none".
func predictFactor(hAbs, hAbsOld, errNorm, errNormOld float64) float64 {
	multiplier := 1.0
	if errNormOld >= 0 && hAbsOld != 0 && errNorm != 0 {
		multiplier = hAbs / hAbsOld * math.Pow(errNormOld/errNorm, 0.25)
	}
	return pyMin(1, multiplier) * math.Pow(errNorm, -0.25)
}

func (s *radauSolver) stepImpl() (bool, string) {
	t := s.t
	n := s.n
	minStep := s.minStep(t)
	var hAbs, hAbsOld, errNormOld float64
	switch {
	case s.hAbs > s.maxStep:
		hAbs, hAbsOld, errNormOld = s.maxStep, 0, -1
	case s.hAbs < minStep:
		hAbs, hAbsOld, errNormOld = minStep, 0, -1
	default:
		hAbs, hAbsOld, errNormOld = s.hAbs, s.hAbsOld, s.errNormOld
	}
	currentJac := s.currentJac
	rejected := false
	scale, errv, ze := s.scale, s.errv, s.ze
	var tNew, h, errNorm, safety, rate float64
	var nIter int
	for {
		// "!(hAbs >= minStep)" also stops on a NaN step, where SciPy would loop
		// forever (a zero error scale makes every estimate NaN).
		if !(hAbs >= minStep) {
			return false, tooSmallStep
		}
		h = hAbs * s.direction
		tNew = t + h
		if s.direction*(tNew-s.tBound) > 0 {
			tNew = s.tBound
		}
		h = tNew - t
		hAbs = math.Abs(h)
		if s.sol == nil {
			for st := 0; st < 3; st++ {
				clear(s.Z0[st])
			}
		} else {
			for st := 0; st < 3; st++ {
				s.sol.eval(t+h*radauC[st], s.Z0[st])
				for i := 0; i < n; i++ {
					s.Z0[st][i] -= s.y[i]
				}
			}
		}
		for i := range scale {
			scale[i] = s.atol[i] + math.Abs(s.y[i])*s.rtol[i]
		}
		converged := false
		for !converged {
			if !s.luValid {
				s.factorize(h)
			}
			converged, nIter, rate = s.collocation(t, h, scale)
			if !converged {
				if currentJac {
					break
				}
				s.jacobian(t, s.y, s.f, s.J)
				currentJac = true
				s.luValid = false
			}
		}
		if !converged {
			hAbs *= 0.5
			s.luValid = false
			continue
		}
		for i := 0; i < n; i++ {
			ze[i] = (s.Z[0][i]*radauE[0] + s.Z[1][i]*radauE[1] + s.Z[2][i]*radauE[2]) / h
			errv[i] = s.f[i] + ze[i]
		}
		s.lu.solve(errv)
		for i := range scale {
			yNew := s.y[i] + s.Z[2][i]
			scale[i] = s.atol[i] + math.Max(math.Abs(s.y[i]), math.Abs(yNew))*s.rtol[i]
		}
		errNorm = scaledNorm(errv, scale)
		safety = 0.9 * (2*radauNewtonMaxIter + 1) / float64(2*radauNewtonMaxIter+nIter)
		if rejected && errNorm > 1 {
			for i := 0; i < n; i++ {
				s.ytmp[i] = s.y[i] + errv[i]
			}
			s.eval(t, s.ytmp, errv)
			for i := 0; i < n; i++ {
				errv[i] += ze[i]
			}
			s.lu.solve(errv)
			errNorm = scaledNorm(errv, scale)
		}
		if errNorm > 1 {
			factor := predictFactor(hAbs, hAbsOld, errNorm, errNormOld)
			hAbs *= pyMax(radauMinFactor, safety*factor)
			s.luValid = false
			rejected = true
			continue
		}
		break
	}
	recomputeJac := nIter > 2 && rate > 1e-3
	factor := predictFactor(hAbs, hAbsOld, errNorm, errNormOld)
	factor = pyMin(radauMaxFactor, safety*factor)
	if !recomputeJac && factor < 1.2 {
		factor = 1
	} else {
		s.luValid = false
	}
	copy(s.yOld, s.y)
	for i := 0; i < n; i++ {
		s.y[i] += s.Z[2][i]
	}
	s.eval(tNew, s.y, s.f)
	if recomputeJac {
		s.jacobian(tNew, s.y, s.f, s.J)
		currentJac = true
	} else {
		currentJac = false
	}
	s.hAbsOld = s.hAbs
	s.errNormOld = errNorm
	s.hAbs = hAbs * factor
	s.t = tNew
	s.currentJac = currentJac
	s.tOld = t
	s.sol = s.newDense()
	return true, ""
}

func (s *radauSolver) newDense() *radauDense {
	n := s.n
	Q := make([]float64, n*3)
	for i := 0; i < n; i++ {
		for j := 0; j < 3; j++ {
			Q[i*3+j] = s.Z[0][i]*radauP[0][j] + s.Z[1][i]*radauP[1][j] + s.Z[2][i]*radauP[2][j]
		}
	}
	return &radauDense{tOld: s.tOld, h: s.t - s.tOld, yOld: append([]float64(nil), s.yOld...), Q: Q}
}

func (s *radauSolver) denseImpl() denseOutput { return s.sol }

// radauDense is y_old + Σ_j Q_j x^(j+1), x = (t - t_old)/h.
type radauDense struct {
	tOld, h float64
	yOld    []float64
	Q       []float64
}

func (d *radauDense) dim() int { return len(d.yOld) }

func (d *radauDense) eval(t float64, y []float64) {
	x := (t - d.tOld) / d.h
	for i := range y {
		x2 := x * x
		y[i] = d.Q[i*3]*x + d.Q[i*3+1]*x2 + d.Q[i*3+2]*(x2*x) + d.yOld[i]
	}
}
