// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Variable-order (1–5) backward differentiation in the numerical
// differentiation formula variant (Shampine & Reichelt, "The MATLAB ODE
// Suite", 1997), with the quasi-constant step size implementation of SciPy's
// BDF class: the history is a table of backward differences rescaled
// whenever the step changes.

const (
	bdfMaxOrder      = 5
	bdfNewtonMaxIter = 4
	bdfMinFactor     = 0.2
	bdfMaxFactor     = 10
)

var bdfKappa = [6]float64{0, -0.185, -1.0 / 9, -0.0823, -0.0415, 0}

var bdfGamma, bdfAlpha, bdfErrorConst [6]float64

func init() {
	for k := 1; k <= bdfMaxOrder; k++ {
		bdfGamma[k] = bdfGamma[k-1] + 1/float64(k)
	}
	for k := 0; k <= bdfMaxOrder; k++ {
		bdfAlpha[k] = (1 - bdfKappa[k]) * bdfGamma[k]
		bdfErrorConst[k] = bdfKappa[k]*bdfGamma[k] + 1/float64(k+1)
	}
}

// computeR is SciPy's compute_R: the matrix that rescales the difference
// table by factor (order+1 ≤ 6 rows used).
func computeR(order int, factor float64, R *[bdfMaxOrder + 1][bdfMaxOrder + 1]float64) {
	m := order + 1
	for j := 0; j < m; j++ {
		R[0][j] = 1
	}
	for i := 1; i < m; i++ {
		R[i][0] = 0
		for j := 1; j < m; j++ {
			R[i][j] = R[i-1][j] * ((float64(i) - 1 - factor*float64(j)) / float64(i))
		}
	}
}

// changeD rescales the first order+1 rows of D for a step multiplied by
// factor (SciPy's change_D); tmp is scratch with at least order+1 rows.
func changeD(D [][]float64, order int, factor float64, tmp [][]float64) {
	var R, U, RU [bdfMaxOrder + 1][bdfMaxOrder + 1]float64
	computeR(order, factor, &R)
	computeR(order, 1, &U)
	m := order + 1
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			s := 0.0
			for k := 0; k < m; k++ {
				s += R[i][k] * U[k][j]
			}
			RU[i][j] = s
		}
	}
	n := len(D[0])
	for k := 0; k < m; k++ {
		for i := 0; i < n; i++ {
			s := 0.0
			for r := 0; r < m; r++ {
				s += RU[r][k] * D[r][i]
			}
			tmp[k][i] = s
		}
	}
	for k := 0; k < m; k++ {
		copy(D[k], tmp[k])
	}
}

type bdfSolver struct {
	*odeCore
	hAbs        float64
	newtonTol   float64
	J           []float64
	lu          *luReal
	luValid     bool
	mat         []float64
	D           [][]float64
	tmpD        [][]float64
	order       int
	nEqualSteps int
	yPredict    []float64
	yNew        []float64
	d           []float64
	dy          []float64
	psi         []float64
	scale       []float64
	ftmp        []float64
	errv        []float64
}

// newBDF builds the BDF solver. f0, if non-nil, is f(t0, y0) and h0 > 0 the
// initial step (the LSODA-style switch hands both over); otherwise f is
// evaluated and the step selected as SciPy does.
func newBDF(core *odeCore, f0 []float64, h0 float64) *bdfSolver {
	n := core.n
	s := &bdfSolver{odeCore: core, order: 1}
	f := make([]float64, n)
	if f0 != nil {
		copy(f, f0)
		s.hAbs = h0
	} else {
		core.eval(core.t, core.y, f)
		s.hAbs = core.initialStep(f, 1)
	}
	s.newtonTol = newtonTolerance(core.rtol)
	s.J = make([]float64, n*n)
	core.jacobian(core.t, core.y, f, s.J)
	s.lu = newLUReal(n)
	s.mat = make([]float64, n*n)
	s.D = make([][]float64, bdfMaxOrder+3)
	s.tmpD = make([][]float64, bdfMaxOrder+1)
	for i := range s.D {
		s.D[i] = make([]float64, n)
	}
	for i := range s.tmpD {
		s.tmpD[i] = make([]float64, n)
	}
	copy(s.D[0], core.y)
	for i := 0; i < n; i++ {
		s.D[1][i] = f[i] * s.hAbs * core.direction
	}
	s.yPredict = make([]float64, n)
	s.yNew = make([]float64, n)
	s.d = make([]float64, n)
	s.dy = make([]float64, n)
	s.psi = make([]float64, n)
	s.scale = make([]float64, n)
	s.ftmp = make([]float64, n)
	s.errv = make([]float64, n)
	return s
}

func (s *bdfSolver) factorize(c float64) {
	n := s.n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s.mat[i*n+j] = -c * s.J[i*n+j]
		}
		s.mat[i*n+i] += 1
	}
	s.lu.factor(s.mat)
	s.nlu++
	s.luValid = true
}

// newton is SciPy's solve_bdf_system; the iterate is left in s.yNew and the
// accumulated correction in s.d.
func (s *bdfSolver) newton(tNew, c float64) (bool, int) {
	n := s.n
	clear(s.d)
	copy(s.yNew, s.yPredict)
	dyNormOld := -1.0
	iters := 0
	for k := 0; k < bdfNewtonMaxIter; k++ {
		iters = k + 1
		s.eval(tNew, s.yNew, s.ftmp)
		if !allFinite(s.ftmp) {
			break
		}
		for i := 0; i < n; i++ {
			s.dy[i] = c*s.ftmp[i] - s.psi[i] - s.d[i]
		}
		s.lu.solve(s.dy)
		dyNorm := scaledNorm(s.dy, s.scale)
		rate := math.NaN()
		if dyNormOld >= 0 {
			rate = dyNorm / dyNormOld
		}
		if !math.IsNaN(rate) && (rate >= 1 || math.Pow(rate, float64(bdfNewtonMaxIter-k))/(1-rate)*dyNorm > s.newtonTol) {
			break
		}
		for i := 0; i < n; i++ {
			s.yNew[i] += s.dy[i]
			s.d[i] += s.dy[i]
		}
		if dyNorm == 0 || (!math.IsNaN(rate) && rate/(1-rate)*dyNorm < s.newtonTol) {
			return true, iters
		}
		dyNormOld = dyNorm
	}
	return false, iters
}

func (s *bdfSolver) stepImpl() (bool, string) {
	t := s.t
	n := s.n
	D := s.D
	minStep := s.minStep(t)
	var hAbs float64
	switch {
	case s.hAbs > s.maxStep:
		hAbs = s.maxStep
		changeD(D, s.order, s.maxStep/s.hAbs, s.tmpD)
		s.nEqualSteps = 0
	case s.hAbs < minStep:
		hAbs = minStep
		changeD(D, s.order, minStep/s.hAbs, s.tmpD)
		s.nEqualSteps = 0
	default:
		hAbs = s.hAbs
	}
	order := s.order
	currentJac := false
	var tNew, errNorm, safety float64
	for {
		// "!(hAbs >= minStep)" also stops on a NaN step, where SciPy would loop
		// forever (a zero error scale makes every estimate NaN).
		if !(hAbs >= minStep) {
			return false, tooSmallStep
		}
		h := hAbs * s.direction
		tNew = t + h
		if s.direction*(tNew-s.tBound) > 0 {
			tNew = s.tBound
			changeD(D, order, math.Abs(tNew-t)/hAbs, s.tmpD)
			s.nEqualSteps = 0
			s.luValid = false
		}
		h = tNew - t
		hAbs = math.Abs(h)
		for i := 0; i < n; i++ {
			yp := 0.0
			for k := 0; k <= order; k++ {
				yp += D[k][i]
			}
			s.yPredict[i] = yp
			s.scale[i] = s.atol[i] + s.rtol[i]*math.Abs(yp)
			ps := 0.0
			for k := 1; k <= order; k++ {
				ps += D[k][i] * bdfGamma[k]
			}
			s.psi[i] = ps / bdfAlpha[order]
		}
		c := h / bdfAlpha[order]
		converged := false
		var nIter int
		for !converged {
			if !s.luValid {
				s.factorize(c)
			}
			converged, nIter = s.newton(tNew, c)
			if !converged {
				if currentJac {
					break
				}
				// The Jacobian is taken at the prediction; the
				// derivative there is not counted in nfev, as in
				// SciPy.
				if s.jac == nil {
					s.nfevJac++
					s.fun(tNew, s.yPredict, s.ftmp)
				}
				s.jacobian(tNew, s.yPredict, s.ftmp, s.J)
				s.luValid = false
				currentJac = true
			}
		}
		if !converged {
			hAbs *= 0.5
			changeD(D, order, 0.5, s.tmpD)
			s.nEqualSteps = 0
			s.luValid = false
			continue
		}
		safety = 0.9 * (2*bdfNewtonMaxIter + 1) / float64(2*bdfNewtonMaxIter+nIter)
		for i := 0; i < n; i++ {
			s.scale[i] = s.atol[i] + s.rtol[i]*math.Abs(s.yNew[i])
			s.errv[i] = bdfErrorConst[order] * s.d[i]
		}
		errNorm = scaledNorm(s.errv, s.scale)
		if errNorm > 1 {
			factor := pyMax(bdfMinFactor, safety*math.Pow(errNorm, -1/float64(order+1)))
			hAbs *= factor
			changeD(D, order, factor, s.tmpD)
			s.nEqualSteps = 0
			continue
		}
		break
	}
	s.nEqualSteps++
	s.t = tNew
	copy(s.y, s.yNew)
	s.hAbs = hAbs
	for i := 0; i < n; i++ {
		D[order+2][i] = s.d[i] - D[order+1][i]
		D[order+1][i] = s.d[i]
	}
	for k := order; k >= 0; k-- {
		for i := 0; i < n; i++ {
			D[k][i] += D[k+1][i]
		}
	}
	if s.nEqualSteps < order+1 {
		return true, ""
	}
	errM := math.Inf(1)
	if order > 1 {
		for i := 0; i < n; i++ {
			s.errv[i] = bdfErrorConst[order-1] * D[order][i]
		}
		errM = scaledNorm(s.errv, s.scale)
	}
	errP := math.Inf(1)
	if order < bdfMaxOrder {
		for i := 0; i < n; i++ {
			s.errv[i] = bdfErrorConst[order+1] * D[order+2][i]
		}
		errP = scaledNorm(s.errv, s.scale)
	}
	norms := [3]float64{errM, errNorm, errP}
	best, bestF := 0, math.Inf(-1)
	for k, e := range norms {
		f := math.Pow(e, -1/float64(order+k))
		if f > bestF {
			best, bestF = k, f
		}
	}
	order += best - 1
	s.order = order
	factor := pyMin(bdfMaxFactor, safety*bestF)
	s.hAbs *= factor
	changeD(D, order, factor, s.tmpD)
	s.nEqualSteps = 0
	s.luValid = false
	return true, ""
}

func (s *bdfSolver) denseImpl() denseOutput {
	h := s.hAbs * s.direction
	D := make([][]float64, s.order+1)
	for k := range D {
		D[k] = append([]float64(nil), s.D[k]...)
	}
	return &bdfDense{t: s.t, h: h, order: s.order, D: D}
}

// bdfDense evaluates the interpolating polynomial of the difference table.
type bdfDense struct {
	t, h  float64
	order int
	D     [][]float64
}

func (d *bdfDense) dim() int { return len(d.D[0]) }

func (d *bdfDense) eval(t float64, y []float64) {
	// p_j = Π_{k≤j} (t - (t_n - k·h)) / ((k+1)·h).
	var p [bdfMaxOrder]float64
	acc := 1.0
	for j := 0; j < d.order; j++ {
		acc *= (t - (d.t - d.h*float64(j))) / (d.h * float64(1+j))
		p[j] = acc
	}
	for i := range y {
		s := 0.0
		for j := 0; j < d.order; j++ {
			s += d.D[j+1][i] * p[j]
		}
		y[i] = s + d.D[0][i]
	}
}
