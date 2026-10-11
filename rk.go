// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Explicit embedded Runge–Kutta pairs with step size control (SciPy's
// RungeKutta, RK23, RK45 and DOP853 classes).

const (
	rkSafety    = 0.9
	rkMinFactor = 0.2
	rkMaxFactor = 10
)

// rkTableau is a Butcher tableau with error and interpolation coefficients.
type rkTableau struct {
	order, errOrder, nStages int
	C                        []float64
	A                        [][]float64
	B                        []float64
	E                        []float64   // error weights (nStages+1), RK23/RK45
	P                        [][]float64 // dense output (nStages+1 × k)
	// stiffBound is the stability boundary used by Hairer's stiffness test
	// (3.25 for DOPRI5, 6.1 for DOP853), 0 when the last internal stage is
	// not at c = 1 and the test does not apply.
	stiffBound float64
}

// Bogacki–Shampine 3(2).
var tabRK23 = &rkTableau{
	order: 3, errOrder: 2, nStages: 3,
	C: []float64{0, 1.0 / 2, 3.0 / 4},
	A: [][]float64{{0, 0, 0}, {1.0 / 2, 0, 0}, {0, 3.0 / 4, 0}},
	B: []float64{2.0 / 9, 1.0 / 3, 4.0 / 9},
	E: []float64{5.0 / 72, -1.0 / 12, -1.0 / 9, 1.0 / 8},
	P: [][]float64{
		{1, -4.0 / 3, 5.0 / 9},
		{0, 1, -2.0 / 3},
		{0, 4.0 / 3, -8.0 / 9},
		{0, -1, 1},
	},
}

// Dormand–Prince 5(4) with Shampine's 4th-order interpolant.
var tabRK45 = &rkTableau{
	order: 5, errOrder: 4, nStages: 6,
	C: []float64{0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1},
	A: [][]float64{
		{0, 0, 0, 0, 0},
		{1.0 / 5, 0, 0, 0, 0},
		{3.0 / 40, 9.0 / 40, 0, 0, 0},
		{44.0 / 45, -56.0 / 15, 32.0 / 9, 0, 0},
		{19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0},
		{9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
	},
	B: []float64{35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84},
	E: []float64{-71.0 / 57600, 0, 71.0 / 16695, -71.0 / 1920, 17253.0 / 339200, -22.0 / 525, 1.0 / 40},
	P: [][]float64{
		{1, -8048581381.0 / 2820520608, 8663915743.0 / 2820520608, -12715105075.0 / 11282082432},
		{0, 0, 0, 0},
		{0, 131558114200.0 / 32700410799, -68118460800.0 / 10900136933, 87487479700.0 / 32700410799},
		{0, -1754552775.0 / 470086768, 14199869525.0 / 1410260304, -10690763975.0 / 1880347072},
		{0, 127303824393.0 / 49829197408, -318862633887.0 / 49829197408, 701980252875.0 / 199316789632},
		{0, -282668133.0 / 205662961, 2019193451.0 / 616988883, -1453857185.0 / 822651844},
		{0, 40617522.0 / 29380423, -110615467.0 / 29380423, 69997945.0 / 29380423},
	},
	stiffBound: 3.25,
}

var tabDOP853 = &rkTableau{
	order: 8, errOrder: 7, nStages: dop853NStages,
	C: dop853C[:dop853NStages], A: dop853A[:dop853NStages], B: dop853B,
	stiffBound: 6.1,
}

// rkSolver integrates with an explicit embedded pair.
type rkSolver struct {
	*odeCore
	tab       *rkTableau
	dop       bool
	f         []float64 // f(t, y)
	hAbs      float64
	hPrevious float64
	K         [][]float64 // nStages+1 rows (DOP853: 16, the extended stages)
	errExp    float64
	yOld      []float64
	yNew      []float64
	fNew      []float64
	yTmp      []float64
	yStiff    []float64 // argument of the last internal stage (c = 1)
	scale     []float64
	errv      []float64
	// hLambda is Hairer's stiffness indicator |h|·‖f(y₁)-f(ỹ)‖/‖y₁-ỹ‖ of
	// the last accepted step, for the LSODA-style switch.
	hLambda float64
}

// newRK builds an explicit solver. f0, if non-nil, is f(t0, y0) already
// evaluated and h0 the first step (the LSODA-style switch hands both over).
func newRK(core *odeCore, m Method, f0 []float64, h0 float64) *rkSolver {
	tab := tabRK45
	switch m {
	case RK23:
		tab = tabRK23
	case DOP853:
		tab = tabDOP853
	}
	n := core.n
	s := &rkSolver{odeCore: core, tab: tab, dop: m == DOP853}
	s.f = make([]float64, n)
	if f0 != nil {
		copy(s.f, f0)
	} else {
		core.eval(core.t, core.y, s.f)
	}
	s.hAbs = h0
	rows := tab.nStages + 1
	if s.dop {
		rows = dop853NStagesExtended
	}
	s.K = make([][]float64, rows)
	for i := range s.K {
		s.K[i] = make([]float64, n)
	}
	s.yOld = make([]float64, n)
	s.yNew = make([]float64, n)
	s.fNew = make([]float64, n)
	s.yTmp = make([]float64, n)
	s.yStiff = make([]float64, n)
	s.scale = make([]float64, n)
	s.errv = make([]float64, n)
	s.errExp = -1 / float64(tab.errOrder+1)
	if f0 == nil {
		s.hAbs = core.initialStep(s.f, float64(tab.errOrder))
	}
	return s
}

// rkStep is SciPy's rk_step: the stages, y_new and f_new.
func (s *rkSolver) rkStep(t, h float64) {
	tab := s.tab
	K := s.K
	n := s.n
	copy(K[0], s.f)
	last := tab.nStages - 1
	for st := 1; st < tab.nStages; st++ {
		a := tab.A[st]
		for i := 0; i < n; i++ {
			dy := 0.0
			for j := 0; j < st; j++ {
				dy += K[j][i] * a[j]
			}
			s.yTmp[i] = s.y[i] + dy*h
		}
		if st == last {
			copy(s.yStiff, s.yTmp)
		}
		s.eval(t+tab.C[st]*h, s.yTmp, K[st])
	}
	for i := 0; i < n; i++ {
		dy := 0.0
		for j := 0; j < tab.nStages; j++ {
			dy += K[j][i] * tab.B[j]
		}
		s.yNew[i] = s.y[i] + h*dy
	}
	s.eval(t+h, s.yNew, s.fNew)
	copy(K[tab.nStages], s.fNew)
}

// errorNorm is the scaled norm of the local error estimate.
func (s *rkSolver) errorNorm(h float64) float64 {
	K := s.K
	n := s.n
	if !s.dop {
		E := s.tab.E
		for i := 0; i < n; i++ {
			e := 0.0
			for j := range E {
				e += K[j][i] * E[j]
			}
			s.errv[i] = e * h
		}
		return scaledNorm(s.errv, s.scale)
	}
	// DOP853 blends its 5th- and 3rd-order estimates (Hairer's err).
	var err5, err3 float64
	for i := 0; i < n; i++ {
		e5, e3 := 0.0, 0.0
		for j := 0; j <= dop853NStages; j++ {
			e5 += K[j][i] * dop853E5[j]
			e3 += K[j][i] * dop853E3[j]
		}
		e5 /= s.scale[i]
		e3 /= s.scale[i]
		err5 += e5 * e5
		err3 += e3 * e3
	}
	if err5 == 0 && err3 == 0 {
		return 0
	}
	denom := err5 + 0.01*err3
	return math.Abs(h) * err5 / math.Sqrt(denom*float64(n))
}

func (s *rkSolver) stepImpl() (bool, string) {
	t := s.t
	minStep := s.minStep(t)
	hAbs := s.hAbs
	if hAbs > s.maxStep {
		hAbs = s.maxStep
	} else if hAbs < minStep {
		hAbs = minStep
	}
	rejected := false
	for {
		// "!(hAbs >= minStep)" also stops on a NaN step, where SciPy would loop
		// forever (a zero error scale makes every estimate NaN).
		if !(hAbs >= minStep) {
			return false, tooSmallStep
		}
		h := hAbs * s.direction
		tNew := t + h
		if s.direction*(tNew-s.tBound) > 0 {
			tNew = s.tBound
		}
		h = tNew - t
		hAbs = math.Abs(h)
		s.rkStep(t, h)
		for i := range s.scale {
			s.scale[i] = s.atol[i] + math.Max(math.Abs(s.y[i]), math.Abs(s.yNew[i]))*s.rtol[i]
		}
		errNorm := s.errorNorm(h)
		if errNorm < 1 {
			factor := float64(rkMaxFactor)
			if errNorm != 0 {
				factor = pyMin(rkMaxFactor, rkSafety*math.Pow(errNorm, s.errExp))
			}
			if rejected {
				factor = pyMin(1, factor)
			}
			hAbs *= factor
			s.hPrevious = h
			s.stiffness(h)
			s.yOld, s.y, s.yNew = s.y, s.yNew, s.yOld
			s.f, s.fNew = s.fNew, s.f
			s.t = tNew
			s.hAbs = hAbs
			return true, ""
		}
		hAbs *= pyMax(rkMinFactor, rkSafety*math.Pow(errNorm, s.errExp))
		rejected = true
	}
}

// stiffness computes Hairer's indicator hλ ≈ |h|·‖f(y₁) - f(ỹ)‖/‖y₁ - ỹ‖,
// where ỹ is the argument of the last internal stage (c = 1): an estimate of
// |h| times the dominant eigenvalue, compared with the stability boundary.
func (s *rkSolver) stiffness(h float64) {
	if s.tab.stiffBound == 0 {
		return
	}
	num, den := 0.0, 0.0
	last := s.K[s.tab.nStages-1]
	for i := 0; i < s.n; i++ {
		d := s.fNew[i] - last[i]
		num += d * d
		e := s.yNew[i] - s.yStiff[i]
		den += e * e
	}
	s.hLambda = 0
	if den > 0 {
		s.hLambda = math.Abs(h) * math.Sqrt(num/den)
	}
}

func (s *rkSolver) denseImpl() denseOutput {
	n := s.n
	h := s.hPrevious
	yOld := append([]float64(nil), s.yOld...)
	if !s.dop {
		P := s.tab.P
		k := len(P[0])
		Q := make([]float64, n*k)
		for i := 0; i < n; i++ {
			for j := 0; j < k; j++ {
				q := 0.0
				for st := range P {
					q += s.K[st][i] * P[st][j]
				}
				Q[i*k+j] = q
			}
		}
		return &rkDense{tOld: s.tOld, h: h, yOld: yOld, Q: Q, k: k}
	}
	K := s.K
	for st := dop853NStages + 1; st < dop853NStagesExtended; st++ {
		a := dop853A[st]
		for i := 0; i < n; i++ {
			dy := 0.0
			for j := 0; j < st; j++ {
				dy += K[j][i] * a[j]
			}
			s.yTmp[i] = s.yOld[i] + dy*h
		}
		s.eval(s.tOld+dop853C[st]*h, s.yTmp, K[st])
	}
	F := make([][]float64, dop853InterpPower)
	for r := range F {
		F[r] = make([]float64, n)
	}
	fOld := K[0]
	for i := 0; i < n; i++ {
		dy := s.y[i] - yOld[i]
		F[0][i] = dy
		F[1][i] = h*fOld[i] - dy
		F[2][i] = 2*dy - h*(s.f[i]+fOld[i])
		for r := 3; r < dop853InterpPower; r++ {
			d := dop853D[r-3]
			v := 0.0
			for st := 0; st < dop853NStagesExtended; st++ {
				v += d[st] * K[st][i]
			}
			F[r][i] = h * v
		}
	}
	return &dop853Dense{tOld: s.tOld, h: h, yOld: yOld, F: F}
}

// rkDense is y_old + h·Σ_j Q_j x^(j+1), x = (t - t_old)/h.
type rkDense struct {
	tOld, h float64
	yOld    []float64
	Q       []float64
	k       int
}

func (d *rkDense) dim() int { return len(d.yOld) }

func (d *rkDense) eval(t float64, y []float64) {
	x := (t - d.tOld) / d.h
	for i := range y {
		s := 0.0
		p := 1.0
		for j := 0; j < d.k; j++ {
			p *= x
			s += d.Q[i*d.k+j] * p
		}
		y[i] = d.h*s + d.yOld[i]
	}
}

// dop853Dense is Hairer's contd8 interpolant.
type dop853Dense struct {
	tOld, h float64
	yOld    []float64
	F       [][]float64
}

func (d *dop853Dense) dim() int { return len(d.yOld) }

func (d *dop853Dense) eval(t float64, y []float64) {
	x := (t - d.tOld) / d.h
	for i := range y {
		v := 0.0
		for r := len(d.F) - 1; r >= 0; r-- {
			v += d.F[r][i]
			if (len(d.F)-1-r)%2 == 0 {
				v *= x
			} else {
				v *= 1 - x
			}
		}
		y[i] = v + d.yOld[i]
	}
}
