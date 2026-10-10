// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

const dblEps = 2.220446049250313e-16

// Finite-difference Jacobian constants (SciPy's num_jac).
var (
	numJacDiffReject = math.Pow(dblEps, 0.875)
	numJacDiffSmall  = math.Pow(dblEps, 0.75)
	numJacDiffBig    = math.Pow(dblEps, 0.25)
)

const (
	numJacMinFactor      = 1e3 * dblEps
	numJacFactorIncrease = 10
	numJacFactorDecrease = 0.1
)

// method is one integrator: stepImpl advances core by one accepted step,
// denseImpl returns the interpolant of the last step.
type method interface {
	stepImpl() (bool, string)
	denseImpl() denseOutput
}

// denseOutput is the local interpolant of one step.
type denseOutput interface {
	eval(t float64, y []float64)
	dim() int
}

// constDense is the interpolant of an empty step (y constant).
type constDense struct{ y []float64 }

func (d *constDense) eval(_ float64, y []float64) { copy(y, d.y) }
func (d *constDense) dim() int                    { return len(d.y) }

// odeCore is the state shared by every method (SciPy's OdeSolver base).
type odeCore struct {
	fun       ODEFunc
	n         int
	t, tOld   float64
	tBound    float64
	direction float64
	y         []float64
	rtol      []float64
	atol      []float64
	maxStep   float64
	firstStep float64
	jac       JacFunc
	jacFactor []float64

	nfev, njev, nlu, nfevJac, nsteps int
	finished                         bool
}

const tooSmallStep = "Required step size is less than spacing between numbers."

func newCore(f ODEFunc, t0 float64, y0 []float64, tf float64, o *IVPOptions) (*odeCore, error) {
	n := len(y0)
	c := &odeCore{fun: f, n: n, t: t0, tBound: tf, y: append([]float64(nil), y0...), jac: o.Jac}
	c.direction = 1
	if tf < t0 {
		c.direction = -1
	}
	c.rtol = make([]float64, n)
	c.atol = make([]float64, n)
	if o.RTolVec != nil && len(o.RTolVec) != n {
		return nil, odeInput("RTolVec has %d entries for %d components", len(o.RTolVec), n)
	}
	if o.ATolVec != nil && len(o.ATolVec) != n {
		return nil, odeInput("ATolVec has %d entries for %d components", len(o.ATolVec), n)
	}
	rtol := o.RTol
	if rtol == 0 {
		rtol = 1e-3
	}
	atol := orDefault(o.ATol, 1e-6)
	for i := 0; i < n; i++ {
		r, a := rtol, atol
		if o.RTolVec != nil {
			r = o.RTolVec[i]
		}
		if o.ATolVec != nil {
			a = o.ATolVec[i]
		}
		if a < 0 {
			return nil, odeInput("atol must be non-negative")
		}
		// SciPy warns and clamps a relative tolerance below 100·eps.
		c.rtol[i] = math.Max(r, 100*dblEps)
		c.atol[i] = a
	}
	c.maxStep = o.MaxStep
	switch {
	case c.maxStep == 0:
		c.maxStep = math.Inf(1)
	case c.maxStep < 0:
		return nil, odeInput("MaxStep must be positive")
	}
	if o.FirstStep < 0 || o.FirstStep > math.Abs(tf-t0) {
		return nil, odeInput("FirstStep must be positive and within the span")
	}
	c.firstStep = o.FirstStep
	return c, nil
}

// eval evaluates the right-hand side, counting it in nfev.
func (c *odeCore) eval(t float64, y, out []float64) {
	c.nfev++
	c.fun(t, y, out)
}

func (c *odeCore) minStep(t float64) float64 {
	return 10 * math.Abs(math.Nextafter(t, c.direction*math.Inf(1))-t)
}

// step is OdeSolver.step.
func (c *odeCore) step(m method) (bool, string) {
	if c.n == 0 || c.t == c.tBound {
		c.tOld = c.t
		c.t = c.tBound
		c.finished = true
		return true, ""
	}
	t := c.t
	if ok, msg := m.stepImpl(); !ok {
		return false, msg
	}
	c.tOld = t
	c.nsteps++
	if c.direction*(c.t-c.tBound) >= 0 {
		c.finished = true
	}
	return true, ""
}

// dense is OdeSolver.dense_output.
func (c *odeCore) dense(m method) denseOutput {
	if c.n == 0 || c.t == c.tOld {
		return &constDense{y: append([]float64(nil), c.y...)}
	}
	return m.denseImpl()
}

// initialStep returns the first step: the user's, or the empirical choice of
// Hairer, Nørsett & Wanner (Solving ODEs I, II.4) as SciPy's
// select_initial_step makes it, for an error estimator of the given order.
func (c *odeCore) initialStep(f0 []float64, order float64) float64 {
	if c.firstStep > 0 {
		return c.firstStep
	}
	n := c.n
	if n == 0 {
		return math.Inf(1)
	}
	interval := math.Abs(c.tBound - c.t)
	if interval == 0 {
		return 0
	}
	scale := make([]float64, n)
	tmp := make([]float64, n)
	for i := range scale {
		scale[i] = c.atol[i] + math.Abs(c.y[i])*c.rtol[i]
	}
	for i := range tmp {
		tmp[i] = c.y[i] / scale[i]
	}
	d0 := rmsNorm(tmp)
	for i := range tmp {
		tmp[i] = f0[i] / scale[i]
	}
	d1 := rmsNorm(tmp)
	h0 := 0.01 * d0 / d1
	if d0 < 1e-5 || d1 < 1e-5 {
		h0 = 1e-6
	}
	h0 = math.Min(h0, interval)
	y1 := make([]float64, n)
	for i := range y1 {
		y1[i] = c.y[i] + h0*c.direction*f0[i]
	}
	f1 := make([]float64, n)
	c.eval(c.t+h0*c.direction, y1, f1)
	for i := range tmp {
		tmp[i] = (f1[i] - f0[i]) / scale[i]
	}
	d2 := rmsNorm(tmp) / h0
	var h1 float64
	if d1 <= 1e-15 && d2 <= 1e-15 {
		h1 = math.Max(1e-6, h0*1e-3)
	} else {
		h1 = math.Pow(0.01/math.Max(d1, d2), 1/(order+1))
	}
	return math.Min(math.Min(100*h0, h1), math.Min(interval, c.maxStep))
}

// rmsNorm is SciPy's norm: ‖x‖₂/√n.
func rmsNorm(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	return math.Sqrt(s) / math.Sqrt(float64(len(x)))
}

// scaledNorm is rmsNorm(x/scale) without a temporary.
func scaledNorm(x, scale []float64) float64 {
	s := 0.0
	for i, v := range x {
		v /= scale[i]
		s += v * v
	}
	return math.Sqrt(s) / math.Sqrt(float64(len(x)))
}

// jacobian evaluates the Jacobian at (t, y) into J, from the user's JacFunc
// or by finite differences; f is f(t, y). It counts njev as SciPy does.
func (c *odeCore) jacobian(t float64, y, f, J []float64) {
	c.njev++
	if c.jac != nil {
		c.jac(t, y, J)
		return
	}
	c.numJac(t, y, f, J)
}

// numJac is SciPy's dense num_jac: forward differences with a step factor per
// column that adapts between calls, and one retry with a larger step for the
// columns whose difference drowned in roundoff.
func (c *odeCore) numJac(t float64, y, f, J []float64) {
	n := c.n
	if c.jacFactor == nil {
		c.jacFactor = make([]float64, n)
		for i := range c.jacFactor {
			c.jacFactor[i] = math.Sqrt(dblEps)
		}
	}
	factor := c.jacFactor
	yScale := make([]float64, n)
	h := make([]float64, n)
	for i := 0; i < n; i++ {
		sign := 1.0
		if f[i] < 0 {
			sign = -1
		}
		yScale[i] = sign * math.Max(c.atol[i], math.Abs(y[i]))
		h[i] = (y[i] + factor[i]*yScale[i]) - y[i]
		for h[i] == 0 {
			factor[i] *= 10
			h[i] = (y[i] + factor[i]*yScale[i]) - y[i]
		}
	}
	ytmp := make([]float64, n)
	col := make([]float64, n)
	diff := make([]float64, n*n) // column-major: diff[j*n+i]
	maxDiff := make([]float64, n)
	scale := make([]float64, n)
	column := func(j int, hj float64) (md, sc float64) {
		copy(ytmp, y)
		ytmp[j] += hj
		c.nfevJac++
		c.fun(t, ytmp, col)
		mi := 0
		for i := 0; i < n; i++ {
			if math.Abs(col[i]-f[i]) > math.Abs(col[mi]-f[mi]) {
				mi = i
			}
		}
		return math.Abs(col[mi] - f[mi]), math.Max(math.Abs(f[mi]), math.Abs(col[mi]))
	}
	for j := 0; j < n; j++ {
		maxDiff[j], scale[j] = column(j, h[j])
		for i := 0; i < n; i++ {
			diff[j*n+i] = col[i] - f[i]
		}
	}
	for j := 0; j < n; j++ {
		if !(maxDiff[j] < numJacDiffReject*scale[j]) {
			continue
		}
		newFactor := numJacFactorIncrease * factor[j]
		hNew := (y[j] + newFactor*yScale[j]) - y[j]
		mdNew, scNew := column(j, hNew)
		if maxDiff[j]*scNew < mdNew*scale[j] {
			factor[j] = newFactor
			h[j] = hNew
			for i := 0; i < n; i++ {
				diff[j*n+i] = col[i] - f[i]
			}
			scale[j] = scNew
			maxDiff[j] = mdNew
		}
	}
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			J[i*n+j] = diff[j*n+i] / h[j]
		}
		if maxDiff[j] < numJacDiffSmall*scale[j] {
			factor[j] *= numJacFactorIncrease
		} else if maxDiff[j] > numJacDiffBig*scale[j] {
			factor[j] *= numJacFactorDecrease
		}
		factor[j] = math.Max(factor[j], numJacMinFactor)
	}
}

// newMethod builds the integrator selected by m on core.
func newMethod(core *odeCore, m Method, o *IVPOptions) (method, error) {
	switch m {
	case RK45, RK23, DOP853:
		return newRK(core, m, nil, 0), nil
	case Radau:
		return newRadau(core), nil
	case BDF:
		return newBDF(core, nil, 0), nil
	case LSODA:
		return newSwitching(core), nil
	}
	return nil, odeInput("unknown method %v", m)
}

// pyMax and pyMin are Python's built-in max and min on two floats, which the
// step-size controllers use: a NaN second argument loses (max(0.2, nan) is
// 0.2), where math.Max would propagate it and stall the step loop.
func pyMax(a, b float64) float64 {
	if b > a {
		return b
	}
	return a
}

func pyMin(a, b float64) float64 {
	if b < a {
		return b
	}
	return a
}
