// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// OdeintOptions tunes Odeint. Zero values select SciPy's odeint defaults.
type OdeintOptions struct {
	// RTol and ATol default to 1.49012e-8. A negative ATol means 0.
	RTol, ATol float64
	// RTolVec and ATolVec give per-component tolerances.
	RTolVec, ATolVec []float64
	// Jac is the Jacobian (SciPy's Dfun, with col_deriv=False).
	Jac JacFunc
	// H0 is the first step (0: automatic) and HMax the largest step
	// (0: unbounded).
	H0, HMax float64
	// MxStep bounds the number of steps between two output times
	// (default 500).
	MxStep int
}

// OdeintInfo is SciPy's odeint infodict (full_output=True): one entry per
// output time after the first.
type OdeintInfo struct {
	// HU is the step size last used, TCur the time reached.
	HU, TCur []float64
	// NST, NFE and NJE are the cumulative numbers of steps, right-hand side
	// evaluations and Jacobian evaluations.
	NST, NFE, NJE []int
	// MUsed is the method of the last step: 1 non-stiff (Runge–Kutta),
	// 2 stiff (BDF).
	MUsed []int
	// Message describes the outcome.
	Message string
}

// Odeint integrates dy/dt = f(t, y) from t[0] and returns y at every time of
// t (which must be monotonic) as a len(t)×n ndarray whose first row is y0.
// It is SciPy's odeint(func, y0, t, Dfun=...) over the LSODA-style switching
// method; SciPy's func(y, t) takes the state first, here f(t, y, dydt) takes
// time first, as with odeint(..., tfirst=True). On failure the rows not
// reached are NaN and the error wraps ErrExcessWork or ErrStepTooSmall.
func Odeint(f ODEFunc, y0 []float64, t []float64, opts *OdeintOptions) (*ndarray.Array, *OdeintInfo, error) {
	var o OdeintOptions
	if opts != nil {
		o = *opts
	}
	n := len(y0)
	if len(t) == 0 {
		return nil, nil, odeInput("Odeint needs at least one output time")
	}
	inc, dec := true, true
	for k := 1; k < len(t); k++ {
		inc = inc && t[k] > t[k-1]
		dec = dec && t[k] < t[k-1]
	}
	if len(t) > 1 && !inc && !dec {
		return nil, nil, odeInput("the output times t must be strictly monotonic")
	}
	rtol := o.RTol
	if rtol == 0 {
		rtol = 1.49012e-8
	}
	atol := o.ATol
	if atol == 0 {
		atol = 1.49012e-8
	}
	mxstep := o.MxStep
	if mxstep == 0 {
		mxstep = 500
	}
	data := make([]float64, len(t)*n)
	copy(data, y0)
	for i := n; i < len(data); i++ {
		data[i] = math.NaN()
	}
	info := &OdeintInfo{Message: "Integration successful."}
	out := func() *ndarray.Array {
		a, _ := ndarray.FromData(data, len(t), n)
		return a
	}
	if len(t) == 1 {
		return out(), info, nil
	}
	ivp := &IVPOptions{Method: LSODA, RTol: rtol, ATol: atol, RTolVec: o.RTolVec, ATolVec: o.ATolVec,
		MaxStep: o.HMax, FirstStep: o.H0, Jac: o.Jac}
	core, err := newCore(f, t[0], y0, t[len(t)-1], ivp)
	if err != nil {
		return nil, nil, err
	}
	sw := newSwitching(core)
	var dense denseOutput
	denseValid := false
	stepsSince := 0
	lastH := 0.0
	for k := 1; k < len(t); k++ {
		tk := t[k]
		for core.direction*(tk-core.t) > 0 {
			if stepsSince >= mxstep {
				info.Message = fmt.Sprintf("Excess work done on this call (MxStep = %d).", mxstep)
				return out(), info, fmt.Errorf("%w (MxStep = %d at t = %g)", ErrExcessWork, mxstep, core.t)
			}
			tPrev := core.t
			if ok, msg := core.step(sw); !ok {
				info.Message = msg
				return out(), info, fmt.Errorf("%w at t = %g", ErrStepTooSmall, core.t)
			}
			lastH = core.t - tPrev
			stepsSince++
			denseValid = false
		}
		if !denseValid {
			dense = core.dense(sw)
			denseValid = true
		}
		dense.eval(tk, data[k*n:(k+1)*n])
		stepsSince = 0
		info.HU = append(info.HU, lastH)
		info.TCur = append(info.TCur, core.t)
		info.NST = append(info.NST, core.nsteps)
		info.NFE = append(info.NFE, core.nfev)
		info.NJE = append(info.NJE, core.njev)
		mused := 1
		if sw.lastStiff {
			mused = 2
		}
		info.MUsed = append(info.MUsed, mused)
	}
	return out(), info, nil
}
