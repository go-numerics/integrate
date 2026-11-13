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
	// HMin is SciPy's hmin, ODEPACK's smallest step (0: none). SciPy 1.18.1
	// ignores it (lsoda.c never stores it); here it is honoured.
	HMin float64
	// MxOrdN and MxOrdS are the largest orders of the Adams (non-stiff,
	// default and maximum 12) and BDF (stiff, default and maximum 5) methods.
	MxOrdN, MxOrdS int
	// TCrit lists the times the integration must not step past (SciPy's
	// tcrit): the solution at a later output time is integrated up to the next
	// critical time first. Sorted in the direction of integration.
	TCrit []float64
	// Band is SciPy's ml and mu (nil: a full Jacobian) and BandJac the banded
	// Jacobian in place of Jac (nil: ODEPACK's finite differences over the
	// band).
	Band    *Bandwidth
	BandJac BandJacFunc
	// Exact selects the path faithful to SciPy's lsoda.c (see IVPOptions.Exact).
	Exact bool
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
	// evaluations (every call, those of finite-difference Jacobians included)
	// and Jacobian evaluations.
	NST, NFE, NJE []int
	// MUsed is the method of the last step: 1 non-stiff (Adams), 2 stiff (BDF).
	MUsed []int
	// Message describes the outcome.
	Message string
}

// Odeint integrates dy/dt = f(t, y) from t[0] and returns y at every time of
// t (which must be monotonic) as a len(t)×n ndarray whose first row is y0.
// It is SciPy's odeint(func, y0, t, Dfun=...), ODEPACK's lsoda (the LSODA
// method of SolveIVP, driven to each output time and interpolated there);
// SciPy's func(y, t) takes the state first, here f(t, y, dydt) takes time
// first, as with odeint(..., tfirst=True). On failure the rows not reached are
// NaN and the error wraps ErrExcessWork (more than MxStep steps between two
// output times), ErrStepTooSmall (repeated error test or corrector failures, or
// a non-finite state) or ErrODEInput (illegal input: output times closer than
// the precision of t, a zero error weight, a tolerance below the machine
// precision).
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
	sv, err := newLsodaConfig(core, lsodaConfig{hmin: o.HMin, mxordn: o.MxOrdN, mxords: o.MxOrdS, band: o.Band, bandJac: o.BandJac, exact: o.Exact})
	if err != nil {
		return nil, nil, err
	}
	sv.mxstep = mxstep
	itask, ci := 1, 0
	if len(o.TCrit) > 0 {
		itask = 4
		sv.tcrit = o.TCrit[0]
	}
	for k := 1; k < len(t); k++ {
		row := data[k*n : (k+1)*n]
		if itask == 4 && t[k] > o.TCrit[ci] {
			// odeint's loop: on to the next critical time, or none left
			if ci++; ci < len(o.TCrit) {
				sv.tcrit = o.TCrit[ci]
			} else {
				itask = 1
			}
		}
		if n > 0 {
			// ODEPACK's lsoda with itask = 1 (4 with critical times): it steps
			// past t[k] and interpolates there, from the Nordsieck array (intdy).
			if _, istate := sv.call(row, t[k], itask); istate < 0 {
				for i := range row {
					row[i] = math.NaN()
				}
				info.Message = lsodaMessages[istate]
				switch istate {
				case -1:
					info.Message = fmt.Sprintf("Excess work done on this call (MxStep = %d).", mxstep)
					return out(), info, fmt.Errorf("%w (MxStep = %d at t = %g)", ErrExcessWork, mxstep, sv.tn)
				case -4, -5, -9:
					return out(), info, fmt.Errorf("%w at t = %g: %s", ErrStepTooSmall, sv.tn, info.Message)
				}
				return out(), info, fmt.Errorf("%w: %s", ErrODEInput, info.Message)
			}
		}
		info.HU = append(info.HU, sv.hu)
		info.TCur = append(info.TCur, sv.tn)
		info.NST = append(info.NST, sv.nst)
		info.NFE = append(info.NFE, sv.nfe)
		info.NJE = append(info.NJE, sv.nje)
		info.MUsed = append(info.MUsed, sv.mused)
	}
	return out(), info, nil
}
