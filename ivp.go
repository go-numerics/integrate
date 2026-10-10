// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/go-ndarray/ndarray"
)

// ODEFunc evaluates the right-hand side of dy/dt = f(t, y), writing f(t, y)
// into dydt. It must not retain y or dydt. SciPy's fun(t, y) returns the
// derivative; here it is written into a caller-owned slice so that a step
// allocates nothing.
type ODEFunc func(t float64, y, dydt []float64)

// JacFunc evaluates the Jacobian ∂f_i/∂y_j of an ODEFunc into jac, an n×n
// row-major matrix: jac[i*n+j] = ∂f_i/∂y_j. It is SciPy's jac argument.
type JacFunc func(t float64, y []float64, jac []float64)

// Method selects the integrator of SolveIVP.
type Method int

const (
	// RK45 is the explicit Dormand–Prince 5(4) pair with its 4th-order
	// free interpolant (SciPy's default, method="RK45").
	RK45 Method = iota
	// RK23 is the explicit Bogacki–Shampine 3(2) pair ("RK23").
	RK23
	// DOP853 is Hairer's explicit Dormand–Prince 8(5,3) method with its
	// 7th-order interpolant ("DOP853").
	DOP853
	// Radau is the implicit Radau IIA method of order 5 ("Radau").
	Radau
	// BDF is the implicit variable-order (1–5) backward differentiation
	// method in its numerical differentiation formula (NDF) variant, the
	// quasi-constant step size implementation of SciPy's "BDF".
	BDF
	// LSODA switches automatically between an explicit Runge–Kutta method
	// on non-stiff stretches and BDF on stiff ones ("LSODA" in spirit; see
	// docs/design.md for how it differs from ODEPACK's Adams/BDF LSODA).
	LSODA
)

var methodNames = [...]string{"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"}

func (m Method) String() string {
	if m < 0 || int(m) >= len(methodNames) {
		return fmt.Sprintf("Method(%d)", int(m))
	}
	return methodNames[m]
}

// Event describes a function whose zeros SolveIVP locates: SciPy's event
// callables with their terminal and direction attributes.
type Event struct {
	// F is the event function g(t, y); events are its zero crossings.
	F func(t float64, y []float64) float64
	// Terminal is 0 for an event that is only recorded; k > 0 stops the
	// integration at the k-th occurrence (SciPy's terminal=True is 1).
	Terminal int
	// Direction > 0 keeps only crossings from negative to positive,
	// < 0 only from positive to negative, 0 both.
	Direction float64
}

// IVPOptions tunes SolveIVP. The zero value gives SciPy's defaults.
type IVPOptions struct {
	// Method is the integrator (default RK45).
	Method Method
	// TEval, if non-nil, lists the times (sorted in the direction of
	// integration, inside the span) at which the solution is stored;
	// otherwise every step is stored.
	TEval []float64
	// DenseOutput requests a continuous solution (Solution.Sol).
	DenseOutput bool
	// Events are located by root finding on the dense output.
	Events []Event
	// RTol and ATol are the relative and absolute tolerances (defaults
	// 1e-3 and 1e-6). The local error estimate e of each component is kept
	// below ATol + RTol·|y|.
	RTol, ATol float64
	// RTolVec and ATolVec give per-component tolerances; when non-nil they
	// replace RTol and ATol.
	RTolVec, ATolVec []float64
	// MaxStep bounds the step size (default +Inf).
	MaxStep float64
	// FirstStep is the initial step size; 0 lets the solver choose.
	FirstStep float64
	// Jac is the Jacobian for Radau, BDF and LSODA; nil uses finite
	// differences.
	Jac JacFunc
}

// Solution is the result of SolveIVP (SciPy's OdeResult).
type Solution struct {
	// T holds the time points.
	T []float64
	// Y holds the states: Y[k] is y(T[k]). SciPy's sol.y is the transpose
	// (component first); YArray returns that layout.
	Y [][]float64
	// Sol is the continuous solution when DenseOutput was requested.
	Sol *ODESolution
	// TEvents[i] and YEvents[i] hold the times and states at which event i
	// occurred.
	TEvents [][]float64
	YEvents [][][]float64
	// NFev, NJev and NLU count right-hand side evaluations, Jacobian
	// evaluations and LU decompositions, as SciPy counts them: NFev does
	// not include the evaluations spent on finite-difference Jacobians,
	// which NFevJac reports.
	NFev, NJev, NLU, NFevJac int
	// NSteps is the number of accepted steps.
	NSteps int
	// Status is 0 when the end of the span was reached, 1 when a terminal
	// event occurred and -1 when integration failed.
	Status int
	// Message describes the reason for termination.
	Message string
	// Success is Status >= 0.
	Success bool
}

// YArray returns the states as an n×len(T) ndarray, the layout of SciPy's
// sol.y.
func (s *Solution) YArray() *ndarray.Array {
	n := 0
	if len(s.Y) > 0 {
		n = len(s.Y[0])
	}
	data := make([]float64, n*len(s.T))
	for k, yk := range s.Y {
		for i, v := range yk {
			data[i*len(s.T)+k] = v
		}
	}
	a, _ := ndarray.FromData(data, n, len(s.T))
	return a
}

// TArray returns the time points as a one-dimensional ndarray.
func (s *Solution) TArray() *ndarray.Array {
	a, _ := ndarray.FromData(append([]float64(nil), s.T...), len(s.T))
	return a
}

// Component returns the history of component i, SciPy's sol.y[i].
func (s *Solution) Component(i int) []float64 {
	out := make([]float64, len(s.Y))
	for k, yk := range s.Y {
		out[k] = yk[i]
	}
	return out
}

// Errors returned by the ODE solvers.
var (
	// ErrStepTooSmall: the required step size fell below the spacing of
	// floating-point numbers at t.
	ErrStepTooSmall = errors.New("integrate: required step size is less than spacing between numbers")
	// ErrODEInput: the problem or the options are invalid.
	ErrODEInput = errors.New("integrate: invalid ODE input")
	// ErrExcessWork: Odeint took more than MxStep steps between two
	// output times.
	ErrExcessWork = errors.New("integrate: excess work done on this call")
)

func odeInput(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrODEInput}, args...)...)
}

// SolveIVP integrates the initial value problem dy/dt = f(t, y), y(t0) = y0
// over tSpan = {t0, tf} (tf < t0 integrates backwards). It is SciPy's
// scipy.integrate.solve_ivp:
//
//	solve_ivp(f, (t0, tf), y0)                        → SolveIVP(f, [2]float64{t0, tf}, y0, nil)
//	solve_ivp(f, span, y0, method="Radau", rtol=1e-6) → SolveIVP(f, span, y0, &IVPOptions{Method: Radau, RTol: 1e-6})
//	solve_ivp(f, span, y0, t_eval=ts, dense_output=True, events=[g])
//	                                                   → SolveIVP(f, span, y0, &IVPOptions{TEval: ts, DenseOutput: true, Events: []Event{{F: g}}})
//
// where SciPy's fun(t, y) returning dydt becomes f(t, y, dydt) writing it.
// Invalid input returns a nil Solution and an error wrapping ErrODEInput. If
// the solver fails mid-way the partial Solution (Success false, Status -1)
// is returned together with the error.
func SolveIVP(f ODEFunc, tSpan [2]float64, y0 []float64, opts *IVPOptions) (*Solution, error) {
	var o IVPOptions
	if opts != nil {
		o = *opts
	}
	t0, tf := tSpan[0], tSpan[1]
	if math.IsNaN(t0) || math.IsNaN(tf) || math.IsInf(t0, 0) || math.IsInf(tf, 0) {
		return nil, odeInput("tSpan must be finite")
	}
	tEval := o.TEval
	if tEval != nil {
		lo, hi := math.Min(t0, tf), math.Max(t0, tf)
		for i, v := range tEval {
			if v < lo || v > hi {
				return nil, odeInput("values in TEval are not within tSpan")
			}
			if i > 0 && ((tf > t0 && v <= tEval[i-1]) || (tf < t0 && v >= tEval[i-1])) {
				return nil, odeInput("values in TEval are not properly sorted")
			}
		}
	}
	core, err := newCore(f, t0, y0, tf, &o)
	if err != nil {
		return nil, err
	}
	solver, err := newMethod(core, o.Method, &o)
	if err != nil {
		return nil, err
	}
	return runIVP(core, solver, tEval, &o)
}

// runIVP is solve_ivp's main loop.
func runIVP(core *odeCore, solver method, tEval []float64, o *IVPOptions) (*Solution, error) {
	t0 := core.t
	sol := &Solution{}
	ne := len(o.Events)
	var g, gNew []float64
	var eventCount []int
	if ne > 0 {
		g = make([]float64, ne)
		gNew = make([]float64, ne)
		eventCount = make([]int, ne)
		sol.TEvents = make([][]float64, ne)
		sol.YEvents = make([][][]float64, ne)
		for i, ev := range o.Events {
			g[i] = ev.F(t0, core.y)
		}
	}
	// t_eval is kept increasing for the search; backwards integration
	// walks it from the end.
	var tev []float64
	tevI := 0
	if tEval != nil {
		tev = tEval
		if core.direction < 0 {
			tev = make([]float64, len(tEval))
			for i, v := range tEval {
				tev[len(tEval)-1-i] = v
			}
			tevI = len(tev)
		}
	} else {
		sol.T = append(sol.T, t0)
		sol.Y = append(sol.Y, append([]float64(nil), core.y...))
	}
	var interps []denseOutput
	var ti []float64
	if o.DenseOutput {
		ti = append(ti, t0)
	}
	status := 0
	running := true
	var failErr error
	for running {
		ok, msg := core.step(solver)
		if !ok {
			status = -1
			sol.Message = msg
			failErr = ErrStepTooSmall
			break
		}
		if core.finished {
			running = false
		}
		tOld, t := core.tOld, core.t
		y := core.y
		var dense denseOutput
		if o.DenseOutput {
			dense = core.dense(solver)
			interps = append(interps, dense)
		}
		if ne > 0 {
			for i, ev := range o.Events {
				gNew[i] = ev.F(t, y)
			}
			active := findActiveEvents(g, gNew, o.Events)
			if len(active) > 0 {
				if dense == nil {
					dense = core.dense(solver)
				}
				for _, e := range active {
					eventCount[e]++
				}
				idx, roots, terminate := handleEvents(dense, o.Events, active, eventCount, tOld, t)
				buf := make([]float64, core.n)
				for k, e := range idx {
					sol.TEvents[e] = append(sol.TEvents[e], roots[k])
					dense.eval(roots[k], buf)
					sol.YEvents[e] = append(sol.YEvents[e], append([]float64(nil), buf...))
				}
				if terminate {
					status = 1
					running = false
					t = roots[len(roots)-1]
					y = make([]float64, core.n)
					dense.eval(t, y)
				}
			}
			g, gNew = gNew, g
		}
		if tEval == nil {
			sol.T = append(sol.T, t)
			sol.Y = append(sol.Y, append([]float64(nil), y...))
		} else {
			var step []float64
			if core.direction > 0 {
				j := sort.Search(len(tev), func(k int) bool { return tev[k] > t })
				step = tev[tevI:j]
				tevI = j
			} else {
				j := sort.Search(len(tev), func(k int) bool { return tev[k] >= t })
				step = make([]float64, 0, tevI-j)
				for k := tevI - 1; k >= j; k-- {
					step = append(step, tev[k])
				}
				tevI = j
			}
			if len(step) > 0 {
				if dense == nil {
					dense = core.dense(solver)
				}
				for _, ts := range step {
					yv := make([]float64, core.n)
					dense.eval(ts, yv)
					sol.T = append(sol.T, ts)
					sol.Y = append(sol.Y, yv)
				}
			}
		}
		if tEval != nil && o.DenseOutput {
			ti = append(ti, t)
		}
	}
	if o.DenseOutput {
		ts := ti
		if tEval == nil {
			ts = sol.T
		}
		sol.Sol = newODESolution(ts, interps, o.Method == BDF || o.Method == LSODA)
	}
	sol.Status = status
	switch status {
	case 0:
		sol.Message = "The solver successfully reached the end of the integration interval."
	case 1:
		sol.Message = "A termination event occurred."
	}
	sol.Success = status >= 0
	sol.NFev, sol.NJev, sol.NLU, sol.NFevJac = core.nfev, core.njev, core.nlu, core.nfevJac
	sol.NSteps = core.nsteps
	return sol, failErr
}

// findActiveEvents returns the events whose sign change between g and gNew
// matches their direction.
func findActiveEvents(g, gNew []float64, events []Event) []int {
	var active []int
	for i, ev := range events {
		up := g[i] <= 0 && gNew[i] >= 0
		down := g[i] >= 0 && gNew[i] <= 0
		if (up && ev.Direction > 0) || (down && ev.Direction < 0) || ((up || down) && ev.Direction == 0) {
			active = append(active, i)
		}
	}
	return active
}

// handleEvents locates the active events in the step and decides whether
// the integration terminates; it returns the events and roots to record, in
// order of occurrence when terminating.
func handleEvents(dense denseOutput, events []Event, active, count []int, tOld, t float64) ([]int, []float64, bool) {
	buf := make([]float64, dense.dim())
	roots := make([]float64, len(active))
	for k, e := range active {
		ev := events[e].F
		roots[k] = brentq(func(s float64) float64 {
			dense.eval(s, buf)
			return ev(s, buf)
		}, tOld, t, 4*dblEps, 4*dblEps, 100)
	}
	terminate := false
	for _, e := range active {
		if events[e].Terminal > 0 && count[e] >= events[e].Terminal {
			terminate = true
		}
	}
	if !terminate {
		return active, roots, false
	}
	order := make([]int, len(active))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		if t > tOld {
			return roots[order[i]] < roots[order[j]]
		}
		return roots[order[i]] > roots[order[j]]
	})
	idx := make([]int, 0, len(active))
	rs := make([]float64, 0, len(active))
	for _, k := range order {
		e := active[k]
		idx = append(idx, e)
		rs = append(rs, roots[k])
		if events[e].Terminal > 0 && count[e] >= events[e].Terminal {
			break
		}
	}
	return idx, rs, true
}

// brentq finds a root of f in [xa, xb] with Brent's method; it is the
// algorithm of SciPy's brentq (Brent 1973, as implemented in SciPy's C
// brentq), which solve_ivp uses for events. f(xa) and f(xb) must bracket a
// root (event detection guarantees it); otherwise xb is returned.
func brentq(f func(float64) float64, xa, xb, xtol, rtol float64, iter int) float64 {
	xpre, xcur := xa, xb
	var xblk, fblk, spre, scur float64
	fpre := f(xpre)
	fcur := f(xcur)
	if fpre == 0 {
		return xpre
	}
	if fcur == 0 {
		return xcur
	}
	for i := 0; i < iter; i++ {
		if fpre != 0 && fcur != 0 && math.Signbit(fpre) != math.Signbit(fcur) {
			xblk = xpre
			fblk = fpre
			spre = xcur - xpre
			scur = spre
		}
		if math.Abs(fblk) < math.Abs(fcur) {
			xpre, xcur, xblk = xcur, xblk, xcur
			fpre, fcur, fblk = fcur, fblk, fcur
		}
		delta := (xtol + rtol*math.Abs(xcur)) / 2
		sbis := (xblk - xcur) / 2
		if fcur == 0 || math.Abs(sbis) < delta {
			return xcur
		}
		if math.Abs(spre) > delta && math.Abs(fcur) < math.Abs(fpre) {
			var stry float64
			if xpre == xblk {
				// interpolate
				stry = -fcur * (xcur - xpre) / (fcur - fpre)
			} else {
				// extrapolate
				dpre := (fpre - fcur) / (xpre - xcur)
				dblk := (fblk - fcur) / (xblk - xcur)
				stry = -fcur * (fblk*dblk - fpre*dpre) / (dblk * dpre * (fblk - fpre))
			}
			if 2*math.Abs(stry) < math.Min(math.Abs(spre), 3*math.Abs(sbis)-delta) {
				// good short step
				spre = scur
				scur = stry
			} else {
				spre = sbis
				scur = sbis
			}
		} else {
			spre = sbis
			scur = sbis
		}
		xpre = xcur
		fpre = fcur
		if math.Abs(scur) > delta {
			xcur += scur
		} else if sbis > 0 {
			xcur += delta
		} else {
			xcur -= delta
		}
		fcur = f(xcur)
	}
	return xcur
}

// ODESolution is a continuous solution assembled from the local
// interpolants of the steps (SciPy's OdeSolution).
type ODESolution struct {
	ts        []float64 // breakpoints, increasing
	interps   []denseOutput
	ascending bool
	altSeg    bool
	n         int
}

func newODESolution(ts []float64, interps []denseOutput, altSegment bool) *ODESolution {
	s := &ODESolution{interps: interps, altSeg: altSegment}
	if len(interps) > 0 {
		s.n = interps[0].dim()
	}
	s.ascending = len(ts) < 2 || ts[len(ts)-1] >= ts[0]
	if s.ascending {
		s.ts = ts
	} else {
		s.ts = make([]float64, len(ts))
		for i, v := range ts {
			s.ts[len(ts)-1-i] = v
		}
	}
	return s
}

// TMin and TMax are the ends of the interval covered.
func (s *ODESolution) TMin() float64 { return s.ts[0] }

// TMax is the upper end of the interval covered.
func (s *ODESolution) TMax() float64 { return s.ts[len(s.ts)-1] }

// EvalInto writes the solution at t into y. Outside [TMin, TMax] the end
// interpolants extrapolate.
func (s *ODESolution) EvalInto(t float64, y []float64) {
	var ind int
	right := s.altSeg == s.ascending
	if right {
		ind = sort.Search(len(s.ts), func(k int) bool { return s.ts[k] > t })
	} else {
		ind = sort.Search(len(s.ts), func(k int) bool { return s.ts[k] >= t })
	}
	nseg := len(s.interps)
	seg := min(max(ind-1, 0), nseg-1)
	if !s.ascending {
		seg = nseg - 1 - seg
	}
	s.interps[seg].eval(t, y)
}

// Eval returns the solution at t.
func (s *ODESolution) Eval(t float64) []float64 {
	y := make([]float64, s.n)
	s.EvalInto(t, y)
	return y
}

// EvalArray evaluates the solution at the times ts and returns an
// n×len(ts) ndarray, as SciPy's sol(ts) does.
func (s *ODESolution) EvalArray(ts []float64) *ndarray.Array {
	data := make([]float64, s.n*len(ts))
	y := make([]float64, s.n)
	for k, t := range ts {
		s.EvalInto(t, y)
		for i, v := range y {
			data[i*len(ts)+k] = v
		}
	}
	a, _ := ndarray.FromData(data, s.n, len(ts))
	return a
}
