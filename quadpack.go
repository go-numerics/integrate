// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

// This file is a port of the QUADPACK adaptive drivers (Piessens, de
// Doncker-Kapenga, Überhuber, Kahaner, 1983; netlib.org/quadpack). The control
// flow, the constants and the order in which subintervals are bisected follow
// the double-precision Fortran routines named in each comment, so that a run
// here and a run of the Fortran code on the same integrand visit the same
// subintervals. Arrays are indexed from 1, as in the Fortran, which keeps the
// translation line-for-line checkable; element 0 is unused.

import "math"

const (
	epmach = 2.220446049250313e-16   // d1mach(4): relative spacing at 1
	uflow  = 2.2250738585072014e-308 // d1mach(1): smallest normal
	oflow  = math.MaxFloat64         // d1mach(2)
)

// qout is what every driver reports: the QUADPACK (result, abserr, neval, ier,
// last) quintuple.
type qout struct {
	result, abserr float64
	neval, ier     int
	last           int
}

// ruleFunc is a local rule applied to [a, b]: it returns the approximation,
// an error estimate, the integral of |f| and the integral of |f - mean|
// (QUADPACK's result, abserr, resabs, resasc).
type ruleFunc func(a, b float64) (result, abserr, resabs, resasc float64)

// workspace holds the subinterval lists of the adaptive drivers.
type workspace struct {
	alist, blist, rlist, elist []float64
	iord, level                []int
}

func newWorkspace(limit int) *workspace {
	n := limit + 1
	return &workspace{
		alist: make([]float64, n), blist: make([]float64, n),
		rlist: make([]float64, n), elist: make([]float64, n),
		iord: make([]int, n), level: make([]int, n),
	}
}

// split stores the two halves of interval maxerr in slots maxerr and last,
// keeping the half with the larger error in maxerr (QUADPACK's labels 20/30).
func (w *workspace) split(maxerr, last int, a1, b1, a2, b2, area1, area2, error1, error2 float64) {
	w.rlist[maxerr] = area1
	w.rlist[last] = area2
	if error2 > error1 {
		w.alist[maxerr] = a2
		w.alist[last] = a1
		w.blist[last] = b1
		w.rlist[maxerr] = area2
		w.rlist[last] = area1
		w.elist[maxerr] = error2
		w.elist[last] = error1
		return
	}
	w.alist[last] = a2
	w.blist[maxerr] = b1
	w.blist[last] = b2
	w.elist[maxerr] = error1
	w.elist[last] = error2
}

func (w *workspace) sum(last int) float64 {
	s := 0.0
	for k := 1; k <= last; k++ {
		s += w.rlist[k]
	}
	return s
}

// tolInvalid is QUADPACK's ier=6 tolerance test.
func tolInvalid(epsabs, epsrel float64) bool {
	return epsabs <= 0 && epsrel < math.Max(50*epmach, 0.5e-28)
}

// tooSmall reports QUADPACK's "interval too small" test for a bisection of
// [a1, b2] at a2.
func tooSmall(a1, a2, b2 float64) bool {
	return math.Max(math.Abs(a1), math.Abs(b2)) <= (1+100*epmach)*(math.Abs(a2)+1000*uflow)
}

// qpsrt maintains the descending ordering of error estimates (dqpsrt).
func qpsrt(limit, last int, maxerr *int, ermax *float64, elist []float64, iord []int, nrmax *int) {
	if last <= 2 {
		iord[1] = 1
		iord[2] = 2
	} else {
		errmax := elist[*maxerr]
		if *nrmax != 1 {
			ido := *nrmax - 1
			for i := 1; i <= ido; i++ {
				isucc := iord[*nrmax-1]
				if errmax <= elist[isucc] {
					break
				}
				iord[*nrmax] = isucc
				*nrmax--
			}
		}
		jupbn := last
		if last > limit/2+2 {
			jupbn = limit + 3 - last
		}
		errmin := elist[last]
		jbnd := jupbn - 1
		ibeg := *nrmax + 1
		i := ibeg
		inserted := false
		for ; i <= jbnd; i++ {
			isucc := iord[i]
			if errmax >= elist[isucc] {
				inserted = true
				break
			}
			iord[i-1] = isucc
		}
		if !inserted {
			iord[jbnd] = *maxerr
			iord[jupbn] = last
		} else {
			iord[i-1] = *maxerr
			k := jbnd
			placed := false
			for j := i; j <= jbnd; j++ {
				isucc := iord[k]
				if errmin < elist[isucc] {
					iord[k+1] = last
					placed = true
					break
				}
				iord[k+1] = isucc
				k--
			}
			if !placed {
				iord[i] = last
			}
		}
	}
	*maxerr = iord[*nrmax]
	*ermax = elist[*maxerr]
}

// epsTable is the state of Wynn's epsilon algorithm (dqelg): the table
// (1-based, 52 entries plus the two scratch slots dqelg writes), the last three
// results and the call counter.
type epsTable struct {
	tab    [55]float64
	res3la [4]float64
	nres   int
}

// qelg performs one step of the epsilon algorithm on the first n entries of
// the table and returns the extrapolated limit and its error estimate (dqelg).
// n is updated in place, as in the Fortran.
func (e *epsTable) qelg(n *int) (result, abserr float64) {
	ep := &e.tab
	e.nres++
	abserr = oflow
	result = ep[*n]
	if *n < 3 {
		return result, math.Max(abserr, 5*epmach*math.Abs(result))
	}
	const limexp = 50
	ep[*n+2] = ep[*n]
	newelm := (*n - 1) / 2
	ep[*n] = oflow
	num := *n
	k1 := *n
	for i := 1; i <= newelm; i++ {
		k2 := k1 - 1
		k3 := k1 - 2
		res := ep[k1+2]
		e0 := ep[k3]
		e1 := ep[k2]
		e2 := res
		e1abs := math.Abs(e1)
		delta2 := e2 - e1
		err2 := math.Abs(delta2)
		tol2 := math.Max(math.Abs(e2), e1abs) * epmach
		delta3 := e1 - e0
		err3 := math.Abs(delta3)
		tol3 := math.Max(e1abs, math.Abs(e0)) * epmach
		if err2 <= tol2 && err3 <= tol3 {
			// e0, e1 and e2 are equal to within machine accuracy:
			// convergence is assumed.
			result = res
			abserr = err2 + err3
			return result, math.Max(abserr, 5*epmach*math.Abs(result))
		}
		e3 := ep[k1]
		ep[k1] = e1
		delta1 := e1 - e3
		err1 := math.Abs(delta1)
		tol1 := math.Max(e1abs, math.Abs(e3)) * epmach
		var ss float64
		irregular := err1 <= tol1 || err2 <= tol2 || err3 <= tol3
		if !irregular {
			ss = 1/delta1 + 1/delta2 - 1/delta3
			irregular = math.Abs(ss*e1) <= 1e-4
		}
		if irregular {
			// Two elements are very close: omit a part of the table by
			// adjusting n.
			*n = i + i - 1
			break
		}
		res = e1 + 1/ss
		ep[k1] = res
		k1 -= 2
		errv := err2 + math.Abs(res-e2) + err3
		if errv <= abserr {
			abserr = errv
			result = res
		}
	}
	if *n == limexp {
		*n = 2*(limexp/2) - 1
	}
	ib := 1
	if (num/2)*2 == num {
		ib = 2
	}
	ie := newelm + 1
	for i := 1; i <= ie; i++ {
		ib2 := ib + 2
		ep[ib] = ep[ib2]
		ib = ib2
	}
	if num != *n {
		indx := num - *n + 1
		for i := 1; i <= *n; i++ {
			ep[i] = ep[indx]
			indx++
		}
	}
	if e.nres < 4 {
		e.res3la[e.nres] = result
		abserr = oflow
	} else {
		abserr = math.Abs(result-e.res3la[3]) + math.Abs(result-e.res3la[2]) + math.Abs(result-e.res3la[1])
		e.res3la[1] = e.res3la[2]
		e.res3la[2] = e.res3la[3]
		e.res3la[3] = result
	}
	return result, math.Max(abserr, 5*epmach*math.Abs(result))
}

// extrapTail is the common epilogue of the extrapolating drivers (labels
// 100-130 of dqagse). It decides between the extrapolated result and the plain
// sum of the subinterval contributions, and may flag divergence (ier=6, which
// ierShift turns into 5). ge selects dqawoe's ">=" in the divergence test.
func extrapTail(w *workspace, last int, result, abserr, area, errsum, correc, defabs float64, ier, ierro, ksgn int, ge bool) (float64, float64, int) {
	useSum := false
	check := true // reach label 110
	switch {
	case abserr == oflow:
		useSum = true
	case ier+ierro != 0:
		if ierro == 3 {
			abserr += correc
		}
		if ier == 0 {
			ier = 3
		}
		switch {
		case result != 0 && area != 0:
			useSum = abserr/math.Abs(result) > errsum/math.Abs(area)
		case abserr > errsum:
			useSum = true
		case area == 0:
			check = false
		}
	}
	if useSum {
		result = w.sum(last)
		abserr = errsum
	} else if check && !(ksgn == -1 && math.Max(math.Abs(result), math.Abs(area)) <= defabs*0.01) {
		ratio := result / area
		divergent := errsum > math.Abs(area)
		if ge {
			divergent = errsum >= math.Abs(area)
		}
		if 0.01 > ratio || ratio > 100 || divergent {
			ier = 6
		}
	}
	return result, abserr, ier
}

// ierShift maps the internal codes of the extrapolating drivers to the
// documented ones (label 130: "if(ier.gt.2) ier=ier-1").
func ierShift(ier int) int {
	if ier > 2 {
		return ier - 1
	}
	return ier
}

// qagsCore is the body shared by dqagse and dqagie: globally adaptive
// bisection with epsilon-algorithm extrapolation. rule integrates one
// subinterval; small0 is the initial "small interval" width (0.375|b-a| for
// dqagse, 0.375 for dqagie on the transformed [0, 1]).
func qagsCore(rule ruleFunc, a, b, epsabs, epsrel float64, limit int, small0 float64) qout {
	if tolInvalid(epsabs, epsrel) {
		return qout{ier: 6}
	}
	w := newWorkspace(limit)
	var et epsTable
	ierro := 0
	result, abserr, defabs, resabs := rule(a, b)
	dres := math.Abs(result)
	errbnd := math.Max(epsabs, epsrel*dres)
	last := 1
	w.alist[1], w.blist[1] = a, b
	w.rlist[1] = result
	w.elist[1] = abserr
	w.iord[1] = 1
	ier := 0
	if abserr <= 100*epmach*defabs && abserr > errbnd {
		ier = 2
	}
	if limit == 1 {
		ier = 1
	}
	if ier != 0 || (abserr <= errbnd && abserr != resabs) || abserr == 0 {
		return qout{result: result, abserr: abserr, ier: ier, last: last}
	}
	et.tab[1] = result
	errmax := abserr
	maxerr := 1
	area := result
	errsum := abserr
	abserr = oflow
	nrmax := 1
	numrl2 := 2
	ktmin := 0
	extrap, noext := false, false
	iroff1, iroff2, iroff3 := 0, 0, 0
	ksgn := -1
	if dres >= (1-50*epmach)*defabs {
		ksgn = 1
	}
	var small, erlarg, ertest, correc float64
	converged := false
	for last = 2; last <= limit; last++ {
		a1 := w.alist[maxerr]
		b1 := 0.5 * (w.alist[maxerr] + w.blist[maxerr])
		a2 := b1
		b2 := w.blist[maxerr]
		erlast := errmax
		area1, error1, _, defab1 := rule(a1, b1)
		area2, error2, _, defab2 := rule(a2, b2)
		area12 := area1 + area2
		erro12 := error1 + error2
		errsum = errsum + erro12 - errmax
		area = area + area12 - w.rlist[maxerr]
		if defab1 != error1 && defab2 != error2 {
			if math.Abs(w.rlist[maxerr]-area12) <= 1e-5*math.Abs(area12) && erro12 >= 0.99*errmax {
				if extrap {
					iroff2++
				} else {
					iroff1++
				}
			}
			if last > 10 && erro12 > errmax {
				iroff3++
			}
		}
		errbnd = math.Max(epsabs, epsrel*math.Abs(area))
		if iroff1+iroff2 >= 10 || iroff3 >= 20 {
			ier = 2
		}
		if iroff2 >= 5 {
			ierro = 3
		}
		if last == limit {
			ier = 1
		}
		if tooSmall(a1, a2, b2) {
			ier = 4
		}
		w.split(maxerr, last, a1, b1, a2, b2, area1, area2, error1, error2)
		qpsrt(limit, last, &maxerr, &errmax, w.elist, w.iord, &nrmax)
		if errsum <= errbnd {
			converged = true
			break
		}
		if ier != 0 {
			break
		}
		if last == 2 {
			small = small0
			erlarg = errsum
			ertest = errbnd
			et.tab[2] = area
			continue
		}
		if noext {
			continue
		}
		erlarg -= erlast
		if math.Abs(b1-a1) > small {
			erlarg += erro12
		}
		if !extrap {
			// Test whether the interval to be bisected next is the
			// smallest interval.
			if math.Abs(w.blist[maxerr]-w.alist[maxerr]) > small {
				continue
			}
			extrap = true
			nrmax = 2
		}
		if ierro != 3 && erlarg > ertest {
			// The smallest interval has the largest error. Before
			// bisecting decrease the sum of the errors over the larger
			// intervals (erlarg) and perform extrapolation.
			jupbnd := last
			if last > 2+limit/2 {
				jupbnd = limit + 3 - last
			}
			large := false
			for k := nrmax; k <= jupbnd; k++ {
				maxerr = w.iord[nrmax]
				errmax = w.elist[maxerr]
				if math.Abs(w.blist[maxerr]-w.alist[maxerr]) > small {
					large = true
					break
				}
				nrmax++
			}
			if large {
				continue
			}
		}
		numrl2++
		et.tab[numrl2] = area
		reseps, abseps := et.qelg(&numrl2)
		ktmin++
		if ktmin > 5 && abserr < 1e-3*errsum {
			ier = 5
		}
		if abseps < abserr {
			ktmin = 0
			abserr = abseps
			result = reseps
			correc = erlarg
			ertest = math.Max(epsabs, epsrel*math.Abs(reseps))
			if abserr <= ertest {
				break
			}
		}
		if numrl2 == 1 {
			noext = true
		}
		if ier == 5 {
			break
		}
		maxerr = w.iord[1]
		errmax = w.elist[maxerr]
		nrmax = 1
		extrap = false
		small *= 0.5
		erlarg = errsum
	}
	if converged {
		result = w.sum(last)
		abserr = errsum
	} else {
		result, abserr, ier = extrapTail(w, last, result, abserr, area, errsum, correc, defabs, ier, ierro, ksgn, false)
	}
	return qout{result: result, abserr: abserr, ier: ierShift(ier), last: last}
}

// qagse integrates f over the finite interval [a, b] (dqagse).
func qagse(f func(float64) float64, a, b, epsabs, epsrel float64, limit int) qout {
	r := qagsCore(func(lo, hi float64) (float64, float64, float64, float64) {
		return qk21(f, lo, hi)
	}, a, b, epsabs, epsrel, limit, math.Abs(b-a)*0.375)
	if r.ier != 6 {
		r.neval = 42*r.last - 21
	}
	return r
}

// qagie integrates f over an infinite range (dqagie). inf is 1 for
// (bound, +inf), -1 for (-inf, bound) and 2 for (-inf, +inf).
func qagie(f func(float64) float64, bound float64, inf int, epsabs, epsrel float64, limit int) qout {
	boun := bound
	if inf == 2 {
		boun = 0
	}
	r := qagsCore(func(lo, hi float64) (float64, float64, float64, float64) {
		return qk15i(f, boun, inf, lo, hi)
	}, 0, 1, epsabs, epsrel, limit, 0.375)
	if r.ier != 6 {
		r.neval = 30*r.last - 15
		if inf == 2 {
			r.neval *= 2
		}
	}
	return r
}

// qagpe integrates f over [a, b] with user-supplied break points (dqagpe).
// points holds the interior break points, sorted and strictly inside (a, b),
// with a < b: Quad orders the limits and filters the points (as SciPy does)
// before calling, so dqagpe's own sorting and sign handling are not needed.
func qagpe(f func(float64) float64, a, b float64, points []float64, epsabs, epsrel float64, limit int) qout {
	npts := len(points)
	npts2 := npts + 2
	if limit <= npts || tolInvalid(epsabs, epsrel) {
		return qout{ier: 6}
	}
	pts := make([]float64, npts2+1)
	pts[1] = a
	copy(pts[2:], points)
	pts[npts+2] = b
	nint := npts + 1
	w := newWorkspace(limit)
	ndin := make([]int, npts2+1)
	var result, abserr, resabs float64
	a1 := pts[1]
	for i := 1; i <= nint; i++ {
		b1 := pts[i+1]
		area1, error1, defabs, resa := qk21(f, a1, b1)
		abserr += error1
		result += area1
		if error1 == resa && error1 != 0 {
			ndin[i] = 1
		}
		resabs += defabs
		w.elist[i] = error1
		w.alist[i] = a1
		w.blist[i] = b1
		w.rlist[i] = area1
		w.iord[i] = i
		a1 = b1
	}
	errsum := 0.0
	for i := 1; i <= nint; i++ {
		if ndin[i] == 1 {
			w.elist[i] = abserr
		}
		errsum += w.elist[i]
	}
	last := nint
	neval := 21 * nint
	dres := math.Abs(result)
	errbnd := math.Max(epsabs, epsrel*dres)
	ier := 0
	if abserr <= 100*epmach*resabs && abserr > errbnd {
		ier = 2
	}
	if nint != 1 {
		for i := 1; i <= npts; i++ {
			ind1 := w.iord[i]
			k := 0
			for j := i + 1; j <= nint; j++ {
				ind2 := w.iord[j]
				if w.elist[ind1] <= w.elist[ind2] {
					ind1 = ind2
					k = j
				}
			}
			if ind1 != w.iord[i] {
				w.iord[k] = w.iord[i]
				w.iord[i] = ind1
			}
		}
		if limit < npts2 {
			ier = 1
		}
	}
	if ier != 0 || abserr <= errbnd {
		return qout{result: result, abserr: abserr, neval: neval, ier: ier, last: last}
	}
	var et epsTable
	et.tab[1] = result
	maxerr := w.iord[1]
	errmax := w.elist[maxerr]
	area := result
	nrmax := 1
	numrl2 := 1
	ktmin := 0
	extrap, noext := false, false
	erlarg := errsum
	ertest := errbnd
	levmax := 1
	iroff1, iroff2, iroff3 := 0, 0, 0
	ierro := 0
	abserr = oflow
	ksgn := -1
	if dres >= (1-50*epmach)*resabs {
		ksgn = 1
	}
	var correc float64
	converged := false
	for last = npts2; last <= limit; last++ {
		levcur := w.level[maxerr] + 1
		a1 := w.alist[maxerr]
		b1 := 0.5 * (w.alist[maxerr] + w.blist[maxerr])
		a2 := b1
		b2 := w.blist[maxerr]
		erlast := errmax
		area1, error1, _, defab1 := qk21(f, a1, b1)
		area2, error2, _, defab2 := qk21(f, a2, b2)
		neval += 42
		area12 := area1 + area2
		erro12 := error1 + error2
		errsum = errsum + erro12 - errmax
		area = area + area12 - w.rlist[maxerr]
		if defab1 != error1 && defab2 != error2 {
			if math.Abs(w.rlist[maxerr]-area12) <= 1e-5*math.Abs(area12) && erro12 >= 0.99*errmax {
				if extrap {
					iroff2++
				} else {
					iroff1++
				}
			}
			if last > 10 && erro12 > errmax {
				iroff3++
			}
		}
		w.level[maxerr] = levcur
		w.level[last] = levcur
		errbnd = math.Max(epsabs, epsrel*math.Abs(area))
		if iroff1+iroff2 >= 10 || iroff3 >= 20 {
			ier = 2
		}
		if iroff2 >= 5 {
			ierro = 3
		}
		if last == limit {
			ier = 1
		}
		if tooSmall(a1, a2, b2) {
			ier = 4
		}
		w.split(maxerr, last, a1, b1, a2, b2, area1, area2, error1, error2)
		qpsrt(limit, last, &maxerr, &errmax, w.elist, w.iord, &nrmax)
		if errsum <= errbnd {
			converged = true
			break
		}
		if ier != 0 {
			break
		}
		if noext {
			continue
		}
		erlarg -= erlast
		if levcur+1 <= levmax {
			erlarg += erro12
		}
		if !extrap {
			if w.level[maxerr]+1 <= levmax {
				continue
			}
			extrap = true
			nrmax = 2
		}
		if ierro != 3 && erlarg > ertest {
			jupbnd := last
			if last > 2+limit/2 {
				jupbnd = limit + 3 - last
			}
			large := false
			for k := nrmax; k <= jupbnd; k++ {
				maxerr = w.iord[nrmax]
				errmax = w.elist[maxerr]
				if w.level[maxerr]+1 <= levmax {
					large = true
					break
				}
				nrmax++
			}
			if large {
				continue
			}
		}
		numrl2++
		et.tab[numrl2] = area
		if numrl2 > 2 {
			reseps, abseps := et.qelg(&numrl2)
			ktmin++
			if ktmin > 5 && abserr < 1e-3*errsum {
				ier = 5
			}
			if abseps < abserr {
				ktmin = 0
				abserr = abseps
				result = reseps
				correc = erlarg
				ertest = math.Max(epsabs, epsrel*math.Abs(reseps))
				if abserr < ertest {
					break
				}
			}
			if numrl2 == 1 {
				noext = true
			}
			if ier >= 5 {
				break
			}
		}
		maxerr = w.iord[1]
		errmax = w.elist[maxerr]
		nrmax = 1
		extrap = false
		levmax++
		erlarg = errsum
	}
	if converged {
		result = w.sum(last)
		abserr = errsum
	} else {
		result, abserr, ier = extrapTail(w, last, result, abserr, area, errsum, correc, resabs, ier, ierro, ksgn, false)
	}
	return qout{result: result, abserr: abserr, neval: neval, ier: ierShift(ier), last: last}
}
