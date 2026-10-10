// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Weighted QUADPACK drivers: dqawoe/dqawfe (cos, sin), dqawce (Cauchy
// principal value), dqawse (algebraic-logarithmic end-point singularities),
// with their Clenshaw–Curtis kernels dqc25f, dqc25c and dqc25s.

// gtsl solves the tridiagonal system with sub-diagonal c[2..n], diagonal
// d[1..n] and super-diagonal e[1..n-1] by Gaussian elimination with partial
// pivoting (LINPACK dgtsl, n ≥ 2). b is overwritten with the solution. The
// systems built by qc25f are diagonally dominant away from the first rows and
// never singular, so the singularity exits of dgtsl are not reproduced.
func gtsl(n int, c, d, e, b []float64) {
	c[1] = d[1]
	nm1 := n - 1
	d[1] = e[1]
	e[1] = 0
	e[n] = 0
	for k := 1; k <= nm1; k++ {
		kp1 := k + 1
		if math.Abs(c[kp1]) >= math.Abs(c[k]) {
			c[kp1], c[k] = c[k], c[kp1]
			d[kp1], d[k] = d[k], d[kp1]
			e[kp1], e[k] = e[k], e[kp1]
			b[kp1], b[k] = b[k], b[kp1]
		}
		t := -c[kp1] / c[k]
		c[kp1] = d[kp1] + t*d[k]
		d[kp1] = e[kp1] + t*e[k]
		e[kp1] = 0
		b[kp1] += t * b[k]
	}
	b[n] /= c[n]
	b[nm1] = (b[nm1] - d[nm1]*b[n]) / c[nm1]
	for kb := 1; kb <= n-2; kb++ {
		k := n - 2 - kb + 1
		b[k] = (b[k] - d[k]*b[k+1] - e[k]*b[k+2]) / c[k]
	}
}

// fourierMoments computes the modified Chebyshev moments
// ∫_{-1}^{1} T_k(x) cos(par·x) dx (odd slots) and ∫ T_k(x) sin(par·x) dx
// (even slots) for k = 0..24, in dqc25f's layout (1-based, 25 entries). For
// |par| ≤ 24 it solves Olver's boundary-value formulation of the recurrence
// (a tridiagonal system), above it it recurses forwards, exactly as dqc25f.
func fourierMoments(parint float64, mom *[26]float64) {
	const noequ = 25
	var v [29]float64
	var d, d1, d2 [26]float64
	par2 := parint * parint
	par22 := par2 + 2
	sinpar := math.Sin(parint)
	cospar := math.Cos(parint)
	// Cosine moments.
	v[1] = 2 * sinpar / parint
	v[2] = (8*cospar + (par2+par2-8)*sinpar/parint) / par2
	v[3] = (32*(par2-12)*cospar + (2*((par2-80)*par2+192)*sinpar)/parint) / (par2 * par2)
	ac := 8 * cospar
	as := 24 * parint * sinpar
	if math.Abs(parint) > 24 {
		an := 4.0
		for i := 4; i <= 13; i++ {
			an2 := an * an
			v[i] = ((an2-4)*(2*(par22-an2-an2)*v[i-1]-ac) + as - par2*(an+1)*(an+2)*v[i-2]) /
				(par2 * (an - 1) * (an - 2))
			an += 2
		}
	} else {
		an := 6.0
		for k := 1; k <= noequ-1; k++ {
			an2 := an * an
			d[k] = -2 * (an2 - 4) * (par22 - an2 - an2)
			d2[k] = (an - 1) * (an - 2) * par2
			d1[k+1] = (an + 3) * (an + 4) * par2
			v[k+3] = as - (an2-4)*ac
			an += 2
		}
		an2 := an * an
		d[noequ] = -2 * (an2 - 4) * (par22 - an2 - an2)
		v[noequ+3] = as - (an2-4)*ac
		v[4] -= 56 * par2 * v[3]
		ass := parint * sinpar
		asap := (((((210*par2-1)*cospar-(105*par2-63)*ass)/an2-(1-15*par2)*cospar+15*ass)/an2-cospar+3*ass)/an2 - cospar) / an2
		v[noequ+3] -= 2 * asap * par2 * (an - 1) * (an - 2)
		gtsl(noequ, d1[:], d[:], d2[:], v[3:])
	}
	for j := 1; j <= 13; j++ {
		mom[2*j-1] = v[j]
	}
	// Sine moments.
	v[1] = 2 * (sinpar - parint*cospar) / par2
	v[2] = (18-48/par2)*sinpar/par2 + (-2+48/par2)*cospar/parint
	ac = -24 * parint * cospar
	as = -8 * sinpar
	if math.Abs(parint) > 24 {
		an := 3.0
		for i := 3; i <= 12; i++ {
			an2 := an * an
			v[i] = ((an2-4)*(2*(par22-an2-an2)*v[i-1]+as) + ac - par2*(an+1)*(an+2)*v[i-2]) /
				(par2 * (an - 1) * (an - 2))
			an += 2
		}
	} else {
		an := 5.0
		for k := 1; k <= noequ-1; k++ {
			an2 := an * an
			d[k] = -2 * (an2 - 4) * (par22 - an2 - an2)
			d2[k] = (an - 1) * (an - 2) * par2
			d1[k+1] = (an + 3) * (an + 4) * par2
			v[k+2] = ac + (an2-4)*as
			an += 2
		}
		an2 := an * an
		d[noequ] = -2 * (an2 - 4) * (par22 - an2 - an2)
		v[noequ+2] = ac + (an2-4)*as
		v[3] -= 42 * par2 * v[2]
		ass := parint * cospar
		asap := (((((105*par2-63)*ass+(210*par2-1)*sinpar)/an2+(15*par2-1)*sinpar-15*ass)/an2-3*ass-sinpar)/an2 - sinpar) / an2
		v[noequ+2] -= 2 * asap * par2 * (an - 1) * (an - 2)
		gtsl(noequ, d1[:], d[:], d2[:], v[2:])
	}
	for j := 1; j <= 12; j++ {
		mom[2*j] = v[j]
	}
}

// momentCache keeps the Chebyshev moments of dqc25f per bisection level, as
// QUADPACK's chebmo/momcom pair does: level nrmom's moments live in row
// nrmom+1 once computed; beyond maxp1 levels they are recomputed in the last
// row.
type momentCache struct {
	chebmo [][26]float64
	momcom int
	m      int
	maxp1  int
}

func newMomentCache(maxp1 int) *momentCache {
	return &momentCache{chebmo: make([][26]float64, maxp1+1), maxp1: maxp1}
}

// qc25f integrates f·cos(ωx) (integr 1) or f·sin(ωx) (integr 2) over
// [a, b] with a 25-point Clenshaw–Curtis rule, or with the 15-point
// Gauss–Kronrod rule when the interval holds less than a period (dqc25f).
func (mc *momentCache) qc25f(f func(float64) float64, a, b, omega float64, integr, nrmom, ksave int) (result, abserr float64, neval int, resabs, resasc float64) {
	centr := 0.5 * (b + a)
	hlgth := 0.5 * (b - a)
	parint := omega * hlgth
	if math.Abs(parint) <= 2 {
		w := math.Cos
		if integr == 2 {
			w = math.Sin
		}
		result, abserr, resabs, resasc = qk15w(f, func(x float64) float64 { return w(omega * x) }, a, b)
		return result, abserr, 15, resabs, resasc
	}
	conc := hlgth * math.Cos(centr*omega)
	cons := hlgth * math.Sin(centr*omega)
	resasc = oflow
	if !(nrmom < mc.momcom || ksave == 1) {
		mc.m = mc.momcom + 1
		fourierMoments(parint, &mc.chebmo[mc.m])
	}
	if nrmom < mc.momcom {
		mc.m = nrmom + 1
	}
	if mc.momcom < mc.maxp1-1 && nrmom >= mc.momcom {
		mc.momcom++
	}
	var fval [26]float64
	var cheb12 [14]float64
	var cheb24 [26]float64
	sample25(f, centr, hlgth, &fval)
	qcheb(&fval, &cheb12, &cheb24)
	cm := &mc.chebmo[mc.m]
	resc12 := cheb12[13] * cm[13]
	ress12 := 0.0
	k := 11
	for j := 1; j <= 6; j++ {
		resc12 += cheb12[k] * cm[k]
		ress12 += cheb12[k+1] * cm[k+1]
		k -= 2
	}
	resc24 := cheb24[25] * cm[25]
	ress24 := 0.0
	resabs = math.Abs(cheb24[25])
	k = 23
	for j := 1; j <= 12; j++ {
		resc24 += cheb24[k] * cm[k]
		ress24 += cheb24[k+1] * cm[k+1]
		resabs += math.Abs(cheb24[k]) + math.Abs(cheb24[k+1])
		k -= 2
	}
	estc := math.Abs(resc24 - resc12)
	ests := math.Abs(ress24 - ress12)
	resabs *= math.Abs(hlgth)
	if integr == 2 {
		result = conc*ress24 + cons*resc24
		abserr = math.Abs(conc*ests) + math.Abs(cons*estc)
	} else {
		result = conc*resc24 - cons*ress24
		abserr = math.Abs(conc*estc) + math.Abs(cons*ests)
	}
	return result, abserr, 25, resabs, resasc
}

// qawoe integrates f(x)·cos(ωx) or f(x)·sin(ωx) over a finite interval
// (dqawoe). icall > 1 reuses the moments already in mc (dqawfe's cycles).
func qawoe(f func(float64) float64, a, b, omega float64, integr int, epsabs, epsrel float64, limit, icall int, mc *momentCache) qout {
	if tolInvalid(epsabs, epsrel) {
		return qout{ier: 6}
	}
	w := newWorkspace(limit)
	nnlog := w.level
	domega := math.Abs(omega)
	nrmom := 0
	if icall <= 1 {
		mc.momcom = 0
	}
	result, abserr, neval, defabs, _ := mc.qc25f(f, a, b, domega, integr, nrmom, 0)
	dres := math.Abs(result)
	errbnd := math.Max(epsabs, epsrel*dres)
	w.alist[1], w.blist[1] = a, b
	w.rlist[1] = result
	w.elist[1] = abserr
	w.iord[1] = 1
	last := 1
	ier := 0
	if abserr <= 100*epmach*defabs && abserr > errbnd {
		ier = 2
	}
	if limit == 1 {
		ier = 1
	}
	sign := func(r float64) float64 {
		if integr == 2 && omega < 0 {
			return -r
		}
		return r
	}
	if ier != 0 || abserr <= errbnd {
		return qout{result: sign(result), abserr: abserr, neval: neval, ier: ier, last: last}
	}
	var et epsTable
	errmax := abserr
	maxerr := 1
	area := result
	errsum := abserr
	abserr = oflow
	nrmax := 1
	extrap, noext := false, false
	ierro := 0
	iroff1, iroff2, iroff3 := 0, 0, 0
	ktmin := 0
	small := math.Abs(b-a) * 0.75
	numrl2 := 0
	extall := false
	if 0.5*math.Abs(b-a)*domega <= 2 {
		numrl2 = 1
		extall = true
		et.tab[1] = result
	}
	if 0.25*math.Abs(b-a)*domega <= 2 {
		extall = true
	}
	ksgn := -1
	if dres >= (1-50*epmach)*defabs {
		ksgn = 1
	}
	var erlarg, ertest, correc float64
	converged := false
	for last = 2; last <= limit; last++ {
		nrmom = nnlog[maxerr] + 1
		a1 := w.alist[maxerr]
		b1 := 0.5 * (w.alist[maxerr] + w.blist[maxerr])
		a2 := b1
		b2 := w.blist[maxerr]
		erlast := errmax
		area1, error1, nev, _, defab1 := mc.qc25f(f, a1, b1, domega, integr, nrmom, 0)
		neval += nev
		area2, error2, nev, _, defab2 := mc.qc25f(f, a2, b2, domega, integr, nrmom, 1)
		neval += nev
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
		nnlog[maxerr] = nrmom
		nnlog[last] = nrmom
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
		if last == 2 && extall {
			small *= 0.5
			numrl2++
			et.tab[numrl2] = area
			ertest = errbnd
			erlarg = errsum
			continue
		}
		if noext {
			continue
		}
		if extall {
			erlarg -= erlast
			if math.Abs(b1-a1) > small {
				erlarg += erro12
			}
		}
		if !(extall && extrap) {
			// Label 50: test whether the interval to be bisected next is
			// the smallest interval.
			width := math.Abs(w.blist[maxerr] - w.alist[maxerr])
			if width > small {
				continue
			}
			if !extall {
				// Test whether we can start with the extrapolation
				// procedure (we do this if we integrate over the next
				// interval with use of a Gauss–Kronrod rule).
				small *= 0.5
				if 0.25*width*domega > 2 {
					continue
				}
				extall = true
				ertest = errbnd
				erlarg = errsum
				continue
			}
			extrap = true
			nrmax = 2
		}
		if ierro != 3 && erlarg > ertest {
			jupbnd := last
			if last > limit/2+2 {
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
		if numrl2 >= 3 {
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
		}
		maxerr = w.iord[1]
		errmax = w.elist[maxerr]
		nrmax = 1
		extrap = false
		small *= 0.5
		erlarg = errsum
	}
	if converged || abserr == oflow || et.nres == 0 {
		result = w.sum(last)
		abserr = errsum
	} else {
		result, abserr, ier = extrapTail(w, last, result, abserr, area, errsum, correc, defabs, ier, ierro, ksgn, true)
	}
	return qout{result: sign(result), abserr: abserr, neval: neval, ier: ierShift(ier), last: last}
}

// qawfe computes the Fourier integral ∫_a^∞ f(x)·cos(ωx) dx (integr 1) or
// with sin (integr 2) by integrating over successive cycles with qawoe and
// accelerating the series of cycle contributions with the epsilon algorithm
// (dqawfe). ierlst receives the per-cycle diagnostics.
func qawfe(f func(float64) float64, a, omega float64, integr int, epsabs float64, limlst, limit, maxp1 int) (qout, []int) {
	if epsabs <= 0 || limlst < 3 {
		return qout{ier: 6}, nil
	}
	if omega == 0 {
		// The QUADPACK source passes 0 as the lower bound here; the
		// integral over (a, ∞) is what is meant.
		if integr == 1 {
			r := qagie(f, a, 1, epsabs, 0, limit)
			return r, []int{r.ier}
		}
		return qout{last: 1}, []int{0}
	}
	const p = 0.9
	l := math.Trunc(math.Abs(omega))
	dl := 2*l + 1
	cycle := dl * math.Pi / math.Abs(omega)
	ier := 0
	ktmin := 0
	neval := 0
	numrl2 := 0
	c1 := a
	c2 := cycle + a
	p1 := 1 - p
	eps := epsabs
	if epsabs > uflow/p1 {
		eps = epsabs * p1
	}
	ep := eps
	fact := 1.0
	var correc, abserr, errsum, result, drl float64
	var et epsTable
	psum := &et.tab
	ierlst := make([]int, 0, limlst)
	mc := newMomentCache(maxp1)
	ll := 0
	useSum := false
	done := false
	for lst := 1; lst <= limlst && !done; lst++ {
		epsa := eps * fact
		r := qawoe(f, c1, c2, omega, integr, epsa, 0, limit, lst, mc)
		ierlst = append(ierlst, r.ier)
		neval += r.neval
		fact *= p
		errsum += r.abserr
		drl = 50 * math.Abs(r.result)
		if errsum+drl <= epsabs && lst >= 6 {
			useSum = true
			break
		}
		correc = math.Max(correc, r.abserr)
		if r.ier != 0 {
			eps = math.Max(ep, correc*p1)
			ier = 7
		}
		if ier == 7 && errsum+drl <= correc*10 && lst > 5 {
			useSum = true
			break
		}
		numrl2++
		if lst == 1 {
			psum[1] = r.result
		} else {
			psum[numrl2] = psum[ll] + r.result
			if lst > 2 {
				if lst == limlst {
					ier = 1
				}
				reseps, abseps := et.qelg(&numrl2)
				ktmin++
				if ktmin >= 15 && abserr <= 1e-3*(errsum+drl) {
					ier = 4
				}
				if abseps <= abserr || lst == 3 {
					abserr = abseps
					result = reseps
					ktmin = 0
					if abserr+10*correc <= epsabs || (abserr <= epsabs && 10*correc >= epsabs) {
						done = true
					}
				}
				if !done && ier != 0 && ier != 7 {
					done = true
				}
			}
		}
		ll = numrl2
		c1 = c2
		c2 += cycle
	}
	if !useSum {
		abserr, useSum = qawfeTail(abserr, correc, result, psum[numrl2], errsum, drl, ier)
	}
	if useSum {
		result = psum[numrl2]
		abserr = errsum + drl
	}
	return qout{result: result, abserr: abserr, neval: neval, ier: ier, last: len(ierlst)}, ierlst
}

// qawfeTail is dqawfe's label 60: it adds the cycle corrections to the
// extrapolated error and decides whether the plain partial sum ps is the
// better answer.
func qawfeTail(abserr, correc, result, ps, errsum, drl float64, ier int) (float64, bool) {
	abserr += 10 * correc
	if ier == 0 {
		return abserr, false
	}
	switch {
	case !(result != 0 && ps != 0) && abserr > errsum:
		return abserr, true
	case ps == 0:
		return abserr, false
	case abserr/math.Abs(result) > (errsum+drl)/math.Abs(ps):
		return abserr, true
	case ier != 7:
		abserr += drl
	}
	return abserr, false
}

// qc25c applies the 25-point Clenshaw–Curtis rule with the Cauchy weight
// 1/(x-c) on [a, b], or the 15-point Gauss–Kronrod rule on f(x)/(x-c) when c
// is far from the interval (dqc25c). It returns whether the Gauss–Kronrod rule
// was used with a non-degenerate error estimate (QUADPACK's krul decrement).
func qc25c(f func(float64) float64, a, b, c float64) (result, abserr float64, neval int, gk bool) {
	cc := (2*c - b - a) / (b - a)
	if math.Abs(cc) >= 1.1 {
		var resasc float64
		result, abserr, _, resasc = qk15w(f, func(x float64) float64 { return 1 / (x - c) }, a, b)
		return result, abserr, 15, resasc != abserr
	}
	hlgth := 0.5 * (b - a)
	centr := 0.5 * (b + a)
	var fval [26]float64
	var cheb12 [14]float64
	var cheb24 [26]float64
	sample25(f, centr, hlgth, &fval)
	qcheb(&fval, &cheb12, &cheb24)
	amom0 := math.Log(math.Abs((1 - cc) / (1 + cc)))
	amom1 := 2 + cc*amom0
	res12 := cheb12[1]*amom0 + cheb12[2]*amom1
	res24 := cheb24[1]*amom0 + cheb24[2]*amom1
	for k := 3; k <= 25; k++ {
		amom2 := 2*cc*amom1 - amom0
		ak22 := float64((k - 2) * (k - 2))
		if k%2 == 0 {
			amom2 -= 4 / (ak22 - 1)
		}
		if k <= 13 {
			res12 += cheb12[k] * amom2
		}
		res24 += cheb24[k] * amom2
		amom0 = amom1
		amom1 = amom2
	}
	return res24, math.Abs(res24 - res12), 25, false
}

// qawce computes the Cauchy principal value of ∫_a^b f(x)/(x-c) dx (dqawce)
// for a < b (Quad orders the limits, so dqawce's swap is not needed) and
// c ≠ a, b (checked by Quad).
func qawce(f func(float64) float64, a, b, c, epsabs, epsrel float64, limit int) qout {
	if tolInvalid(epsabs, epsrel) {
		return qout{ier: 6}
	}
	w := newWorkspace(limit)
	result, abserr, neval, _ := qc25c(f, a, b, c)
	last := 1
	ier := 0
	errbnd := math.Max(epsabs, epsrel*math.Abs(result))
	if limit == 1 {
		ier = 1
	}
	if abserr < math.Min(0.01*math.Abs(result), errbnd) || ier == 1 {
		return qout{result: result, abserr: abserr, neval: neval, ier: ier, last: last}
	}
	w.alist[1], w.blist[1] = a, b
	w.rlist[1] = result
	w.elist[1] = abserr
	w.iord[1] = 1
	errmax := abserr
	maxerr := 1
	area := result
	errsum := abserr
	nrmax := 1
	iroff1, iroff2 := 0, 0
	for last = 2; last <= limit; last++ {
		a1 := w.alist[maxerr]
		b1 := 0.5 * (w.alist[maxerr] + w.blist[maxerr])
		b2 := w.blist[maxerr]
		if c <= b1 && c > a1 {
			b1 = 0.5 * (c + b2)
		}
		if c > b1 && c < b2 {
			b1 = 0.5 * (a1 + c)
		}
		a2 := b1
		area1, error1, nev, gk1 := qc25c(f, a1, b1, c)
		neval += nev
		area2, error2, nev, gk2 := qc25c(f, a2, b2, c)
		neval += nev
		krule0 := gk1 && gk2
		area12 := area1 + area2
		erro12 := error1 + error2
		errsum = errsum + erro12 - errmax
		area = area + area12 - w.rlist[maxerr]
		if math.Abs(w.rlist[maxerr]-area12) < 1e-5*math.Abs(area12) && erro12 >= 0.99*errmax && krule0 {
			iroff1++
		}
		if last > 10 && erro12 > errmax && krule0 {
			iroff2++
		}
		errbnd = math.Max(epsabs, epsrel*math.Abs(area))
		if errsum > errbnd {
			if iroff1 >= 6 && iroff2 > 20 {
				ier = 2
			}
			if last == limit {
				ier = 1
			}
			if tooSmall(a1, a2, b2) {
				ier = 3
			}
		}
		w.split(maxerr, last, a1, b1, a2, b2, area1, area2, error1, error2)
		qpsrt(limit, last, &maxerr, &errmax, w.elist, w.iord, &nrmax)
		if ier != 0 || errsum <= errbnd {
			break
		}
	}
	return qout{result: w.sum(last), abserr: errsum, neval: neval, ier: ier, last: last}
}

// algMoments holds the modified Chebyshev moments of dqmomo for the weight
// (x-a)^α (b-x)^β and its logarithmic variants.
type algMoments struct {
	ri, rj, rg, rh [26]float64
}

// qmomo computes the moments of dqmomo.
func qmomo(alfa, beta float64, integr int) *algMoments {
	m := &algMoments{}
	ri, rj, rg, rh := &m.ri, &m.rj, &m.rg, &m.rh
	alfp1 := alfa + 1
	betp1 := beta + 1
	alfp2 := alfa + 2
	betp2 := beta + 2
	ralf := math.Pow(2, alfp1)
	rbet := math.Pow(2, betp1)
	ri[1] = ralf / alfp1
	rj[1] = rbet / betp1
	ri[2] = ri[1] * alfa / alfp2
	rj[2] = rj[1] * beta / betp2
	an := 2.0
	anm1 := 1.0
	for i := 3; i <= 25; i++ {
		ri[i] = -(ralf + an*(an-alfp2)*ri[i-1]) / (anm1 * (an + alfp1))
		rj[i] = -(rbet + an*(an-betp2)*rj[i-1]) / (anm1 * (an + betp1))
		anm1 = an
		an++
	}
	if integr == 2 || integr == 4 {
		rg[1] = -ri[1] / alfp1
		rg[2] = -(ralf+ralf)/(alfp2*alfp2) - rg[1]
		an = 2
		anm1 = 1
		im1 := 2
		for i := 3; i <= 25; i++ {
			rg[i] = -(an*(an-alfp2)*rg[im1] - an*ri[im1] + anm1*ri[i]) / (anm1 * (an + alfp1))
			anm1 = an
			an++
			im1 = i
		}
	}
	if integr == 3 || integr == 4 {
		rh[1] = -rj[1] / betp1
		rh[2] = -(rbet+rbet)/(betp2*betp2) - rh[1]
		an = 2
		anm1 = 1
		im1 := 2
		for i := 3; i <= 25; i++ {
			rh[i] = -(an*(an-betp2)*rh[im1] - an*rj[im1] + anm1*rj[i]) / (anm1 * (an + betp1))
			anm1 = an
			an++
			im1 = i
		}
		for i := 2; i <= 25; i += 2 {
			rh[i] = -rh[i]
		}
	}
	for i := 2; i <= 25; i += 2 {
		rj[i] = -rj[i]
	}
	return m
}

// algWeight is dqwgts: (x-a)^α (b-x)^β times the logarithms selected by
// integr.
func algWeight(x, a, b, alfa, beta float64, integr int) float64 {
	xma := x - a
	bmx := b - x
	w := math.Pow(xma, alfa) * math.Pow(bmx, beta)
	switch integr {
	case 2:
		w *= math.Log(xma)
	case 3:
		w *= math.Log(bmx)
	case 4:
		w *= math.Log(xma) * math.Log(bmx)
	}
	return w
}

// qc25s integrates f·w over [bl, br] ⊂ [a, b] for the algebraic-logarithmic
// weight of dqawse: with modified Clenshaw–Curtis moments on a subinterval
// touching a singular end point, with 15-point Gauss–Kronrod elsewhere
// (dqc25s). The netlib source accumulates "res24 = res12 + ..." in one loop
// of the integr=2 branch; the intended "res24 = res24 + ..." is used here.
func qc25s(f func(float64) float64, a, b, bl, br, alfa, beta float64, mo *algMoments, integr int) (result, abserr, resasc float64, nev int) {
	leftSing := bl == a && (alfa != 0 || integr == 2 || integr == 4)
	rightSing := br == b && (beta != 0 || integr == 3 || integr == 4)
	if !leftSing && !rightSing {
		result, abserr, _, resasc = qk15w(f, func(x float64) float64 {
			return algWeight(x, a, b, alfa, beta, integr)
		}, bl, br)
		return result, abserr, resasc, 15
	}
	hlgth := 0.5 * (br - bl)
	centr := 0.5 * (br + bl)
	var fval [26]float64
	var cheb12 [14]float64
	var cheb24 [26]float64
	var factor float64
	var plain, logged *[26]float64 // moments without/with the log factor
	var withLog bool               // the log at the singular end is present
	if leftSing {
		// (x-a)^α is expanded by the moments; (b-x)^β and log(b-x) are
		// smooth on this subinterval and go into the samples.
		sampleSing(f, centr, hlgth, b-centr, -1, beta, integr > 2, &fval)
		factor = math.Pow(hlgth, alfa+1)
		plain, logged = &mo.ri, &mo.rg
		withLog = integr == 2 || integr == 4
	} else {
		sampleSing(f, centr, hlgth, centr-a, 1, alfa, integr == 2 || integr == 4, &fval)
		factor = math.Pow(hlgth, beta+1)
		plain, logged = &mo.rj, &mo.rh
		withLog = integr == 3 || integr == 4
	}
	qcheb(&fval, &cheb12, &cheb24)
	res12, res24 := chebSums(&cheb12, &cheb24, plain)
	if withLog {
		dc := math.Log(br - bl)
		result = res24 * dc
		abserr = math.Abs((res24 - res12) * dc)
		res12, res24 = chebSums(&cheb12, &cheb24, logged)
	}
	result = (result + res24) * factor
	abserr = (abserr + math.Abs(res24-res12)) * factor
	return result, abserr, 0, 25
}

// sampleSing fills fval like sample25 with f(x)·arg^p (times log(arg) when
// withLog), where arg = fix + s·(x - centr) is the distance to the regular end
// point, computed as dqc25s computes it.
func sampleSing(f func(float64) float64, centr, hlgth, fix, s, p float64, withLog bool, fval *[26]float64) {
	fval[1] = 0.5 * f(centr+hlgth) * math.Pow(fix+s*hlgth, p)
	fval[13] = f(centr) * math.Pow(fix, p)
	fval[25] = 0.5 * f(centr-hlgth) * math.Pow(fix-s*hlgth, p)
	for i := 2; i <= 12; i++ {
		u := hlgth * chebX[i-1]
		fval[i] = f(u+centr) * math.Pow(fix+s*u, p)
		fval[26-i] = f(centr-u) * math.Pow(fix-s*u, p)
	}
	if withLog {
		fval[1] *= math.Log(fix + s*hlgth)
		fval[13] *= math.Log(fix)
		fval[25] *= math.Log(fix - s*hlgth)
		for i := 2; i <= 12; i++ {
			u := hlgth * chebX[i-1]
			fval[i] *= math.Log(fix + s*u)
			fval[26-i] *= math.Log(fix - s*u)
		}
	}
}

// qawse integrates f(x)·w(x) over [a, b] for the algebraic-logarithmic
// end-point weights w (dqawse); integr selects 1: (x-a)^α (b-x)^β,
// 2: ·log(x-a), 3: ·log(b-x), 4: ·log(x-a)·log(b-x).
func qawse(f func(float64) float64, a, b, alfa, beta float64, integr int, epsabs, epsrel float64, limit int) qout {
	// b > a, α, β > -1 and limit ≥ 2 are checked by Quad.
	if epsabs == 0 && epsrel < math.Max(50*epmach, 0.5e-28) {
		return qout{ier: 6}
	}
	mo := qmomo(alfa, beta, integr)
	w := newWorkspace(limit)
	centre := 0.5 * (b + a)
	area1, error1, resas1, nev := qc25s(f, a, b, a, centre, alfa, beta, mo, integr)
	neval := nev
	area2, error2, resas2, nev := qc25s(f, a, b, centre, b, alfa, beta, mo, integr)
	neval += nev
	_, _ = resas1, resas2
	last := 2
	result := area1 + area2
	abserr := error1 + error2
	errbnd := math.Max(epsabs, epsrel*math.Abs(result))
	w.alist[1], w.blist[1] = a, b
	w.split(1, 2, a, centre, centre, b, area1, area2, error1, error2)
	w.iord[1] = 1
	w.iord[2] = 2
	ier := 0
	if limit == 2 {
		ier = 1
	}
	if abserr <= errbnd || ier == 1 {
		return qout{result: result, abserr: abserr, neval: neval, ier: ier, last: last}
	}
	errmax := w.elist[1]
	maxerr := 1
	nrmax := 1
	area := result
	errsum := abserr
	iroff1, iroff2 := 0, 0
	for last = 3; last <= limit; last++ {
		a1 := w.alist[maxerr]
		b1 := 0.5 * (w.alist[maxerr] + w.blist[maxerr])
		a2 := b1
		b2 := w.blist[maxerr]
		area1, error1, resas1, nev := qc25s(f, a, b, a1, b1, alfa, beta, mo, integr)
		neval += nev
		area2, error2, resas2, nev := qc25s(f, a, b, a2, b2, alfa, beta, mo, integr)
		neval += nev
		area12 := area1 + area2
		erro12 := error1 + error2
		errsum = errsum + erro12 - errmax
		area = area + area12 - w.rlist[maxerr]
		if a != a1 && b != b2 && resas1 != error1 && resas2 != error2 {
			if math.Abs(w.rlist[maxerr]-area12) < 1e-5*math.Abs(area12) && erro12 >= 0.99*errmax {
				iroff1++
			}
			if last > 10 && erro12 > errmax {
				iroff2++
			}
		}
		errbnd = math.Max(epsabs, epsrel*math.Abs(area))
		if errsum > errbnd {
			if last == limit {
				ier = 1
			}
			if iroff1 >= 6 || iroff2 >= 20 {
				ier = 2
			}
			if tooSmall(a1, a2, b2) {
				ier = 3
			}
		}
		w.split(maxerr, last, a1, b1, a2, b2, area1, area2, error1, error2)
		qpsrt(limit, last, &maxerr, &errmax, w.elist, w.iord, &nrmax)
		if ier != 0 || errsum <= errbnd {
			break
		}
	}
	return qout{result: w.sum(last), abserr: errsum, neval: neval, ier: ier, last: last}
}
