// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// ddPow is x**y correctly rounded in all but astronomically rare cases, from
// double-double arithmetic (about 100 bits). It is what the Exact LSODA uses
// for the step-ratio powers: lsoda.c calls the C library's pow, which is
// within an ulp of the correctly rounded value, where Go's math.Pow is not, and
// ODEPACK's step selection turns a last-bit difference into a different step
// sequence (docs/design.md); with ddPow the two agree to the bit.

type dd struct{ hi, lo float64 }

func twoSum(a, b float64) dd {
	s := a + b
	bb := s - a
	return dd{s, (a - (s - bb)) + (b - bb)}
}

func ddAdd(a, b dd) dd {
	s := twoSum(a.hi, b.hi)
	t := twoSum(a.lo, b.lo)
	s.lo += t.hi
	s = twoSum(s.hi, s.lo)
	s.lo += t.lo
	return twoSum(s.hi, s.lo)
}

func ddMul(a, b dd) dd {
	// The explicit conversions keep the compiler from fusing the product into
	// the sum of twoSum (arm64 does, across the inlined call), which would
	// break the error-free transformation.
	p := float64(a.hi * b.hi)
	e := math.FMA(a.hi, b.hi, -p)
	e += float64(a.hi*b.lo) + float64(a.lo*b.hi)
	return twoSum(p, e)
}

// ddDivF is a/f to double-double precision: the quotient of the high part and
// that of the remainder.
func ddDivF(a dd, f float64) dd {
	q := a.hi / f
	r := math.FMA(-q, f, a.hi) + a.lo
	return twoSum(q, r/f)
}

func ddMulF(a dd, f float64) dd { return ddMul(a, dd{f, 0}) }

var ddLn2 = dd{0.6931471805599453, 2.3190468138462996e-17}

// ddExp is exp(a): a = k ln2 + r, exp(r/256) by its Taylor series, squared 8
// times.
func ddExp(a dd) dd {
	k := math.Round(a.hi / ddLn2.hi)
	r := ddAdd(a, ddMulF(ddLn2, -k))
	r = dd{r.hi / 256, r.lo / 256}
	sum := dd{1, 0}
	term := dd{1, 0}
	for i := 1; i <= 12; i++ {
		term = ddMul(term, r)
		term = ddDivF(term, float64(i))
		sum = ddAdd(sum, term)
	}
	for i := 0; i < 8; i++ {
		sum = ddMul(sum, sum)
	}
	return dd{math.Ldexp(sum.hi, int(k)), math.Ldexp(sum.lo, int(k))}
}

// ddLog is log x for x > 0 (one Newton step from math.Log).
func ddLog(x float64) dd {
	l0 := math.Log(x)
	t := ddMul(ddExp(dd{-l0, 0}), dd{x, 0}) // x e^{-l0} = 1 + u
	u := ddAdd(t, dd{-1, 0})
	u2 := ddMul(u, u)
	corr := ddAdd(u, dd{-u2.hi / 2, -u2.lo / 2})
	return ddAdd(dd{l0, 0}, corr)
}

func ddPow(x, y float64) float64 {
	if !(x > 0) || math.IsInf(x, 0) || y == 0 || x == 1 || math.IsNaN(y) || math.IsInf(y, 0) {
		return math.Pow(x, y)
	}
	z := ddMulF(ddLog(x), y)
	if math.Abs(z.hi) > 700 {
		return math.Pow(x, y)
	}
	e := ddExp(z)
	return e.hi + e.lo
}

// powRecip is x**e for e = fl(1/k), k = 1..16, which is every exponent of
// lsoda's step ratios, correctly rounded like ddPow but much cheaper: the k-th
// root of x refined by one Newton step on the exact (double-double) k-th power
// of math.Pow's estimate, times the factor x**(e-1/k) = 1 + (e-1/k) log x that
// the rounding of e to a double changes it by. Anything else goes to ddPow.
func powRecip(x, e float64) float64 {
	if !(x > 0) || math.IsInf(x, 0) || !(e > 0 && e <= 1) {
		return math.Pow(x, e)
	}
	k := math.Round(1 / e)
	if k < 1 || k > 16 || 1/k != e {
		return ddPow(x, e)
	}
	if k == 1 {
		return x
	}
	if k == 2 {
		return math.Sqrt(x) // fl(1/2) is exact and sqrt is correctly rounded
	}
	r := math.Pow(x, e)
	rk1 := dd{1, 0}
	for i := 1; i < int(k); i++ {
		rk1 = ddMulF(rk1, r)
	}
	p := ddMulF(rk1, r) // r^k
	res := ddAdd(p, dd{-x, 0})
	c := res.hi * r / (k * p.hi) // Newton: r - (r^k - x) / (k r^(k-1))
	root := twoSum(r, -c)
	// e - 1/k, with 1/k = q + qlo: e = q, so e - 1/k = -qlo
	qlo := math.FMA(-e, k, 1) / k
	root.lo += root.hi * (-math.Log(x) * qlo)
	root = twoSum(root.hi, root.lo)
	return root.hi + root.lo
}
