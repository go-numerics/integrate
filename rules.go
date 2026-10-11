// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "math"

// Gauss–Kronrod 21-point nodes and weights (dqk21). xgk21[1], xgk21[3], ...
// are the 10-point Gauss nodes; the Kronrod nodes interleave them.
var (
	xgk21 = [11]float64{
		0.995657163025808080735527280689003,
		0.973906528517171720077964012084452,
		0.930157491355708226001207180059508,
		0.865063366688984510732096688423493,
		0.780817726586416897063717578345042,
		0.679409568299024406234327365114874,
		0.562757134668604683339000099272694,
		0.433395394129247190799265943165784,
		0.294392862701460198131126603103866,
		0.148874338981631210884826001129720,
		0,
	}
	wgk21 = [11]float64{
		0.011694638867371874278064396062192,
		0.032558162307964727478818972459390,
		0.054755896574351996031381300244580,
		0.075039674810919952767043140916190,
		0.093125454583697605535065465083366,
		0.109387158802297641899210590325805,
		0.123491976262065851077208980223051,
		0.134709217311473325928054001771707,
		0.142775938577060080797094273138717,
		0.147739104901338491374841515972068,
		0.149445554002916905664936468389821,
	}
	wg10 = [5]float64{
		0.066671344308688137593568809893332,
		0.149451349150580593145776339657697,
		0.219086362515982043995534934228163,
		0.269266719309996355091226921569469,
		0.295524224714752870173892994651338,
	}
)

// Gauss–Kronrod 15-point rule (dqk15i, dqk15w). xgk15[1], xgk15[3], xgk15[5]
// and the centre are the 7-point Gauss nodes.
var (
	xgk15 = [8]float64{
		0.991455371120812639206854697526329,
		0.949107912342758524526189684047851,
		0.864864423359769072789712788640926,
		0.741531185599394439863864773280788,
		0.586087235467691130294144845693013,
		0.405845151377397166906606412076961,
		0.207784955007898467600689403773245,
		0,
	}
	wgk15 = [8]float64{
		0.022935322010529224963732008058970,
		0.063092092629978553290700663189204,
		0.104790010322250183839876322541518,
		0.140653259715525918745189590510238,
		0.169004726639267902826583426598550,
		0.190350578064785409913256402421014,
		0.204432940075298892414161999234649,
		0.209482141084727828012999174891714,
	}
	// wg7i is the 7-point Gauss weight vector laid over the 15 Kronrod
	// nodes (zero where the node is Kronrod-only), as dqk15i stores it.
	wg7i = [8]float64{
		0, 0.129484966168869693270611432679082,
		0, 0.279705391489276667901467771423780,
		0, 0.381830050505118944950369775488975,
		0, 0.417959183673469387755102040816327,
	}
)

// scaleErr applies QUADPACK's error-estimate transformation shared by every
// Gauss–Kronrod rule.
func scaleErr(abserr, resabs, resasc float64) float64 {
	if resasc != 0 && abserr != 0 {
		abserr = resasc * math.Min(1, math.Pow(200*abserr/resasc, 1.5))
	}
	if resabs > uflow/(50*epmach) {
		abserr = math.Max(epmach*50*resabs, abserr)
	}
	return abserr
}

// qk21 applies the 21-point Gauss–Kronrod rule on [a, b] (dqk21).
func qk21(f func(float64) float64, a, b float64) (result, abserr, resabs, resasc float64) {
	var fv1, fv2 [10]float64
	centr := 0.5 * (a + b)
	hlgth := 0.5 * (b - a)
	dhlgth := math.Abs(hlgth)
	resg := 0.0
	fc := f(centr)
	resk := wgk21[10] * fc
	resabs = math.Abs(resk)
	for j := 1; j <= 5; j++ {
		jtw := 2*j - 1 // Fortran index 2j, zero-based
		absc := hlgth * xgk21[jtw]
		fval1 := f(centr - absc)
		fval2 := f(centr + absc)
		fv1[jtw] = fval1
		fv2[jtw] = fval2
		fsum := fval1 + fval2
		resg += wg10[j-1] * fsum
		resk += wgk21[jtw] * fsum
		resabs += wgk21[jtw] * (math.Abs(fval1) + math.Abs(fval2))
	}
	for j := 1; j <= 5; j++ {
		jtwm1 := 2*j - 2 // Fortran index 2j-1, zero-based
		absc := hlgth * xgk21[jtwm1]
		fval1 := f(centr - absc)
		fval2 := f(centr + absc)
		fv1[jtwm1] = fval1
		fv2[jtwm1] = fval2
		fsum := fval1 + fval2
		resk += wgk21[jtwm1] * fsum
		resabs += wgk21[jtwm1] * (math.Abs(fval1) + math.Abs(fval2))
	}
	reskh := resk * 0.5
	resasc = wgk21[10] * math.Abs(fc-reskh)
	for j := 0; j < 10; j++ {
		resasc += wgk21[j] * (math.Abs(fv1[j]-reskh) + math.Abs(fv2[j]-reskh))
	}
	result = resk * hlgth
	resabs *= dhlgth
	resasc *= dhlgth
	abserr = scaleErr(math.Abs((resk-resg)*hlgth), resabs, resasc)
	return result, abserr, resabs, resasc
}

// qk15i applies the 15-point rule to the transformed integrand of an
// infinite range, on the subinterval [a, b] of (0, 1] (dqk15i).
func qk15i(f func(float64) float64, boun float64, inf int, a, b float64) (result, abserr, resabs, resasc float64) {
	var fv1, fv2 [7]float64
	dinf := float64(min(1, inf))
	centr := 0.5 * (a + b)
	hlgth := 0.5 * (b - a)
	tabsc1 := boun + dinf*(1-centr)/centr
	fval1 := f(tabsc1)
	if inf == 2 {
		fval1 += f(-tabsc1)
	}
	fc := (fval1 / centr) / centr
	resg := wg7i[7] * fc
	resk := wgk15[7] * fc
	resabs = math.Abs(resk)
	for j := 0; j < 7; j++ {
		absc := hlgth * xgk15[j]
		absc1 := centr - absc
		absc2 := centr + absc
		t1 := boun + dinf*(1-absc1)/absc1
		t2 := boun + dinf*(1-absc2)/absc2
		v1 := f(t1)
		v2 := f(t2)
		if inf == 2 {
			v1 += f(-t1)
			v2 += f(-t2)
		}
		v1 = (v1 / absc1) / absc1
		v2 = (v2 / absc2) / absc2
		fv1[j] = v1
		fv2[j] = v2
		fsum := v1 + v2
		resg += wg7i[j] * fsum
		resk += wgk15[j] * fsum
		resabs += wgk15[j] * (math.Abs(v1) + math.Abs(v2))
	}
	reskh := resk * 0.5
	resasc = wgk15[7] * math.Abs(fc-reskh)
	for j := 0; j < 7; j++ {
		resasc += wgk15[j] * (math.Abs(fv1[j]-reskh) + math.Abs(fv2[j]-reskh))
	}
	result = resk * hlgth
	resasc *= hlgth
	resabs *= hlgth
	abserr = scaleErr(math.Abs((resk-resg)*hlgth), resabs, resasc)
	return result, abserr, resabs, resasc
}

// qk15w applies the 15-point rule to f(x)·w(x) on [a, b] (dqk15w).
func qk15w(f, w func(float64) float64, a, b float64) (result, abserr, resabs, resasc float64) {
	var fv1, fv2 [7]float64
	centr := 0.5 * (a + b)
	hlgth := 0.5 * (b - a)
	dhlgth := math.Abs(hlgth)
	fc := f(centr) * w(centr)
	resg := wg7i[7] * fc
	resk := wgk15[7] * fc
	resabs = math.Abs(resk)
	for j := 0; j < 7; j++ {
		absc := hlgth * xgk15[j]
		absc1 := centr - absc
		absc2 := centr + absc
		v1 := f(absc1) * w(absc1)
		v2 := f(absc2) * w(absc2)
		fv1[j] = v1
		fv2[j] = v2
		fsum := v1 + v2
		resg += wg7i[j] * fsum
		resk += wgk15[j] * fsum
		resabs += wgk15[j] * (math.Abs(v1) + math.Abs(v2))
	}
	reskh := resk * 0.5
	resasc = wgk15[7] * math.Abs(fc-reskh)
	for j := 0; j < 7; j++ {
		resasc += wgk15[j] * (math.Abs(fv1[j]-reskh) + math.Abs(fv2[j]-reskh))
	}
	result = resk * hlgth
	resabs *= dhlgth
	resasc *= dhlgth
	abserr = scaleErr(math.Abs((resk-resg)*hlgth), resabs, resasc)
	return result, abserr, resabs, resasc
}

// chebX holds cos(k·π/24), k = 1..11 (1-based), the abscissae of the
// Clenshaw–Curtis rules of dqc25c, dqc25f and dqc25s.
var chebX = [12]float64{0,
	0.991444861373810411144557526928563,
	0.965925826289068286749743199728897,
	0.923879532511286756128183189396788,
	0.866025403784438646763723170752936,
	0.793353340291235164579776961501299,
	0.707106781186547524400844362104849,
	0.608761429008720639416097542898164,
	0.5,
	0.382683432365089771728459984030399,
	0.258819045102520762348898837624048,
	0.130526192220051591548406227895489,
}

// sample25 fills fval (1-based, 25 entries) with g at the 25
// Chebyshev–Lobatto points of [centr-hlgth, centr+hlgth], end values halved,
// in dqc25*'s order.
func sample25(g func(float64) float64, centr, hlgth float64, fval *[26]float64) {
	fval[1] = 0.5 * g(centr+hlgth)
	fval[13] = g(centr)
	fval[25] = 0.5 * g(centr-hlgth)
	for i := 2; i <= 12; i++ {
		u := hlgth * chebX[i-1]
		fval[i] = g(centr + u)
		fval[26-i] = g(centr - u)
	}
}

// qcheb computes the Chebyshev series coefficients of the 12th and 24th
// degree interpolants from the 25 samples (dqcheb). fval is overwritten.
func qcheb(fval *[26]float64, cheb12 *[14]float64, cheb24 *[26]float64) {
	x := &chebX
	var v [13]float64
	for i := 1; i <= 12; i++ {
		j := 26 - i
		v[i] = fval[i] - fval[j]
		fval[i] += fval[j]
	}
	alam1 := v[1] - v[9]
	alam2 := x[6] * (v[3] - v[7] - v[11])
	cheb12[4] = alam1 + alam2
	cheb12[10] = alam1 - alam2
	alam1 = v[2] - v[8] - v[10]
	alam2 = v[4] - v[6] - v[12]
	alam := x[3]*alam1 + x[9]*alam2
	cheb24[4] = cheb12[4] + alam
	cheb24[22] = cheb12[4] - alam
	alam = x[9]*alam1 - x[3]*alam2
	cheb24[10] = cheb12[10] + alam
	cheb24[16] = cheb12[10] - alam
	part1 := x[4] * v[5]
	part2 := x[8] * v[9]
	part3 := x[6] * v[7]
	alam1 = v[1] + part1 + part2
	alam2 = x[2]*v[3] + part3 + x[10]*v[11]
	cheb12[2] = alam1 + alam2
	cheb12[12] = alam1 - alam2
	alam = x[1]*v[2] + x[3]*v[4] + x[5]*v[6] + x[7]*v[8] + x[9]*v[10] + x[11]*v[12]
	cheb24[2] = cheb12[2] + alam
	cheb24[24] = cheb12[2] - alam
	alam = x[11]*v[2] - x[9]*v[4] + x[7]*v[6] - x[5]*v[8] + x[3]*v[10] - x[1]*v[12]
	cheb24[12] = cheb12[12] + alam
	cheb24[14] = cheb12[12] - alam
	alam1 = v[1] - part1 + part2
	alam2 = x[10]*v[3] - part3 + x[2]*v[11]
	cheb12[6] = alam1 + alam2
	cheb12[8] = alam1 - alam2
	alam = x[5]*v[2] - x[9]*v[4] - x[1]*v[6] - x[11]*v[8] + x[3]*v[10] + x[7]*v[12]
	cheb24[6] = cheb12[6] + alam
	cheb24[20] = cheb12[6] - alam
	alam = x[7]*v[2] - x[3]*v[4] - x[11]*v[6] + x[1]*v[8] - x[9]*v[10] - x[5]*v[12]
	cheb24[8] = cheb12[8] + alam
	cheb24[18] = cheb12[8] - alam
	for i := 1; i <= 6; i++ {
		j := 14 - i
		v[i] = fval[i] - fval[j]
		fval[i] += fval[j]
	}
	alam1 = v[1] + x[8]*v[5]
	alam2 = x[4] * v[3]
	cheb12[3] = alam1 + alam2
	cheb12[11] = alam1 - alam2
	cheb12[7] = v[1] - v[5]
	alam = x[2]*v[2] + x[6]*v[4] + x[10]*v[6]
	cheb24[3] = cheb12[3] + alam
	cheb24[23] = cheb12[3] - alam
	alam = x[6] * (v[2] - v[4] - v[6])
	cheb24[7] = cheb12[7] + alam
	cheb24[19] = cheb12[7] - alam
	alam = x[10]*v[2] - x[6]*v[4] + x[2]*v[6]
	cheb24[11] = cheb12[11] + alam
	cheb24[15] = cheb12[11] - alam
	for i := 1; i <= 3; i++ {
		j := 8 - i
		v[i] = fval[i] - fval[j]
		fval[i] += fval[j]
	}
	cheb12[5] = v[1] + x[8]*v[3]
	cheb12[9] = fval[1] - x[8]*fval[3]
	alam = x[4] * v[2]
	cheb24[5] = cheb12[5] + alam
	cheb24[21] = cheb12[5] - alam
	alam = x[8]*fval[2] - fval[4]
	cheb24[9] = cheb12[9] + alam
	cheb24[17] = cheb12[9] - alam
	cheb12[1] = fval[1] + fval[3]
	alam = fval[2] + fval[4]
	cheb24[1] = cheb12[1] + alam
	cheb24[25] = cheb12[1] - alam
	cheb12[13] = v[1] - v[3]
	cheb24[13] = cheb12[13]
	alam = 1.0 / 6.0
	for i := 2; i <= 12; i++ {
		cheb12[i] *= alam
	}
	alam *= 0.5
	cheb12[1] *= alam
	cheb12[13] *= alam
	for i := 2; i <= 24; i++ {
		cheb24[i] *= alam
	}
	cheb24[1] = 0.5 * alam * cheb24[1]
	cheb24[25] = 0.5 * alam * cheb24[25]
}

// chebSums returns Σ cheb12[i]·mom[i] (i ≤ 13) and Σ cheb24[i]·mom[i]
// (i ≤ 25), the two Clenshaw–Curtis estimates against a moment vector.
func chebSums(cheb12 *[14]float64, cheb24 *[26]float64, mom *[26]float64) (res12, res24 float64) {
	for i := 1; i <= 13; i++ {
		res12 += cheb12[i] * mom[i]
		res24 += cheb24[i] * mom[i]
	}
	for i := 14; i <= 25; i++ {
		res24 += cheb24[i] * mom[i]
	}
	return res12, res24
}
