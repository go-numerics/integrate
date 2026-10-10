// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"
)

// quadCase mirrors one entry of the QUAD table of testdata/gen/fixtures.py.
type quadCase struct {
	f    func(float64) float64
	opts *QuadOptions
}

var quadCases = map[string]quadCase{
	"exp":           {math.Exp, nil},
	"sin":           {math.Sin, nil},
	"runge":         {func(x float64) float64 { return 1 / (1 + 25*x*x) }, nil},
	"log_over_sqrt": {func(x float64) float64 { return math.Log(x) / math.Sqrt(x) }, nil},
	"inv_sqrt":      {func(x float64) float64 { return 1 / math.Sqrt(x) }, nil},
	"x_pow_m09":     {func(x float64) float64 { return math.Pow(x, -0.9) }, nil},
	"log":           {math.Log, nil},
	"exp_cos50":     {func(x float64) float64 { return math.Exp(-x) * math.Cos(50*x) }, nil},
	"tight_runge": {func(x float64) float64 { return 1 / (1 + 25*x*x) },
		&QuadOptions{EpsAbs: -1, EpsRel: 1e-13, Limit: 200}},
	"gauss_inf":     {func(x float64) float64 { return math.Exp(-x * x) }, nil},
	"lorentz_half":  {func(x float64) float64 { return 1 / (1 + x*x) }, nil},
	"exp_left":      {math.Exp, nil},
	"x2_exp":        {func(x float64) float64 { return x * x * math.Exp(-x) }, nil},
	"lorentz_shift": {func(x float64) float64 { return 1 / ((x-1)*(x-1) + 1) }, nil},
	"kink_points":   {func(x float64) float64 { return math.Abs(x - 1.0/3) }, &QuadOptions{Points: []float64{1.0 / 3}}},
	"step_points":   {stepFn, &QuadOptions{Points: []float64{0.5}}},
	"step_nopoints": {stepFn, nil},
	"sqrt_kink_points": {func(x float64) float64 { return math.Sqrt(math.Abs(x - 0.25)) },
		&QuadOptions{Points: []float64{0.25, 0.7, 0.25}}},
	"cos_weight":        {math.Exp, &QuadOptions{Weight: WeightCos, Omega: 100}},
	"sin_weight_log1p":  {math.Log1p, &QuadOptions{Weight: WeightSin, Omega: 10 * math.Pi}},
	"cos_weight_small":  {func(x float64) float64 { return x * x }, &QuadOptions{Weight: WeightCos, Omega: 1.5}},
	"fourier_cos":       {func(x float64) float64 { return math.Exp(-x) }, &QuadOptions{Weight: WeightCos, Omega: 2}},
	"fourier_sin":       {func(x float64) float64 { return 1 / (1 + x*x) }, &QuadOptions{Weight: WeightSin, Omega: 1}},
	"fourier_sinc":      {func(x float64) float64 { return 1 / x }, &QuadOptions{Weight: WeightSin, Omega: 1}},
	"fourier_left":      {math.Exp, &QuadOptions{Weight: WeightCos, Omega: 2}},
	"fourier_left_sin":  {math.Exp, &QuadOptions{Weight: WeightSin, Omega: 2}},
	"alg_chebyshev":     {one, &QuadOptions{Weight: WeightAlg, Alpha: -0.5, Beta: -0.5}},
	"alg_cos":           {math.Cos, &QuadOptions{Weight: WeightAlg, Alpha: -0.5, Beta: 0.3}},
	"alg_loga":          {one, &QuadOptions{Weight: WeightAlgLogA}},
	"alg_logb":          {one, &QuadOptions{Weight: WeightAlgLogB, Alpha: 0.5}},
	"alg_log":           {one, &QuadOptions{Weight: WeightAlgLog}},
	"alg_log_exp":       {math.Exp, &QuadOptions{Weight: WeightAlgLog, Alpha: 0.25, Beta: -0.4}},
	"cauchy_one":        {one, &QuadOptions{Weight: WeightCauchy, C: 0}},
	"cauchy_exp":        {math.Exp, &QuadOptions{Weight: WeightCauchy, C: 0}},
	"cauchy_far":        {math.Exp, &QuadOptions{Weight: WeightCauchy, C: 3}},
	"divergent":         {func(x float64) float64 { return 1 / x }, nil},
	"sin_inv":           {func(x float64) float64 { return math.Sin(1 / x) }, nil},
	"limit_small":       {func(x float64) float64 { return math.Sin(1 / x) }, &QuadOptions{Limit: 5}},
	"inv_x_lim200":      {func(x float64) float64 { return 1 / x }, &QuadOptions{Limit: 200}},
	"x_pow_m12":         {posOnly(func(x float64) float64 { return math.Pow(x, -1.2) }), &QuadOptions{Limit: 200}},
	"noise":             {func(x float64) float64 { return math.Sin(1e7*x)*1e-3 + 1 }, nil},
	"tiny_exp":          {math.Exp, tiny(0)},
	"tiny_sqrt":         {math.Sqrt, tiny(200)},
	"tiny_log":          {math.Log, tiny(200)},
	"inf_div":           {func(x float64) float64 { return 1 / (1 + math.Abs(x)) }, &QuadOptions{Limit: 200}},
	"inf_osc":           {math.Sin, nil},
	"inf_slow":          {func(x float64) float64 { return 1 / math.Pow(1+x, 1.05) }, &QuadOptions{Limit: 200}},
	"pts_div":           {except(0.5, func(x float64) float64 { return 1 / math.Abs(x-0.5) }), &QuadOptions{Points: []float64{0.5}, Limit: 200}},
	"pts_log_tiny":      {except(0.3, func(x float64) float64 { return math.Log(math.Abs(x - 0.3)) }), &QuadOptions{Points: []float64{0.3}, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 200}},
	"pts_sqrt_tiny":     {func(x float64) float64 { return math.Sqrt(math.Abs(x - 0.3)) }, &QuadOptions{Points: []float64{0.3, 0.6}, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 200}},
	"pts_lim":           {posOnly(func(x float64) float64 { return math.Sin(1 / x) }), &QuadOptions{Points: []float64{0.5}, Limit: 10}},
	"pts_x_m12":         {except(0.3, func(x float64) float64 { return math.Pow(math.Abs(x-0.3), -1.2) }), &QuadOptions{Points: []float64{0.3}, Limit: 500}},
	"qawo_long":         {func(x float64) float64 { return math.Exp(-x/30) / (1 + x) }, &QuadOptions{Weight: WeightCos, Omega: 10}},
	"qawo_long_sin":     {func(x float64) float64 { return 1 / (1 + x*x) }, &QuadOptions{Weight: WeightSin, Omega: 1}},
	"qawo_log":          {posOnly(math.Log), &QuadOptions{Weight: WeightCos, Omega: 50}},
	"qawo_rsqrt":        {posOnly(rsqrt), &QuadOptions{Weight: WeightCos, Omega: 200}},
	"qawo_rsqrt_negsin": {posOnly(rsqrt), &QuadOptions{Weight: WeightSin, Omega: -200}},
	"qawo_lim":          {posOnly(rsqrt), &QuadOptions{Weight: WeightCos, Omega: 200, Limit: 4}},
	"qawo_tiny":         {math.Sqrt, &QuadOptions{Weight: WeightCos, Omega: 20, EpsAbs: -1, EpsRel: 1.2e-14}},
	"qawo_maxp1":        {posOnly(rsqrt), &QuadOptions{Weight: WeightCos, Omega: 200, MaxP1: 2}},
	"qawf_limlst":       {rsqrt, &QuadOptions{Weight: WeightCos, Omega: 1, LimLst: 3}},
	"qawf_rsqrt":        {rsqrt, &QuadOptions{Weight: WeightCos, Omega: 1}},
	"qawf_shift": {func(x float64) float64 {
		if x > 1 {
			return 1 / math.Sqrt(x-1)
		}
		return 0
	}, &QuadOptions{Weight: WeightSin, Omega: 3}},
	"qawf_omega0":     {func(x float64) float64 { return math.Exp(-x) }, &QuadOptions{Weight: WeightCos}},
	"qawf_omega0_sin": {func(x float64) float64 { return math.Exp(-x) }, &QuadOptions{Weight: WeightSin}},
	"qawf_slow_omega": {func(x float64) float64 { return math.Exp(-x / 5) }, &QuadOptions{Weight: WeightCos, Omega: 0.3}},
	"qawc_sqrt":       {func(x float64) float64 { return math.Sqrt(x + 1) }, &QuadOptions{Weight: WeightCauchy, C: 0.2}},
	"qawc_peak":       {func(x float64) float64 { return 1 / (1 + 100*(x-0.6)*(x-0.6)) }, &QuadOptions{Weight: WeightCauchy, C: 0.1}},
	"qawc_lim":        {func(x float64) float64 { return math.Sqrt(x + 1) }, &QuadOptions{Weight: WeightCauchy, C: 0.2, Limit: 3}},
	"qawc_flip":       {func(x float64) float64 { return math.Sqrt(x + 1) }, &QuadOptions{Weight: WeightCauchy, C: 0.2}},
	"qawc_tiny":       {func(x float64) float64 { return math.Sqrt(x + 1) }, &QuadOptions{Weight: WeightCauchy, C: 0.2, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 200}},
	"qaws_cos30":      {cos30, &QuadOptions{Weight: WeightAlg, Alpha: -0.5}},
	"qaws_lim":        {cos30, &QuadOptions{Weight: WeightAlg, Alpha: -0.5, Limit: 3}},
	"qaws_lim2":       {cos30, &QuadOptions{Weight: WeightAlg, Alpha: -0.5, Limit: 2}},
	"qaws_logb":       {sin20, &QuadOptions{Weight: WeightAlgLogB, Alpha: 0.3, Beta: -0.5}},
	"qaws_loga":       {sin20, &QuadOptions{Weight: WeightAlgLogA, Alpha: 0.3, Beta: -0.5}},
	"qaws_tiny":       {cos30, &QuadOptions{Weight: WeightAlg, Alpha: -0.5, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 200}},
	"qaws_rough":      {func(x float64) float64 { return math.Sqrt(math.Abs(x - 0.37)) }, &QuadOptions{Weight: WeightAlg, Alpha: 0.5, Beta: 0.5, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 200}},
	"x_m2":            {posOnly(func(x float64) float64 { return 1 / (x * x) }), &QuadOptions{Limit: 200}},
	"log_div": {func(x float64) float64 {
		if x > 0 && x < 1 {
			return (1 / x) * math.Pow(1/math.Abs(math.Log(x)), 1.01)
		}
		return 0
	}, &QuadOptions{Limit: 200}},
	"inf_x_m09":          {func(x float64) float64 { return math.Pow(x+1, -0.9) }, &QuadOptions{Limit: 500}},
	"inf_osc_slow":       {func(x float64) float64 { return math.Sin(x) / (1 + x) }, &QuadOptions{Limit: 500}},
	"inf_fresnel":        {func(x float64) float64 { return math.Cos(x * x) }, &QuadOptions{Limit: 200}},
	"qawo_div":           {posOnly(func(x float64) float64 { return 1 / x }), &QuadOptions{Weight: WeightCos, Omega: 30, Limit: 200}},
	"qawo_x_m0999":       {posOnly(func(x float64) float64 { return math.Pow(x, -0.999) }), &QuadOptions{Weight: WeightCos, Omega: 30, Limit: 200}},
	"qawo_noise":         {func(x float64) float64 { return math.Exp(x) + 1e-12*math.Sin(1e6*x) }, &QuadOptions{Weight: WeightSin, Omega: 30, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 200}},
	"qawo_round":         {func(x float64) float64 { return math.RoundToEven(math.Exp(x)*1e10) / 1e10 }, &QuadOptions{Weight: WeightCos, Omega: 30, EpsAbs: -1, EpsRel: 1e-13, Limit: 200}},
	"qawo_small_w":       {posOnly(rsqrt), &QuadOptions{Weight: WeightCos, Omega: 3, Limit: 200}},
	"qawo_small_w_log":   {posOnly(math.Log), &QuadOptions{Weight: WeightCos, Omega: 1, Limit: 200}},
	"qawo_mid_w":         {posOnly(rsqrt), &QuadOptions{Weight: WeightCos, Omega: 6, Limit: 200}},
	"qawf_tight":         {func(x float64) float64 { return math.Exp(-x) }, &QuadOptions{Weight: WeightCos, Omega: 10, EpsAbs: 1e-15}},
	"qawf_tight_lorentz": {func(x float64) float64 { return 1 / (1 + x*x) }, &QuadOptions{Weight: WeightCos, Omega: 10, EpsAbs: 1e-15}},
	"qawf_loose":         {func(x float64) float64 { return math.Exp(-x) }, &QuadOptions{Weight: WeightCos, Omega: 10, EpsAbs: 1e-3}},
	"qawf_linear":        {func(x float64) float64 { return x }, &QuadOptions{Weight: WeightCos, Omega: 2}},
	"qawc_tough": {func(x float64) float64 {
		if math.Abs(x) < 1 {
			return 1 / math.Sqrt(1-x*x)
		}
		return 0
	}, &QuadOptions{Weight: WeightCauchy, C: 0.5, Limit: 200}},
	"qawc_rough":      {func(x float64) float64 { return math.Pow(math.Abs(x-0.7), 0.1) }, &QuadOptions{Weight: WeightCauchy, C: 0.2, Limit: 500, EpsAbs: -1, EpsRel: 1e-13}},
	"qaws_alglog_cos": {cos30, &QuadOptions{Weight: WeightAlgLog, Alpha: -0.5, Beta: 0.2}},
	"qaws_inner_sing": {except(0.37, func(x float64) float64 { return math.Pow(math.Abs(x-0.37), -0.5) }), &QuadOptions{Weight: WeightAlg, Alpha: 0.5, Beta: 0.5, Limit: 200}},
	"br_x_m12_rel":    {powAbs(0, -1.2, 0), &QuadOptions{EpsAbs: -1, EpsRel: 1.2e-14, Limit: 50}},
	"br_x_m12_abs":    {powAbs(0, -1.2, 0), &QuadOptions{EpsAbs: 1e-14, EpsRel: -1, Limit: 50}},
	"br_sqrt_noise":   {powAbs(0, 0.5, 1e-6), &QuadOptions{EpsAbs: -1, EpsRel: 1e-12, Limit: 50}},
	"br_round_lim1": {func(x float64) float64 {
		v := math.RoundToEven(math.Exp(x)*1e10) / 1e10
		if x > 0 {
			v++
		}
		return v
	}, &QuadOptions{EpsAbs: 1e-14, EpsRel: -1, Limit: 1}},
	"br_lim1":            {powAbs(0, -1.2, 0), &QuadOptions{Limit: 1}},
	"br_sin_ier2":        {math.Sin, &QuadOptions{EpsAbs: 1e-300, EpsRel: 1.2e-14}},
	"br_pts_x_m12_12":    {powAbs(0, -1.2, 0), &QuadOptions{Points: []float64{0.3, 0.5}, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 12}},
	"br_pts_x_m12_50":    {powAbs(0, -1.2, 0), &QuadOptions{Points: []float64{0.3, 0.5}, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 50}},
	"br_pts_x_m12_noise": {powAbs(0, -1.2, 1e-13), &QuadOptions{Points: []float64{0.3, 0.5}, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 50}},
	"br_pts_p01_noise":   {powAbs(0, 0.1, 1e-6), &QuadOptions{Points: []float64{0.3, 0.5}, EpsAbs: -1, EpsRel: 1e-12, Limit: 50}},
	"br_pts_p01_loose":   {powAbs(0, 0.1, 0), &QuadOptions{Points: []float64{0.3, 0.5}, EpsAbs: -1, EpsRel: 1e-3, Limit: 12}},
	"br_pts_lim3":        {powAbs(0, -1.2, 0), &QuadOptions{Points: []float64{0.3, 0.5}, Limit: 3}},
	"br_pts_sin_ier2":    {math.Sin, &QuadOptions{Points: []float64{math.Pi}, EpsAbs: 1e-300, EpsRel: 1.2e-14}},
	"br_qawo_ier2":       {powAbs(0, -1, 0), &QuadOptions{Weight: WeightSin, Omega: 1, EpsAbs: 1e-14, EpsRel: -1, Limit: 1}},
	"br_qawo_lim1":       {powAbs(0, -1.2, 0), &QuadOptions{Weight: WeightCos, Omega: 1, Limit: 1}},
	"br_qawo_small":      {powAbs(0.3, -1, 0), &QuadOptions{Weight: WeightCos, Omega: 1, Limit: 200}},
	"br_qawo_jup":        {powAbs(0, -1.2, 0), &QuadOptions{Weight: WeightCos, Omega: 1, EpsAbs: -1, EpsRel: 1e-12, Limit: 12}},
	"br_qawo_div":        {powAbs(0, -1.2, 0), &QuadOptions{Weight: WeightCos, Omega: 1, EpsAbs: -1, EpsRel: 1.2e-14, Limit: 50}},
	"br_qawo_x_m105":     {powAbs(0, -1.05, 0), &QuadOptions{Weight: WeightCos, Omega: 1}},
	"br_qawo_ext":        {posOnly(func(x float64) float64 { return math.Sin(1/x) / x }), &QuadOptions{Weight: WeightCos, Omega: 3, EpsAbs: -1, EpsRel: 1e-13, Limit: 50}},
	"br_qawf_limexp":     {invPow(0.5), &QuadOptions{Weight: WeightCos, Omega: 1, EpsAbs: 1e-15, Limit: 1, LimLst: 50}},
	"br_qawf_ier7":       {invPow(0.9), &QuadOptions{Weight: WeightCos, Omega: 10, EpsAbs: 1e-3, Limit: 1, LimLst: 10}},
	"br_qawf_ier7b":      {func(x float64) float64 { return math.Sin(x) * math.Pow(1+math.Abs(x), -0.5) }, &QuadOptions{Weight: WeightSin, Omega: 0.3, EpsAbs: 1e-3, Limit: 1, LimLst: 50}},
	"br_qawf_ier4":       {invPow(0.5), &QuadOptions{Weight: WeightCos, Omega: 0.3, EpsAbs: 1e-15, Limit: 1, LimLst: 50}},
	"br_qawf_drl":        {invPow(0.5), &QuadOptions{Weight: WeightCos, Omega: 0.3, EpsAbs: 1e-8, Limit: 1, LimLst: 10}},
	"br_qawc_lim1":       {func(x float64) float64 { return math.Sqrt(x + 1) }, &QuadOptions{Weight: WeightCauchy, C: 0.2, Limit: 1}},
}

func powAbs(c, p, noise float64) func(float64) float64 {
	return func(x float64) float64 {
		if x == c {
			return 0
		}
		return math.Pow(math.Abs(x-c), p) * (1 + noise*math.Sin(1e6*x))
	}
}

func invPow(q float64) func(float64) float64 {
	return func(x float64) float64 { return math.Pow(1+math.Abs(x), -q) }
}

func rsqrt(x float64) float64 { return 1 / math.Sqrt(x) }
func cos30(x float64) float64 { return math.Cos(30 * x) }
func sin20(x float64) float64 { return math.Sin(20 * x) }

// posOnly is f on x > 0 and 0 elsewhere, as the fixture integrands guard
// their singular end points.
func posOnly(f func(float64) float64) func(float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return f(x)
		}
		return 0
	}
}

// except is f except at the singular point p, where it is 0.
func except(p float64, f func(float64) float64) func(float64) float64 {
	return func(x float64) float64 {
		if x == p {
			return 0
		}
		return f(x)
	}
}

func tiny(limit int) *QuadOptions {
	return &QuadOptions{EpsAbs: -1, EpsRel: 1.2e-14, Limit: limit}
}

func one(float64) float64 { return 1 }

func stepFn(x float64) float64 {
	if x < 0.5 {
		return 1
	}
	return 2
}

type quadFixture struct {
	Meta  map[string]string `json:"meta"`
	Quad  []quadRecord      `json:"quad"`
	Multi []struct {
		Name   string  `json:"name"`
		Value  float64 `json:"value"`
		AbsErr float64 `json:"abserr"`
		Ref    float64 `json:"ref"`
	} `json:"multi"`
	Fixed map[string]json.RawMessage `json:"fixed"`
}

type quadRecord struct {
	Name   string   `json:"name"`
	A      jsonF    `json:"a"`
	B      jsonF    `json:"b"`
	Value  float64  `json:"value"`
	AbsErr float64  `json:"abserr"`
	NEval  int      `json:"neval"`
	Calls  int      `json:"calls"`
	Last   int      `json:"last"`
	Ier    int      `json:"ier"`
	Ref    *float64 `json:"ref"`
}

// jsonF decodes a JSON number, or the strings "inf" and "-inf" the fixture
// generator writes for the infinities JSON cannot express.
type jsonF float64

func (f *jsonF) UnmarshalJSON(b []byte) error {
	switch string(b) {
	case `"inf"`:
		*f = jsonF(math.Inf(1))
		return nil
	case `"-inf"`:
		*f = jsonF(math.Inf(-1))
		return nil
	}
	var v float64
	err := json.Unmarshal(b, &v)
	*f = jsonF(v)
	return err
}

func loadQuadFixture(t *testing.T) *quadFixture {
	t.Helper()
	b, err := os.ReadFile("testdata/quad.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx quadFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	return &fx
}

// Fixture integrals on which SciPy's run is not the reference QUADPACK
// behaviour. Each was rerun with the netlib Fortran compiled by gfortran
// (testdata/gen/netlib_harness.f).
//
// netlibOracle: SciPy 1.18.1's C translation of QUADPACK departs from the
// Fortran; this port follows the Fortran (TestNetlibOracle pins the values).
//
// pathSensitive: the tolerance is at the roundoff limit or the integral
// diverges, and the subdivision path depends on the last bit of f and of each
// rule (Go's math.Pow, FMA contraction, summation order). Perturbing f by one
// ulp moves SciPy itself from 777 to 945 or 735 evaluations on br_x_m12_*; the
// Fortran, SciPy and this port each take their own path. Only the presence of
// a diagnostic and the agreement of the values within the error estimates are
// checked.
var (
	netlibOracle  = map[string]bool{"qawo_maxp1": true, "qawo_div": true, "br_qawo_small": true}
	pathSensitive = map[string]bool{"qawc_tiny": true, "br_x_m12_rel": true, "br_x_m12_abs": true,
		"br_pts_x_m12_50": true, "br_qawo_div": true, "br_qawf_limexp": true, "br_qawf_ier4": true}
)

// TestQuadAgainstSciPy runs every fixture integral and compares with SciPy's
// QUADPACK: the value to within a few units in the last place of the error
// estimate, the evaluation count exactly (the bisection sequence is the
// same), and the diagnostic.
func TestQuadAgainstSciPy(t *testing.T) {
	fx := loadQuadFixture(t)
	if len(fx.Quad) != len(quadCases) {
		t.Fatalf("fixture has %d integrals, Go table %d", len(fx.Quad), len(quadCases))
	}
	sameNeval := 0
	for _, rec := range fx.Quad {
		c, ok := quadCases[rec.Name]
		if !ok {
			t.Fatalf("no Go integrand for %q", rec.Name)
		}
		calls := 0
		f := func(x float64) float64 { calls++; return c.f(x) }
		r, err := Quad(f, float64(rec.A), float64(rec.B), c.opts)
		// Tolerances at the roundoff limit and diverging or roundoff-
		// diagnosed integrals follow a path that the last bit decides,
		// and the last bit differs between architectures (FMA on arm64,
		// ppc64le, s390x, riscv64, loong64; none on amd64, 386, arm).
		sensitive := pathSensitive[rec.Name] || rec.Ier == 2 || rec.Ier == 4 || rec.Ier == 5 ||
			(c.opts != nil && c.opts.EpsRel > 0 && c.opts.EpsRel < 1e-12)
		deviates := netlibOracle[rec.Name] || sensitive
		ier := 0
		var qe *QuadError
		if errors.As(err, &qe) {
			ier = qe.Ier
		} else if err != nil {
			t.Fatalf("%s: %v", rec.Name, err)
		}
		if ier != rec.Ier && !deviates {
			t.Errorf("%s: ier %d, SciPy %d", rec.Name, ier, rec.Ier)
		}
		if sensitive {
			if (ier == 0) != (rec.Ier == 0) {
				t.Errorf("%s: ier %d, SciPy %d", rec.Name, ier, rec.Ier)
			}
			if d := math.Abs(r.Value - rec.Value); d > 10*math.Max(r.AbsErr, rec.AbsErr) {
				t.Errorf("%s: value %.17g, SciPy %.17g, beyond the error estimates", rec.Name, r.Value, rec.Value)
			}
		}
		if calls == rec.Calls {
			sameNeval++
		} else if !deviates {
			t.Errorf("%s: %d integrand calls, SciPy %d", rec.Name, calls, rec.Calls)
		}
		if r.NEval != calls {
			t.Errorf("%s: NEval %d but %d calls", rec.Name, r.NEval, calls)
		}
		tol := 1e-12 * math.Max(1, math.Abs(rec.Value))
		if rec.Ier != 0 {
			tol *= 1000 // a diagnosed result is only as stable as its path
		}
		if math.Abs(r.Value-rec.Value) > tol && !deviates {
			t.Errorf("%s: value %.17g, SciPy %.17g (diff %.3g)", rec.Name, r.Value, rec.Value, r.Value-rec.Value)
		}
		if rec.Ref != nil && ier == 0 {
			// The estimate must bound the true error (QUADPACK's error
			// estimates are pessimistic by design).
			trueErr := math.Abs(r.Value - *rec.Ref)
			if trueErr > math.Max(r.AbsErr, 2e-15*math.Abs(*rec.Ref)) {
				t.Errorf("%s: true error %.3g exceeds estimate %.3g", rec.Name, trueErr, r.AbsErr)
			}
		}
	}
	t.Logf("SciPy %s: identical evaluation counts on %d/%d integrals", fx.Meta["scipy"], sameNeval, len(fx.Quad))
}

func TestMultiAgainstSciPy(t *testing.T) {
	fx := loadQuadFixture(t)
	got := map[string]func() (QuadResult, error){
		"dbl_rect": func() (QuadResult, error) {
			return DblQuad(func(y, x float64) float64 { return x * y * y }, 0, 2, Const(0), Const(1), nil)
		},
		"dbl_tri": func() (QuadResult, error) {
			return DblQuad(func(y, x float64) float64 { return math.Exp(-x * y) }, 0, 1, Const(0), func(x float64) float64 { return x }, nil)
		},
		"tpl_box": func() (QuadResult, error) {
			return TplQuad(func(z, y, x float64) float64 { return x * y * z }, 1, 2, Const(2), Const(3), Const2(0), Const2(1), nil)
		},
		"tpl_simplex": func() (QuadResult, error) {
			return TplQuad(func(z, y, x float64) float64 { return 1 }, 0, 1, Const(0), func(x float64) float64 { return 1 - x },
				Const2(0), func(x, y float64) float64 { return 1 - x - y }, nil)
		},
		"nquad_4": func() (QuadResult, error) {
			r := ConstRange(0, 1)
			return NQuad(func(x []float64) float64 { return x[0] * x[1] * x[2] * x[3] }, []Range{r, r, r, r}, nil)
		},
	}
	for _, m := range fx.Multi {
		r, err := got[m.Name]()
		if err != nil {
			t.Fatalf("%s: %v", m.Name, err)
		}
		if math.Abs(r.Value-m.Value) > 1e-13*math.Max(1, math.Abs(m.Value)) {
			t.Errorf("%s: %.17g, SciPy %.17g", m.Name, r.Value, m.Value)
		}
		if math.Abs(r.Value-m.Ref) > math.Max(r.AbsErr, 1e-15) {
			t.Errorf("%s: error %.3g above estimate %.3g", m.Name, math.Abs(r.Value-m.Ref), r.AbsErr)
		}
	}
}

func TestFixedAgainstSciPy(t *testing.T) {
	fx := loadQuadFixture(t)
	num := func(k string) float64 {
		var v float64
		if err := json.Unmarshal(fx.Fixed[k], &v); err != nil {
			t.Fatal(k, err)
		}
		return v
	}
	vec := func(k string) []float64 {
		var v []float64
		if err := json.Unmarshal(fx.Fixed[k], &v); err != nil {
			t.Fatal(k, err)
		}
		return v
	}
	x := make([]float64, 9)
	y := make([]float64, 9)
	for i := range x {
		x[i] = 0.25 * float64(i)
		y[i] = math.Exp(x[i])
	}
	xu := []float64{0.0, 0.1, 0.35, 0.5, 0.9, 1.3, 1.31, 2.0}
	yu := make([]float64, len(xu))
	for i, v := range xu {
		yu[i] = math.Exp(v)
	}
	check := func(name string, got float64, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want := num(name)
		if math.Abs(got-want) > 1e-14*math.Max(1, math.Abs(want)) {
			t.Errorf("%s: %.17g, SciPy %.17g", name, got, want)
		}
	}
	v, err := Trapezoid(y, x, 0)
	check("trapezoid", v, err)
	v, err = Trapezoid(y, nil, 0.25)
	check("trapezoid_dx", v, err)
	v, err = Simpson(y, x, 0)
	check("simpson_odd", v, err)
	v, err = Simpson(y, nil, 0.25)
	check("simpson_odd_dx", v, err)
	v, err = Simpson(yu, xu, 0)
	check("simpson_even", v, err)
	v, err = Simpson(y[:8], nil, 0.25)
	check("simpson_even_dx", v, err)
	v, err = Simpson(y[:2], x[:2], 0)
	check("simpson_two", v, err)
	ys := make([]float64, 17)
	for i := range ys {
		ys[i] = math.Exp(float64(i) / 16)
	}
	v, err = Romb(ys, 1.0/16)
	check("romb", v, err)
	v, err = FixedQuad(math.Cos, 0, 1, 5)
	check("fixed_quad5", v, err)
	v, err = FixedQuad(math.Cos, 0, 1, 20)
	check("fixed_quad20", v, err)
	ct, err := CumulativeTrapezoid(yu, xu, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	ct0, _ := CumulativeTrapezoid(yu, xu, 0, true)
	for name, got := range map[string][]float64{"cumtrapz": ct, "cumtrapz0": ct0} {
		want := vec(name)
		if len(got) != len(want) {
			t.Fatalf("%s: %d entries, SciPy %d", name, len(got), len(want))
		}
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-14 {
				t.Errorf("%s[%d]: %.17g, SciPy %.17g", name, i, got[i], want[i])
			}
		}
	}
	gx, gw := GaussLegendre(7)
	wx, ww := vec("legendre7_x"), vec("legendre7_w")
	for i := range gx {
		if math.Abs(gx[i]-wx[i]) > 2e-16 || math.Abs(gw[i]-ww[i]) > 4e-16 {
			t.Errorf("Gauss–Legendre 7 node %d: (%.17g, %.17g), SciPy (%.17g, %.17g)", i, gx[i], gw[i], wx[i], ww[i])
		}
	}
}

// TestNetlibOracle pins the results of the netlib Fortran QUADPACK (gfortran,
// testdata/gen/netlib_harness.f) on the integrals where SciPy departs from it.
func TestNetlibOracle(t *testing.T) {
	for _, c := range []struct {
		name  string
		value float64
		neval int
		ier   int
		last  int
	}{
		{"qawo_maxp1", 8.42501986376901629e-02, 485, 0, 13},
		{"qawo_div", 1.40956644439315767e+02, 6035, 1, 200},
		{"br_qawo_small", 6.76282900830324536e+01, 2685, 3, 90},
	} {
		q := quadCases[c.name]
		r, err := Quad(q.f, 0, 1, q.opts)
		ier := 0
		var qe *QuadError
		if errors.As(err, &qe) {
			ier = qe.Ier
		}
		if math.Abs(r.Value-c.value) > 1e-13*math.Abs(c.value) || r.NEval != c.neval || ier != c.ier || r.NIntervals != c.last {
			t.Errorf("%s: (%.17g, %d, ier %d, last %d), netlib (%.17g, %d, ier %d, last %d)",
				c.name, r.Value, r.NEval, ier, r.NIntervals, c.value, c.neval, c.ier, c.last)
		}
	}
}
