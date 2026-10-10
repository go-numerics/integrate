// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate_test

import (
	"fmt"
	"math"

	"github.com/go-numerics/integrate"
)

// quad(lambda x: math.log(x) / math.sqrt(x), 0, 1)
func ExampleQuad() {
	r, err := integrate.Quad(func(x float64) float64 { return math.Log(x) / math.Sqrt(x) }, 0, 1, nil)
	fmt.Printf("%.12f %v %d\n", r.Value, err, r.NEval)
	// Output: -4.000000000000 <nil> 315
}

// quad(lambda x: math.exp(-x*x), -np.inf, np.inf)
func ExampleQuad_infinite() {
	r, _ := integrate.Quad(func(x float64) float64 { return math.Exp(-x * x) }, math.Inf(-1), math.Inf(1), nil)
	fmt.Printf("%.15f\n", r.Value)
	// Output: 1.772453850905516
}

// quad(lambda x: math.exp(-x), 0, np.inf, weight="cos", wvar=2)
func ExampleQuad_fourier() {
	r, _ := integrate.Quad(func(x float64) float64 { return math.Exp(-x) }, 0, math.Inf(1),
		&integrate.QuadOptions{Weight: integrate.WeightCos, Omega: 2})
	fmt.Printf("%.12f\n", r.Value)
	// Output: 0.200000000000
}

// quad(f, 0, 1) on 1/x reports a diagnostic, as SciPy's IntegrationWarning.
func ExampleQuadError() {
	_, err := integrate.Quad(func(x float64) float64 { return 1 / x }, 0, 1, nil)
	fmt.Println(err)
	// Output: integrate: maximum number of subdivisions reached
}

// dblquad(lambda y, x: x*y*y, 0, 2, 0, 1)
func ExampleDblQuad() {
	r, _ := integrate.DblQuad(func(y, x float64) float64 { return x * y * y }, 0, 2,
		integrate.Const(0), integrate.Const(1), nil)
	fmt.Printf("%.12f\n", r.Value)
	// Output: 0.666666666667
}

// simpson(y, x=x)
func ExampleSimpson() {
	x := []float64{0, 0.5, 1, 1.5, 2}
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = v * v * v
	}
	v, _ := integrate.Simpson(y, x, 0)
	fmt.Println(v)
	// Output: 4
}

// solve_ivp(lotka, (0, 15), [10, 5])
func ExampleSolveIVP() {
	lotka := func(t float64, y, dydt []float64) {
		dydt[0] = 1.5*y[0] - y[0]*y[1]
		dydt[1] = -3*y[1] + y[0]*y[1]
	}
	sol, _ := integrate.SolveIVP(lotka, [2]float64{0, 15}, []float64{10, 5}, nil)
	fmt.Println(sol.Message, sol.NFev, len(sol.T))
	// Output: The solver successfully reached the end of the integration interval. 470 57
}

// solve_ivp(vdp, (0, 3000), [2, 0], method="Radau") with μ = 1000 (stiff)
func ExampleSolveIVP_stiff() {
	const mu = 1000
	vdp := func(t float64, y, dydt []float64) {
		dydt[0] = y[1]
		dydt[1] = mu*(1-y[0]*y[0])*y[1] - y[0]
	}
	sol, err := integrate.SolveIVP(vdp, [2]float64{0, 3000}, []float64{2, 0}, &integrate.IVPOptions{Method: integrate.Radau})
	fmt.Println(err, sol.Success, sol.NSteps < 300)
	// Output: <nil> true true
}

// A terminal event: solve_ivp(ball, (0, 10), [10, 5], events=hit) with
// hit.terminal = True, hit.direction = -1.
func ExampleSolveIVP_events() {
	ball := func(t float64, y, dydt []float64) { dydt[0], dydt[1] = y[1], -9.81 }
	hit := integrate.Event{F: func(t float64, y []float64) float64 { return y[0] }, Terminal: 1, Direction: -1}
	sol, _ := integrate.SolveIVP(ball, [2]float64{0, 10}, []float64{10, 5},
		&integrate.IVPOptions{Events: []integrate.Event{hit}, RTol: 1e-8})
	fmt.Printf("%.6f %d\n", sol.TEvents[0][0], sol.Status)
	// Output: 2.025769 1
}

// odeint(robertson, [1, 0, 0], t, tfirst=True)
func ExampleOdeint() {
	robertson := func(t float64, y, dydt []float64) {
		dydt[0] = -0.04*y[0] + 1e4*y[1]*y[2]
		dydt[1] = 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1]
		dydt[2] = 3e7 * y[1] * y[1]
	}
	y, info, _ := integrate.Odeint(robertson, []float64{1, 0, 0}, []float64{0, 40, 4e4}, nil)
	fmt.Printf("%.4f %.4f %d\n", y.At(1, 0), y.At(2, 2), info.MUsed[1])
	// Output: 0.7158 0.9610 2
}
