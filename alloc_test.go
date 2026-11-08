package integrate

import (
	"math"
	"testing"
)

// The pooled QUADPACK workspace and the reused Jacobian and Radau buffers must
// not change a bit of any result: the SciPy fixtures of quad_fixtures_test.go
// and ode_fixtures_test.go pin the values, and these tests pin what only reuse
// can break (stale contents, aliasing between steps).

// TestPooledWorkspaceBitIdentical: a quadrature gives the same bits whatever the
// pool holds, here a workspace dirtied by a larger, different quadrature.
func TestPooledWorkspaceBitIdentical(t *testing.T) {
	f := func(x float64) float64 { return math.Log(x) / math.Sqrt(x) }
	g := func(x float64) float64 { return math.Sin(37*x) / (1 + x*x) }
	want, err := Quad(f, 0, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	for round := 0; round < 5; round++ {
		// Dirty the pool: many subintervals, a bigger limit, other weights.
		if _, err := Quad(g, 0, 200, &QuadOptions{Limit: 400}); err != nil && round == 0 {
			t.Log("dirtying quadrature:", err)
		}
		if _, err := Quad(f, 0, 1, &QuadOptions{Points: []float64{0.5}}); err != nil {
			t.Fatal(err)
		}
		got, err := Quad(f, 0, 1, nil)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float64bits(got.Value) != math.Float64bits(want.Value) ||
			math.Float64bits(got.AbsErr) != math.Float64bits(want.AbsErr) || got.NEval != want.NEval {
			t.Fatalf("round %d: %+v, want %+v", round, got, want)
		}
	}
}

// TestQuadAllocs: a quadrature takes its workspace from the pool, so a plain Quad
// allocates nothing and a triple quadrature (hundreds of inner Quad calls, 2780
// allocations before) a constant few.
func TestQuadAllocs(t *testing.T) {
	if raceOn {
		t.Skip("sync.Pool drops items at random under the race detector")
	}
	f := func(x float64) float64 { return 1 / (1 + 25*x*x) }
	Quad(f, -1, 1, nil)
	if a := testing.AllocsPerRun(50, func() { Quad(f, -1, 1, nil) }); a != 0 {
		t.Errorf("Quad: %v allocations per call, want 0", a)
	}
	f3 := func(z, y, x float64) float64 { return math.Exp(-(x*x + y*y + z*z)) }
	call := func() {
		TplQuad(f3, 0, 1, Const(0), Const(1), Const2(0), Const2(1), nil)
	}
	call()
	if a := testing.AllocsPerRun(20, call); a > 6 {
		t.Errorf("TplQuad: %v allocations per call, want at most 6", a)
	}
}

// TestRadauDenseKeepsEveryStep: the interpolant of a step is refreshed in place
// for the next step's predictor, so the continuous solution must hold a copy per
// step: evaluated at the end of each step it gives that step's state.
func TestRadauDenseKeepsEveryStep(t *testing.T) {
	sol, err := SolveIVP(vdpF(100), [2]float64{0, 300}, []float64{2, 0}, &IVPOptions{Method: Radau, DenseOutput: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(sol.T) < 20 {
		t.Fatalf("only %d steps", len(sol.T))
	}
	for k := 1; k < len(sol.T); k++ {
		y := sol.Sol.Eval(sol.T[k])
		for i := range y {
			if d := math.Abs(y[i] - sol.Y[k][i]); d > 1e-8*(1+math.Abs(sol.Y[k][i])) {
				t.Fatalf("step %d component %d: dense %v, state %v", k, i, y[i], sol.Y[k][i])
			}
		}
	}
	// Two runs agree to the bit, dense or not: reuse changes no result.
	plain, err := SolveIVP(vdpF(100), [2]float64{0, 300}, []float64{2, 0}, &IVPOptions{Method: Radau})
	if err != nil {
		t.Fatal(err)
	}
	if len(plain.T) != len(sol.T) {
		t.Fatalf("%d steps against %d", len(plain.T), len(sol.T))
	}
	for k := range plain.Y {
		for i := range plain.Y[k] {
			if math.Float64bits(plain.Y[k][i]) != math.Float64bits(sol.Y[k][i]) {
				t.Fatalf("step %d component %d differs", k, i)
			}
		}
	}
}

// TestRadauAllocsPerStep: a Radau step allocates the state it stores and the
// growth of the lists, not an interpolant, and a finite-difference Jacobian none.
func TestRadauAllocsPerStep(t *testing.T) {
	o := &IVPOptions{Method: Radau}
	f := vdpF(1000)
	var steps int
	call := func() {
		s, err := SolveIVP(f, [2]float64{0, 3000}, []float64{2, 0}, o)
		if err != nil {
			t.Fatal(err)
		}
		steps = len(s.T)
	}
	call()
	a := testing.AllocsPerRun(10, call)
	if perStep := a / float64(steps); perStep > 2 {
		t.Errorf("%v allocations for %d steps: %.2f per step, want at most 2", a, steps, perStep)
	}
}
