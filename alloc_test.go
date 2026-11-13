package integrate

import (
	"math"
	"runtime"
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

// odeintCall is one of the two Odeint rows of bench/perf_test.go.
func odeintCall(robertson bool) func() {
	if robertson {
		ts := []float64{0, 0.4, 4, 40, 400, 4000, 40000}
		o := &OdeintOptions{RTol: 1e-8, ATol: 1e-12}
		return func() {
			if _, _, err := Odeint(robertsonF, []float64{1, 0, 0}, ts, o); err != nil {
				panic(err)
			}
		}
	}
	ts := make([]float64, 101)
	for i := range ts {
		ts[i] = 15 * float64(i) / 100
	}
	return func() {
		if _, _, err := Odeint(lotkaF, []float64{10, 5}, ts, nil); err != nil {
			panic(err)
		}
	}
}

// odeintWant: the allocations of one Odeint call (the output array, the info
// lists and the solver's own state, no interpolant per output time), exact on
// amd64 and arm64 (the two measured: 146 and 829 before the interpolant was
// reused). Another architecture steps differently where the compiler fuses
// multiply-adds, so it is held to a bound a per-time interpolant would break.
var odeintWant = map[string][2]float64{ // GOARCH: {Robertson, Lotka 101 times}
	"amd64": {106, 99},
	"arm64": {106, 99},
}

// TestOdeintAllocs: Odeint refills one interpolant per solver rather than
// allocating a fresh one (and copies of its coefficients) per output time.
func TestOdeintAllocs(t *testing.T) {
	if raceOn {
		t.Skip("the race detector changes allocation counts")
	}
	for i, name := range []string{"Robertson", "Lotka 101 times"} {
		call := odeintCall(i == 0)
		call()
		a := testing.AllocsPerRun(10, call)
		if want, ok := odeintWant[runtime.GOARCH]; ok {
			if a != want[i] {
				t.Errorf("%s: %v allocations per call on %s, want exactly %v", name, a, runtime.GOARCH, want[i])
			}
		} else if a > 130 {
			t.Errorf("%s: %v allocations per call, want at most 130", name, a)
		}
	}
}

// TestDenseReusedBitIdentical: the interpolant a solver refills evaluates to the
// same bits as a fresh one, step after step (stale coefficients, a changed order
// or method), and refilling it leaves the fresh copies taken earlier untouched.
func TestDenseReusedBitIdentical(t *testing.T) {
	for _, m := range []Method{RK23, RK45, DOP853, BDF, Radau, LSODA} {
		o := &IVPOptions{RTol: 1e-6, ATol: 1e-9}
		core, err := newCore(vdpF(30), 0, []float64{2, 0}, 120, o)
		if err != nil {
			t.Fatal(err)
		}
		sm, err := newMethod(core, m, o)
		if err != nil {
			t.Fatal(err)
		}
		type kept struct {
			d  denseOutput
			ts []float64
			ys [][]float64
		}
		var all []kept
		for steps := 0; !core.finished && steps < 400; steps++ {
			if ok, msg := core.step(sm); !ok {
				t.Fatalf("%v: %s", m, msg)
			}
			r := core.denseReused(sm)
			var ts []float64
			var ys [][]float64
			yr, yf := make([]float64, 2), make([]float64, 2)
			for _, x := range []float64{0, 0.25, 0.5, 0.9, 1} {
				tt := core.tOld + x*(core.t-core.tOld)
				r.eval(tt, yr)
				f := core.dense(sm)
				f.eval(tt, yf)
				for i := range yr {
					if math.Float64bits(yr[i]) != math.Float64bits(yf[i]) {
						t.Fatalf("%v step %d t=%v: reused %v, fresh %v", m, steps, tt, yr, yf)
					}
				}
				ts = append(ts, tt)
				ys = append(ys, append([]float64(nil), yf...))
				if x == 0.5 {
					all = append(all, kept{f, ts, ys})
				}
			}
		}
		if len(all) < 10 {
			t.Fatalf("%v: only %d steps", m, len(all))
		}
		y := make([]float64, 2)
		for k, c := range all {
			for j, tt := range c.ts {
				c.d.eval(tt, y)
				for i := range y {
					if math.Float64bits(y[i]) != math.Float64bits(c.ys[j][i]) {
						t.Fatalf("%v: fresh copy of step %d changed after later reuse", m, k)
					}
				}
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
