package bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/integrate"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with scipy.integrate (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, inventory
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run TestDataParity -perf.parity .       both sides get the same bytes (needs uv)
//	go test -run TestWorkParity -perf.parity .       both sides call the integrand as often (needs uv)
//	go test -run TestRefSmoke -perf.smoke .          every SciPy call runs once (needs uv)
//
// What is compared. Every row times one call on both sides, on the same
// problem: the Go body is the public function, the SciPy body is the
// scipy.integrate function it ports (SciPy 1.18.1, NumPy 2.5.4, pinned with
// their hashes in refs.txt). The sample rules take arrays (100 000 points; the
// ND rules a 1000 x 100 grid), the same bytes on both sides.
//
// Callbacks. SciPy calls its integrand or right-hand side from Python, which is
// slow; each row says what it does about it:
//
//   - quad rows named "(libm)" integrate a C function of libm (sin, log, exp)
//     handed to SciPy as a scipy.LowLevelCallable built with ctypes, so no
//     Python runs inside the integrator; the Go side integrates math.Sin,
//     math.Log, math.Exp. The twin rows named "(Python)" integrate the same
//     function written in Python, so the cost of the callback is the difference.
//   - fixed_quad calls its function once on the array of nodes: the SciPy row
//     passes numpy.cos, a vectorised callback; the Go side calls math.Cos at
//     each node.
//   - dblquad, tplquad, nquad, solve_ivp and odeint have no compiled or
//     vectorised form of the callback in SciPy (a ctypes function must have
//     the signature double(double) or double(int, double*)), so those rows
//     call a Python function, and TestWorkParity checks that SciPy and Go call
//     it the same number of times (the same work), the equal counts of the
//     table in docs/perf.md. (LSODA and odeint are ODEPACK's lsoda, ported
//     from SciPy's lsoda.c: no row is pinned any more; a row whose algorithms
//     differ by design would carry its two counts in the test and report them.)

const (
	gold    = 0.6180339887498949
	plastic = 0.7548776662466927
)

func frac(i int, m float64) float64 { return math.Mod(float64(i)*m, 1) }

// The sample rules' data: x = 10 t^2 on t = i/(n-1), y a smooth signal with a
// little noise. No transcendental function, so that the bytes are NumPy's;
// every product is wrapped in float64 so arm64 does not fuse it.
func samples(n int) (y, x []float64) {
	x = make([]float64, n)
	y = make([]float64, n)
	for i := range x {
		t := float64(i) / float64(n-1)
		x[i] = float64(10*t) * t
		y[i] = float64(x[i]*(10-x[i]))/25 + float64(0.001*(frac(i, gold)-0.5))
	}
	return y, x
}

func grid() []float64 {
	g := make([]float64, 1000*100)
	for i := range g {
		g[i] = frac(i, gold)
	}
	return g
}

func ndGrid() *ndarray.Array {
	a, err := ndarray.FromData(grid(), 1000, 100)
	if err != nil {
		panic(err)
	}
	return a
}

const (
	pySamples = "(lambda i: (lambda x: [(x*(10.0-x))/25.0 + 0.001*(np.mod(i*%v,1.0)-0.5), x])((10.0*(i/(n-1)))*(i/(n-1))))(np.arange(n, dtype=np.float64))"
	pyGrid    = "[np.mod(np.arange(100000, dtype=np.float64)*%v, 1.0).reshape(1000, 100)]"
)

// The problems.
func lotkaF(_ float64, y, d []float64) {
	d[0] = 1.5*y[0] - y[0]*y[1]
	d[1] = -3*y[1] + y[0]*y[1]
}

func vdpF(_ float64, y, d []float64) {
	d[0] = y[1]
	d[1] = 1000*(1-y[0]*y[0])*y[1] - y[0]
}

func vdpJ(_ float64, y, j []float64) {
	j[0], j[1] = 0, 1
	j[2], j[3] = -2*1000*y[0]*y[1]-1, 1000*(1-y[0]*y[0])
}

func robertsonF(_ float64, y, d []float64) {
	d[0] = -0.04*y[0] + 1e4*y[1]*y[2]
	d[1] = 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1]
	d[2] = 3e7 * y[1] * y[1]
}

func linspace(a, b float64, m int) []float64 {
	v := make([]float64, m)
	for i := range v {
		v[i] = float64(float64(i)*(b-a)/float64(m-1)) + a
	}
	return v
}

var (
	sink    float64
	sinkS   []float64
	sinkErr error
	sinkAny any
)

// row is one case.
type row struct {
	name, fn, ref, parity string
	setup, names          string // the Python expression of the arguments, their names
	data                  [][]float64
	n                     int
	body                  string                 // the SciPy call, in the names
	run                   func(cb string) string // the SciPy call as a function of its callback (a callback row)
	raw                   string                 // the callback's Python source
	obj, use              string                 // a method: the object built outside the loop, the call timed
	mk                    func() func()
	goN                   func() int // calls of the integrand by Go (the work), with run: those of SciPy
	goNExact              func() int // the same on the Exact path (LSODA), which equals SciPy's; goN is then held to at most SciPy's
	skipWork              bool       // no count: a compiled callback cannot be counted
	pin                   [2]int     // a pinned (Go, SciPy) pair where the algorithms differ by design
	within                float64    // the relative difference of the counts tolerated (rounding of a stiff solve)
}

const imp = `__import__("scipy.integrate", fromlist=["_"])`

const prelude = "(lambda si, math, ct, sc, sp: "

func tail() string {
	return ")(" + imp + `, __import__("math"), __import__("ctypes"), __import__("scipy"), __import__("scipy.special", fromlist=["_"]))`
}

func (r row) setupExpr() string {
	if r.setup == "" {
		return "[]"
	}
	return r.setup
}

func (r row) call() string {
	switch {
	case r.run != nil:
		return prelude + "lambda: " + r.run(r.raw) + tail()
	case r.body != "":
		return prelude + "lambda " + r.names + ": " + r.body + tail()
	}
	return prelude + "(lambda s: lambda *_: " + r.use + ")(" + r.obj + ")" + tail()
}

// count is the Python expression of the number of calls SciPy makes of the callback.
func (r row) count() string {
	w := "(lambda g: lambda *a: (c.__setitem__(0, c[0]+1), g(*a))[1])(" + r.raw + ")"
	return prelude + "(lambda c: (" + r.run(w) + ", c[0])[1])([0])" + tail()
}

func (r row) size() int {
	if r.n != 0 {
		return r.n
	}
	return 1
}

// libm is a function of the C library as a scipy.LowLevelCallable.
func libm(name string) string {
	return "sc.LowLevelCallable((lambda f: (setattr(f, 'restype', ct.c_double), setattr(f, 'argtypes', [ct.c_double]), f)[2])(ct.CDLL(None)." + name + "))"
}

func rows() []row {
	var rs []row
	add := func(r row) { rs = append(rs, r) }
	const ex, tol = refbench.ParityExact, refbench.ParityTolerance

	// A quad row with a Python callback.
	quad := func(name, ref, raw string, f func(float64) float64, a, b float64, o *integrate.QuadOptions, args string) {
		add(row{name: name, fn: "Quad", ref: ref, parity: tol, raw: raw,
			run: func(cb string) string { return "si.quad(" + cb + ", " + args + ")" },
			mk: func() func() {
				return func() { r, err := integrate.Quad(f, a, b, o); sink, sinkErr = r.Value, err }
			},
			goN: func() int {
				n := 0
				r, _ := integrate.Quad(func(x float64) float64 { n++; return f(x) }, a, b, o)
				_ = r
				return n
			}})
	}
	// A quad row with a compiled (libm) callback: the same function as a twin row.
	quadC := func(name, ref, cname string, f func(float64) float64, a, b float64, o *integrate.QuadOptions, args string) {
		add(row{name: name, fn: "Quad", ref: ref, parity: tol, skipWork: true,
			body: "si.quad(" + libm(cname) + ", " + args + ")",
			mk: func() func() {
				return func() { r, err := integrate.Quad(f, a, b, o); sink, sinkErr = r.Value, err }
			}})
	}

	quad("Quad smooth (Python)", "quad(1/(1+25x^2), -1, 1)", "lambda x: 1/(1+25*x*x)",
		func(x float64) float64 { return 1 / (1 + 25*x*x) }, -1, 1, nil, "-1, 1")
	quad("Quad singular (Python)", "quad(log(x)/sqrt(x), 0, 1)", "lambda x: math.log(x)/math.sqrt(x)",
		func(x float64) float64 { return math.Log(x) / math.Sqrt(x) }, 0, 1, nil, "0, 1")
	quad("Quad infinite (Python)", "quad(exp(-x^2), -inf, inf)", "lambda x: math.exp(-x*x)",
		func(x float64) float64 { return math.Exp(-x * x) }, math.Inf(-1), math.Inf(1), nil, "-math.inf, math.inf")
	expm := func(x float64) float64 { return math.Exp(-x) }
	quad("Quad cos weight (Python)", "quad(exp(-x), 0, 10, weight='cos', wvar=20)", "lambda x: math.exp(-x)",
		expm, 0, 10, &integrate.QuadOptions{Weight: integrate.WeightCos, Omega: 20}, "0, 10, weight='cos', wvar=20")
	quad("Quad sin weight, infinite (Python)", "quad(exp(-x), 0, inf, weight='sin', wvar=1)", "lambda x: math.exp(-x)",
		expm, 0, math.Inf(1), &integrate.QuadOptions{Weight: integrate.WeightSin, Omega: 1}, "0, math.inf, weight='sin', wvar=1")
	quad("Quad alg-loga weight (Python)", "quad(1/(1+x), 0, 1, weight='alg-loga', wvar=(-0.5, 0))", "lambda x: 1/(1+x)",
		func(x float64) float64 { return 1 / (1 + x) }, 0, 1,
		&integrate.QuadOptions{Weight: integrate.WeightAlgLogA, Alpha: -0.5}, "0, 1, weight='alg-loga', wvar=(-0.5, 0)")
	quad("Quad cauchy weight (Python)", "quad(1/(1+x^2), -1, 2, weight='cauchy', wvar=0.5)", "lambda x: 1/(1+x*x)",
		func(x float64) float64 { return 1 / (1 + x*x) }, -1, 2,
		&integrate.QuadOptions{Weight: integrate.WeightCauchy, C: 0.5}, "-1, 2, weight='cauchy', wvar=0.5")
	quad("Quad points (Python)", "quad(sqrt|x-0.3|, 0, 1, points=[0.3])", "lambda x: math.sqrt(abs(x-0.3))",
		func(x float64) float64 { return math.Sqrt(math.Abs(x - 0.3)) }, 0, 1,
		&integrate.QuadOptions{Points: []float64{0.3}}, "0, 1, points=[0.3]")
	quad("Quad log (Python)", "quad(log, 0, 1)", "lambda x: math.log(x)", math.Log, 0, 1, nil, "0, 1")
	quad("Quad sin 0..100 (Python)", "quad(sin, 0, 100)", "lambda x: math.sin(x)", math.Sin, 0, 100, nil, "0, 100")
	quadC("Quad log (libm)", "quad(libm log, 0, 1)", "log", math.Log, 0, 1, nil, "0, 1")
	quadC("Quad sin 0..100 (libm)", "quad(libm sin, 0, 100)", "sin", math.Sin, 0, 100, nil, "0, 100")
	quadC("Quad cos weight (libm)", "quad(libm exp, 0, 5, weight='cos', wvar=20)", "exp", math.Exp, 0, 5,
		&integrate.QuadOptions{Weight: integrate.WeightCos, Omega: 20}, "0, 5, weight='cos', wvar=20")

	add(row{name: "DblQuad", fn: "DblQuad", ref: "dblquad(exp(-xy), 0, 1, 0, x)", parity: tol,
		raw: "lambda y, x: math.exp(-x*y)",
		run: func(cb string) string { return "si.dblquad(" + cb + ", 0, 1, 0, lambda x: x)" },
		mk: func() func() {
			f := func(y, x float64) float64 { return math.Exp(-x * y) }
			return func() {
				r, err := integrate.DblQuad(f, 0, 1, integrate.Const(0), func(x float64) float64 { return x }, nil)
				sink, sinkErr = r.Value, err
			}
		},
		goN: func() int {
			n := 0
			f := func(y, x float64) float64 { n++; return math.Exp(-x * y) }
			integrate.DblQuad(f, 0, 1, integrate.Const(0), func(x float64) float64 { return x }, nil)
			return n
		}})
	add(row{name: "TplQuad", fn: "TplQuad", ref: "tplquad(exp(-(x^2+y^2+z^2)), unit cube)", parity: tol,
		raw: "lambda z, y, x: math.exp(-(x*x+y*y+z*z))",
		run: func(cb string) string { return "si.tplquad(" + cb + ", 0, 1, 0, 1, 0, 1)" },
		mk: func() func() {
			f := func(z, y, x float64) float64 { return math.Exp(-(x*x + y*y + z*z)) }
			return func() {
				r, err := integrate.TplQuad(f, 0, 1, integrate.Const(0), integrate.Const(1), integrate.Const2(0), integrate.Const2(1), nil)
				sink, sinkErr = r.Value, err
			}
		},
		goN: func() int {
			n := 0
			f := func(z, y, x float64) float64 { n++; return math.Exp(-(x*x + y*y + z*z)) }
			integrate.TplQuad(f, 0, 1, integrate.Const(0), integrate.Const(1), integrate.Const2(0), integrate.Const2(1), nil)
			return n
		}})
	add(row{name: "NQuad", fn: "NQuad", ref: "nquad(exp(-|x|^2), unit cube)", parity: tol,
		raw: "lambda x0, x1, x2: math.exp(-(x0*x0+x1*x1+x2*x2))",
		run: func(cb string) string { return "si.nquad(" + cb + ", [[0, 1], [0, 1], [0, 1]])" },
		mk: func() func() {
			f := func(x []float64) float64 { return math.Exp(-(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])) }
			r := []integrate.Range{integrate.ConstRange(0, 1), integrate.ConstRange(0, 1), integrate.ConstRange(0, 1)}
			return func() { q, err := integrate.NQuad(f, r, nil); sink, sinkErr = q.Value, err }
		},
		goN: func() int {
			n := 0
			f := func(x []float64) float64 { n++; return math.Exp(-(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])) }
			r := []integrate.Range{integrate.ConstRange(0, 1), integrate.ConstRange(0, 1), integrate.ConstRange(0, 1)}
			integrate.NQuad(f, r, nil)
			return n
		}})
	add(row{name: "FixedQuad (vectorised numpy.cos)", fn: "FixedQuad", ref: "fixed_quad(np.cos, 0, 1, n=5)", parity: tol,
		body: "si.fixed_quad(np.cos, 0, 1, n=5)",
		mk: func() func() {
			return func() { sink, sinkErr = integrate.FixedQuad(math.Cos, 0, 1, 5) }
		}})
	add(row{name: "GaussLegendre 50 (cached)", fn: "GaussLegendre", ref: "_cached_roots_legendre(50)", parity: ex,
		body: "si._quadrature._cached_roots_legendre(50)",
		mk: func() func() {
			integrate.GaussLegendre(50)
			return func() { x, w := integrate.GaussLegendre(50); sink = x[0] + w[0] }
		}})

	// The sample rules.
	const m = 100_000
	y, x := samples(m)
	d5 := func(n int) (string, [][]float64) {
		yy, xx := samples(n)
		return fmt.Sprintf(pySamples, gold), [][]float64{yy, xx}
	}
	s5, dat5 := d5(m)
	arr := func(name, fn, ref, body string, n int, setup string, data [][]float64, names string, mk func() func()) {
		add(row{name: name, fn: fn, ref: ref, parity: ex, setup: setup, names: names, data: data, n: n, body: body, mk: mk})
	}
	arr("Trapezoid 1e5", "Trapezoid", "trapezoid(y, x)", "si.trapezoid(y, x)", m, s5, dat5, "y, x",
		func() func() { return func() { sink, sinkErr = integrate.Trapezoid(y, x, 0) } })
	arr("Simpson 1e5", "Simpson", "simpson(y, x=x)", "si.simpson(y, x=x)", m, s5, dat5, "y, x",
		func() func() { return func() { sink, sinkErr = integrate.Simpson(y, x, 0) } })
	arr("CumulativeTrapezoid 1e5", "CumulativeTrapezoid", "cumulative_trapezoid(y, x, initial=0)", "si.cumulative_trapezoid(y, x, initial=0)", m, s5, dat5, "y, x",
		func() func() { return func() { sinkS, sinkErr = integrate.CumulativeTrapezoid(y, x, 0, true) } })
	const mr = 1<<16 + 1
	yr, _ := samples(mr)
	sr, datr := d5(mr)
	arr("Romb 65537", "Romb", "romb(y, dx=0.001)", "si.romb(y, dx=0.001)", mr, sr, datr, "y, x",
		func() func() { return func() { sink, sinkErr = integrate.Romb(yr, 0.001) } })
	g := ndGrid()
	pg := fmt.Sprintf(pyGrid, gold)
	gd := [][]float64{grid()}
	ndr := func(name, fn, ref, body string, mk func() func()) {
		arr(name, fn, ref, body, 100000, pg, gd, "g", mk)
	}
	ndr("TrapezoidND axis 0", "TrapezoidND", "trapezoid(g, dx=0.1, axis=0)", "si.trapezoid(g, dx=0.1, axis=0)",
		func() func() { return func() { sinkAny, sinkErr = integrate.TrapezoidND(g, nil, 0.1, 0) } })
	ndr("TrapezoidND axis 1", "TrapezoidND", "trapezoid(g, dx=0.1, axis=1)", "si.trapezoid(g, dx=0.1, axis=1)",
		func() func() { return func() { sinkAny, sinkErr = integrate.TrapezoidND(g, nil, 0.1, 1) } })
	ndr("SimpsonND axis 1", "SimpsonND", "simpson(g, dx=0.1, axis=1)", "si.simpson(g, dx=0.1, axis=1)",
		func() func() { return func() { sinkAny, sinkErr = integrate.SimpsonND(g, nil, 0.1, 1) } })
	ndr("SimpsonND axis 0", "SimpsonND", "simpson(g, dx=0.1, axis=0)", "si.simpson(g, dx=0.1, axis=0)",
		func() func() { return func() { sinkAny, sinkErr = integrate.SimpsonND(g, nil, 0.1, 0) } })
	ndr("CumulativeTrapezoidND axis 0", "CumulativeTrapezoidND", "cumulative_trapezoid(g, dx=0.1, axis=0, initial=0)",
		"si.cumulative_trapezoid(g, dx=0.1, axis=0, initial=0)",
		func() func() {
			return func() { sinkAny, sinkErr = integrate.CumulativeTrapezoidND(g, nil, 0.1, 0, true) }
		})

	// Initial value problems.
	lotka := "lambda t, y: [1.5*y[0] - y[0]*y[1], -3*y[1] + y[0]*y[1]]"
	vdp := "lambda t, y: [y[1], 1000.0*(1 - y[0]**2)*y[1] - y[0]]"
	vdpjac := "lambda t, y: [[0, 1], [-2*1000.0*y[0]*y[1] - 1, 1000.0*(1 - y[0]**2)]]"
	// ivp adds a SolveIVP row: the SciPy call is run(cb), the Go call solves and returns the solution.
	ivp := func(name, ref, raw string, run func(cb string) string, f integrate.ODEFunc, span [2]float64, y0 []float64, o *integrate.IVPOptions, pin [2]int) {
		var exact func() int
		if o != nil && o.Method == integrate.LSODA {
			oe := *o
			oe.Exact = true
			exact = func() int {
				n := 0
				integrate.SolveIVP(func(t float64, y, d []float64) { n++; f(t, y, d) }, span, y0, &oe)
				return n
			}
		}
		add(row{name: name, fn: "SolveIVP", ref: ref, parity: tol, raw: raw, run: run, pin: pin, goNExact: exact,
			mk: func() func() {
				return func() { s, err := integrate.SolveIVP(f, span, y0, o); sinkAny, sinkErr = s, err }
			},
			goN: func() int {
				n := 0
				integrate.SolveIVP(func(t float64, y, d []float64) { n++; f(t, y, d) }, span, y0, o)
				return n
			}})
	}
	l0, v0 := []float64{10, 5}, []float64{2, 0}
	ivp("SolveIVP Lotka RK45", "solve_ivp(lotka, (0, 15), [10, 5])", lotka,
		func(cb string) string { return "si.solve_ivp(" + cb + ", (0, 15), [10, 5])" }, lotkaF, [2]float64{0, 15}, l0, nil, [2]int{})
	ivp("SolveIVP Lotka RK23", "solve_ivp(lotka, method='RK23')", lotka,
		func(cb string) string { return "si.solve_ivp(" + cb + ", (0, 15), [10, 5], method='RK23')" }, lotkaF, [2]float64{0, 15}, l0,
		&integrate.IVPOptions{Method: integrate.RK23}, [2]int{})
	ivp("SolveIVP Lotka DOP853 tight", "solve_ivp(lotka, method='DOP853', rtol=1e-10, atol=1e-12)", lotka,
		func(cb string) string {
			return "si.solve_ivp(" + cb + ", (0, 15), [10, 5], method='DOP853', rtol=1e-10, atol=1e-12)"
		}, lotkaF, [2]float64{0, 15}, l0, &integrate.IVPOptions{Method: integrate.DOP853, RTol: 1e-10, ATol: 1e-12}, [2]int{})
	te := linspace(0, 15, 1001)
	ivp("SolveIVP Lotka dense t_eval", "solve_ivp(lotka, t_eval=1001 points, dense_output=True)", lotka,
		func(cb string) string {
			return "si.solve_ivp(" + cb + ", (0, 15), [10, 5], t_eval=15.0*np.arange(1001)/1000, dense_output=True)"
		}, lotkaF, [2]float64{0, 15}, l0, &integrate.IVPOptions{TEval: te, DenseOutput: true}, [2]int{})
	ivp("SolveIVP Lotka event", "solve_ivp(lotka, events=y0-5)", lotka,
		func(cb string) string {
			return "si.solve_ivp(" + cb + ", (0, 15), [10, 5], events=lambda t, y: y[0] - 5)"
		}, lotkaF, [2]float64{0, 15}, l0,
		&integrate.IVPOptions{Events: []integrate.Event{{F: func(_ float64, y []float64) float64 { return y[0] - 5 }}}}, [2]int{})
	vd := func(name, method string, m integrate.Method, jac bool, pin [2]int, within float64) {
		o := &integrate.IVPOptions{Method: m}
		py := "si.solve_ivp(%s, (0, 3000), [2, 0], method='" + method + "'"
		if jac {
			o.Jac = vdpJ
			py += ", jac=" + vdpjac
		}
		ivp(name, "solve_ivp(vdp mu=1000, method='"+method+"'"+map[bool]string{true: ", jac", false: ""}[jac]+")", vdp,
			func(cb string) string { return fmt.Sprintf(py, cb) + ")" }, vdpF, [2]float64{0, 3000}, v0, o, pin)
		rs[len(rs)-1].within = within
	}
	// Radau and BDF: the same algorithm as SciPy's; the stiff Newton iteration is
	// sensitive to the rounding of the LU solves, so a few steps differ (Radau:
	// 285 against 283 and 282 against 285 steps; BDF: one call).
	vd("SolveIVP VdP1000 Radau", "Radau", integrate.Radau, false, [2]int{}, 0.02)
	vd("SolveIVP VdP1000 Radau jac", "Radau", integrate.Radau, true, [2]int{}, 0.02)
	vd("SolveIVP VdP1000 BDF", "BDF", integrate.BDF, false, [2]int{}, 0.01)
	vd("SolveIVP VdP1000 BDF jac", "BDF", integrate.BDF, true, [2]int{}, 0)
	// LSODA is ODEPACK's (SciPy 1.18.1's lsoda.c): the same evaluation count.
	vd("SolveIVP VdP1000 LSODA", "LSODA", integrate.LSODA, false, [2]int{}, 0)

	// Odeint.
	robertson := "lambda t, y: [-0.04*y[0] + 1e4*y[1]*y[2], 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]**2, 3e7*y[1]**2]"
	tsR := []float64{0, 0.4, 4, 40, 400, 4000, 40000}
	oR := &integrate.OdeintOptions{RTol: 1e-8, ATol: 1e-12}
	odeint := func(name, ref, raw string, run func(cb string) string, f integrate.ODEFunc, y0, ts []float64, o *integrate.OdeintOptions, pin [2]int) {
		var oe integrate.OdeintOptions
		if o != nil {
			oe = *o
		}
		oe.Exact = true
		add(row{name: name, fn: "Odeint", ref: ref, parity: tol, raw: raw, run: run, pin: pin,
			goNExact: func() int {
				n := 0
				integrate.Odeint(func(t float64, y, d []float64) { n++; f(t, y, d) }, y0, ts, &oe)
				return n
			},
			mk: func() func() {
				return func() { a, _, err := integrate.Odeint(f, y0, ts, o); sinkAny, sinkErr = a, err }
			},
			goN: func() int {
				n := 0
				integrate.Odeint(func(t float64, y, d []float64) { n++; f(t, y, d) }, y0, ts, o)
				return n
			}})
	}
	odeint("Odeint Robertson", "odeint(robertson, rtol=1e-8, atol=1e-12)", robertson,
		func(cb string) string {
			return "si.odeint(" + cb + ", [1, 0, 0], np.array([0, 0.4, 4, 40, 400, 4000, 40000]), rtol=1e-8, atol=1e-12, tfirst=True)"
		}, robertsonF, []float64{1, 0, 0}, tsR, oR, [2]int{})
	odeint("Odeint Lotka 101 times", "odeint(lotka, 101 times)", lotka,
		func(cb string) string { return "si.odeint(" + cb + ", [10, 5], 15.0*np.arange(101)/100, tfirst=True)" },
		lotkaF, l0, linspace(0, 15, 101), nil, [2]int{})

	// The dense output of a solution.
	dense, _ := integrate.SolveIVP(lotkaF, [2]float64{0, 15}, l0, &integrate.IVPOptions{DenseOutput: true})
	const dobj = "si.solve_ivp(lambda t, y: [1.5*y[0] - y[0]*y[1], -3*y[1] + y[0]*y[1]], (0, 15), [10, 5], dense_output=True)"
	buf := make([]float64, 2)
	add(row{name: "ODESolution.Eval scalar", fn: "ODESolution.Eval", ref: "sol.sol(7.5)", parity: tol, obj: dobj, use: "s.sol(7.5)",
		mk: func() func() { return func() { sinkS = dense.Sol.Eval(7.5) } }})
	add(row{name: "ODESolution.EvalInto scalar", fn: "ODESolution.EvalInto", ref: "sol.sol(7.5)", parity: tol, obj: dobj, use: "s.sol(7.5)",
		mk: func() func() { return func() { dense.Sol.EvalInto(7.5, buf); sink = buf[0] } }})
	add(row{name: "ODESolution.EvalArray 1001 points", fn: "ODESolution.EvalArray", ref: "sol.sol(t_array)", parity: tol,
		obj: dobj, use: "s.sol(15.0*np.arange(1001)/1000)",
		mk: func() func() { return func() { sinkAny = dense.Sol.EvalArray(te) } }})
	return rs
}

func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range rows() {
		body := r.mk()
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(r.setupExpr(), r.call(), r.size())
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: "scipy 1.18.1 integrate." + r.ref,
			Parity:    r.parity,
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					body()
				}
			},
			Ref: ref,
		})
	}
	return out
}

var (
	parityFlag = flag.Bool("perf.parity", false, "TestDataParity and TestWorkParity: compare with SciPy's (starts the worker through uv)")
	smokeFlag  = flag.Bool("perf.smoke", false, "TestRefSmoke: run every SciPy call once")
)

func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

// archConfig: the stiff rows store one state per step and take a different number
// of steps where the compiler fuses multiply-adds (arm64), so their allocation
// counts are recorded per GOARCH, as exact values: docs/perf.json and
// docs/perf-table.md for amd64 (and any other architecture without its own
// files), docs/perf-arm64.json and docs/perf-table-arm64.md for arm64.
// refbench v0.1.2 holds one report per file, hence one file pair per GOARCH
// (go-scikit/refbench asks for a per-GOARCH report).
func archConfig(cs []refbench.Case) refbench.CIConfig {
	cfg := refbench.CIConfig{Root: "..", Cases: cs}
	if runtime.GOARCH == "arm64" {
		cfg.JSON = filepath.Join("..", "docs", "perf-arm64.json")
		cfg.Table = filepath.Join("..", "docs", "perf-table-arm64.md")
	}
	return cfg
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	opt := refbench.Options{Repo: "go-numerics/integrate"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, archConfig(cases(w)), opt)
}

func digest(arrs [][]float64) string {
	h := sha256.New()
	for _, a := range arrs {
		for _, v := range a {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TestDataParity checks, for every row with array data, that the Go arrays and
// SciPy's are the same bytes: a ratio between two functions on different data
// means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	seen := map[string]bool{}
	for _, r := range rows() {
		if r.setup == "" {
			continue
		}
		key := fmt.Sprint(r.setup, r.size())
		if seen[key] {
			continue
		}
		seen[key] = true
		got, err := w.DataHash(r.setup, r.size())
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		if want := digest(r.data); got != want {
			t.Errorf("%s: SciPy data %s, Go data %s", r.name, got, want)
		}
	}
}

// TestWorkParity checks that both sides do the same work: the number of calls of
// the integrand or right-hand side (the quantity that decides the work of an
// adaptive integrator; SciPy's nfev excludes the calls of finite-difference
// Jacobians, so both sides count every call). The Python side wraps the
// callback in a counter and hands the count back as the argument to hash.
// Where the algorithm differs by design the pair (Go, SciPy) is pinned, so that
// a change on either side is noticed; none does now (LSODA and odeint call the
// callback exactly as often as SciPy's: 1803, 1683 and 1243 times, were 2594,
// 2222 and 1880 before the port of ODEPACK's lsoda).
func TestWorkParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, r := range rows() {
		if r.run == nil || r.skipWork {
			continue
		}
		setup := "[np.array([float(" + r.count() + ")])]"
		got, err := w.DataHash(setup, 1)
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		goN := r.goN()
		if r.goNExact != nil {
			// LSODA: the Exact path is lsoda.c's (the same count), the default
			// path may do better (never worse); its error is held to the
			// accuracy contract by the tests of the package.
			sci := findCount(got)
			if want := digest([][]float64{{float64(r.goNExact())}}); got != want {
				t.Errorf("%s: SciPy calls the callback %d times, Go (Exact) %d", r.name, sci, r.goNExact())
			}
			if goN > sci {
				t.Errorf("%s: SciPy calls the callback %d times, Go %d: more", r.name, sci, goN)
			}
			t.Logf("%s: Go %d calls, Exact %d, SciPy %d", r.name, goN, r.goNExact(), sci)
			continue
		}
		if r.pin[0] > 0 {
			// pinned: both counts must be the recorded ones
			if goN != r.pin[0] || got != digest([][]float64{{float64(r.pin[1])}}) {
				t.Errorf("%s: pinned (Go %d, SciPy %d) no longer holds: Go calls %d times, SciPy %d", r.name, r.pin[0], r.pin[1], goN, findCount(got))
			}
			t.Logf("%s: Go %d calls, SciPy %d (by design: a different method switch)", r.name, goN, r.pin[1])
			continue
		}
		if r.within > 0 {
			// the same algorithm; a few steps of a stiff problem differ with the rounding of the linear algebra
			sci := findCount(got)
			if d := math.Abs(float64(goN-sci)) / float64(sci); d > r.within {
				t.Errorf("%s: SciPy calls the callback %d times, Go %d (%.1f %% apart, over %.1f %%)", r.name, sci, goN, 100*d, 100*r.within)
			}
			t.Logf("%s: Go %d calls, SciPy %d", r.name, goN, sci)
			continue
		}
		if want := digest([][]float64{{float64(goN)}}); got != want {
			t.Errorf("%s: SciPy calls the callback %d times, Go %d", r.name, findCount(got), goN)
		}
	}
}

// findCount recovers the count behind a digest of one number (for a failure
// message only): the digest is the SHA-256 of the float64's bytes.
func findCount(h string) int {
	for n := 0; n <= 200_000; n++ {
		if digest([][]float64{{float64(n)}}) == h {
			return n
		}
	}
	return -1
}

// TestRefSmoke runs every SciPy call once: a typo in a reference expression is a
// failure here, not in a timing run.
func TestRefSmoke(t *testing.T) {
	if !*smokeFlag {
		t.Skip("set -perf.smoke (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, c := range cases(w) {
		if _, err := c.Ref(1); err != nil {
			t.Errorf("%s: %v", c.Name, err)
		}
	}
}
