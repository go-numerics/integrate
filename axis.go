// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "github.com/go-ndarray/ndarray"

// alongAxis applies fn to every one-dimensional lane of y along axis
// (negative values count from the end). fn writes outLen values; with
// outLen 0 it is a reduction writing one value and the axis is dropped from
// the result, as numpy reductions drop it.
func alongAxis(y *ndarray.Array, axis, outLen int, fn func(lane, out []float64) error) (*ndarray.Array, error) {
	shape := y.Shape()
	nd := len(shape)
	if axis < 0 {
		axis += nd
	}
	if axis < 0 || axis >= nd {
		return nil, samplesErr("axis %d out of range for %d dimensions", axis, nd)
	}
	n := shape[axis]
	outer, inner := 1, 1
	for _, d := range shape[:axis] {
		outer *= d
	}
	for _, d := range shape[axis+1:] {
		inner *= d
	}
	flat := y.Flatten()
	m := outLen
	if m == 0 {
		m = 1
	}
	out := make([]float64, outer*m*inner)
	lane := make([]float64, n)
	res := make([]float64, m)
	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			for k := 0; k < n; k++ {
				lane[k] = flat.At(o*n*inner + k*inner + i)
			}
			if err := fn(lane, res); err != nil {
				return nil, err
			}
			for k := 0; k < m; k++ {
				out[o*m*inner+k*inner+i] = res[k]
			}
		}
	}
	oshape := append([]int(nil), shape[:axis]...)
	if outLen > 0 {
		oshape = append(oshape, outLen)
	}
	oshape = append(oshape, shape[axis+1:]...)
	return ndarray.FromData(out, oshape...)
}

// TrapezoidND integrates y along axis with the trapezoidal rule, at
// abscissae x (nil: spacing dx). It is SciPy's trapezoid(y, x, dx, axis) for
// an ndarray; the integrated axis is removed from the result.
func TrapezoidND(y *ndarray.Array, x []float64, dx float64, axis int) (*ndarray.Array, error) {
	return alongAxis(y, axis, 0, func(lane, out []float64) error {
		v, err := Trapezoid(lane, x, dx)
		out[0] = v
		return err
	})
}

// SimpsonND integrates y along axis with Simpson's rule (SciPy's
// simpson(y, x=x, dx=dx, axis=axis)).
func SimpsonND(y *ndarray.Array, x []float64, dx float64, axis int) (*ndarray.Array, error) {
	return alongAxis(y, axis, 0, func(lane, out []float64) error {
		v, err := Simpson(lane, x, dx)
		out[0] = v
		return err
	})
}

// CumulativeTrapezoidND is SciPy's cumulative_trapezoid(y, x, dx, axis,
// initial=0 if initial else None) for an ndarray.
func CumulativeTrapezoidND(y *ndarray.Array, x []float64, dx float64, axis int, initial bool) (*ndarray.Array, error) {
	shape := y.Shape()
	ax := axis
	if ax < 0 {
		ax += len(shape)
	}
	if ax < 0 || ax >= len(shape) {
		return nil, samplesErr("axis %d out of range for %d dimensions", axis, len(shape))
	}
	m := shape[ax]
	if !initial {
		m--
	}
	if m < 1 {
		return nil, samplesErr("CumulativeTrapezoidND needs at least 2 samples along the axis")
	}
	return alongAxis(y, axis, m, func(lane, out []float64) error {
		v, err := CumulativeTrapezoid(lane, x, dx, initial)
		copy(out, v)
		return err
	})
}
