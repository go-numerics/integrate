// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

import "github.com/go-ndarray/ndarray"

// NumPy reduces a contiguous lane (the last axis) with its pairwise sum, but
// any other axis by adding whole rows one after the other: each output
// element is then a running sum from 0.0 in index order. The ND rules follow
// both orders, so that they return SciPy's values, and the row order lets
// every lane advance in the same pass.

// axisLayout is y's data in row-major order, viewed as outer × n × inner with
// n the length of the integrated axis.
type axisLayout struct {
	flat            []float64
	outer, n, inner int
	shape           []int
	axis            int
}

func layout(y *ndarray.Array, axis int) (axisLayout, error) {
	shape := y.Shape()
	nd := len(shape)
	if axis < 0 {
		axis += nd
	}
	if axis < 0 || axis >= nd {
		return axisLayout{}, samplesErr("axis %d out of range for %d dimensions", axis, nd)
	}
	l := axisLayout{n: shape[axis], outer: 1, inner: 1, shape: shape, axis: axis}
	for _, d := range shape[:axis] {
		l.outer *= d
	}
	for _, d := range shape[axis+1:] {
		l.inner *= d
	}
	// The array's own storage when it is a contiguous float64 array (it is
	// only read), else a converted copy.
	if y.DType() == ndarray.Float64 {
		l.flat, _ = ndarray.Data[float64](y)
	} else {
		f := y.Flatten()
		l.flat = make([]float64, f.Size())
		for k := range l.flat {
			l.flat[k] = f.At(k)
		}
	}
	return l, nil
}

// row returns y[o, k, :] (inner values).
func (l axisLayout) row(o, k int) []float64 {
	s := (o*l.n + k) * l.inner
	return l.flat[s : s+l.inner]
}

// result shapes out as y's shape with the axis replaced by outLen samples
// (removed when outLen is 0).
func (l axisLayout) result(out []float64, outLen int) (*ndarray.Array, error) {
	oshape := append([]int(nil), l.shape[:l.axis]...)
	if outLen > 0 {
		oshape = append(oshape, outLen)
	}
	oshape = append(oshape, l.shape[l.axis+1:]...)
	return ndarray.FromData(out, oshape...)
}

// lanes applies fn to every contiguous lane (inner == 1); fn writes m
// values.
func (l axisLayout) lanes(m int, fn func(lane, out []float64) error) ([]float64, error) {
	out := make([]float64, l.outer*m)
	for o := 0; o < l.outer; o++ {
		if err := fn(l.flat[o*l.n:o*l.n+l.n], out[o*m:o*m+m]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// TrapezoidND integrates y along axis with the trapezoidal rule, at
// abscissae x (nil: spacing dx). It is SciPy's trapezoid(y, x, dx, axis) for
// an ndarray; the integrated axis is removed from the result.
func TrapezoidND(y *ndarray.Array, x []float64, dx float64, axis int) (*ndarray.Array, error) {
	l, err := layout(y, axis)
	if err != nil {
		return nil, err
	}
	var out []float64
	if l.inner == 1 {
		out, err = trapezoidLanes(l, x, dx)
	} else {
		out, err = trapezoidRows(l, x, dx)
	}
	if err != nil {
		return nil, err
	}
	return l.result(out, 0)
}

// trapezoidLanes integrates every contiguous lane, sharing one buffer for
// their terms.
func trapezoidLanes(l axisLayout, x []float64, dx float64) ([]float64, error) {
	out := make([]float64, l.outer)
	if l.outer == 0 {
		return out, nil
	}
	if err := checkLen(l.n, x, 0, "Trapezoid"); err != nil {
		return nil, err
	}
	scratch := make([]float64, l.n)
	for o := range out {
		out[o] = trapezoidLane(l.flat[o*l.n:o*l.n+l.n], x, dx, scratch)
	}
	return out, nil
}

func trapezoidRows(l axisLayout, x []float64, dx float64) ([]float64, error) {
	out := make([]float64, l.outer*l.inner)
	if l.outer == 0 || l.inner == 0 { // no lanes
		return out, nil
	}
	if err := checkLen(l.n, x, 0, "Trapezoid"); err != nil {
		return nil, err
	}
	for o := 0; o < l.outer; o++ {
		acc := out[o*l.inner : (o+1)*l.inner]
		for k := 0; k+1 < l.n; k++ {
			d := spacing(x, dx, k)
			r0, r1 := l.row(o, k), l.row(o, k+1)
			for i := range acc {
				acc[i] += float64(d*(r1[i]+r0[i])) / 2
			}
		}
	}
	return out, nil
}

// SimpsonND integrates y along axis with Simpson's rule (SciPy's
// simpson(y, x=x, dx=dx, axis=axis)).
func SimpsonND(y *ndarray.Array, x []float64, dx float64, axis int) (*ndarray.Array, error) {
	l, err := layout(y, axis)
	if err != nil {
		return nil, err
	}
	var out []float64
	if l.inner == 1 {
		out, err = simpsonLanes(l, x, dx)
	} else {
		out, err = simpsonRows(l, x, dx)
	}
	if err != nil {
		return nil, err
	}
	return l.result(out, 0)
}

// simpsonLanes is trapezoidLanes for Simpson's rule.
func simpsonLanes(l axisLayout, x []float64, dx float64) ([]float64, error) {
	out := make([]float64, l.outer)
	if l.outer == 0 {
		return out, nil
	}
	if err := checkLen(l.n, x, 2, "Simpson"); err != nil {
		return nil, err
	}
	scratch := make([]float64, l.n)
	for o := range out {
		out[o] = simpsonLane(l.flat[o*l.n:o*l.n+l.n], x, dx, scratch)
	}
	return out, nil
}

func simpsonRows(l axisLayout, x []float64, dx float64) ([]float64, error) {
	out := make([]float64, l.outer*l.inner)
	if l.outer == 0 || l.inner == 0 { // no lanes
		return out, nil
	}
	n := l.n
	if err := checkLen(n, x, 2, "Simpson"); err != nil {
		return nil, err
	}
	for o := 0; o < l.outer; o++ {
		acc := out[o*l.inner : (o+1)*l.inner]
		if n == 2 {
			c := float64(0.5 * spacing(x, dx, 0))
			r0, r1 := l.row(o, 0), l.row(o, 1)
			for i := range acc {
				acc[i] = 0 + c*(r1[i]+r0[i])
			}
			continue
		}
		stop := n - 2
		if n%2 == 0 {
			stop = n - 3
		}
		// _basic_simpson's panel terms, summed row after row.
		for p := 0; p < stop; p += 2 {
			r0, r1, r2 := l.row(o, p), l.row(o, p+1), l.row(o, p+2)
			if x == nil {
				for i := range acc {
					acc[i] += r0[i] + float64(4*r1[i]) + r2[i]
				}
				continue
			}
			h0 := x[p+1] - x[p]
			h1 := x[p+2] - x[p+1]
			hsum := h0 + h1
			hprod := h0 * h1
			h0divh1 := div0(h0, h1)
			c0, c1, c2 := 2-div0(1, h0divh1), float64(hsum*div0(hsum, hprod)), 2-h0divh1
			for i := range acc {
				acc[i] += float64(hsum / 6 * (float64(r0[i]*c0) + float64(r1[i]*c1) + float64(r2[i]*c2)))
			}
		}
		if x == nil {
			for i := range acc {
				acc[i] *= float64(dx / 3)
			}
		}
		if n%2 == 0 {
			alpha, beta, eta := simpsonEnd(x, dx, n)
			y1, y2, y3 := l.row(o, n-1), l.row(o, n-2), l.row(o, n-3)
			for i := range acc {
				corr := float64(alpha*y1[i]) + float64(beta*y2[i]) - float64(eta*y3[i])
				acc[i] = acc[i] + corr + 0
			}
		}
	}
	return out, nil
}

// CumulativeTrapezoidND is SciPy's cumulative_trapezoid(y, x, dx, axis,
// initial=0 if initial else None) for an ndarray.
func CumulativeTrapezoidND(y *ndarray.Array, x []float64, dx float64, axis int, initial bool) (*ndarray.Array, error) {
	l, err := layout(y, axis)
	if err != nil {
		return nil, err
	}
	m := l.n
	if !initial {
		m--
	}
	if m < 1 {
		return nil, samplesErr("CumulativeTrapezoidND needs at least 2 samples along the axis")
	}
	var out []float64
	if l.inner == 1 {
		out, err = l.lanes(m, func(lane, o []float64) error {
			v, err := CumulativeTrapezoid(lane, x, dx, initial)
			copy(o, v)
			return err
		})
	} else {
		out, err = cumulativeRows(l, x, dx, initial, m)
	}
	if err != nil {
		return nil, err
	}
	return l.result(out, m)
}

func cumulativeRows(l axisLayout, x []float64, dx float64, initial bool, m int) ([]float64, error) {
	out := make([]float64, l.outer*m*l.inner)
	if l.outer == 0 || l.inner == 0 { // no lanes
		return out, nil
	}
	if err := checkLen(l.n, x, 1, "CumulativeTrapezoid"); err != nil {
		return nil, err
	}
	off := 0
	if initial {
		off = 1 // the first row stays 0
	}
	for o := 0; o < l.outer; o++ {
		base := out[o*m*l.inner : (o+1)*m*l.inner]
		for k := 0; k+1 < l.n; k++ {
			d := spacing(x, dx, k)
			r0, r1 := l.row(o, k), l.row(o, k+1)
			dst := base[(off+k)*l.inner : (off+k+1)*l.inner]
			if k == 0 {
				for i := range dst {
					dst[i] = float64(d*(r1[i]+r0[i])) / 2
				}
				continue
			}
			prev := base[(off+k-1)*l.inner : (off+k)*l.inner]
			for i := range dst {
				dst[i] = prev[i] + float64(d*(r1[i]+r0[i]))/2
			}
		}
	}
	return out, nil
}
