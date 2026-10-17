// Copyright (c) 2026, the go-numerics/integrate authors
// All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package integrate

// npSum returns the sum of the n terms that fill writes (fill(lo, dst) sets
// dst[k] to term lo+k), in the order of numpy.add.reduce on a contiguous
// float64 array: 0.0 plus NumPy's pairwise sum, which adds blocks of at most
// 128 terms with eight running accumulators and combines the blocks
// pairwise. SciPy's trapezoid and simpson sum their terms this way, so the
// rules that use npSum return SciPy's value to the last bit when their terms
// are formed as SciPy forms them. Its error grows as O(log n) instead of
// O(n), and its eight independent accumulators make it several times faster
// than a running sum.
func npSum(n int, fill func(lo int, dst []float64)) float64 {
	var buf [pwBlock]float64
	return 0.0 + pairwise(0, n, fill, &buf)
}

// pwBlock is NumPy's PW_BLOCKSIZE.
const pwBlock = 128

// pairwise is numpy's pairwise_sum_DOUBLE over terms lo..lo+n-1, formed
// block by block into buf.
func pairwise(lo, n int, fill func(lo int, dst []float64), buf *[pwBlock]float64) float64 {
	if n <= pwBlock {
		a := buf[:n]
		fill(lo, a)
		return pairwiseSlice(a)
	}
	n2 := n / 2
	n2 -= n2 % 8
	return pairwise(lo, n2, fill, buf) + pairwise(lo+n2, n-n2, fill, buf)
}

// pairwiseSlice is numpy's pairwise_sum_DOUBLE over the terms in a (without
// the leading 0.0 of add.reduce).
func pairwiseSlice(a []float64) float64 {
	n := len(a)
	switch {
	case n < 8:
		res := 0.0
		for _, v := range a {
			res += v
		}
		return res
	case n <= pwBlock:
		r0, r1, r2, r3, r4, r5, r6, r7 := a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
		i := 8
		for ; i < n-n%8; i += 8 {
			b := a[i : i+8 : i+8]
			r0 += b[0]
			r1 += b[1]
			r2 += b[2]
			r3 += b[3]
			r4 += b[4]
			r5 += b[5]
			r6 += b[6]
			r7 += b[7]
		}
		res := ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return pairwiseSlice(a[:n2]) + pairwiseSlice(a[n2:])
}
