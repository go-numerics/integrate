# Values SciPy returns on inputs that both sides form exactly.
import numpy as np
from scipy import integrate
def data(n):
    i = np.arange(n, dtype=float)
    y = ((i * i) % 97) / 7.0 - 3.0
    x = i + (i % 3) / 8.0
    return y, x
out = []
for n in (2, 3, 4, 7, 8, 9, 16, 17, 129, 130, 1000, 1001, 4097, 100000, 100001):
    y, x = data(n)
    out.append((n, "trapezoid_x", integrate.trapezoid(y, x)))
    out.append((n, "trapezoid_dx", integrate.trapezoid(y, dx=0.375)))
    out.append((n, "simpson_x", integrate.simpson(y, x=x)))
    out.append((n, "simpson_dx", integrate.simpson(y, dx=0.375)))
    c = integrate.cumulative_trapezoid(y, x, initial=0)
    out.append((n, "cumtrapz_last", c[-1]))
    out.append((n, "cumtrapz_mid", c[len(c) // 2]))
for k in (1, 2, 3, 4, 10, 16):
    y, _ = data(2 ** k + 1)
    out.append((2 ** k + 1, "romb", integrate.romb(y, dx=0.375)))
for n, name, v in out:
    print(f'\t{{{n}, "{name}", {float(v).hex()}}},  // {float(v)!r}')

# Along an axis of a 3-D array: NumPy sums a contiguous lane pairwise and
# any other axis row after row.
print("// ND")
for shape in ((3, 9, 5), (4, 8, 3), (2, 2, 3)):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=float)
    a = (((i * i) % 97) / 7.0 - 3.0).reshape(shape)
    for axis in range(3):
        m = shape[axis]
        j = np.arange(m, dtype=float)
        x = j + (j % 3) / 8.0
        res = {
            "trapezoid_x": integrate.trapezoid(a, x, axis=axis),
            "trapezoid_dx": integrate.trapezoid(a, dx=0.375, axis=axis),
            "simpson_x": integrate.simpson(a, x=x, axis=axis),
            "simpson_dx": integrate.simpson(a, dx=0.375, axis=axis),
            "cumtrapz_x": integrate.cumulative_trapezoid(a, x, axis=axis, initial=0),
            "cumtrapz_dx": integrate.cumulative_trapezoid(a, dx=0.375, axis=axis),
        }
        for name, v in res.items():
            vals = ", ".join(float(t).hex() for t in np.ravel(v))
            print(f'\t{{[]int{{{shape[0]}, {shape[1]}, {shape[2]}}}, {axis}, "{name}", []float64{{{vals}}}}},')
