"""Numerical integration (Lecture Notes Ch. 5).

Data-based  : composite trapezoidal, Simpson 1/3, Simpson 3/8, mixed 3/8 + 1/3,
              Romberg built from sub-sampled data (Richardson extrapolation).
Function    : Gauss-Legendre quadrature (2..6 points), composite on sub-intervals,
              Romberg on a function.
"""
import numpy as np


# ------------------------------------------------------------------------ Newton-Cotes
def _check_uniform(x):
    x = np.asarray(x, float)
    h = np.diff(x)
    if not np.allclose(h, h[0]):
        raise ValueError("equally spaced data required")
    return x, h[0]


def trapezoid(x, f):
    """Composite trapezoidal rule (works for unequal segments too)."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    return float(np.sum(np.diff(x) * (f[1:] + f[:-1]) / 2.0))


def simpson13(x, f):
    x, h = _check_uniform(x)
    f = np.asarray(f, float)
    n = len(x) - 1
    if n % 2:
        raise ValueError("Simpson 1/3 needs an even number of segments")
    return float(h / 3.0 * (f[0] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum() + f[-1]))


def simpson38(x, f):
    x, h = _check_uniform(x)
    f = np.asarray(f, float)
    n = len(x) - 1
    if n % 3:
        raise ValueError("Simpson 3/8 needs a multiple of 3 segments")
    s = 0.0
    for k in range(0, n, 3):
        s += 3 * h / 8 * (f[k] + 3 * f[k + 1] + 3 * f[k + 2] + f[k + 3])
    return float(s)


def simpson_mixed(x, f, first="38"):
    """Odd number of segments: 3/8 on the first (or last) 3 segments, 1/3 on the rest.
    For an even number of segments with first='38' we use 3/8 + trapezoid on the
    leftover segment (lower order) - included only to show the accuracy loss."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    n = len(x) - 1
    if n < 3:
        raise ValueError("need at least 3 segments")
    if n % 2 == 1:
        a = simpson38(x[:4], f[:4])
        return a + (simpson13(x[3:], f[3:]) if n > 3 else 0.0)
    return simpson38(x[:4], f[:4]) + trapezoid(x[3:], f[3:])


def romberg_from_data(x, f):
    """Romberg table from data with 2^m segments: trapezoid on every 2^k-th point, then
    Richardson extrapolation  I_{j,k} = (4^(k-1) I_{j+1,k-1} - I_{j,k-1}) / (4^(k-1) - 1)."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    n = len(x) - 1
    m = int(round(np.log2(n)))
    if 2 ** m != n:
        raise ValueError("number of segments must be a power of 2")
    col = [trapezoid(x[::2 ** (m - k)], f[::2 ** (m - k)]) for k in range(m + 1)]  # coarse -> fine
    table = [col]
    for lev in range(1, m + 1):
        prev = table[-1]
        c = 4 ** lev
        table.append([(c * prev[i + 1] - prev[i]) / (c - 1) for i in range(len(prev) - 1)])
    return table[-1][0], table


def romberg_function(func, a, b, levels=6):
    table = []
    for k in range(levels):
        n = 2 ** k
        x = np.linspace(a, b, n + 1)
        table.append([trapezoid(x, [func(t) for t in x])])
    for lev in range(1, levels):
        c = 4 ** lev
        for j in range(lev, levels):
            table[j].append((c * table[j][lev - 1] - table[j - 1][lev - 1]) / (c - 1))
    return table[-1][-1], table


# --------------------------------------------------------------------- Gauss-Legendre
_GL = {
    1: ([0.0], [2.0]),
    2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
    3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
    4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
    5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
    6: ([-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
         0.2386191860831969, 0.6612093864662645, 0.9324695142031521],
        [0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
         0.4679139345726910, 0.3607615730481386, 0.1713244923791704]),
}


def gauss_legendre(func, a, b, n=3):
    """n-point Gauss-Legendre: change of variable x = (b+a)/2 + (b-a)/2 * xd."""
    xd, w = _GL[n]
    return float(sum(wi * func((b + a) / 2 + (b - a) / 2 * xi) for xi, wi in zip(xd, w)) * (b - a) / 2)


def composite_gauss_legendre(func, breaks, n=2):
    return float(sum(gauss_legendre(func, breaks[k], breaks[k + 1], n) for k in range(len(breaks) - 1)))
