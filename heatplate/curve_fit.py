"""Curve fitting / interpolation of the k(t) data (Lecture Notes Ch. 2).

* polynomial least-squares regression (normal equations, Sec. 2.2)
* Lagrange interpolating polynomial (Sec. 2.3.2)
* natural cubic spline (Sec. 2.3.3)
* non-linear regression with the Gauss-Newton method (Sec. 2.2) for the saturating model
        k(t) = a - b exp(-c t)
"""
import math
from dataclasses import dataclass

import numpy as np

from .linear_solvers import gauss_elimination


# ---------------------------------------------------------------- polynomial regression
@dataclass
class PolyFit:
    coeffs: np.ndarray          # a0 + a1 t + a2 t^2 + ...
    r2: float
    sy_x: float                 # standard error of the estimate

    def __call__(self, t):
        t = np.asarray(t, float)
        return sum(a * t ** k for k, a in enumerate(self.coeffs))

    def integral(self, a, b):
        return float(sum(c / (k + 1) * (b ** (k + 1) - a ** (k + 1)) for k, c in enumerate(self.coeffs)))


def _r2(y, yhat):
    St = np.sum((y - y.mean()) ** 2)
    Sr = np.sum((y - yhat) ** 2)
    return (1.0 - Sr / St) if St > 0 else 1.0, Sr


def polyfit(t, y, m):
    """Least-squares polynomial of order m: solve the (m+1)x(m+1) normal equations."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    n = len(t)
    if m >= n:
        raise ValueError("order too high for the number of points")
    N = np.array([[np.sum(t ** (r + c)) for c in range(m + 1)] for r in range(m + 1)])
    rhs = np.array([np.sum(y * t ** r) for r in range(m + 1)])
    a = gauss_elimination(N, rhs)
    fit = PolyFit(a, 0.0, 0.0)
    r2, Sr = _r2(y, fit(t))
    fit.r2 = r2
    fit.sy_x = math.sqrt(Sr / (n - (m + 1))) if n > m + 1 else 0.0
    return fit


# --------------------------------------------------------------------- Lagrange
def lagrange(t_data, y_data, t):
    t_data, y_data = np.asarray(t_data, float), np.asarray(y_data, float)
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for i in range(len(t_data)):
        Li = np.ones_like(t)
        for j in range(len(t_data)):
            if j != i:
                Li *= (t - t_data[j]) / (t_data[i] - t_data[j])
        out += y_data[i] * Li
    return out


# ---------------------------------------------------------------- natural cubic spline
class NaturalCubicSpline:
    """S_i(t) = y_i + b_i (t-t_i) + c_i (t-t_i)^2 + d_i (t-t_i)^3, with S'' = 0 at both ends."""

    def __init__(self, t, y):
        t, y = np.asarray(t, float), np.asarray(y, float)
        n = len(t) - 1
        h = np.diff(t)
        # tridiagonal system for c_1..c_{n-1}  (c_0 = c_n = 0), solved with the Thomas algorithm
        sub = h[:-1].copy(); diag = 2 * (h[:-1] + h[1:]); sup = h[1:].copy()
        rhs = 3 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        m = len(diag)
        for k in range(1, m):
            w = sub[k] / diag[k - 1]
            diag[k] -= w * sup[k - 1]
            rhs[k] -= w * rhs[k - 1]
        cin = np.zeros(m)
        for k in range(m - 1, -1, -1):
            cin[k] = (rhs[k] - (sup[k] * cin[k + 1] if k < m - 1 else 0.0)) / diag[k]
        c = np.concatenate([[0.0], cin, [0.0]])
        self.b = (y[1:] - y[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3
        self.d = (c[1:] - c[:-1]) / (3 * h)
        self.c = c[:-1]
        self.t, self.y, self.h, self.n = t, y, h, n

    def __call__(self, tq):
        tq = np.atleast_1d(np.asarray(tq, float))
        k = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, self.n - 1)
        dt = tq - self.t[k]
        out = self.y[k] + self.b[k] * dt + self.c[k] * dt ** 2 + self.d[k] * dt ** 3
        return out if out.size > 1 else float(out[0])

    def integral(self):
        """Exact integral over [t_0, t_n] (sum of integrated cubics)."""
        h = self.h
        return float(np.sum(self.y[:-1] * h + self.b * h ** 2 / 2 + self.c * h ** 3 / 3 + self.d * h ** 4 / 4))


# --------------------------------------------------- non-linear regression (Gauss-Newton)
@dataclass
class ExpFit:
    a: float
    b: float
    c: float
    r2: float
    iterations: int

    def __call__(self, t):
        return self.a - self.b * np.exp(-self.c * np.asarray(t, float))

    def integral(self, t0, t1):
        a, b, c = self.a, self.b, self.c
        return float(a * (t1 - t0) + b / c * (math.exp(-c * t1) - math.exp(-c * t0)))


def fit_saturating_exp(t, y, guess=(0.45, 0.35, 0.5), tol=1e-10, max_iter=100):
    """Gauss-Newton: linearise f about current parameters, solve (Z^T Z) dA = Z^T D."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    p = np.array(guess, float)
    for it in range(1, max_iter + 1):
        a, b, c = p
        e = np.exp(-c * t)
        f = a - b * e
        Z = np.column_stack([np.ones_like(t), -e, b * t * e])      # partial derivatives
        D = y - f
        dp = gauss_elimination(Z.T @ Z, Z.T @ D)
        p = p + dp
        if np.max(np.abs(dp / p)) < tol:
            break
    fit = ExpFit(*p, 0.0, it)
    fit.r2 = _r2(y, fit(t))[0]
    return fit
