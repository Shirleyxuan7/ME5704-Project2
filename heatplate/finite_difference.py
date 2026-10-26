"""Numerical differentiation (Lecture Notes Sec. 5.5) on the plate grid.

All finite-difference formulas are generated from Taylor series / Lagrange interpolation:
for sample points at offsets d_0..d_m around x (not necessarily equally spaced) the weights
w_k of   f'(x) ~ sum_k w_k f(x + d_k)   satisfy

        sum_k w_k d_k^q / q! = delta_{q,1},     q = 0..m,

which reproduces the textbook formulas for equal spacing, e.g.
  forward  O(h)  : (f1 - f0)/h
  forward  O(h^2): (-f2 + 4f1 - 3f0)/(2h)
  central  O(h^2): (f1 - f-1)/(2h)
and gives the correct *unequal-spacing* versions next to the off-grid heat source.

Source awareness: the temperature gradient is discontinuous across the red line, so a
stencil must never reach across it.  ``line_points`` walks from a node in one direction and
stops at the source (or at the outer boundary), returning the usable sample points.
"""
import math

import numpy as np

from .discretisation import Discretisation
from .linear_solvers import gauss_elimination

DIRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}


def fd_weights(offsets, deriv=1):
    """Weights for the `deriv`-th derivative at 0 from samples at `offsets`."""
    d = np.asarray(offsets, float)
    m = len(d)
    M = np.array([[dk ** q / math.factorial(q) for dk in d] for q in range(m)])
    rhs = np.zeros(m); rhs[deriv] = 1.0
    return gauss_elimination(M, rhs)


def _is_source_node(disc: Discretisation, i, j):
    return disc.fixed[i, j] and 0 < i < disc.n and 0 < j < disc.n


def line_points(disc: Discretisation, T, i, j, direction, max_points=2):
    """Walk from node (i, j) in `direction`; return (offsets, values), stopping at the source/boundary."""
    di, dj = DIRS[direction]
    offs, vals = [], []
    ci, cj, dist = i, j, 0.0
    for _ in range(max_points):
        og = disc.offgrid.get((ci, cj), {})
        if direction in og:                                   # off-grid source in between
            alpha, Ts = og[direction]
            offs.append(dist + alpha * disc.h); vals.append(Ts)
            break
        ci, cj = ci + di, cj + dj
        dist += disc.h
        offs.append(dist); vals.append(T[ci, cj])
        if disc.fixed[ci, cj]:                                # reached boundary or source node
            break
    sign = 1.0 if direction in ("E", "N") else -1.0
    return [sign * o for o in offs], vals


def derivative(disc, T, i, j, axis, scheme="central"):
    """dT/dx (axis='x') or dT/dy (axis='y') at grid node (i, j)."""
    plus, minus = ("E", "W") if axis == "x" else ("N", "S")
    f0 = T[i, j]
    if scheme == "naive_central":                           # ignores the source (for comparison)
        di, dj = DIRS[plus]
        return (T[i + di, j + dj] - T[i - di, j - dj]) / (2 * disc.h)
    if scheme == "central":
        op, vp = line_points(disc, T, i, j, plus, 1)
        om, vm = line_points(disc, T, i, j, minus, 1)
        offs, vals = [om[0], 0.0, op[0]], [vm[0], f0, vp[0]]
    elif scheme in ("forward", "forward2", "backward", "backward2"):
        d = plus if scheme.startswith("forward") else minus
        npts = 2 if scheme.endswith("2") else 1
        o, v = line_points(disc, T, i, j, d, npts)
        offs, vals = [0.0] + o, [f0] + v
    else:
        raise ValueError(scheme)
    return float(fd_weights(offs) @ np.asarray(vals))


def gradient(disc, T, i, j, scheme="central"):
    return derivative(disc, T, i, j, "x", scheme), derivative(disc, T, i, j, "y", scheme)
