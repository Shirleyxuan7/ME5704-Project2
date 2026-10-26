"""Part (b): heat-flux density  q = -k grad T,  and the total heat flow Phi used in part (c).

Because k is spatially uniform, every flux is reported per unit k:  q/k = -grad T.

Total heat flow (per unit k, per unit thickness) out of the source:
        Q(t) = closed-integral q.n dA = k(t) * Phi,   Phi = closed-integral (-dT/dn) ds * thickness
Laplace's equation is source-free away from the red line, so Phi must be the SAME through
any closed surface enclosing the source.  Three surfaces are used as a cross-check
(energy conservation):
  * 'outer'  : the plate edges (far from the singular source tips -> best convergence)
  * 'box'    : a rectangle [1,2] x [1.5,3.5] around the source (needs h <= 1/2)
  * 'source' : the two faces of the red line itself (affected by the r^(1/2) tip singularity)
"""
import math

import numpy as np

from . import config
from .discretisation import Discretisation
from .finite_difference import derivative, fd_weights, line_points
from .integration import simpson13, trapezoid


# ----------------------------------------------------------------- pointwise flux, part (b)
def flux_at_points(disc: Discretisation, T, scheme="central", points=config.POINTS_9):
    """Return list of dicts with dT/dx, dT/dy, q/k components, |q|/k and direction (deg)."""
    rows = []
    for (xp, yp) in points:
        i, j = disc.node_of(xp, yp)
        gx = derivative(disc, T, i, j, "x", scheme)
        gy = derivative(disc, T, i, j, "y", scheme)
        qx, qy = -gx, -gy
        rows.append({"x": xp, "y": yp, "T": T[i, j], "dTdx": gx, "dTdy": gy,
                     "qx_k": qx, "qy_k": qy, "qmag_k": math.hypot(qx, qy),
                     "angle_deg": math.degrees(math.atan2(qy, qx))})
    return rows


def flux_field(disc, T):
    """q/k on the whole grid with plain central differences (for plotting only)."""
    gx, gy = np.gradient(T, disc.h, disc.h)
    return -gx, -gy


# --------------------------------------------------------------- integrated heat flow Phi
def _integrate(s, g, rule):
    n = len(s) - 1
    if rule == "simpson" and n % 2 == 0 and n >= 2:
        return simpson13(s, g)
    return trapezoid(s, g)


def phi_outer(disc: Discretisation, T, rule="simpson", order=2):
    """Heat leaving through the four edges: sum of  integral (dT/dn_inward) ds  (O(h^order) one-sided)."""
    n, h = disc.n, disc.h
    s = np.linspace(0, n * h, n + 1)
    total = 0.0
    sides = {"W": [(0, k) for k in range(n + 1)], "E": [(n, k) for k in range(n + 1)],
             "S": [(k, 0) for k in range(n + 1)], "N": [(k, n) for k in range(n + 1)]}
    inward = {"W": "E", "E": "W", "S": "N", "N": "S"}
    for side, nodes in sides.items():
        g = []
        for (i, j) in nodes:
            if (i, j) in ((0, 0), (0, n), (n, 0), (n, n)):
                g.append(0.0)                       # corner: T constant along both edges
                continue
            o, v = line_points(disc, T, i, j, inward[side], order)
            offs = [0.0] + [abs(x) for x in o]
            g.append(float(fd_weights(offs) @ np.asarray([T[i, j]] + v)))
        total += _integrate(s, np.array(g), rule)
    return total


def phi_box(disc: Discretisation, T, box=(1.0, 2.0, 1.5, 3.5), rule="simpson"):
    """Heat leaving a rectangle whose edges lie on grid lines (central differences)."""
    x0, x1, y0, y1 = box
    h = disc.h
    for v in box:
        if abs(v / h - round(v / h)) > 1e-9:
            raise ValueError("box edges must be grid lines")
    i0, i1, j0, j1 = (int(round(v / h)) for v in box)
    total = 0.0
    # vertical edges: outward normal -x (left) / +x (right);  -dT/dn_out
    ys = disc.y[j0:j1 + 1]
    gl = [-(-derivative(disc, T, i0, j, "x")) for j in range(j0, j1 + 1)]
    gr = [-(derivative(disc, T, i1, j, "x")) for j in range(j0, j1 + 1)]
    xs = disc.x[i0:i1 + 1]
    gb = [-(-derivative(disc, T, i, j0, "y")) for i in range(i0, i1 + 1)]
    gt = [-(derivative(disc, T, i, j1, "y")) for i in range(i0, i1 + 1)]
    for s, g in ((ys, gl), (ys, gr), (xs, gb), (xs, gt)):
        total += _integrate(s, np.array(g), rule)
    return total


def phi_source(disc: Discretisation, T, rule="simpson", order=2):
    """Heat leaving the two faces of the red line: integral of -dT/d(distance from line)."""
    if disc.source_x_eff is None:
        return 0.0
    h = disc.h
    y0, y1 = config.SOURCE_Y
    js = [j for j in range(disc.n + 1) if y0 - 1e-9 <= disc.y[j] <= y1 + 1e-9]
    xs = disc.source_x_eff
    s = np.array([disc.y[j] for j in js])
    total = 0.0
    for side in ("W", "E"):
        g = []
        for j in js:
            if disc.source_on_grid:
                i_s = int(round(xs / h))
                Ts = T[i_s, j]
                o, v = line_points(disc, T, i_s, j, side, order)
            else:
                iL = int(math.floor(xs / h))
                alpha = xs / h - iL
                start = iL if side == "W" else iL + 1
                a0 = alpha if side == "W" else 1 - alpha
                Ts = list(disc.offgrid[(start, j)].values())[0][1]
                o, v = line_points(disc, T, start, j, side, order - 1)
                o = [a0 * h] + [a0 * h + abs(x) for x in o]
                v = [T[start, j]] + v
            offs = [0.0] + [abs(x) for x in o]
            dTds = float(fd_weights(offs) @ np.asarray([Ts] + v))   # derivative away from the line
            g.append(-dTds)
        total += _integrate(s, np.array(g), rule)
    return total
