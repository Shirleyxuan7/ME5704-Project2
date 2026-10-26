"""Grid-refinement study, observed order of accuracy and Richardson extrapolation.

For a scheme of order p, the discretisation error behaves like  e(h) ~ C h^p.  With three
grids h, h/2, h/4 the *observed* order is

        p_obs = log( |f_h - f_h/2| / |f_h/2 - f_h/4| ) / log 2

and the generalised Richardson extrapolation (Lecture Notes Sec. 5.3.2 with general p) is

        f_*  =  f_h/2 + (f_h/2 - f_h) / (2^p - 1).

The lecture formula 4/3 f_h/2 - 1/3 f_h is the special case p = 2.  The 5-point scheme is
O(h^2) for smooth solutions, but the end points of the line source are singular points
(T ~ r^(1/2), like a crack tip), so p_obs < 2 here; using p = 2 blindly would be wrong.
"""
import math

import numpy as np

from . import config
from .discretisation import Problem, assignment_problem, discretise, sample
from .grid_solver import solve_grid


def solve(problem: Problem, h: float, source_mode="exact", method="sor", tol=1e-11):
    disc = discretise(problem, h, source_mode)
    res = solve_grid(disc, method=method, tol=tol)
    return disc, res


def observed_order(f_coarse, f_mid, f_fine, r=2.0):
    f_coarse, f_mid, f_fine = map(np.asarray, (f_coarse, f_mid, f_fine))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.abs(f_coarse - f_mid) / np.abs(f_mid - f_fine)) / math.log(r)


def richardson(f_coarse, f_fine, p, r=2.0):
    f_coarse, f_fine = np.asarray(f_coarse), np.asarray(f_fine)
    return f_fine + (f_fine - f_coarse) / (r ** p - 1.0)


def refinement_study(hs=config.GRID_SPACINGS, problem: Problem = None, source_mode="exact",
                     extra=None, verbose=False):
    """Solve on every grid; return the 9-point values (and any extra functionals).

    extra : optional dict name -> f(disc, T) evaluated on each grid (e.g. the heat flow Phi).
    """
    problem = problem or assignment_problem()
    out = {"h": [], "T9": [], "iters": [], "fields": {}, "extra": {k: [] for k in (extra or {})}}
    for h in hs:
        disc, res = solve(problem, h, source_mode)
        out["h"].append(h)
        out["T9"].append(sample(res.T, disc))
        out["iters"].append(res.iterations)
        out["fields"][h] = (disc, res.T)
        for k, f in (extra or {}).items():
            out["extra"][k].append(f(disc, res.T))
        if verbose:
            print(f"    h = 1/{round(1 / h):<3d} unknowns = {np.sum(~disc.fixed):6d}  "
                  f"SOR iterations = {res.iterations}")
    out["T9"] = np.array(out["T9"])
    return out


def extrapolate_last_three(values):
    """values ordered coarse -> fine with ratio 2.  Returns (p_obs, extrapolated)."""
    v = np.asarray(values)
    p = observed_order(v[-3], v[-2], v[-1])
    return p, richardson(v[-2], v[-1], p)


# ----------------------------------------------------------- analytic verification case
def harmonic_test_problem(kind="sinh"):
    """Problems with known exact solution (used in tests and in the verification figure).

    'quadratic' : T = 300 + (x^2 - y^2)/2  -> 5-point (and unequal 3-point) scheme is EXACT.
    'sinh'      : T = 200 + 150 sin(pi x/4) sinh(pi y/4) / sinh(pi)  -> smooth, O(h^2) expected.
    The internal 'source' line carries the exact solution, so the same code path is tested.
    """
    if kind == "quadratic":
        f = lambda x, y: 300.0 + 0.5 * (x ** 2 - y ** 2)
    elif kind == "sinh":
        f = lambda x, y: 200.0 + 150.0 * math.sin(math.pi * x / 4) * math.sinh(math.pi * y / 4) / math.sinh(math.pi)
    else:
        raise ValueError(kind)
    return Problem(boundary_T=f, source_T=f), f
