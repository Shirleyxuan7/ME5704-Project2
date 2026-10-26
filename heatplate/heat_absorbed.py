"""Part (c): heat absorbed from t = 0 to 4.

Quasi-steady argument: k cancels out of Laplace's equation, so T(x, y) does not depend on k.
Hence q(x, y, t) = k(t) * (-grad T(x, y)) and

    h = int_0^4  closed-int q.n dA  dt  =  Phi * thickness * int_0^4 k(t) dt ,

where Phi = closed-int (-dT/dn) ds is the heat flow per unit k (module ``flux``).
This module evaluates  I_k = int_0^4 k dt  with every method of Ch. 5 / Ch. 2.
"""
import numpy as np

from . import config
from .curve_fit import NaturalCubicSpline, fit_saturating_exp, lagrange, polyfit
from .integration import (composite_gauss_legendre, gauss_legendre, romberg_from_data,
                          simpson13, simpson_mixed, trapezoid)


def integrate_k_all(t=config.T_DATA, k=config.K_DATA):
    """Return list of (method, family, value, comment)."""
    t, k = np.asarray(t, float), np.asarray(k, float)
    a, b = t[0], t[-1]
    rows = []
    # --- directly on the data (Newton-Cotes)
    rows.append(("Trapezoidal, h = 4 (1 segment)", "data", trapezoid(t[::4], k[::4]), "O(h^2)"))
    rows.append(("Trapezoidal, h = 2 (2 segments)", "data", trapezoid(t[::2], k[::2]), "O(h^2)"))
    rows.append(("Trapezoidal, h = 1 (4 segments)", "data", trapezoid(t, k), "O(h^2)"))
    rows.append(("Simpson 1/3, h = 2 (2 segments)", "data", simpson13(t[::2], k[::2]), "O(h^4), uses 3 of 5 points"))
    rows.append(("Simpson 1/3, h = 1 (4 segments)", "data", simpson13(t, k), "O(h^4)  <- main method"))
    rows.append(("Simpson 3/8 on [0,3] + trapezoid on [3,4]", "data", simpson_mixed(t, k), "mixed order, for illustration"))
    I_rom, table = romberg_from_data(t, k)
    rows.append(("Romberg (trap. h = 4,2,1 + Richardson)", "data", I_rom, "O(h^6) = Boole's rule"))
    # --- fit / interpolate first, then integrate the function
    for m in (1, 2, 3):
        fit = polyfit(t, k, m)
        rows.append((f"Polynomial regression m = {m} (analytic)", "fit", fit.integral(a, b), f"R^2 = {fit.r2:.4f}"))
    p2 = polyfit(t, k, 2)
    rows.append(("Polynomial m = 2 + 2-pt Gauss-Legendre", "fit", gauss_legendre(p2, a, b, 2), "exact for degree <= 3"))
    rows.append(("Lagrange (4th order) + 3-pt Gauss-Legendre", "fit",
                 gauss_legendre(lambda s: float(lagrange(t, k, s)), a, b, 3), "exact for degree <= 5"))
    sp = NaturalCubicSpline(t, k)
    rows.append(("Natural cubic spline (analytic)", "fit", sp.integral(), "interpolates all points"))
    rows.append(("Natural cubic spline + 2-pt Gauss per interval", "fit",
                 composite_gauss_legendre(sp, t, 2), "should equal the line above"))
    ef = fit_saturating_exp(t, k)
    rows.append(("Gauss-Newton fit k = a - b e^(-ct) (analytic)", "fit", ef.integral(a, b),
                 f"a={ef.a:.4f}, b={ef.b:.4f}, c={ef.c:.4f}, R^2={ef.r2:.4f}"))
    return rows, {"romberg_table": table, "poly": {m: polyfit(t, k, m) for m in (1, 2, 3)},
                  "spline": sp, "expfit": ef}


def heat_absorbed(phi: float, I_k: float, thickness=config.THICKNESS):
    return phi * thickness * I_k
