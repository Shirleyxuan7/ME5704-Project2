"""Tests for numerical integration, curve fitting and part (c)."""
import numpy as np
import pytest

from heatplate import config
from heatplate.curve_fit import NaturalCubicSpline, fit_saturating_exp, lagrange, polyfit
from heatplate.heat_absorbed import heat_absorbed, integrate_k_all
from heatplate.integration import (gauss_legendre, romberg_from_data, romberg_function, simpson13,
                                   simpson38, simpson_mixed, trapezoid)

f7 = lambda x: x ** 7          # Lecture Notes example: int_1^2 x^7 dx = 31.875


def test_lecture_gauss_legendre_example():
    assert gauss_legendre(f7, 1, 2, 2) == pytest.approx(31.201, abs=1e-3)
    assert gauss_legendre(f7, 1, 2, 3) == pytest.approx(31.871, abs=1e-3)
    assert gauss_legendre(f7, 1, 2, 4) == pytest.approx(31.875, abs=1e-9)


def test_lecture_trapezoid_and_romberg_example():
    x2, x4 = np.linspace(1, 2, 3), np.linspace(1, 2, 5)
    assert trapezoid(x2, f7(x2)) == pytest.approx(40.793, abs=1e-3)
    assert trapezoid(x4, f7(x4)) == pytest.approx(34.155, abs=1e-3)
    assert romberg_function(f7, 1, 2, levels=5)[0] == pytest.approx(31.875, abs=1e-9)


def test_degree_of_precision():
    x = np.linspace(0, 4, 5)
    cubic = lambda t: 1 + t - 2 * t ** 2 + 0.5 * t ** 3
    exact = 4 + 8 - 2 * 64 / 3 + 0.5 * 64
    assert simpson13(x, cubic(x)) == pytest.approx(exact)
    x3 = np.linspace(0, 3, 4)
    assert simpson38(x3, cubic(x3)) == pytest.approx(3 + 4.5 - 18 + 0.5 * 81 / 4)
    quintic = lambda t: t ** 5 - t ** 4
    assert romberg_from_data(x, quintic(x))[0] == pytest.approx(4 ** 6 / 6 - 4 ** 5 / 5)   # Boole exact
    assert gauss_legendre(quintic, 0, 4, 3) == pytest.approx(4 ** 6 / 6 - 4 ** 5 / 5)


def test_mixed_simpson_odd_segments():
    x = np.linspace(0, 5, 6)
    assert simpson_mixed(x, x ** 3) == pytest.approx(5 ** 4 / 4)


def test_polyfit_recovers_quadratic():
    t = np.linspace(0, 4, 5)
    fit = polyfit(t, 0.1 + 0.2 * t - 0.03 * t ** 2, 2)
    assert np.allclose(fit.coeffs, [0.1, 0.2, -0.03]) and fit.r2 == pytest.approx(1.0)


def test_spline_and_lagrange_interpolate():
    t, k = config.T_DATA, config.K_DATA
    sp = NaturalCubicSpline(t, k)
    assert np.allclose([sp(v) for v in t], k)
    assert np.allclose(lagrange(t, k, t), k)
    lin = NaturalCubicSpline(t, 2 * t + 1)
    assert lin.integral() == pytest.approx(20.0)


def test_gauss_newton_recovers_parameters():
    t = np.linspace(0, 4, 9)
    y = 0.5 - 0.4 * np.exp(-0.7 * t)
    fit = fit_saturating_exp(t, y)
    assert (fit.a, fit.b, fit.c) == pytest.approx((0.5, 0.4, 0.7), rel=1e-6)


def test_k_integral_methods_agree():
    rows, _ = integrate_k_all()
    vals = {r[0]: r[2] for r in rows}
    assert vals["Simpson 1/3, h = 1 (4 segments)"] == pytest.approx(1.27333, abs=1e-5)
    assert vals["Trapezoidal, h = 1 (4 segments)"] == pytest.approx(1.26)
    good = [r[2] for r in rows if not any(s in r[0] for s in ("h = 4", "h = 2", "m = 1"))]
    assert max(good) - min(good) < 0.02        # all methods that use all 5 points: within ~1 %
    assert heat_absorbed(100.0, 1.5) == pytest.approx(150.0)
