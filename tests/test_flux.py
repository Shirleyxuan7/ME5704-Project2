"""Tests for numerical differentiation, heat flux and the energy balance."""
import numpy as np
import pytest

from heatplate.discretisation import assignment_problem, discretise
from heatplate.finite_difference import fd_weights
from heatplate.flux import flux_at_points, phi_box, phi_outer, phi_source
from heatplate.grid_solver import solve_grid
from heatplate.grid_study import harmonic_test_problem


def test_textbook_formulas():
    assert np.allclose(fd_weights([-1, 0, 1]), [-0.5, 0, 0.5])            # central O(h^2)
    assert np.allclose(fd_weights([0, 1]), [-1, 1])                        # forward O(h)
    assert np.allclose(fd_weights([0, 1, 2]), [-1.5, 2, -0.5])             # forward O(h^2)
    assert np.allclose(fd_weights([-1, 0, 1], deriv=2), [1, -2, 1])        # 2nd derivative


def test_unequal_spacing_exact_for_quadratics():
    w = fd_weights([-1.0, 0.0, 0.5])
    f = lambda x: 2 + 3 * x - 4 * x ** 2
    assert w @ np.array([f(-1.0), f(0.0), f(0.5)]) == pytest.approx(3.0)


def test_gradient_exact_on_quadratic_field():
    """grad T = (x, -y) for T = 300 + (x^2 - y^2)/2, including nodes next to the off-grid source."""
    prob, _ = harmonic_test_problem("quadratic")
    d = discretise(prob, 1.0)
    T = solve_grid(d, tol=1e-13).T
    for r in flux_at_points(d, T, "central"):
        assert r["dTdx"] == pytest.approx(r["x"], abs=1e-6)
        assert r["dTdy"] == pytest.approx(-r["y"], abs=1e-6)
    # 3-point one-sided formulas are exact too wherever two points are available
    r = flux_at_points(d, T, "forward2", points=[(2, 1)])[0]
    assert (r["dTdx"], r["dTdy"]) == pytest.approx((2.0, -1.0), abs=1e-6)
    # ... but next to the source only one point is left -> falls back to O(h): not exact
    r = flux_at_points(d, T, "forward2", points=[(1, 2)])[0]
    assert r["dTdx"] != pytest.approx(1.0, abs=1e-3)


def test_first_order_schemes_have_first_order_error():
    f = np.sin
    errs = [abs((f(1 + h) - f(1)) / h - np.cos(1)) for h in (0.1, 0.05)]
    assert np.log2(errs[0] / errs[1]) == pytest.approx(1.0, abs=0.1)


def test_source_free_field_has_zero_net_flux():
    prob, _ = harmonic_test_problem("quadratic")
    for h in (1.0, 0.5):
        d = discretise(prob, h)
        T = solve_grid(d, tol=1e-13).T
        assert phi_outer(d, T) == pytest.approx(0.0, abs=1e-6)


def test_energy_conservation_assignment():
    d = discretise(assignment_problem(), 1 / 16)
    T = solve_grid(d, tol=1e-12).T
    po, pb, ps = phi_outer(d, T), phi_box(d, T), phi_source(d, T)
    assert abs(po - pb) / po < 0.01                       # same heat through two different surfaces
    assert ps < po                                        # tip singularity -> faces underestimate
    assert po > 0


def test_flux_points_away_from_source():
    d = discretise(assignment_problem(), 1.0)
    A, b, _ = d.to_matrix()
    from heatplate.linear_solvers import gauss_elimination
    T = d.to_field(gauss_elimination(A, b))
    rows = {(r["x"], r["y"]): r for r in flux_at_points(d, T)}
    assert rows[(1, 2)]["qx_k"] < 0 < rows[(2, 2)]["qx_k"]   # left of line flows left, right flows right
