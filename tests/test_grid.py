"""Tests for the source discretisation, the grid solver and grid convergence."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from heatplate.discretisation import assignment_problem, discretise, sample
from heatplate.grid_solver import solve_grid
from heatplate.grid_study import harmonic_test_problem, observed_order, richardson
from heatplate.linear_solvers import gauss_elimination


@pytest.mark.parametrize("h", [1.0, 1 / 2, 1 / 3, 1 / 4])
def test_quadratic_field_is_reproduced_exactly(h):
    """T = 300 + (x^2 - y^2)/2 is harmonic and quadratic: the 5-point AND the unequal-spacing
    stencil are exact, so the discrete solution must equal the exact one (on- and off-grid source)."""
    prob, f = harmonic_test_problem("quadratic")
    d = discretise(prob, h)
    T = solve_grid(d, tol=1e-13).T
    X, Y = np.meshgrid(d.x, d.y, indexing="ij")
    assert np.max(np.abs(T - np.vectorize(f)(X, Y))) < 1e-8


def test_second_order_for_smooth_solution():
    prob, f = harmonic_test_problem("sinh")
    errs = []
    for h in (1 / 4, 1 / 8, 1 / 16):
        d = discretise(prob, h)
        T = solve_grid(d, tol=1e-13).T
        X, Y = np.meshgrid(d.x, d.y, indexing="ij")
        errs.append(np.max(np.abs(T - np.vectorize(f)(X, Y))))
    p = np.log(errs[-2] / errs[-1]) / np.log(2)
    assert 1.8 < p < 2.2


@pytest.mark.parametrize("method", ["jacobi", "gauss_seidel", "sor"])
def test_grid_solver_matches_direct_solution(method):
    d = discretise(assignment_problem(), 1.0)
    A, b, _ = d.to_matrix()
    ref = d.to_field(gauss_elimination(A, b))
    T = solve_grid(d, method=method, tol=1e-13).T
    assert np.allclose(T, ref, atol=1e-8)


def test_fine_grid_against_scipy():
    d = discretise(assignment_problem(), 1 / 8)
    A, b, _ = d.to_matrix()
    ref = d.to_field(spsolve(csr_matrix(A), b))
    assert np.allclose(solve_grid(d, tol=1e-13).T, ref, atol=1e-7)


def test_maximum_principle_and_sampling():
    d = discretise(assignment_problem(), 1 / 4)
    T = solve_grid(d).T
    assert T.min() >= 200 - 1e-9 and T.max() <= 350 + 1e-9
    assert sample(T, d).shape == (3, 3)


def test_snap_modes_fix_the_right_nodes():
    d = discretise(assignment_problem(), 1.0, "snap_left")
    assert d.fixed[1, 2] and d.fixed[1, 3] and not d.fixed[2, 2]
    d = discretise(assignment_problem(), 1.0, "snap_right")
    assert d.fixed[2, 2] and d.fixed[2, 3] and not d.fixed[1, 2]


def test_richardson_helpers():
    h = np.array([0.4, 0.2, 0.1])
    f = 3.0 + 0.7 * h ** 2                      # exact second-order sequence
    assert observed_order(*f) == pytest.approx(2.0)
    assert richardson(f[1], f[2], 2.0) == pytest.approx(3.0)
