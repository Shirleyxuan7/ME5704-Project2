"""Tests for the linear solvers.  Run:  pytest tests/test_linear_solvers.py -v"""
import numpy as np
import pytest

from heatplate.discretisation import Problem, assignment_problem, discretise
from heatplate.linear_solvers import (gauss_elimination, gauss_seidel, is_diagonally_dominant, jacobi,
                                      lu_decompose, lu_solve, sor)


def lecture_exercise():
    """Lecture Notes 6.3.1 exercise: unit square, 4 interior points,
    T = 0 (bottom), 100 (top), 75 (left), 50 (right)."""
    def bc(x, y):
        if x < 1e-9: return 75.0
        if x > 1 - 1e-9: return 50.0
        return 0.0 if y < 1e-9 else 100.0
    d = discretise(Problem(L=1.0, boundary_T=bc, source_x=None), 1 / 3)
    return d.to_matrix()


def test_lecture_liebmann_exercise():
    """Initial values 0, lam = 1.1, eps_s = 1 %  ->  46.9, 40.6, 71.9, 65.6 after 4 iterations."""
    A, b, nodes = lecture_exercise()
    assert nodes == [(1, 1), (2, 1), (1, 2), (2, 2)]
    r = sor(A, b, lam=1.1, eps_s=1.0)
    assert r.iterations == 4
    assert np.allclose(np.round(r.x, 1), [46.9, 40.6, 71.9, 65.6])


def test_lecture_exact_solution():
    A, b, _ = lecture_exercise()
    assert np.allclose(gauss_elimination(A, b), [46.875, 40.625, 71.875, 65.625])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_solvers_against_numpy(seed):
    rng = np.random.default_rng(seed)
    n = 12
    A = rng.uniform(-1, 1, (n, n))
    A += np.diag(np.sum(np.abs(A), axis=1) + 1.0)          # strictly diagonally dominant
    b = rng.uniform(-5, 5, n)
    ref = np.linalg.solve(A, b)
    assert np.allclose(gauss_elimination(A, b), ref)
    assert np.allclose(lu_solve(*lu_decompose(A), b), ref)
    for r in (jacobi(A, b, eps_s=1e-10), gauss_seidel(A, b, eps_s=1e-10), sor(A, b, 1.2, eps_s=1e-10)):
        assert r.converged and np.allclose(r.x, ref, atol=1e-8)


def test_pivoting_needed():
    A = np.array([[0.0, 2.0], [3.0, 1.0]])
    assert np.allclose(gauss_elimination(A, [4.0, 5.0]), [1.0, 2.0])


def test_assignment_system_structure():
    d = discretise(assignment_problem(), 1.0)
    A, b, nodes = d.to_matrix()
    assert A.shape == (9, 9) and is_diagonally_dominant(A)
    k = nodes.index((1, 2))                    # left of the source: 4/3*200 + 8/3*350, aP = 6
    assert A[k, k] == pytest.approx(6.0)
    assert b[k] == pytest.approx(4 / 3 * 200 + 8 / 3 * 350)


def test_gauss_seidel_faster_than_jacobi():
    A, b, _ = discretise(assignment_problem(), 1.0).to_matrix()
    assert gauss_seidel(A, b, eps_s=1e-8).iterations < jacobi(A, b, eps_s=1e-8).iterations
