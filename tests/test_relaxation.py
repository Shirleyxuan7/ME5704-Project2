"""Tests for the optimal over-relaxation factor (1-D optimisation)."""
import math

import pytest

from heatplate.discretisation import Problem, assignment_problem, discretise
from heatplate.relaxation import (golden_section_min, optimise_lambda, parabolic_min,
                                  square_grid_optimum, young_optimum)


def test_golden_section_quadratic():
    x, fx, _, _ = golden_section_min(lambda x: (x - 2.0) ** 2 + 1.0, 0.0, 5.0, tol=1e-8)
    assert x == pytest.approx(2.0, abs=1e-5) and fx == pytest.approx(1.0)


def test_parabolic_quadratic_one_step():
    x, _, it, _ = parabolic_min(lambda x: (x - 2.0) ** 2, 0.0, 1.0, 4.0)
    assert x == pytest.approx(2.0, abs=1e-10) and it <= 2      # exact for a parabola


def test_young_formula_square_plate():
    """Without the source the h = 1 plate is the classic 3x3 Laplace problem: lam = 2/(1+sin(pi/4))."""
    A, _, _ = discretise(Problem(source_x=None), 1.0).to_matrix()
    assert young_optimum(A) == pytest.approx(square_grid_optimum(4), rel=1e-10)
    assert square_grid_optimum(4) == pytest.approx(2 / (1 + math.sin(math.pi / 4)))


def test_golden_section_finds_young_optimum():
    A, _, _ = discretise(assignment_problem(), 1.0).to_matrix()
    info = optimise_lambda(A)
    assert info["lam_golden"] == pytest.approx(info["lam_young"], abs=1e-4)
    assert info["rho_golden"] < info["rho_gauss_seidel"] < info["rho_jacobi"] < 1
