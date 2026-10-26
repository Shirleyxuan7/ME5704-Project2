"""Choosing the over-relaxation factor lam (link between Ch. 4 Optimisation and Ch. 6 PDEs).

The asymptotic convergence speed of SOR is governed by the spectral radius rho(G_lam) of its
iteration matrix  G_lam = (D - lam*L)^-1 [(1 - lam) D + lam*U],   with A = D - L - U.
Error per iteration shrinks by ~rho, so  iterations ~ ln(eps) / ln(rho).

Minimising rho(lam) on [1, 2) is a 1-D unconstrained optimisation problem, solved here with
the two 1-D methods of Lecture Notes Sec. 4.2 (golden-section search, parabolic interpolation)
and compared with the classical theoretical optimum (Young):
        lam_opt = 2 / (1 + sqrt(1 - mu^2)),   mu = rho(G_Jacobi).
Note: rho(lam) has a *kink* (non-smooth minimum) at lam_opt - golden section does not care,
parabolic interpolation assumes smoothness.  This is one of the critical-thinking points.
"""
import math

import numpy as np

from .linear_solvers import sor


def split(A):
    A = np.asarray(A, float)
    D = np.diag(np.diag(A))
    Lw = -np.tril(A, -1)
    Up = -np.triu(A, 1)
    return D, Lw, Up


def spectral_radius(M) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))   # eigenvalues used for analysis only


def jacobi_spectral_radius(A) -> float:
    D, Lw, Up = split(A)
    return spectral_radius(np.linalg.solve(D, Lw + Up))


def sor_spectral_radius(A, lam: float) -> float:
    D, Lw, Up = split(A)
    G = np.linalg.solve(D - lam * Lw, (1.0 - lam) * D + lam * Up)
    return spectral_radius(G)


def young_optimum(A) -> float:
    mu = jacobi_spectral_radius(A)
    return 2.0 / (1.0 + math.sqrt(1.0 - mu ** 2))


def square_grid_optimum(n_intervals: int) -> float:
    """lam_opt for the 5-point Laplacian on a square with n intervals per side."""
    return 2.0 / (1.0 + math.sin(math.pi / n_intervals))


# ----------------------------------------------------------------- 1-D optimisers (Ch. 4)
GR = (math.sqrt(5.0) - 1.0) / 2.0          # golden ratio 0.618...


def golden_section_min(f, xl, xu, tol=1e-6, max_iter=200):
    """Golden-section search for a minimum of a unimodal f on [xl, xu]."""
    history = []
    d = GR * (xu - xl)
    x1, x2 = xl + d, xu - d
    f1, f2 = f(x1), f(x2)
    for it in range(1, max_iter + 1):
        if f1 < f2:                      # minimum in [x2, xu]
            xl, x2, f2 = x2, x1, f1
            x1 = xl + GR * (xu - xl); f1 = f(x1)
        else:                            # minimum in [xl, x1]
            xu, x1, f1 = x1, x2, f2
            x2 = xu - GR * (xu - xl); f2 = f(x2)
        xopt = x1 if f1 < f2 else x2
        history.append(xopt)
        if (1 - GR) * abs((xu - xl) / xopt) < tol:
            break
    return xopt, f(xopt), it, history


def parabolic_min(f, x0, x1, x2, tol=1e-6, max_iter=100):
    """Successive parabolic interpolation (Lecture Notes Sec. 4.2.1)."""
    f0, f1, f2 = f(x0), f(x1), f(x2)
    history = []
    x3_old = x1
    for it in range(1, max_iter + 1):
        num = f0 * (x1 ** 2 - x2 ** 2) + f1 * (x2 ** 2 - x0 ** 2) + f2 * (x0 ** 2 - x1 ** 2)
        den = 2 * f0 * (x1 - x2) + 2 * f1 * (x2 - x0) + 2 * f2 * (x0 - x1)
        if den == 0:
            break
        x3 = num / den
        f3 = f(x3)
        history.append(x3)
        # keep the three best points bracketing the minimum (sequential strategy)
        if x3 > x1:
            if f3 < f1:
                x0, f0, x1, f1 = x1, f1, x3, f3
            else:
                x2, f2 = x3, f3
        else:
            if f3 < f1:
                x2, f2, x1, f1 = x1, f1, x3, f3
            else:
                x0, f0 = x3, f3
        if abs((x3 - x3_old) / x3) < tol:
            break
        x3_old = x3
    return x1, f1, it, history


# -------------------------------------------------------------------- experiment helpers
def sweep_iterations(A, b, lams, eps_s=1e-8):
    """Actual number of SOR iterations needed for each lam (what the user really cares about)."""
    return np.array([sor(A, b, lam, eps_s=eps_s).iterations for lam in lams])


def optimise_lambda(A):
    f = lambda lam: sor_spectral_radius(A, lam)
    gs = golden_section_min(f, 1.0, 1.99, tol=1e-6)
    pb = parabolic_min(f, 1.0, 1.5, 1.99, tol=1e-6)
    return {
        "rho_jacobi": jacobi_spectral_radius(A),
        "rho_gauss_seidel": f(1.0),
        "lam_young": young_optimum(A),
        "lam_golden": gs[0], "rho_golden": gs[1], "evals_golden": gs[2] + 2, "hist_golden": gs[3],
        "lam_parabolic": pb[0], "rho_parabolic": pb[1], "evals_parabolic": pb[2] + 3, "hist_parabolic": pb[3],
    }
