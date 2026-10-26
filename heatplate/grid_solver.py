"""Matrix-free iterative solvers on the full grid (used for the grid-refinement study).

For fine grids (h = 1/64 -> 65 000 unknowns) building a dense matrix is impossible, so the
point-iterations are applied directly to the stencil (this *is* Liebmann's method):

    T_hat[i,j] = (aE T[i+1,j] + aW T[i-1,j] + aN T[i,j+1] + aS T[i,j-1] + b) / aP
    T_new      = lam * T_hat + (1 - lam) * T_old

To vectorise Gauss-Seidel in numpy we use *red-black* ordering: nodes with (i + j) even are
updated first, then the odd ones (each colour only depends on the other colour, so every
update immediately uses the newest neighbour values, as Gauss-Seidel requires).  The
convergence rate is the same as for the lexicographic ordering used in the lecture.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .discretisation import Discretisation
from .relaxation import square_grid_optimum


@dataclass
class GridResult:
    T: np.ndarray
    iterations: int
    converged: bool
    lam: float
    method: str
    history: List[float] = field(default_factory=list)


def _update(disc, T):
    c = (slice(1, -1), slice(1, -1))
    return (disc.aE[c] * T[2:, 1:-1] + disc.aW[c] * T[:-2, 1:-1] + disc.aN[c] * T[1:-1, 2:]
            + disc.aS[c] * T[1:-1, :-2] + disc.b[c]) / disc.aP[c]


def solve_grid(disc: Discretisation, method: str = "sor", lam=None, tol=1e-10,
               max_iter=200_000, T0=None) -> GridResult:
    """method: 'jacobi' | 'gauss_seidel' | 'sor'.  lam=None -> theoretical optimum for SOR."""
    if method == "sor" and lam is None:
        lam = square_grid_optimum(disc.n)
    if method in ("jacobi", "gauss_seidel"):
        lam = 1.0
    T = disc.initial_field() if T0 is None else T0.copy()
    free = ~disc.fixed[1:-1, 1:-1]
    ii, jj = np.meshgrid(np.arange(1, disc.n), np.arange(1, disc.n), indexing="ij")
    colours = [free & ((ii + jj) % 2 == 0), free & ((ii + jj) % 2 == 1)]
    res = GridResult(T, 0, False, lam, method)
    Tin = T[1:-1, 1:-1]                       # view into T
    for it in range(1, max_iter + 1):
        T_old = Tin.copy()
        if method == "jacobi":
            new = _update(disc, T)
            Tin[free] = new[free]
        else:
            for mask in colours:
                new = _update(disc, T)        # uses the latest values of the other colour
                Tin[mask] = lam * new[mask] + (1.0 - lam) * Tin[mask]
        change = float(np.max(np.abs(Tin - T_old)) / np.max(np.abs(Tin)))
        res.history.append(change)
        if change < tol:
            res.converged = True
            break
    res.T, res.iterations = T, it
    return res
