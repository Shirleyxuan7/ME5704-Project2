"""Finite-difference discretisation of Laplace's equation on the plate.

Lecture Notes Sec. 6.3.1: the 2nd derivatives are replaced by central differences,

    (T[i-1,j] - 2T[i,j] + T[i+1,j]) / dx^2 + (T[i,j-1] - 2T[i,j] + T[i,j+1]) / dy^2 = 0 .

The red-line heat source (x = 1.5, 2 <= y <= 3) is *not* on the h = 1 grid.  Three
ways of bringing it into the equations are implemented (``source_mode``):

``"exact"``       If x = 1.5 is a grid line (h = 1/2, 1/4, ...) the source nodes are
                  simply fixed at 350 K.  If it lies between two grid lines (h = 1, 1/3,
                  ...) the nodes next to it use the *unequal-spacing* 3-point formula
                  for the 2nd derivative (Chapra & Canale, irregular boundaries):

                    d2T/dx2 ~ 2/h^2 [ T_W/(aW(aW+aE)) + T_E/(aE(aW+aE)) - T_P/(aW aE) ]

                  where aW*h, aE*h are the distances to the west / east neighbour
                  (one of which is the source at 350 K).
``"snap_left"``   Move the source to the nearest grid line on its left (x = 1 for h = 1).
``"snap_right"``  Move the source to the nearest grid line on its right (x = 2 for h = 1).

Every interior equation is written in the "a-coefficient" form

    aP*T[i,j] = aE*T[i+1,j] + aW*T[i-1,j] + aN*T[i,j+1] + aS*T[i,j-1] + b[i,j]

(multiplied through by h^2).  aE..aS are *links* to grid values (which may be fixed
boundary / source values); b holds the contribution of an off-grid source point.
Array index convention throughout the project: T[i, j] with x = i*h, y = j*h.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from . import config


# ----------------------------------------------------------------------------- problem
@dataclass
class Problem:
    """Geometry + Dirichlet data.  Callables make it easy to plug in analytic test cases."""
    L: float = config.L
    boundary_T: Callable[[float, float], float] = lambda x, y: config.T_BOUNDARY
    source_x: Optional[float] = config.SOURCE_X          # None -> no internal source
    source_y: Tuple[float, float] = config.SOURCE_Y
    source_T: Callable[[float, float], float] = lambda x, y: config.T_SOURCE


def assignment_problem() -> Problem:
    """The plate exactly as specified in the assignment."""
    return Problem()


# ----------------------------------------------------------------------- discretisation
@dataclass
class Discretisation:
    h: float
    n: int                       # number of intervals per side
    x: np.ndarray
    y: np.ndarray
    fixed: np.ndarray            # bool (n+1, n+1): Dirichlet node?
    T_fixed: np.ndarray          # values at fixed nodes (NaN elsewhere)
    aP: np.ndarray
    aE: np.ndarray
    aW: np.ndarray
    aN: np.ndarray
    aS: np.ndarray
    b: np.ndarray
    source_mode: str
    source_x_eff: Optional[float]                      # x-position actually used
    source_on_grid: bool
    # (i, j) -> {"E" or "W": (alpha, T_source)} for nodes whose neighbour is the off-grid source
    offgrid: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = field(default_factory=dict)

    # ---- bookkeeping -------------------------------------------------------------
    def unknowns(self):
        """Unknown nodes in lecture ordering: i (x) runs fastest, then j (y)."""
        return [(i, j) for j in range(self.n + 1) for i in range(self.n + 1) if not self.fixed[i, j]]

    def index_map(self):
        return {node: k for k, node in enumerate(self.unknowns())}

    def initial_field(self, guess: float = None) -> np.ndarray:
        T = self.T_fixed.copy()
        if guess is None:
            guess = float(np.nanmean(self.T_fixed))
        T[~self.fixed] = guess
        return T

    def node_of(self, xp: float, yp: float) -> Tuple[int, int]:
        i, j = xp / self.h, yp / self.h
        if abs(i - round(i)) > 1e-9 or abs(j - round(j)) > 1e-9:
            raise ValueError(f"({xp}, {yp}) is not a grid node for h = {self.h}")
        return int(round(i)), int(round(j))

    # ---- matrix form A T = b (used by the direct / lexicographic solvers) ----------
    def to_matrix(self):
        """Assemble the dense system A x = rhs for the unknown nodes (small grids only)."""
        nodes = self.unknowns()
        idx = {node: k for k, node in enumerate(nodes)}
        N = len(nodes)
        A = np.zeros((N, N))
        rhs = np.zeros(N)
        for k, (i, j) in enumerate(nodes):
            A[k, k] = self.aP[i, j]
            rhs[k] = self.b[i, j]
            for (ii, jj), a in (((i + 1, j), self.aE[i, j]), ((i - 1, j), self.aW[i, j]),
                                ((i, j + 1), self.aN[i, j]), ((i, j - 1), self.aS[i, j])):
                if a == 0.0:
                    continue
                if self.fixed[ii, jj]:
                    rhs[k] += a * self.T_fixed[ii, jj]
                else:
                    A[k, idx[(ii, jj)]] -= a
        return A, rhs, nodes

    def to_field(self, vec: np.ndarray) -> np.ndarray:
        T = self.T_fixed.copy()
        for k, (i, j) in enumerate(self.unknowns()):
            T[i, j] = vec[k]
        return T

    def residual(self, T: np.ndarray) -> np.ndarray:
        """r = aE T_E + aW T_W + aN T_N + aS T_S + b - aP T_P on unknown nodes (0 elsewhere)."""
        r = np.zeros_like(T)
        c = (slice(1, -1), slice(1, -1))
        r[c] = (self.aE[c] * T[2:, 1:-1] + self.aW[c] * T[:-2, 1:-1] + self.aN[c] * T[1:-1, 2:]
                + self.aS[c] * T[1:-1, :-2] + self.b[c] - self.aP[c] * T[c])
        r[self.fixed] = 0.0
        return r


def discretise(problem: Problem, h: float, source_mode: str = "exact") -> Discretisation:
    n_float = problem.L / h
    n = int(round(n_float))
    if abs(n - n_float) > 1e-9:
        raise ValueError("L/h must be an integer")
    x = np.linspace(0.0, problem.L, n + 1)
    y = np.linspace(0.0, problem.L, n + 1)
    shape = (n + 1, n + 1)

    fixed = np.zeros(shape, bool)
    T_fixed = np.full(shape, np.nan)
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    for i in range(n + 1):
        for j in range(n + 1):
            if fixed[i, j]:
                T_fixed[i, j] = problem.boundary_T(x[i], y[j])

    # standard 5-point stencil (multiplied by h^2)
    aE = np.ones(shape); aW = np.ones(shape); aN = np.ones(shape); aS = np.ones(shape)
    aP = 4.0 * np.ones(shape)
    b = np.zeros(shape)
    offgrid = {}
    xs_eff, on_grid = None, False

    if problem.source_x is not None:
        xs = problem.source_x
        if source_mode == "snap_left":
            xs = np.floor(xs / h + 1e-12) * h
        elif source_mode == "snap_right":
            xs = np.ceil(xs / h - 1e-12) * h
        elif source_mode != "exact":
            raise ValueError(f"unknown source_mode {source_mode!r}")
        xs_eff = xs
        y0, y1 = problem.source_y
        rows = [j for j in range(n + 1) if y0 - 1e-9 <= y[j] <= y1 + 1e-9 and 0 < j < n]
        s = xs / h
        on_grid = abs(s - round(s)) < 1e-9
        if on_grid:                                            # source nodes are Dirichlet nodes
            i_s = int(round(s))
            for j in rows:
                fixed[i_s, j] = True
                T_fixed[i_s, j] = problem.source_T(xs, y[j])
        else:                                                  # unequal-spacing stencil
            iL = int(np.floor(s))
            aWest = s - iL                                     # distance of source from node iL (in h)
            for j in rows:
                Ts = problem.source_T(xs, y[j])
                for i, side, alpha in ((iL, "E", aWest), (iL + 1, "W", 1.0 - aWest)):
                    if fixed[i, j]:
                        continue
                    a_w, a_e = (1.0, alpha) if side == "E" else (alpha, 1.0)
                    cW = 2.0 / (a_w * (a_w + a_e))
                    cE = 2.0 / (a_e * (a_w + a_e))
                    aP[i, j] = 2.0 / (a_w * a_e) + 2.0
                    if side == "E":
                        aW[i, j], aE[i, j] = cW, 0.0
                        b[i, j] += cE * Ts
                    else:
                        aE[i, j], aW[i, j] = cE, 0.0
                        b[i, j] += cW * Ts
                    offgrid[(i, j)] = {side: (alpha, Ts)}

    for arr in (aE, aW, aN, aS, aP, b):
        arr[fixed] = 0.0
    aP[fixed] = 1.0
    return Discretisation(h, n, x, y, fixed, T_fixed, aP, aE, aW, aN, aS, b,
                          source_mode, xs_eff, on_grid, offgrid)


def sample(T: np.ndarray, disc: Discretisation, points=config.POINTS_9) -> np.ndarray:
    """Temperatures at physical points (x, y) -> 3x3 array laid out like the sketch
    (row 0 = top, y = 3; column 0 = left, x = 1)."""
    vals = {p: T[disc.node_of(*p)] for p in points}
    return np.array([[vals[(xx, yy)] for xx in (1, 2, 3)] for yy in (3, 2, 1)])
