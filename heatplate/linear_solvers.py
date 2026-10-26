"""Solvers for the linear system A x = b (Lecture Notes Sec. 3.5.2 and 6.3.1).

Direct methods : Gauss elimination with partial pivoting, LU (Doolittle) decomposition.
Iterative      : Jacobi, Gauss-Seidel, Gauss-Seidel + over-relaxation (SOR / Liebmann).

The iterative solvers follow the lecture exactly:
  * Jacobi updates all unknowns at the end of an iteration;
  * Gauss-Seidel uses each new value as soon as it is available;
  * over-relaxation:  x_new = lam * x_hat + (1 - lam) * x_old ,  1 <= lam < 2;
  * stopping criterion: max_i |(x_new - x_old) / x_new| * 100% < eps_s.
All algorithms are written out by hand; numpy is only used for array storage.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np


# ------------------------------------------------------------------------ direct methods
def gauss_elimination(A, b):
    """Naive Gauss elimination with partial pivoting + back substitution."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))          # partial pivoting
        if A[p, k] == 0.0:
            raise ZeroDivisionError("matrix is singular")
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            A[i, k:] -= f * A[k, k:]
            b[i] -= f * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def lu_decompose(A):
    """Doolittle LU decomposition (L has unit diagonal), no pivoting
    (safe here because the Laplace matrix is diagonally dominant)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    Lm, U = np.eye(n), np.zeros((n, n))
    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - Lm[i, :i] @ U[:i, k]
        for k in range(i + 1, n):
            Lm[k, i] = (A[k, i] - Lm[k, :i] @ U[:i, i]) / U[i, i]
    return Lm, U


def lu_solve(Lm, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):                                   # forward substitution  L y = b
        y[i] = b[i] - Lm[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                       # back substitution     U x = y
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def is_diagonally_dominant(A) -> bool:
    A = np.abs(np.asarray(A, float))
    off = A.sum(axis=1) - np.diag(A)
    return bool(np.all(np.diag(A) >= off) and np.any(np.diag(A) > off))


# --------------------------------------------------------------------- iterative methods
@dataclass
class IterResult:
    x: np.ndarray
    iterations: int
    converged: bool
    err_history: List[float] = field(default_factory=list)   # max approx. relative error [%]
    res_history: List[float] = field(default_factory=list)   # ||b - A x||_inf


def _rel_err(x_new, x_old):
    denom = np.where(np.abs(x_new) > 0, np.abs(x_new), 1.0)
    return float(np.max(np.abs((x_new - x_old) / denom)) * 100.0)


def jacobi(A, b, x0=None, eps_s=1e-8, max_iter=100_000):
    A = np.asarray(A, float); b = np.asarray(b, float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    D = np.diag(A)
    R = A - np.diag(D)
    res = IterResult(x, 0, False)
    for it in range(1, max_iter + 1):
        x_new = (b - R @ x) / D                       # all values from the previous iteration
        err = _rel_err(x_new, x)
        x = x_new
        res.err_history.append(err)
        res.res_history.append(float(np.max(np.abs(b - A @ x))))
        if err < eps_s:
            res.converged = True
            break
    res.x, res.iterations = x, it
    return res


def sor(A, b, lam=1.0, x0=None, eps_s=1e-8, max_iter=100_000):
    """Gauss-Seidel (lam = 1) with over-relaxation applied to each new value (lecture form)."""
    A = np.asarray(A, float); b = np.asarray(b, float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    res = IterResult(x, 0, False)
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            x_hat = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
            x[i] = lam * x_hat + (1.0 - lam) * x_old[i]
        err = _rel_err(x, x_old)
        res.err_history.append(err)
        res.res_history.append(float(np.max(np.abs(b - A @ x))))
        if err < eps_s:
            res.converged = True
            break
    res.x, res.iterations = x, it
    return res


def gauss_seidel(A, b, x0=None, eps_s=1e-8, max_iter=100_000):
    return sor(A, b, 1.0, x0, eps_s, max_iter)
