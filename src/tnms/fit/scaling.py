"""Fit L(N, D) = E + A / N^alpha + B / D^beta to the grid.

The law is written as a sum of power-law terms plus a constant:

    L_hat = E + sum_k exp(c_k) / x_k^s_k

with x = (N, D), exp(c) = (A, B) and s = (alpha, beta). In log space this is

    log L_hat = logsumexp(c_k - s_k * log x_k, ..., e)

which is always defined and keeps A, B, E positive without bounds. The same
code also fits a one-term law (only N or only D), used as the naive baseline.

The objective is a Huber loss (delta 1e-3) between log L_hat and log L, summed
over runs. Huber behaves like least squares for small residuals and like an
absolute error for large ones, so one odd run cannot pull the whole fit.
L-BFGS is started from a grid of initial values and the best end point kept,
because the objective has several local minima.
"""

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

HUBER_DELTA = 1e-3
COEF_INITS = (0, 5, 10, 15, 20)  # c = log A or log B
EXP_INITS = (0, 0.5, 1, 1.5, 2)  # alpha, beta
E_INITS = (-1, -0.5, 0, 0.5, 1)  # e = log E


def _terms(params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
    """(runs, K + 1) matrix of c_k - s_k log x_k, with e as the last column."""
    k = log_x.shape[1]
    c, s, e = params[:k], params[k : 2 * k], params[-1]
    return np.column_stack([c - s * log_x, np.full(len(log_x), e)])


def predict_log(params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
    return logsumexp(_terms(params, log_x), axis=1)


def objective(params: np.ndarray, log_x: np.ndarray, log_loss: np.ndarray):
    """Summed Huber loss on log L and its gradient."""
    terms = _terms(params, log_x)
    r = logsumexp(terms, axis=1) - log_loss
    small = np.abs(r) <= HUBER_DELTA
    value = np.where(small, 0.5 * r**2, HUBER_DELTA * (np.abs(r) - 0.5 * HUBER_DELTA)).sum()
    # d value / d log L_hat, then the chain rule through logsumexp (its gradient is a softmax).
    d_r = np.clip(r, -HUBER_DELTA, HUBER_DELTA)
    w = softmax(terms, axis=1) * d_r[:, None]
    k = log_x.shape[1]
    grad = np.concatenate([w[:, :k].sum(0), -(w[:, :k] * log_x).sum(0), [w[:, k].sum()]])
    return value, grad


def init_grid(k: int) -> list[np.ndarray]:
    """Every combination of the initial values, for k power-law terms."""
    grids = [COEF_INITS] * k + [EXP_INITS] * k + [E_INITS]
    return [np.array(p, dtype=float) for p in itertools.product(*grids)]


def fit(log_x: np.ndarray, loss: np.ndarray, inits: list[np.ndarray] | None = None) -> np.ndarray:
    """Best parameters [c_1..c_k, s_1..s_k, e] over all starting points."""
    log_x = np.asarray(log_x, dtype=float).reshape(len(loss), -1)
    log_loss = np.log(np.asarray(loss, dtype=float))
    inits = init_grid(log_x.shape[1]) if inits is None else inits
    best = None
    for p0 in inits:
        res = minimize(objective, p0, args=(log_x, log_loss), jac=True, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def named(params: np.ndarray) -> dict:
    """The two-term parameters by their usual names."""
    a, b, alpha, beta, e = params
    return {"E": np.exp(e), "A": np.exp(a), "alpha": alpha, "B": np.exp(b), "beta": beta}


def fit_nd(N, D, loss, inits=None) -> np.ndarray:
    return fit(np.column_stack([np.log(N), np.log(D)]), loss, inits)


def predict_nd(params, N, D) -> np.ndarray:
    return np.exp(predict_log(params, np.column_stack([np.log(N), np.log(D)])))


def bootstrap(N, D, loss, best: np.ndarray, n_resamples: int = 200, seed: int = 0) -> np.ndarray:
    """Refit on runs drawn with replacement. Returns (n_resamples, 5) parameters.

    A full grid of 3,125 starts takes about two minutes, too slow to repeat
    200 times. So each refit starts from the full-data optimum plus 30 random
    points of the grid.
    """
    rng = np.random.default_rng(seed)
    N, D, loss = map(np.asarray, (N, D, loss))
    grid = init_grid(2)
    out = []
    for _ in range(n_resamples):
        idx = rng.integers(len(loss), size=len(loss))
        starts = [best] + [grid[i] for i in rng.choice(len(grid), size=30, replace=False)]
        out.append(fit_nd(N[idx], D[idx], loss[idx], starts))
    return np.array(out)


def compute_optimal(params: np.ndarray, C) -> tuple[np.ndarray, np.ndarray]:
    """N*(C) and D*(C) that minimise the fitted law at compute C = 6 N D.

    Setting the derivative of A N^-alpha + B (C / 6N)^-beta to zero gives
    N* = G (C/6)^(beta / (alpha + beta)) with G = (alpha A / (beta B))^(1 / (alpha + beta)).
    """
    p = named(params)
    alpha, beta = p["alpha"], p["beta"]
    G = (alpha * p["A"] / (beta * p["B"])) ** (1 / (alpha + beta))
    n_opt = G * (np.asarray(C) / 6) ** (beta / (alpha + beta))
    return n_opt, np.asarray(C) / 6 / n_opt


def mean_abs_rel_error(pred, obs) -> float:
    return float(np.mean(np.abs(np.asarray(pred) / np.asarray(obs) - 1)))
