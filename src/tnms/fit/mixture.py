"""A simple per-domain law for loss as a function of the mixture h.

For domain i, at fixed N and D:

    L_i(h) = E_i + B_i * h_i^(-beta_i) + sum over j != i of c_ij * h_j

The power-law term says more of a domain's own data helps with diminishing
returns. The linear terms let the other domains help (c_ij < 0) or hurt
(c_ij > 0). This is a stand-in written for this study, much simpler than the
laws in "Scaling Laws for Optimal Data Mixtures" (see docs/DESIGN.md).
"""

import itertools

import numpy as np
from scipy.optimize import least_squares

DOMAINS = ("image", "audio", "text")
BETA_INITS = (0.1, 0.3, 1.0, 2.0)


def predict(params: np.ndarray, H: np.ndarray, i: int) -> np.ndarray:
    """params = [E, B, beta, c for each other domain in order]."""
    E, B, beta, *c = params
    others = [j for j in range(H.shape[1]) if j != i]
    return E + B * H[:, i] ** (-beta) + H[:, others] @ np.array(c)


def fit_domain(H: np.ndarray, loss: np.ndarray, i: int) -> np.ndarray:
    """Bounded least squares on the loss, from a few starting exponents.

    E, B and beta are kept non-negative, the cross terms are free.
    """
    k = H.shape[1]
    lower = [0.0, 0.0, 1e-3] + [-np.inf] * (k - 1)
    upper = [np.inf, np.inf, 5.0] + [np.inf] * (k - 1)
    best = None
    for beta in BETA_INITS:
        p0 = [loss.min() * 0.5, 0.01, beta] + [0.0] * (k - 1)
        res = least_squares(lambda p: predict(p, H, i) - loss, p0, bounds=(lower, upper))
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def fit_linear(H: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Baseline: L_i = w . h. The shares sum to 1, so no separate intercept is needed."""
    return np.linalg.lstsq(H, loss, rcond=None)[0]


def cross_validate(H: np.ndarray, losses: np.ndarray, n_out: int = 3) -> dict:
    """Leave-n_out-mixtures-out, over every possible held-out set.

    Returns the mean absolute error per domain for the law and the linear
    baseline, averaged over all held-out predictions.
    """
    errors = {"law": np.zeros(H.shape[1]), "linear": np.zeros(H.shape[1])}
    splits = list(itertools.combinations(range(len(H)), n_out))
    for held in splits:
        train = np.setdiff1d(np.arange(len(H)), held)
        for i in range(H.shape[1]):
            law = predict(fit_domain(H[train], losses[train, i], i), H[list(held)], i)
            lin = H[list(held)] @ fit_linear(H[train], losses[train, i])
            errors["law"][i] += np.abs(law - losses[list(held), i]).mean()
            errors["linear"][i] += np.abs(lin - losses[list(held), i]).mean()
    return {name: (e / len(splits)).tolist() for name, e in errors.items()} | {"n_splits": len(splits)}


def simplex_grid(step: float = 0.01, min_share: float = 0.05) -> np.ndarray:
    """Every mixture of three domains on a grid of `step`, with each share >= min_share."""
    n = round(1 / step)
    points = [(a, b, n - a - b) for a in range(n + 1) for b in range(n + 1 - a)]
    H = np.array(points) / n
    return H[(H >= min_share - 1e-9).all(axis=1)]
