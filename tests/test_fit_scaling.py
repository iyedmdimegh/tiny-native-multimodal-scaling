"""The scaling-law fit recovers known parameters from synthetic data."""

import numpy as np
import pytest

from tnms.fit import scaling

# Same N and D values as the real grid, with a made-up law of a similar shape.
N = np.array([97_600, 791_680, 2_656_704, 6_279_424, 14_162_304], dtype=float)
D = np.array([5e6, 10e6, 20e6, 40e6, 80e6])
TRUE = np.array([np.log(300), np.log(3000), 0.4, 0.45, np.log(1.0)])  # a, b, alpha, beta, e


@pytest.mark.parametrize("seed", [0, 1])
def test_fit_recovers_law_with_one_percent_noise(seed):
    NN, DD = (x.ravel() for x in np.meshgrid(N, D))
    clean = scaling.predict_nd(TRUE, NN, DD)
    noisy = clean * (1 + 0.01 * np.random.default_rng(seed).standard_normal(len(clean)))
    # Every 25th start of the grid keeps the test fast. The real fits use all 3,125.
    params = scaling.fit_nd(NN, DD, noisy, scaling.init_grid(2)[::25])

    got, true = scaling.named(params), scaling.named(TRUE)
    # Tolerance: with 25 points and 1% noise the exponents are only loosely pinned down,
    # but the fitted surface must stay close to the true one.
    assert abs(got["alpha"] - true["alpha"]) < 0.15
    assert abs(got["beta"] - true["beta"]) < 0.15
    assert abs(got["E"] / true["E"] - 1) < 0.2
    assert scaling.mean_abs_rel_error(scaling.predict_nd(params, NN, DD), clean) < 0.01


def test_compute_optimal_split_minimises_the_law():
    C = 1e15
    n_opt, d_opt = scaling.compute_optimal(TRUE, C)
    assert np.isclose(6 * n_opt * d_opt, C)
    best = scaling.predict_nd(TRUE, n_opt, d_opt)
    for factor in (0.5, 2.0):  # moving N either way at the same compute is worse
        assert scaling.predict_nd(TRUE, n_opt * factor, d_opt / factor) > best
