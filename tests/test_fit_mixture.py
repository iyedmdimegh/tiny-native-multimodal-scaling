"""The mixture law fit and the simplex grid."""

import numpy as np
import yaml

from tnms.config import CONFIG_DIR
from tnms.fit import mixture

H = np.array(yaml.safe_load(open(CONFIG_DIR / "mixtures.yaml"))["mixtures"], dtype=float)


def test_fit_recovers_known_mixture_law():
    true = np.array([0.3, 0.02, 0.8, -0.05, 0.1])  # E, B, beta, c_audio, c_text for the image domain
    loss = mixture.predict(true, H, 0) * (1 + 0.002 * np.random.default_rng(0).standard_normal(len(H)))
    params = mixture.fit_domain(H, loss, 0)
    grid = mixture.simplex_grid()
    # The fitted law must agree with the true one over the whole search region, not only at the 13 points.
    err = np.abs(mixture.predict(params, grid, 0) / mixture.predict(true, grid, 0) - 1)
    assert err.max() < 0.05


def test_simplex_grid():
    grid = mixture.simplex_grid(step=0.01, min_share=0.05)
    assert np.allclose(grid.sum(axis=1), 1)
    assert grid.min() >= 0.05 - 1e-9
    assert len(grid) == 3741  # (85 + 1) * (85 + 2) / 2 points with every share >= 5 steps
