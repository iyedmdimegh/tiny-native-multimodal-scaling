"""Fit the mixture law to the 13 sweep runs and pick the mixtures to test.

The law is fitted on VALIDATION losses, because it is used to choose
mixtures. Test losses are only read when the fresh runs are reported.
Writes results/mixture_fit.json.
"""

import json

import numpy as np
import pandas as pd

from tnms.config import REPO_ROOT
from tnms.fit import mixture
from tnms.fit.mixture import DOMAINS

RESULTS = REPO_ROOT / "results"


def main():
    table = pd.read_csv(RESULTS / "mixtures.csv")
    H = table[[f"h_{d}" for d in DOMAINS]].to_numpy()
    losses = table[[f"val_{d}" for d in DOMAINS]].to_numpy()

    params = {d: mixture.fit_domain(H, losses[:, i], i) for i, d in enumerate(DOMAINS)}
    for i, d in enumerate(DOMAINS):
        E, B, beta, c1, c2 = params[d]
        others = [o for o in DOMAINS if o != d]
        fit_err = np.abs(mixture.predict(params[d], H, i) - losses[:, i]).mean()
        print(f"{d}: E {E:.3f}, B {B:.4f}, beta {beta:.3f}, c_{others[0]} {c1:+.3f}, c_{others[1]} {c2:+.3f}, "
              f"in-sample MAE {fit_err:.4f}")

    cv = mixture.cross_validate(H, losses)
    for name in ("law", "linear"):
        print(f"leave-3-out MAE ({cv['n_splits']} splits), {name}: "
              + ", ".join(f"{d} {e:.4f}" for d, e in zip(DOMAINS, cv[name])))

    grid = mixture.simplex_grid()
    pred = np.column_stack([mixture.predict(params[d], grid, i) for i, d in enumerate(DOMAINS)])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    pred_uniform = [float(mixture.predict(params[d], uniform, i)[0]) for i, d in enumerate(DOMAINS)]
    optimum = {}
    for target, score in (("mean", pred.mean(axis=1)), ("image", pred[:, 0])):
        best = int(np.argmin(score))
        optimum[target] = {"h": [round(float(x), 2) for x in grid[best]],
                           "predicted_val": dict(zip(DOMAINS, map(float, pred[best])))}
        print(f"best for {target}: h = {optimum[target]['h']}, predicted val {score[best]:.4f} "
              f"(uniform predicted {np.mean(pred_uniform) if target == 'mean' else pred_uniform[0]:.4f})")

    (RESULTS / "mixture_fit.json").write_text(json.dumps({
        "fitted_on": "validation loss of the 13 sweep runs",
        "params": {d: p.tolist() for d, p in params.items()},
        "param_names": ["E", "B", "beta", "c_other_1", "c_other_2"],
        "cross_validation_mae": cv,
        "predicted_uniform_val": dict(zip(DOMAINS, pred_uniform)),
        "optimum": optimum,
    }, indent=2))


if __name__ == "__main__":
    main()
