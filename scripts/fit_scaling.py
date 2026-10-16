"""Fit the scaling law to the grid and test it on held-out runs.

Reads results/grid.csv (the seed-0 grid runs, test losses) and writes:
- results/fits.json: fitted parameters per target, bootstrap intervals,
  compute-optimal exponents and holdout errors.
- results/holdout.csv: every held-out prediction next to its observed loss.
- results/noise.csv: spread of the final test loss across seeds.

Targets are the test loss of each domain and their mean.
"""

import json

import numpy as np
import pandas as pd

from tnms.config import REPO_ROOT
from tnms.fit import scaling

RESULTS = REPO_ROOT / "results"
TARGETS = ("image", "audio", "text", "mean")
N_BOOTSTRAP = 200


def one_term_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str, by: str, x: str) -> np.ndarray:
    """Naive baseline: a separate one-term law L = E + A / x^s for each value of `by`.

    For the size holdout `by` is D and x is N, for the budget holdout the other
    way round. Each curve is fitted on its own runs and extrapolated.
    """
    pred = pd.Series(np.nan, index=test.index)
    for key, rows in test.groupby(by):
        fit_rows = train[train[by] == key]
        params = scaling.fit(np.log(fit_rows[x].to_numpy()), fit_rows[target].to_numpy())
        pred[rows.index] = np.exp(scaling.predict_log(params, np.log(rows[x].to_numpy()).reshape(-1, 1)))
    return pred.to_numpy()


def holdouts(grid: pd.DataFrame, target: str) -> tuple[list[dict], dict]:
    """Fit on part of the grid, predict the rest, with the law and the baseline."""
    splits = {
        "largest N": (grid["size"] != "S5", "D_budget", "N"),
        "largest D": (grid["D_budget"] < grid["D_budget"].max(), "size", "D"),
    }
    rows, errors = [], {}
    for name, (in_train, by, x) in splits.items():
        train, test = grid[in_train], grid[~in_train]
        params = scaling.fit_nd(train["N"], train["D"], train[target])
        law = scaling.predict_nd(params, test["N"], test["D"])
        base = one_term_baseline(train, test, target, by, x)
        errors[name] = {"law": scaling.mean_abs_rel_error(law, test[target]),
                        "baseline": scaling.mean_abs_rel_error(base, test[target])}
        rows += [{"target": target, "holdout": name, "size": s, "D_budget": d, "observed": o,
                  "law": p, "baseline": b}
                 for s, d, o, p, b in zip(test["size"], test["D_budget"], test[target], law, base)]
    return rows, errors


def fit_target(grid: pd.DataFrame, target: str) -> dict:
    N, D, L = grid["N"].to_numpy(), grid["D"].to_numpy(), grid[target].to_numpy()
    best = scaling.fit_nd(N, D, L)
    boot = scaling.bootstrap(N, D, L, best, N_BOOTSTRAP)

    def describe(params):
        p = scaling.named(params)
        # N* grows as C^(beta / (alpha + beta)), D* as C^(alpha / (alpha + beta)).
        p["n_opt_exponent"] = p["beta"] / (p["alpha"] + p["beta"])
        return p

    fitted = describe(best)
    samples = pd.DataFrame([describe(p) for p in boot])
    flops = 6 * N * D
    n_lo, d_lo = scaling.compute_optimal(best, flops.min())
    n_hi, d_hi = scaling.compute_optimal(best, flops.max())
    return {
        "params": best.tolist(),
        "fitted": {k: float(v) for k, v in fitted.items()},
        "ci95": {k: [float(samples[k].quantile(0.025)), float(samples[k].quantile(0.975))] for k in fitted},
        "residual_mare": scaling.mean_abs_rel_error(scaling.predict_nd(best, N, D), L),
        "d_over_n_at_min_C": float(d_lo / n_lo),
        "d_over_n_at_max_C": float(d_hi / n_hi),
    }


def noise_table(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of the test loss over seeds, for cells run with more than one."""
    cols = [f"test_{t}" for t in TARGETS]
    groups = table.groupby(["size", "D_budget"])
    stats = groups[cols].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    stats["seeds"] = groups.size()
    return stats[stats["seeds"] > 1].reset_index()


def main():
    table = pd.read_csv(RESULTS / "grid.csv")
    grid = table[table["kind"] == "grid"].rename(columns={f"test_{t}": t for t in TARGETS}).reset_index(drop=True)
    assert len(grid) == 25, f"expected 25 grid runs, found {len(grid)}"

    fits, holdout_rows = {}, []
    for target in TARGETS:
        print(f"fitting {target} ...")
        fits[target] = fit_target(grid, target)
        rows, errors = holdouts(grid, target)
        fits[target]["holdout_mare"] = errors
        holdout_rows += rows
        f, ci = fits[target]["fitted"], fits[target]["ci95"]
        print("  " + ", ".join(f"{k} {f[k]:.3g} [{ci[k][0]:.3g}, {ci[k][1]:.3g}]" for k in f))
        print("  holdout error " + ", ".join(f"{k}: law {v['law']:.1%}, baseline {v['baseline']:.1%}"
                                             for k, v in errors.items()))

    (RESULTS / "fits.json").write_text(json.dumps(fits, indent=2))
    pd.DataFrame(holdout_rows).to_csv(RESULTS / "holdout.csv", index=False, float_format="%.5g")
    noise = noise_table(table)
    noise.to_csv(RESULTS / "noise.csv", index=False, float_format="%.5g")
    print(noise.to_string(index=False))


if __name__ == "__main__":
    main()
