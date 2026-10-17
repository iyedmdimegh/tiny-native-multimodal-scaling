"""Draw every figure from the committed tables in results/ into figures/.

A figure whose tables do not exist yet is skipped with a message, so this can
run at any point of the study.
"""

import json

import pandas as pd

from tnms import plots
from tnms.config import REPO_ROOT, load_yaml

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
LEARNING_CURVE_SIZE = "S4"


def main():
    FIGURES.mkdir(exist_ok=True)
    have = lambda *names: all((RESULTS / n).exists() for n in names)

    if have("grid.csv", "curves.csv", "floors.json"):
        floors = json.loads((RESULTS / "floors.json").read_text())
        curves = pd.read_csv(RESULTS / "curves.csv")
        plots.learning_curves(curves, LEARNING_CURVE_SIZE, floors, FIGURES / "fig1_learning_curves.png")
        print("fig1 done")
    else:
        print("fig1 skipped: run the grid first")

    if have("grid.csv", "fits.json", "holdout.csv", "floors.json"):
        table = pd.read_csv(RESULTS / "grid.csv")
        grid = table[table["kind"] == "grid"].rename(columns={f"test_{t}": t for t in plots.TARGETS})
        fits = json.loads((RESULTS / "fits.json").read_text())
        noise = pd.read_csv(RESULTS / "noise.csv") if have("noise.csv") else None
        plots.loss_vs_data(grid, fits, noise, floors, FIGURES / "fig2_loss_vs_data.png")
        plots.loss_vs_params(grid, fits, floors, FIGURES / "fig3_loss_vs_params.png")
        plots.holdout_predictions(pd.read_csv(RESULTS / "holdout.csv"), fits, FIGURES / "fig4_holdout.png")
        plots.compute_frontier(grid, fits, FIGURES / "fig5_compute.png")
        print("fig2 to fig5 done")
    else:
        print("fig2 to fig5 skipped: run scripts/fit_scaling.py first")

    if have("mixtures.csv", "mixture_fit.json"):
        fit = json.loads((RESULTS / "mixture_fit.json").read_text())
        plots.mixture_sweep(pd.read_csv(RESULTS / "mixtures.csv"), fit, FIGURES / "fig6_mixture_sweep.png")
        print("fig6 done")
    else:
        print("fig6 skipped: run the mixture sweep and scripts/fit_mixture.py first")

    if have("mixture_validation.csv", "mixture_fit.json", "noise.csv"):
        # Seed spread of the uniform mixture at the mixture experiment's size and budget.
        noise = pd.read_csv(RESULTS / "noise.csv")
        exp = load_yaml("mixtures")
        cell = noise[(noise["size"] == exp["size"]) & (noise["D_budget"] == exp["tokens"])].iloc[0]
        std = {t: cell[f"test_{t}_std"] for t in ("mean", "image")}
        fit = json.loads((RESULTS / "mixture_fit.json").read_text())
        plots.mixture_validation(pd.read_csv(RESULTS / "mixture_validation.csv"), fit, std, FIGURES / "fig7_mixture_check.png")
        print("fig7 done")
    else:
        print("fig7 skipped: run scripts/run_mixture.py validate first")

    if have("fusion.csv"):
        plots.fusion(pd.read_csv(RESULTS / "fusion.csv"), FIGURES / "fig8_fusion.png")
        print("fig8 done")
    else:
        print("fig8 skipped: run scripts/run_fusion.py first")


if __name__ == "__main__":
    main()
