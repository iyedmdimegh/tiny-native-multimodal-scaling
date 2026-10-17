"""Experiment B runs. Resumable, like the grid.

    python scripts/run_mixture.py sweep      # the 13 mixtures of configs/mixtures.yaml
    python scripts/run_mixture.py validate   # predicted-best mixtures and uniform, fresh seed

`validate` reads the mixtures chosen by scripts/fit_mixture.py from
results/mixture_fit.json. Both write a table to results/.
"""

import argparse
import json

import pandas as pd

from tnms.config import REPO_ROOT, RUNS_DIR, load_yaml, run_config
from tnms.fit.mixture import DOMAINS
from tnms.train import run

RESULTS = REPO_ROOT / "results"


def train_mixture(shares: list[float], seed: int, label: str) -> dict:
    """One run at the experiment's size and budget with token shares (image, audio, text)."""
    exp, preset = load_yaml("mixtures"), load_yaml("full")
    sweep = pd.read_csv(RESULTS / "lr_sweep.csv")
    lr = float(sweep[sweep["best"] & (sweep["size"] == exp["size"])]["lr"].iloc[0])
    name = "_".join(f"{d[0]}{h:g}" for d, h in zip(DOMAINS, shares))
    run_dir = RUNS_DIR / "full" / "mixtures" / f"{exp['size']}_D{exp['tokens']}_{name}_seed{seed}"
    if (run_dir / "summary.json").exists():
        s = json.loads((run_dir / "summary.json").read_text())
    else:
        cfg = run_config(preset, exp["size"], exp["tokens"], lr, seed=seed, shares=dict(zip(DOMAINS, shares)))
        s = run(cfg, run_dir)
    row = {"label": label, "seed": seed, **{f"h_{d}": h for d, h in zip(DOMAINS, shares)}, "D": s["D"]}
    for split in ("val", "test"):
        row.update({f"{split}_{d}": v for d, v in s[split].items()})
    row.update({f"epochs_{d}": round(v, 2) for d, v in s["epochs"].items()})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["sweep", "validate"])
    args = parser.parse_args()
    exp = load_yaml("mixtures")

    if args.step == "sweep":
        rows = []
        for shares in exp["mixtures"]:
            rows.append(train_mixture(shares, exp["seed"], "sweep"))
            pd.DataFrame(rows).to_csv(RESULTS / "mixtures.csv", index=False, float_format="%.5g")
    else:
        chosen = json.loads((RESULTS / "mixture_fit.json").read_text())["optimum"]
        runs = {"uniform": exp["mixtures"][0], "best mean": chosen["mean"]["h"], "best image": chosen["image"]["h"]}
        rows = []
        for label, shares in runs.items():
            rows.append(train_mixture(shares, exp["validation_seed"], label))
            pd.DataFrame(rows).to_csv(RESULTS / "mixture_validation.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
