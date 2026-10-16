"""Run every (size, budget) pair of a preset. Resumable.

A run whose summary.json exists is skipped, so the script can be stopped and
restarted at any time. Runs are ordered by budget first, so all sizes at the
smallest D finish before any larger D starts.
"""

import argparse

import pandas as pd

from tnms.config import RUNS_DIR, load_yaml, results_dir, run_config
from tnms.train import run


def learning_rates(preset_name: str, preset: dict) -> dict:
    """LR per size: written in the preset, or the best one from the LR sweep."""
    if preset["lr"] != "sweep":
        return preset["lr"]
    table = pd.read_csv(results_dir(preset_name) / "lr_sweep.csv")
    return dict(zip(table[table["best"]]["size"], table[table["best"]]["lr"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke")
    args = parser.parse_args()
    preset = load_yaml(args.preset)
    lrs = learning_rates(args.preset, preset)

    for tokens in preset["budgets"]:
        for size in preset["sizes"]:
            run_id = f"{size}_D{tokens}_seed{preset['seed']}"
            run_dir = RUNS_DIR / args.preset / run_id
            if (run_dir / "summary.json").exists():
                print(f"skip {run_id}, already done")
                continue
            run(run_config(preset, size, tokens, lrs[size]), run_dir)


if __name__ == "__main__":
    main()
