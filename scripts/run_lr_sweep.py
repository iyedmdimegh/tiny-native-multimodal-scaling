"""Pick a learning rate per model size. Resumable.

For each size, train at a fixed budget with every LR in the preset's list and
keep the one with the lowest validation loss averaged over the three domains.
If the best LR is at the edge of the list, add one more LR in that direction
(half or double) and try again, so the chosen LR is never an untested edge.
Writes lr_sweep.csv with one row per run and a `best` column.
"""

import argparse
import json

import pandas as pd

from tnms.config import RUNS_DIR, load_yaml, results_dir, run_config
from tnms.train import run

MAX_EXTENSIONS = 2


def sweep_run(preset_name: str, preset: dict, size: str, lr: float) -> dict:
    run_dir = RUNS_DIR / preset_name / "lr_sweep" / f"{size}_lr{lr:g}"
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text())
    else:
        summary = run(run_config(preset, size, preset["lr_sweep"]["tokens"], lr), run_dir)
    return {
        "size": size, "N": summary["N"], "lr": lr, "D": summary["D"],
        **{f"val_{d}": v for d, v in summary["val"].items()},
    }


def sweep_size(preset_name: str, preset: dict, size: str) -> list[dict]:
    lrs = list(preset["lr_sweep"]["lrs"])
    for _ in range(MAX_EXTENSIONS + 1):
        rows = [sweep_run(preset_name, preset, size, lr) for lr in lrs]
        best = min(rows, key=lambda r: r["val_mean"])["lr"]
        if best == lrs[0]:
            lrs.insert(0, lrs[0] / 2)
        elif best == lrs[-1]:
            lrs.append(lrs[-1] * 2)
        else:
            break
    else:
        print(f"warning: best LR for {size} is still at the edge after {MAX_EXTENSIONS} extensions")
    for r in rows:
        r["best"] = r["lr"] == best
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke")
    args = parser.parse_args()
    preset = load_yaml(args.preset)

    rows = []
    for size in preset["sizes"]:
        rows += sweep_size(args.preset, preset, size)
        table = pd.DataFrame(rows)
        out = results_dir(args.preset) / "lr_sweep.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False, float_format="%.5g")  # written after each size, so partial results are kept
    print(table[table["best"]][["size", "N", "lr", "val_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
