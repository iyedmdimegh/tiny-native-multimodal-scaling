"""Run every (size, budget) pair of a preset. Resumable.

A run whose summary.json exists is skipped, so the script can be stopped and
restarted at any time. Runs are ordered by budget first, so all sizes at the
smallest D finish before any larger D starts.
"""

import argparse

from tnms.config import RUNS_DIR, load_yaml, model_shape
from tnms.train import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke")
    args = parser.parse_args()
    preset = load_yaml(args.preset)

    for tokens in preset["budgets"]:
        for size in preset["sizes"]:
            run_id = f"{size}_D{tokens}_seed{preset['seed']}"
            run_dir = RUNS_DIR / args.preset / run_id
            if (run_dir / "summary.json").exists():
                print(f"skip {run_id}, already done")
                continue
            cfg = {
                "size": size,
                **model_shape(size),
                "tokens": tokens,
                "lr": preset["lr"][size],
                "batch_size": preset["batch_size"],
                "shares": preset["shares"],
                "seed": preset["seed"],
                "eval_examples": preset["eval_examples"],
                "log_every": preset["log_every"],
                "device": preset.get("device", "auto"),
            }
            run(cfg, run_dir)


if __name__ == "__main__":
    main()
