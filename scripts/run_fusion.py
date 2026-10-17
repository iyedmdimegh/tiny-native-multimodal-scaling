"""Experiment C (optional): early against late fusion at matched N. Resumable.

Early fusion is the grid model. Late fusion gives image and audio their own
2-layer encoder and takes 4 layers off the decoder, so N (counted over encoders
plus decoder) is exactly the same. Both use the uniform mixture, the swept LR
of the early model, and three seeds. Early-fusion runs reuse the grid and noise
runs where they exist. Writes results/fusion.csv.
"""

import json

import pandas as pd

from tnms.config import REPO_ROOT, RUNS_DIR, load_yaml, model_shape, run_config
from tnms.train import run

SIZES = ["S3", "S4"]
TOKENS = 40_000_000
SEEDS = [0, 1, 2]
ENCODER_LAYERS = 2


def main():
    preset = load_yaml("full")
    sweep = pd.read_csv(REPO_ROOT / "results" / "lr_sweep.csv")
    rows = []
    for size in SIZES:
        lr = float(sweep[sweep["best"] & (sweep["size"] == size)]["lr"].iloc[0])
        for seed in SEEDS:
            for variant in ("early", "late"):
                cfg = run_config(preset, size, TOKENS, lr, seed=seed)
                if variant == "early":
                    run_dir = RUNS_DIR / "full" / f"{size}_D{TOKENS}_seed{seed}"  # same name as the grid
                else:
                    cfg.update(n_layers=model_shape(size)["n_layers"] - 2 * ENCODER_LAYERS,
                               encoder_layers=ENCODER_LAYERS)
                    run_dir = RUNS_DIR / "full" / "fusion" / f"{size}_late_D{TOKENS}_seed{seed}"
                if (run_dir / "summary.json").exists():
                    s = json.loads((run_dir / "summary.json").read_text())
                else:
                    s = run(cfg, run_dir)
                rows.append({"size": size, "variant": variant, "seed": seed, "N": s["N"], "D": s["D"],
                             "wall_time_s": round(s["wall_time_s"]),
                             **{f"test_{d}": v for d, v in s["test"].items()}})
                pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "fusion.csv", index=False, float_format="%.5g")

    table = pd.DataFrame(rows)
    summary = table.groupby(["size", "variant"])[["test_image", "test_audio", "test_text", "test_mean"]].agg(["mean", "std"])
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
