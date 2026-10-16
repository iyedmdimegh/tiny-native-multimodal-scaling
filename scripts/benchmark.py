"""Measure training throughput per size and estimate the GPU time of every experiment.

Each size trains for 400 steps through the real `train.run`. The first 200
steps reuse batches built before the timer starts (the schedule probe), so
only steps 200 to 400 are counted: they include building the batches.

Also prints how many epochs of each domain a uniform-mixture run sees at each
budget D, and writes that table to results/epochs_per_domain.csv.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from tnms.config import REPO_ROOT, model_shape
from tnms.data.mixture_sampler import domain_probs, estimate_mean_tokens
from tnms.data.tokenizer import load_tokenizer
from tnms.train import load_datasets, run

SIZES = ["S1", "S2", "S3", "S4", "S5"]
UNIFORM = {"image": 1 / 3, "audio": 1 / 3, "text": 1 / 3}
GRID_BUDGETS = [5e6, 10e6, 20e6, 40e6, 80e6]


def measure(size: str, datasets: dict, tokens_per_step: float) -> float:
    cfg = {
        "size": size, **model_shape(size), "tokens": int(400 * tokens_per_step), "lr": 1e-3,
        "batch_size": 32, "shares": UNIFORM, "seed": 0, "eval_examples": 64, "log_every": 20, "device": "auto",
    }
    with tempfile.TemporaryDirectory() as tmp:
        run(cfg, Path(tmp) / size, datasets)
        metrics = pd.read_csv(Path(tmp) / size / "metrics.csv")
    return float(metrics[metrics["step"] > 220]["tokens_per_sec"].median())


def main():
    tokenizer = load_tokenizer()
    datasets = {split: load_datasets(split) for split in ("train", "val", "test")}
    mean_tokens = estimate_mean_tokens(datasets["train"], tokenizer)
    probs = domain_probs(UNIFORM, mean_tokens)
    tokens_per_example = sum(probs[d] * mean_tokens[d] for d in probs)

    speed = {size: measure(size, datasets, 32 * tokens_per_example) for size in SIZES}
    for size, s in speed.items():
        print(f"{size}: {s:,.0f} tokens/s")

    # Time for every planned run, in hours, from the measured speed.
    hours = lambda size, tokens: tokens / speed[size] / 3600
    plan = {
        "lr_sweep (5 sizes x 5 LRs at 10M)": sum(5 * hours(s, 10e6) for s in SIZES),
        "grid (5 sizes x 5 budgets)": sum(hours(s, d) for s in SIZES for d in GRID_BUDGETS),
        "noise seeds (2 extra each: S2 10M, S3 20M, S4 40M)": 2 * (hours("S2", 10e6) + hours("S3", 20e6) + hours("S4", 40e6)),
        "LR check at 80M (2 neighbours for S1 and S5)": 2 * (hours("S1", 80e6) + hours("S5", 80e6)),
        "shuffled-label control (S2 10M)": hours("S2", 10e6),
        "mixture sweep (13 runs S4 40M)": 13 * hours("S4", 40e6),
        "mixture validation (3 fresh runs S4 40M)": 3 * hours("S4", 40e6),
    }
    for name, h in plan.items():
        print(f"{name}: {h:.2f} h")
    print(f"total, without evaluation overhead: {sum(plan.values()):.1f} h")

    rows = []
    for d in GRID_BUDGETS:
        examples = d / tokens_per_example
        rows.append({"D": int(d), **{dom: round(examples * probs[dom] / len(datasets["train"][dom]), 2) for dom in probs}})
    epochs = pd.DataFrame(rows)
    print("\nepochs per domain, uniform mixture:\n", epochs.to_string(index=False))
    epochs.to_csv(REPO_ROOT / "results" / "epochs_per_domain.csv", index=False)
    (REPO_ROOT / "results" / "benchmark.json").write_text(json.dumps(
        {"tokens_per_sec": speed, "mean_tokens_per_example": mean_tokens, "hours": plan}, indent=2))


if __name__ == "__main__":
    main()
