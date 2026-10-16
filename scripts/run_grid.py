"""Run every training job of experiment A for a preset. Resumable.

Three kinds of job, all at the uniform mixture:
- grid: every (size, budget) pair, seed from the preset.
- noise: a few (size, budget) cells again with extra seeds.
- lr_check: the largest budget again for some sizes, with the two LRs next to
  the swept one, to see whether the best LR drifts with D.

A run whose summary.json exists is skipped, so the script can be stopped and
restarted at any time. Jobs are ordered by budget first, so all sizes at the
smallest D finish before any larger D starts. After every run the result
tables are rewritten from the summaries on disk.
"""

import argparse
import json
import time

import pandas as pd

from tnms.config import REPO_ROOT, RUNS_DIR, load_yaml, results_dir, run_config
from tnms.data.tokenizer import load_tokenizer
from tnms.evaluate import caption_entropy_floor
from tnms.train import load_datasets, run


def sweep_table(preset_name: str) -> pd.DataFrame:
    return pd.read_csv(results_dir(preset_name) / "lr_sweep.csv")


def learning_rates(preset_name: str, preset: dict) -> dict:
    """LR per size: written in the preset, or the best one from the LR sweep."""
    if preset["lr"] != "sweep":
        return preset["lr"]
    table = sweep_table(preset_name)
    return dict(zip(table[table["best"]]["size"], table[table["best"]]["lr"]))


def neighbour_lrs(preset_name: str, size: str) -> list[float]:
    """The LRs just below and just above the chosen one in the sweep."""
    table = sweep_table(preset_name)
    lrs = sorted(table[table["size"] == size]["lr"])
    i = lrs.index(float(table[(table["size"] == size) & table["best"]]["lr"].iloc[0]))
    return [lrs[j] for j in (i - 1, i + 1) if 0 <= j < len(lrs)]


def plan_jobs(preset_name: str, preset: dict) -> list[dict]:
    """Every job as {kind, size, tokens, lr, seed, dir}, smallest budget first."""
    lrs = learning_rates(preset_name, preset)
    root = RUNS_DIR / preset_name
    seed = preset["seed"]
    jobs = [{"kind": "grid", "size": s, "tokens": d, "lr": lrs[s], "seed": seed}
            for d in preset["budgets"] for s in preset["sizes"]]
    for size, tokens in preset.get("noise_cells", []):
        jobs += [{"kind": "noise", "size": size, "tokens": tokens, "lr": lrs[size], "seed": extra}
                 for extra in preset["noise_seeds"]]
    if "lr_check" in preset:
        tokens = preset["lr_check"]["tokens"]
        jobs += [{"kind": "lr_check", "size": size, "tokens": tokens, "lr": lr, "seed": seed}
                 for size in preset["lr_check"]["sizes"] for lr in neighbour_lrs(preset_name, size)]
    for job in jobs:
        name = f"{job['size']}_D{job['tokens']}_seed{job['seed']}"
        job["dir"] = root / "lr_check" / f"{name}_lr{job['lr']:g}" if job["kind"] == "lr_check" else root / name
    return sorted(jobs, key=lambda j: (j["tokens"], j["size"]))


def expected_seconds(job: dict, speed: dict) -> float:
    return job["tokens"] / speed[job["size"]] if speed else 0.0


def print_eta(todo: list[dict], speed: dict, slowdown: float):
    """Remaining time from the benchmark speed, scaled by how far off it has been so far."""
    if speed:
        hours = sum(expected_seconds(j, speed) for j in todo) * slowdown / 3600
        print(f"{len(todo)} runs left, about {hours:.1f} h (benchmark speed x {slowdown:.2f})")


def summary_row(job: dict, floors: dict) -> dict:
    s = json.loads((job["dir"] / "summary.json").read_text())
    row = {"kind": job["kind"], "size": job["size"], "N": s["N"], "D_budget": job["tokens"], "D": s["D"],
           "seed": job["seed"], "lr": job["lr"], "loss_tokens": s["loss_tokens"], "flops_6ND": s["flops_6ND"]}
    for split, key in (("test", "test"), ("val", "val"), ("train", "train_eval")):
        row.update({f"{split}_{d}": v for d, v in s[key].items()})
    for domain in ("image", "audio"):
        # Below the no-input floor means the model is reading the patches, not just the caption prior.
        row[f"{domain}_below_floor"] = s["test"][domain] < floors[domain]
    row.update({f"epochs_{d}": round(v, 3) for d, v in s["epochs"].items()})
    row.update({"wall_time_s": round(s["wall_time_s"]), "gpu": s["device"], "git_commit": s["git_commit"][:7]})
    return row


def curve_rows(job: dict) -> list[dict]:
    s = json.loads((job["dir"] / "summary.json").read_text())
    return [{"size": job["size"], "D_budget": job["tokens"], "seed": job["seed"], "fraction": p["fraction"],
             "tokens": p["tokens"], "split": split, "domain": d, "loss": v}
            for p in s["curve"] for split in ("val", "test") for d, v in p[split].items()]


def write_tables(preset_name: str, jobs: list[dict], floors: dict):
    done = [j for j in jobs if (j["dir"] / "summary.json").exists()]
    out = results_dir(preset_name)
    out.mkdir(parents=True, exist_ok=True)
    grid = [j for j in done if j["kind"] in ("grid", "noise")]
    if grid:
        pd.DataFrame([summary_row(j, floors) for j in grid]).to_csv(out / "grid.csv", index=False, float_format="%.5g")
        pd.DataFrame([r for j in grid for r in curve_rows(j)]).to_csv(out / "curves.csv", index=False, float_format="%.5g")
    checks = [j for j in done if j["kind"] == "lr_check"]
    if checks:
        # The chosen LR's run at that budget is the grid run, so it joins the comparison.
        cells = {(j["size"], j["tokens"]) for j in checks}
        rows = [summary_row(j, floors) for j in done
                if j["kind"] in ("grid", "lr_check") and (j["size"], j["tokens"]) in cells]
        table = pd.DataFrame(rows)[["size", "D_budget", "lr", "kind", "val_mean", "test_mean"]]
        table["best_val"] = table["val_mean"] == table.groupby("size")["val_mean"].transform("min")
        table.sort_values(["size", "lr"]).to_csv(out / "lr_check.csv", index=False, float_format="%.5g")
    (out / "floors.json").write_text(json.dumps(floors, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke")
    args = parser.parse_args()
    preset = load_yaml(args.preset)
    jobs = plan_jobs(args.preset, preset)

    tokenizer, test = load_tokenizer(), load_datasets("test")
    floors = {d: caption_entropy_floor(test[d], tokenizer, preset["eval_examples"])["without_input"]
              for d in ("image", "audio")}

    bench = REPO_ROOT / "results" / "benchmark.json"
    speed = json.loads(bench.read_text())["tokens_per_sec"] if bench.exists() else {}
    todo = [j for j in jobs if not (j["dir"] / "summary.json").exists()]
    print(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} already done")
    expected = actual = 0.0
    for job in todo:
        print_eta(todo[todo.index(job):], speed, actual / expected if expected else 1.0)
        t = time.time()
        run(run_config(preset, job["size"], job["tokens"], job["lr"], seed=job["seed"]), job["dir"])
        actual += time.time() - t
        expected += expected_seconds(job, speed)
        write_tables(args.preset, jobs, floors)
    write_tables(args.preset, jobs, floors)


if __name__ == "__main__":
    main()
