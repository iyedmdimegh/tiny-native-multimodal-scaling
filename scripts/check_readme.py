"""Check that every number in the README's "What I found" section comes from results/.

`claims()` recomputes each reported number from the committed tables and
formats it the way the README writes it. The check fails when a claim is
missing from the README, or when the findings section holds a number that is
not one of the claims. Model names (S1 to S5), token budgets (5M to 80M) and
figure numbers are not counted as numbers.
"""

import json
import re
import sys

import pandas as pd

from tnms.config import REPO_ROOT

RESULTS = REPO_ROOT / "results"
SECTION = "## What I found"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(fit: dict, name: str, digits: int = 2) -> str:
    lo, hi = fit["ci95"][name]
    return f"{fit['fitted'][name]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def grid_claims() -> dict:
    grid = pd.read_csv(RESULTS / "grid.csv")
    runs = grid[grid["kind"] == "grid"].set_index(["size", "D_budget"])
    noise = pd.read_csv(RESULTS / "noise.csv")
    floors = json.loads((RESULTS / "floors.json").read_text())
    fits = json.loads((RESULTS / "fits.json").read_text())
    controls = json.loads((RESULTS / "controls.json").read_text())
    lr = pd.read_csv(RESULTS / "lr_check.csv")

    c = {}
    c["text loss, S1 at 5M"] = f"{runs.loc[('S1', 5_000_000), 'test_text']:.2f}"
    c["text loss, S5 at 80M"] = f"{runs.loc[('S5', 80_000_000), 'test_text']:.2f}"
    c["seed std of mean loss, lowest"] = f"{noise['test_mean_std'].min():.3f}"
    c["seed std of mean loss, highest"] = f"{noise['test_mean_std'].max():.3f}"

    # A domain reads its input once it is below the floor by twice its largest seed std (docs/DESIGN.md).
    for domain in ("image", "audio"):
        margin = 2 * noise[f"test_{domain}_std"].max()
        c[f"{domain} floor"] = f"{floors[domain]:.3f}"
        c[f"{domain} margin"] = f"{margin:.3f}"
        reads = runs[runs[f"test_{domain}"] < floors[domain] - margin]
        c[f"{domain} cells reading input"] = f"{len(reads)} of {len(runs)}"

    for target in ("image", "audio", "text", "mean"):
        c[f"{target} fit error"] = pct(fits[target]["residual_mare"])
    for target, holdout in (("mean", "largest N"), ("mean", "largest D"), ("audio", "largest D")):
        h = fits[target]["holdout_mare"][holdout]
        c[f"{target} {holdout}, law"] = pct(h["law"])
        c[f"{target} {holdout}, baseline"] = pct(h["baseline"])
    wins = sum(f["holdout_mare"][h]["law"] < f["holdout_mare"][h]["baseline"]
               for f in fits.values() for h in ("largest N", "largest D"))
    c["holdouts the law wins"] = f"{wins} of {2 * len(fits)}"
    for target in ("mean", "text"):
        c[f"{target} alpha"] = ci(fits[target], "alpha")
        c[f"{target} beta"] = ci(fits[target], "beta")
    c["mean N* exponent"] = ci(fits["mean"], "n_opt_exponent")
    c["mean D/N at largest C"] = f"{fits['mean']['d_over_n_at_max_C']:,.0f}"
    c["largest D/N in grid"] = f"{(grid['D'] / grid['N']).max():,.0f}"

    s5 = lr[lr["size"] == "S5"].set_index("kind")
    best = lr[(lr["size"] == "S5") & lr["best_val"]].iloc[0]
    c["S5 grid LR"] = f"{s5.loc['grid', 'lr']:g}"
    c["S5 best LR at 80M"] = f"{best['lr']:g}"
    c["S5 val gain from LR"] = f"{s5.loc['grid', 'val_mean'] - best['val_mean']:.3f}"

    ab = controls["patch_ablation/S4_D80000000_seed0"]
    for domain in ("image", "audio"):
        for mode in ("real", "other", "zero"):
            c[f"S4 80M {domain} {mode}"] = f"{ab[domain]['loss'][mode]:.2f}"
    c["shuffled image loss"] = f"{controls['shuffled_labels']['test_image']:.3f}"
    c["normal S2 10M image loss"] = f"{runs.loc[('S2', 10_000_000), 'test_image']:.3f}"
    gaps = pd.DataFrame(controls["train_val_gap"])
    c["largest train/val gap"] = f"{gaps[['image', 'audio', 'text']].to_numpy().max():.3f}"
    return c


def mixture_claims() -> dict:
    fit = json.loads((RESULTS / "mixture_fit.json").read_text())
    check = pd.read_csv(RESULTS / "mixture_validation.csv").set_index("label")
    noise = pd.read_csv(RESULTS / "noise.csv").set_index(["size", "D_budget"]).loc[("S4", 40_000_000)]

    c = {"mixtures in the sweep": f"{len(pd.read_csv(RESULTS / 'mixtures.csv'))} mixtures",
         "cross-validation splits": f"{fit['cross_validation_mae']['n_splits']} splits"}
    cv = fit["cross_validation_mae"]
    for i, domain in enumerate(("image", "audio", "text")):
        c[f"{domain} CV error, law"] = f"{cv['law'][i]:.3f}"
        c[f"{domain} CV error, linear"] = f"{cv['linear'][i]:.3f}"
    c["audio seed std at S4 40M"] = f"{noise['test_audio_std']:.3f}"
    c["mean seed std at S4 40M"] = f"{noise['test_mean_std']:.3f}"
    for target, label in (("mean", "best mean"), ("image", "best image")):
        h = fit["optimum"][target]["h"]
        c[f"{label} shares"] = "(" + ", ".join(f"{x:g}" for x in h) + ")"
        p = fit["optimum"][target]["predicted_val"]
        c[f"{label} predicted val"] = f"{(sum(p.values()) / 3 if target == 'mean' else p[target]):.3f}"
        c[f"{label} observed val"] = f"{check.loc[label, f'val_{target}']:.3f}"
        c[f"{label} test"] = f"{check.loc[label, f'test_{target}']:.3f}"
        c[f"uniform {target} test"] = f"{check.loc['uniform', f'test_{target}']:.3f}"
    c["best image epochs"] = f"{check.loc['best image', 'epochs_image']:.1f}"
    c["best image text test"] = f"{check.loc['best image', 'test_text']:.2f}"
    return c


def claims() -> dict:
    return {**grid_claims(), **mixture_claims()}


def numbers_in(text: str) -> list[str]:
    text = re.sub(r"\bS[1-5]\b|\b\d+M\b|[Ff]igure \d", " ", text)
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", text)


def main():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    c = claims()
    missing = [f"{name}: {value}" for name, value in c.items() if value not in readme]
    start = readme.find(SECTION)
    section = readme[start:readme.find("\n## ", start + 1)] if start >= 0 else ""
    allowed = {n for value in c.values() for n in numbers_in(value)} | {"95%"}  # the bootstrap interval level
    stray = sorted(set(numbers_in(section)) - allowed)

    for line in missing:
        print("claim not in README:", line)
    for n in stray:
        print("number in findings with no claim:", n)
    if start < 0:
        print(f"README has no '{SECTION}' section")
    ok = not missing and not stray and start >= 0
    print("README numbers match results/" if ok else "README check failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
