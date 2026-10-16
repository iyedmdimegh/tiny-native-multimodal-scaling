"""Sanity controls that show the model and the training setup behave as expected.

    python scripts/run_controls.py overfit              # control 1, all sizes
    python scripts/run_controls.py ablation RUN_DIR     # control 2, one trained model
    python scripts/run_controls.py shuffled             # control 3, trains S2 at 10M
    python scripts/run_controls.py gap                  # control 4, from results/grid.csv
    python scripts/run_controls.py samples RUN_DIR      # control 5, writes results/samples.txt

Results are merged into results/controls.json.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from tnms.config import REPO_ROOT, RUNS_DIR, load_yaml, model_shape, run_config
from tnms.data.mixture_sampler import MixtureSampler, estimate_mean_tokens
from tnms.data.sequence import DOMAINS, IGNORE, MAX_LEN, collate
from tnms.data.tokenizer import BOS, EOS, load_tokenizer
from tnms.evaluate import caption_entropy_floor, domain_loss, load_model
from tnms.model import Model
from tnms.train import load_datasets, overfit_one_batch, run

RESULTS = REPO_ROOT / "results" / "controls.json"
SIZES = ["S1", "S2", "S3", "S4", "S5"]
# Written before any sample was generated. All five go into samples.txt.
PROMPTS = [
    "Once upon a time, there was a little girl named",
    "Tom and his dog went to the park. They",
    "One day, a small bird found",
    "The sun was hot, so Lily wanted to",
    "Ben had a red ball. He",
]


def save(key: str, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2))


def first_token_floor(batch) -> float:
    """The part of the one-batch loss that cannot be memorised.

    Every text row starts with the same input (<bos>) but a different first
    story token. No model can tell these rows apart at position 0, so the best
    it can do there is the empirical distribution of those first tokens.
    Returned in nats per loss token, like the loss itself.
    """
    text_rows = batch["domain"] == DOMAINS.index("text")
    firsts = batch["targets"][text_rows, 0].tolist()
    k = len(firsts)
    nats = -sum(c * math.log(c / k) for c in Counter(firsts).values())
    return nats / (batch["targets"] != IGNORE).sum().item()


def overfit(steps: int = 600, lr: float = 1e-3, tolerance: float = 0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    train = load_datasets("train")
    shares = {"image": 1 / 3, "audio": 1 / 3, "text": 1 / 3}
    sampler = MixtureSampler(train, shares, estimate_mean_tokens(train, tokenizer), seed=0)
    batch = collate([sampler.next_example() for _ in range(32)], tokenizer)
    floor = first_token_floor(batch)
    print(f"irreducible floor for this batch: {floor:.4f} nats per token")

    results, failed = {}, []
    for size in SIZES:
        torch.manual_seed(0)
        model = Model(**model_shape(size)).to(device)
        losses = overfit_one_batch(model, batch, device, steps=steps, lr=lr)
        final = min(losses[-50:])  # Adam can spike for a step, take the best of the last 50
        ok = final <= floor + tolerance
        results[size] = {"start": losses[0], "final": final, "passed": ok}
        print(f"{size}: {losses[0]:.3f} -> {final:.4f}  {'ok' if ok else 'FAILED'}")
        if not ok:
            failed.append(size)
    save("overfit_one_batch", {"steps": steps, "lr": lr, "floor": floor, "tolerance": tolerance, "sizes": results})
    if failed:
        raise SystemExit(f"overfit control FAILED for {failed}: loss did not reach floor + {tolerance}")


def ablation(run_dir: Path, n_examples: int = 2000):
    """Control 2: replace the patches and check that the caption loss goes up."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model, _ = load_model(run_dir, device)
    test = load_datasets("test")
    out = {}
    for domain in ("image", "audio"):
        ds = test[domain]
        losses = {mode: domain_loss(model, ds, tokenizer, n_examples, device, patches=mode)
                  for mode in ("real", "zero", "other")}
        floors = caption_entropy_floor(ds, tokenizer, n_examples)
        out[domain] = {"loss": losses, "floor": floors}
        print(f"{domain}: real {losses['real']:.3f}, zero {losses['zero']:.3f}, other {losses['other']:.3f} "
              f"| floor with input {floors['with_input']:.3f}, without input {floors['without_input']:.3f}")
    save(f"patch_ablation/{run_dir.name}", {"run": str(run_dir.name), "n_examples": n_examples, **out})


def shuffled(size: str = "S2", tokens: int = 10_000_000, tolerance: float = 0.01):
    """Control 3: train with image labels shuffled across images.

    The captions then carry no information about the image, so the held-out
    image loss should sit near the no-input floor. Clearly lower would mean
    the label leaks in some other way.
    """
    preset = load_yaml("full")
    lr = float(pd.read_csv(REPO_ROOT / "results" / "lr_sweep.csv").query("best and size == @size")["lr"].iloc[0])
    cfg = {**run_config(preset, size, tokens, lr), "shuffle_image_labels": True}
    run_dir = RUNS_DIR / "full" / "controls" / f"shuffled_{size}_D{tokens}"
    summary = run(cfg, run_dir)
    floors = caption_entropy_floor(load_datasets("test")["image"], load_tokenizer(), preset["eval_examples"])
    loss = summary["test"]["image"]
    leak = loss < floors["without_input"] - tolerance
    print(f"shuffled labels: test image loss {loss:.3f}, floor without input {floors['without_input']:.3f}"
          f"{'  LOWER THAN FLOOR, possible leak' if leak else ''}")
    save("shuffled_labels", {"size": size, "tokens": tokens, "lr": lr, "test_image": loss, "floor": floors,
                             "below_floor_by_more_than_tolerance": leak, "tolerance": tolerance})


def train_val_gap():
    """Control 4: validation minus train loss at the end of every grid run, per domain."""
    table = pd.read_csv(REPO_ROOT / "results" / "grid.csv").query("kind == 'grid'")
    rows = []
    for _, r in table.iterrows():
        gaps = {d: r[f"val_{d}"] - r[f"train_{d}"] for d in ("image", "audio", "text")}
        rows.append({"size": r["size"], "D_budget": int(r["D_budget"]), **{d: round(g, 4) for d, g in gaps.items()}})
        print(f"{r['size']} D={r['D_budget']:>9,}: " + ", ".join(f"{d} {g:+.3f}" for d, g in gaps.items()))
    save("train_val_gap", rows)


@torch.no_grad()
def greedy(model, tokenizer, prompt: str, device: str) -> str:
    """Extend a text prompt one most-likely token at a time, up to <eos> or 128 tokens."""
    ids = [BOS] + tokenizer.encode(prompt).ids
    empty = torch.empty(0, dtype=torch.long, device=device)
    zero = torch.zeros(1, dtype=torch.long, device=device)
    while len(ids) < MAX_LEN:
        batch = {"input_ids": torch.tensor([ids], device=device), "lengths": torch.tensor([len(ids)], device=device),
                 "prefix_start": zero, "prefix_end": zero, "image_rows": empty, "audio_rows": empty}
        next_id = int(model(batch)[0, -1].argmax())
        if next_id == EOS:
            break
        ids.append(next_id)
    return tokenizer.decode(ids[1:])


def samples(run_dir: Path):
    """Control 5: greedy continuations of the fixed prompts, all of them, in order."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model, _ = load_model(run_dir, device)
    model.eval()
    lines = [f"Greedy samples from {run_dir.name}. Fixed prompts, every sample kept.", ""]
    for i, prompt in enumerate(PROMPTS, 1):
        lines += [f"{i}. Prompt: {prompt}", f"   Output: {greedy(model, tokenizer, prompt, device)}", ""]
    text = "\n".join(lines)
    (REPO_ROOT / "results" / "samples.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="control", required=True)
    sub.add_parser("overfit")
    p = sub.add_parser("ablation")
    p.add_argument("run_dir", type=Path)
    sub.add_parser("shuffled")
    sub.add_parser("gap")
    p = sub.add_parser("samples")
    p.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.control == "overfit":
        overfit()
    elif args.control == "ablation":
        ablation(args.run_dir)
    elif args.control == "shuffled":
        shuffled()
    elif args.control == "gap":
        train_val_gap()
    else:
        samples(args.run_dir)


if __name__ == "__main__":
    main()
