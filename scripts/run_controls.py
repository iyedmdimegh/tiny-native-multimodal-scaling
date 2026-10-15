"""Sanity controls that show the model and the training setup behave as expected.

    python scripts/run_controls.py overfit              # control 1, all sizes
    python scripts/run_controls.py ablation RUN_DIR     # control 2, one trained model

Results are merged into results/controls.json.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from tnms.config import REPO_ROOT, model_shape
from tnms.data.mixture_sampler import MixtureSampler, estimate_mean_tokens
from tnms.data.sequence import DOMAINS, IGNORE, collate
from tnms.data.tokenizer import load_tokenizer
from tnms.evaluate import caption_entropy_floor, domain_loss, load_model
from tnms.model import Model
from tnms.train import load_datasets, overfit_one_batch

RESULTS = REPO_ROOT / "results" / "controls.json"
SIZES = ["S1", "S2", "S3", "S4", "S5"]


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


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="control", required=True)
    sub.add_parser("overfit")
    p = sub.add_parser("ablation")
    p.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.control == "overfit":
        overfit()
    else:
        ablation(args.run_dir)


if __name__ == "__main__":
    main()
