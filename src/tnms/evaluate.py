"""Held-out loss per domain.

Each domain is scored on a fixed subset of its held-out split: the same
examples, in the same order, with no augmentation, for every run. The loss is
the mean cross-entropy in nats over the loss-bearing tokens of that domain.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from tnms.data.sequence import IGNORE, build_sequence, collate
from tnms.model import Model

EVAL_SUBSET_SEED = 1234  # fixed: which held-out examples are scored


def fixed_subset(n_total: int, n: int) -> list[int]:
    idx = np.random.default_rng(EVAL_SUBSET_SEED).permutation(n_total)[:n]
    return sorted(int(i) for i in idx)


def autocast(device: str):
    """bf16 autocast on GPUs that support it, fp16 on older GPUs, off on CPU."""
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def domain_loss(
    model, dataset, tokenizer, n_examples: int, device: str, batch_size: int = 64, patches: str = "real"
) -> float:
    """Mean next-token loss (nats) over the loss tokens of a fixed held-out subset.

    `patches` is only for the ablation control:
    - "real": the normal evaluation.
    - "zero": every patch vector replaced by zeros.
    - "other": each example gets the patches of the next example in the batch,
      so the caption no longer matches the input.
    """
    model.eval()
    idx = fixed_subset(len(dataset), n_examples)
    total, count = 0.0, 0
    for start in range(0, len(idx), batch_size):
        batch = collate([dataset.get(i) for i in idx[start : start + batch_size]], tokenizer)
        key = f"{dataset.domain}_patches"
        if patches == "zero":
            batch[key] = torch.zeros_like(batch[key])
        elif patches == "other":
            batch[key] = batch[key].roll(1, dims=0)
        batch = to_device(batch, device)
        with autocast(device):
            logits = model(batch)
        total += F.cross_entropy(
            logits.float().flatten(0, 1), batch["targets"].flatten(), ignore_index=IGNORE, reduction="sum"
        ).item()
        count += (batch["targets"] != IGNORE).sum().item()
    model.train()
    return total / count


def evaluate(model, datasets: dict, tokenizer, n_examples: int, device: str) -> dict:
    """Loss for each domain plus their plain average ("mean")."""
    losses = {d: domain_loss(model, ds, tokenizer, n_examples, device) for d, ds in datasets.items()}
    losses["mean"] = float(np.mean(list(losses.values())))
    return losses


def caption_entropy_floor(dataset, tokenizer, n_examples: int) -> dict:
    """The lowest caption loss possible on the fixed eval subset, with and without the input.

    A caption is a random template plus the label. Without seeing the input,
    the best a model can do is pay log(templates) + H(label) nats per
    caption. Seeing the input removes H(label) but not the template choice.
    Dividing by the number of loss tokens gives nats per token, the same
    unit as the reported loss. This assumes each (template, label) pair has
    one tokenization, so it is a rough reference, not an exact bound.
    """
    idx = fixed_subset(len(dataset), n_examples)
    labels = np.array([int(dataset.labels[i]) for i in idx])
    p = np.bincount(labels) / len(labels)
    p = p[p > 0]
    h_label = float(-(p * np.log(p)).sum())
    h_template = float(np.log(len(dataset.templates)))
    loss_tokens = sum(sum(build_sequence(dataset.get(i), tokenizer)["loss"]) for i in idx)
    return {
        "without_input": len(idx) * (h_template + h_label) / loss_tokens,
        "with_input": len(idx) * h_template / loss_tokens,
    }


def load_model(run_dir, device: str):
    """Rebuild a trained model from its run directory."""
    cfg = json.loads((run_dir / "summary.json").read_text())["config"]
    model = Model(cfg["d_model"], cfg["n_layers"], cfg["n_heads"]).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return model, cfg
