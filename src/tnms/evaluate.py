"""Held-out loss per domain.

Each domain is scored on a fixed subset of its held-out split: the same
examples, in the same order, with no augmentation, for every run. The loss is
the mean cross-entropy in nats over the loss-bearing tokens of that domain.
"""

import numpy as np
import torch
import torch.nn.functional as F

from tnms.data.sequence import IGNORE, collate

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
def domain_loss(model, dataset, tokenizer, n_examples: int, device: str, batch_size: int = 64) -> float:
    """Mean next-token loss (nats) over the loss tokens of a fixed held-out subset."""
    model.eval()
    idx = fixed_subset(len(dataset), n_examples)
    total, count = 0.0, 0
    for start in range(0, len(idx), batch_size):
        batch = collate([dataset.get(i) for i in idx[start : start + batch_size]], tokenizer)
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
