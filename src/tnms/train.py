"""One training run: config in, metrics, summary and checkpoint out.

Every (N, D) point of the study is its own complete run. The learning-rate
schedule always finishes at the run's own token budget D. We never cut a
longer run short and call it a smaller-budget run, because a run stopped in
the middle of its cosine decay has a higher loss than one whose schedule
was planned for that budget.
"""

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tnms.config import REPO_ROOT
from tnms.data.cifar100 import Cifar100
from tnms.data.mixture_sampler import MixtureSampler, estimate_mean_tokens
from tnms.data.sequence import IGNORE, collate
from tnms.data.speechcommands import SpeechCommands
from tnms.data.tinystories import TinyStories
from tnms.data.tokenizer import load_tokenizer
from tnms.evaluate import autocast, evaluate, to_device
from tnms.flops import count_non_embedding_params, train_flops
from tnms.model import Model

SCHEDULE_PROBE_BATCHES = 200
EVAL_FRACTIONS = (0.25, 0.5, 0.75)


def load_datasets(split: str) -> dict:
    return {"image": Cifar100(split), "audio": SpeechCommands(split), "text": TinyStories(split)}


def warmup_steps(total_steps: int) -> int:
    """5% of the steps, at least 100, but never more than 10% of a very short run."""
    return min(max(100, round(0.05 * total_steps)), max(1, total_steps // 10))


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay from peak to 10% of peak at `total` steps.

    Batches vary in length, so the run can take a few more steps than
    planned. Those extra steps stay at the final 10% value.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def param_groups(model, weight_decay: float = 0.1):
    """Weight decay on the transformer's matrices only.

    No decay on norms, and none on the embedding-like input layers (token and
    position embeddings, patch projections), which are also the parameters
    left out of N.
    """
    embeddings = {model.tok_emb.weight, model.pos_emb.weight, model.image_proj.weight, model.audio_proj.weight}
    decay = [p for p in model.parameters() if p.dim() == 2 and p not in embeddings]
    no_decay = [p for p in model.parameters() if p.dim() != 2 or p in embeddings]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_json_atomic(path: Path, data: dict):
    """Write to a temp file then rename, so a crash never leaves half a file.

    The grid runner treats an existing summary.json as "this run is done".
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def shuffle_labels(dataset, seed: int):
    """Control 3: give every training image the label of another, fixed for the whole run.

    The captions keep their real distribution, but they no longer say anything
    about the image. A model trained this way can at best learn the label
    frequencies, so its held-out image loss should sit at the no-input floor.
    """
    dataset.labels = np.random.default_rng(seed).permutation(dataset.labels)


def overfit_one_batch(model, batch: dict, device: str, steps: int, lr: float) -> list[float]:
    """Sanity control: train on one fixed batch again and again.

    Any working model, loss and optimiser can memorise 32 sequences, so the
    loss should fall close to zero. If it stalls, gradients are not reaching
    the parameters or the targets cannot be predicted from the inputs.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    batch = to_device(batch, device)
    losses = []
    for _ in range(steps):
        with autocast(device):
            logits = model(batch)
        loss = F.cross_entropy(logits.float().flatten(0, 1), batch["targets"].flatten(), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def run(cfg: dict, run_dir: Path, datasets: dict | None = None) -> dict:
    """Train one model and return its summary.

    cfg keys: size, d_model, n_layers, n_heads, tokens (D), lr, batch_size,
    shares, seed, eval_examples, log_every, device ("cpu" or "auto").
    `datasets` maps "train"/"val"/"test" to {domain: dataset}. It is only
    passed by the tests, which use small synthetic datasets.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    tokenizer = load_tokenizer()
    if datasets is None:
        datasets = {split: load_datasets(split) for split in ("train", "val", "test")}
    if cfg.get("shuffle_image_labels"):
        shuffle_labels(datasets["train"]["image"], cfg["seed"])

    model = Model(cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg.get("encoder_layers", 0)).to(device)
    n_params = count_non_embedding_params(model)
    print(f"[{run_dir.name}] N = {n_params:,} non-embedding parameters, device = {device}")

    mean_tokens = estimate_mean_tokens(datasets["train"], tokenizer)
    sampler = MixtureSampler(datasets["train"], cfg["shares"], mean_tokens, cfg["seed"])

    def next_batch():
        return collate([sampler.next_example() for _ in range(cfg["batch_size"])], tokenizer)

    # Fix the schedule length from the mean token count of the first batches.
    # These batches are then used for training, in order.
    probe = [next_batch() for _ in range(SCHEDULE_PROBE_BATCHES)]
    tokens_per_batch = float(np.mean([b["lengths"].sum().item() for b in probe]))
    total_steps = math.ceil(cfg["tokens"] / tokens_per_batch)
    warmup = warmup_steps(total_steps)

    opt = torch.optim.AdamW(param_groups(model), lr=cfg["lr"], betas=(0.9, 0.95))
    # GradScaler is only needed for fp16. bf16 has the same range as fp32.
    scaler = torch.amp.GradScaler(enabled=device == "cuda" and not torch.cuda.is_bf16_supported())

    def eval_all(splits=("val", "test")):
        return {s: evaluate(model, datasets[s], tokenizer, cfg["eval_examples"], device) for s in splits}

    curve, fractions = [], list(EVAL_FRACTIONS)
    step = tokens = loss_tokens = 0
    t_start = t_log = time.time()
    tokens_at_log = 0
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step", "tokens", "train_loss", "lr", "tokens_per_sec"])
        while tokens < cfg["tokens"]:
            cpu_batch = probe[step] if step < len(probe) else next_batch()
            batch = to_device(cpu_batch, device)
            lr = lr_at(step, cfg["lr"], warmup, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            with autocast(device):
                logits = model(batch)
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch["targets"].flatten(), ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            step += 1
            tokens += int(cpu_batch["lengths"].sum())
            loss_tokens += int((cpu_batch["targets"] != IGNORE).sum())

            if step % cfg["log_every"] == 0:
                now = time.time()
                speed = (tokens - tokens_at_log) / (now - t_log)
                log.writerow([step, tokens, f"{loss.item():.5f}", f"{lr:.3e}", f"{speed:.0f}"])
                f.flush()
                t_log, tokens_at_log = now, tokens
            if fractions and tokens >= fractions[0] * cfg["tokens"]:
                curve.append({"fraction": fractions.pop(0), "step": step, "tokens": tokens, **eval_all()})
                t_log, tokens_at_log = time.time(), tokens  # do not count eval time as training speed

    wall_time = time.time() - t_start
    final = eval_all(("train", "val", "test"))  # train loss (no augmentation) for the overfitting gap
    curve.append({"fraction": 1.0, "step": step, "tokens": tokens, "val": final["val"], "test": final["test"]})
    torch.save(model.state_dict(), run_dir / "model.pt")

    summary = {
        "run_id": run_dir.name,
        "config": cfg,
        "N": n_params,
        "D": tokens,
        "loss_tokens": loss_tokens,
        "steps": step,
        "planned_steps": total_steps,
        "warmup_steps": warmup,
        "flops_6ND": train_flops(n_params, tokens),
        "mean_tokens_per_example": mean_tokens,
        "epochs": sampler.epochs(),
        "train_eval": final["train"],
        "val": final["val"],
        "test": final["test"],
        "curve": curve,
        "wall_time_s": wall_time,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "git_commit": git_commit(),
    }
    write_json_atomic(run_dir / "summary.json", summary)
    print(f"[{run_dir.name}] done in {wall_time:.0f}s, D = {tokens:,}, val mean loss = {final['val']['mean']:.4f}")
    return summary
