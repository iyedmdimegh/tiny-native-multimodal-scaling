"""Turn examples from any domain into one padded batch for the model.

Sequence formats:
    image: <bos> <img> [64 patch slots] </img> caption <eos>
    audio: <bos> <aud> [96 patch slots] </aud> caption <eos>
    text:  <bos> 126 story tokens <eos>

Patch slots hold a placeholder id (<pad>). The model overwrites their
embeddings with the projected patch vectors, so the id is never used.

The loss is only on text: caption tokens plus their <eos> for image and audio,
every token after <bos> for text.
"""

import torch

from tnms.data.tokenizer import AUD, AUD_END, BOS, EOS, IMG, IMG_END, PAD

DOMAINS = ("image", "audio", "text")
PATCH_START = 2  # patch slots always start right after <bos> and <img>/<aud>
MAX_LEN = 128
IGNORE = -100  # target value that cross_entropy skips


def build_sequence(ex: dict, tokenizer) -> dict:
    """Build ids, loss mask and bidirectional prefix range for one example.

    `prefix` is (start, end) with end exclusive: the open token, the patch
    slots and the close token. Those positions attend to each other in both
    directions. Text examples have an empty prefix.
    """
    if ex["domain"] == "text":
        ids = [BOS] + ex["ids"] + [EOS]
        return {"ids": ids, "loss": [0] + [1] * (len(ids) - 1), "prefix": (0, 0)}

    open_tok, close_tok = (IMG, IMG_END) if ex["domain"] == "image" else (AUD, AUD_END)
    n_patches = ex["patches"].shape[0]
    caption = tokenizer.encode(ex["caption"]).ids + [EOS]
    head = [BOS, open_tok] + [PAD] * n_patches + [close_tok]
    return {
        "ids": head + caption,
        "loss": [0] * len(head) + [1] * len(caption),
        "prefix": (1, PATCH_START + n_patches + 1),
    }


def collate(examples: list[dict], tokenizer) -> dict:
    """Right-pad a list of examples into one batch of tensors.

    targets[b, t] is the token the model should predict at position t, i.e.
    ids[b, t + 1], or IGNORE when position t + 1 carries no loss.
    """
    seqs = [build_sequence(ex, tokenizer) for ex in examples]
    B, T = len(seqs), max(len(s["ids"]) for s in seqs)
    assert T <= MAX_LEN, f"sequence of length {T} is longer than {MAX_LEN}"

    ids = torch.full((B, T), PAD, dtype=torch.long)
    loss_mask = torch.zeros(B, T, dtype=torch.bool)
    for b, s in enumerate(seqs):
        ids[b, : len(s["ids"])] = torch.tensor(s["ids"])
        loss_mask[b, : len(s["loss"])] = torch.tensor(s["loss"], dtype=torch.bool)

    targets = torch.full((B, T), IGNORE, dtype=torch.long)
    targets[:, :-1] = torch.where(loss_mask[:, 1:], ids[:, 1:], IGNORE)

    batch = {
        "input_ids": ids,
        "targets": targets,
        "loss_mask": loss_mask,
        "lengths": torch.tensor([len(s["ids"]) for s in seqs]),
        "prefix_start": torch.tensor([s["prefix"][0] for s in seqs]),
        "prefix_end": torch.tensor([s["prefix"][1] for s in seqs]),
        "domain": torch.tensor([DOMAINS.index(ex["domain"]) for ex in examples]),
    }
    for name in ("image", "audio"):
        rows = [b for b, ex in enumerate(examples) if ex["domain"] == name]
        batch[f"{name}_rows"] = torch.tensor(rows, dtype=torch.long)
        batch[f"{name}_patches"] = (
            torch.stack([examples[b]["patches"] for b in rows]) if rows else torch.empty(0)
        )
    return batch
