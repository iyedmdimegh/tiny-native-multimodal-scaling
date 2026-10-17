"""Attention mask: prefix is bidirectional, text is causal, padding is invisible."""

import torch

from tnms.data.sequence import collate
from tnms.model import Model, attention_mask


def test_mask_rules():
    # One image-like row (prefix 1..66, length 80) and one text row (length 128).
    lengths = torch.tensor([80, 128])
    m = attention_mask(lengths, torch.tensor([1, 0]), torch.tensor([67, 0]), 128)
    img = m[0]
    assert img[2, 65] and img[66, 1]  # patches see later patches and </img>
    assert not img[0, 1]  # <bos> stays causal
    assert not img[67, 68] and img[68, 67]  # caption tokens are causal
    assert not img[:, 80:].any()  # padding is never attended
    txt = m[1]
    assert torch.equal(txt, torch.tril(torch.ones(128, 128, dtype=torch.bool)))


def _logits(model, batch):
    with torch.no_grad():
        return model(batch)


def test_future_text_does_not_change_past_logits(fake_datasets, tokenizer):
    torch.manual_seed(0)
    model = Model(64, 2, 1).eval()
    train = fake_datasets["train"]
    batch = collate([train["image"].get(0), train["text"].get(1)], tokenizer)
    before = _logits(model, batch)

    changed = {k: v.clone() for k, v in batch.items()}
    changed["input_ids"][0, 70] = 100  # a caption token of the image row
    changed["input_ids"][1, 50] = 100  # a story token of the text row
    after = _logits(model, changed)
    assert torch.allclose(before[0, :70], after[0, :70], atol=1e-6)
    assert torch.allclose(before[1, :50], after[1, :50], atol=1e-6)
    assert not torch.allclose(before[0, 70:], after[0, 70:])


def test_later_patch_changes_earlier_patch_output(fake_datasets, tokenizer):
    """The prefix is bidirectional, so the first patch sees the last one."""
    torch.manual_seed(0)
    model = Model(64, 2, 1).eval()
    batch = collate([fake_datasets["train"]["image"].get(0)], tokenizer)
    before = _logits(model, batch)
    batch["image_patches"][0, -1] += 1.0
    after = _logits(model, batch)
    assert not torch.allclose(before[0, 2], after[0, 2])


def test_late_fusion_forward_runs_and_reads_patches(fake_datasets):
    from tnms.data.sequence import collate
    from tnms.data.tokenizer import load_tokenizer

    torch.manual_seed(0)
    model = Model(64, 1, 1, encoder_layers=2).eval()
    train = fake_datasets["train"]
    batch = collate([train["image"].get(0), train["audio"].get(0), train["text"].get(0)], load_tokenizer())
    logits = model(batch)
    assert logits.shape[:2] == batch["input_ids"].shape
    batch["image_patches"] = torch.zeros_like(batch["image_patches"])
    assert not torch.allclose(model(batch)[0], logits[0])  # changing the patches changes the image row
