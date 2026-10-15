"""The loss mask: loss only on text, never on patches or modality tokens."""

import torch

from tnms.data.sequence import IGNORE, collate
from tnms.data.tokenizer import BOS, EOS, IMG, IMG_END


def test_image_loss_positions_are_caption_and_eos(fake_datasets, tokenizer):
    ex = fake_datasets["train"]["image"].get(3)
    b = collate([ex], tokenizer)
    caption = tokenizer.encode(ex["caption"]).ids
    n = 1 + 1 + 64 + 1  # <bos> <img> patches </img>
    ids = b["input_ids"][0]
    assert ids[0] == BOS and ids[1] == IMG and ids[n - 1] == IMG_END
    assert ids[n:].tolist() == caption + [EOS]
    expected = torch.zeros(len(ids), dtype=torch.bool)
    expected[n:] = True
    assert torch.equal(b["loss_mask"][0], expected)
    assert (b["prefix_start"][0], b["prefix_end"][0]) == (1, n)


def test_audio_prefix_covers_96_patches(fake_datasets, tokenizer):
    b = collate([fake_datasets["train"]["audio"].get(0)], tokenizer)
    assert (b["prefix_start"][0], b["prefix_end"][0]) == (1, 1 + 1 + 96 + 1)
    assert not b["loss_mask"][0, :99].any() and b["loss_mask"][0, 99:].all()


def test_text_loss_on_everything_after_bos(fake_datasets, tokenizer):
    b = collate([fake_datasets["train"]["text"].get(0)], tokenizer)
    assert b["input_ids"].shape == (1, 128)
    assert not b["loss_mask"][0, 0] and b["loss_mask"][0, 1:].all()
    assert b["input_ids"][0, -1] == EOS


def test_targets_are_shifted_ids_on_loss_positions(fake_datasets, tokenizer):
    train = fake_datasets["train"]
    b = collate([train["image"].get(1), train["text"].get(2), train["audio"].get(3)], tokenizer)
    ids, targets, mask = b["input_ids"], b["targets"], b["loss_mask"]
    for row in range(3):
        for t in range(ids.shape[1] - 1):
            want = ids[row, t + 1].item() if mask[row, t + 1] else IGNORE
            assert targets[row, t].item() == want
    # Padding never carries loss.
    for row, n in enumerate(b["lengths"].tolist()):
        assert not mask[row, n:].any()
