"""TinyStories as the text-only domain.

Stories are tokenized once, joined into one long token stream with <eos>
between stories, and stored as uint16 (the vocabulary has 2048 entries). A
training example is one window of 126 consecutive tokens from that stream.

Splits:
- train: stories from the start of the HF train split, until TRAIN_TOKENS.
- val: the last VAL_STORIES stories of the HF train split (never trained on).
- test: the HF validation split.
"""

import numpy as np
from datasets import load_dataset

from tnms.config import DATA_DIR
from tnms.data.tokenizer import EOS

WINDOW = 126  # 128 minus <bos> and <eos>
# Largest text need: 0.8 of a 40M-token mixture run, or 1/3 of an 80M grid
# run. 40M tokens covers both without repeating any text.
TRAIN_TOKENS = 40_000_000
VAL_STORIES = 20_000
TOKENIZER_TEXT_BYTES = 20_000_000

CACHE_DIR = "tinystories_cache"


def load_raw(root=DATA_DIR):
    return load_dataset("roneneldan/TinyStories", cache_dir=str(root / "hf"))


def tokenizer_corpus(raw, chunk=10_000):
    """Roughly the first 20 MB of train stories, used to train the BPE tokenizer."""
    size = 0
    for start in range(0, len(raw["train"]), chunk):
        for text in raw["train"][start : start + chunk]["text"]:
            if size >= TOKENIZER_TEXT_BYTES:
                return
            size += len(text.encode())
            yield text


def _tokenize(ds, start, stop, tokenizer, max_tokens=None, chunk=10_000) -> np.ndarray:
    """Encode stories ds[start:stop] and join them with <eos>.

    Reads the dataset in chunks so we never hold all stories in memory.
    Stops early once max_tokens is reached.
    """
    parts, total = [], 0
    for a in range(start, stop, chunk):
        texts = ds[a : min(a + chunk, stop)]["text"]
        for enc in tokenizer.encode_batch(texts):
            ids = enc.ids + [EOS]
            parts.append(np.array(ids, dtype=np.uint16))
            total += len(ids)
            if max_tokens is not None and total >= max_tokens:
                return np.concatenate(parts)
    return np.concatenate(parts)


def prepare(raw, tokenizer, root=DATA_DIR):
    """Tokenize the three splits and save them as .npy token streams."""
    out_dir = root / CACHE_DIR
    if (out_dir / "test_tokens.npy").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = len(raw["train"])
    val_start = n_train - VAL_STORIES

    # The train stream reads from the front and never reaches the val tail.
    train = _tokenize(raw["train"], 0, val_start, tokenizer, max_tokens=TRAIN_TOKENS)
    val = _tokenize(raw["train"], val_start, n_train, tokenizer)
    test = _tokenize(raw["validation"], 0, len(raw["validation"]), tokenizer)
    for name, arr in [("train", train), ("val", val), ("test", test)]:
        np.save(out_dir / f"{name}_tokens.npy", arr)


class TinyStories:
    """One split of the text domain. Example i is window i of the token stream."""

    domain = "text"

    def __init__(self, split: str, root=DATA_DIR):
        assert split in ("train", "val", "test")
        self.tokens = np.load(root / CACHE_DIR / f"{split}_tokens.npy", mmap_mode="r")

    def __len__(self):
        return len(self.tokens) // WINDOW

    def get(self, i: int, rng=None) -> dict:
        ids = self.tokens[i * WINDOW : (i + 1) * WINDOW].astype(np.int64).tolist()
        return {"domain": self.domain, "ids": ids}
