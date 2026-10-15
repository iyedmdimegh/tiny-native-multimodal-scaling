"""Small synthetic datasets with the same interface as the real ones.

They let the tests run in seconds on a machine without the downloaded data.
Captions come from the real templates, so sequence lengths are realistic.
"""

import numpy as np
import pytest
import torch

from tnms.data import cifar100, speechcommands
from tnms.data.tokenizer import load_tokenizer

FINE = list(cifar100.FINE_TO_COARSE)


class FakeImage:
    domain = "image"

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def get(self, i, rng=None):
        t = int(rng.integers(5)) if rng is not None else i % 5
        patches = torch.randn(64, 48, generator=torch.Generator().manual_seed(i))
        return {"domain": "image", "patches": patches, "caption": cifar100.caption(FINE[i % 100], t)}


class FakeAudio:
    domain = "audio"

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def get(self, i, rng=None):
        t = int(rng.integers(4)) if rng is not None else i % 4
        word = speechcommands.WORDS[i % 35]
        patches = torch.randn(96, 64, generator=torch.Generator().manual_seed(i))
        return {"domain": "audio", "patches": patches, "caption": speechcommands.TEMPLATES[t].format(word=word)}


class FakeText:
    domain = "text"

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def get(self, i, rng=None):
        ids = np.random.default_rng(i).integers(7, 2048, size=126).tolist()
        return {"domain": "text", "ids": ids}


def make_fake(n):
    return {"image": FakeImage(n), "audio": FakeAudio(n), "text": FakeText(n)}


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()


@pytest.fixture
def fake_datasets():
    return {"train": make_fake(200), "val": make_fake(40), "test": make_fake(40)}
