import numpy as np
import pytest

from tnms.data.mixture_sampler import MixtureSampler, estimate_mean_tokens
from tnms.data.sequence import build_sequence


@pytest.mark.parametrize("shares", [
    {"image": 1 / 3, "audio": 1 / 3, "text": 1 / 3},
    {"image": 0.6, "audio": 0.3, "text": 0.1},
    {"image": 0.05, "audio": 0.475, "text": 0.475},
])
def test_realised_token_shares_match_target(fake_datasets, tokenizer, shares):
    train = fake_datasets["train"]
    sampler = MixtureSampler(train, shares, estimate_mean_tokens(train, tokenizer), seed=0)
    tokens = {d: 0 for d in shares}
    for _ in range(20_000):
        ex = sampler.next_example()
        tokens[ex["domain"]] += len(build_sequence(ex, tokenizer)["ids"])
    total = sum(tokens.values())
    for d, h in shares.items():
        assert abs(tokens[d] / total - h) / h < 0.05, (d, tokens[d] / total, h)


def test_same_seed_same_stream(fake_datasets, tokenizer):
    train = fake_datasets["train"]
    means = estimate_mean_tokens(train, tokenizer)
    shares = {"image": 0.4, "audio": 0.4, "text": 0.2}
    a = MixtureSampler(train, shares, means, seed=7)
    b = MixtureSampler(train, shares, means, seed=7)
    for _ in range(200):
        ea, eb = a.next_example(), b.next_example()
        assert ea["domain"] == eb["domain"]
        assert ea.get("caption") == eb.get("caption")
        assert np.array_equal(np.asarray(ea.get("ids", [])), np.asarray(eb.get("ids", [])))
