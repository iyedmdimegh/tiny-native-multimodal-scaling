"""Draw a stream of training examples from several domains at a target token share.

The mixture h is defined by token share: with h = (1/3, 1/3, 1/3), each domain
should make up a third of all tokens the model sees. Examples from different
domains have different lengths (a text example is 128 tokens, an image example
about 78), so drawing domains with probability h would give text more than its
share. Instead we draw domain i with probability proportional to
h_i / mean_tokens_i, which makes the expected token share equal to h_i.
"""

import numpy as np

from tnms.data.sequence import build_sequence


def estimate_mean_tokens(datasets: dict, tokenizer, n: int = 1000, seed: int = 0) -> dict:
    """Average sequence length (non-pad tokens) per domain over n random examples.

    Uses its own rng so it does not change the training data order.
    """
    rng = np.random.default_rng(seed)
    means = {}
    for name, ds in datasets.items():
        idx = rng.integers(len(ds), size=n)
        means[name] = float(np.mean([len(build_sequence(ds.get(int(i), rng), tokenizer)["ids"]) for i in idx]))
    return means


def domain_probs(shares: dict, mean_tokens: dict) -> dict:
    """Probability of drawing each domain so that token shares match `shares`."""
    w = {d: shares[d] / mean_tokens[d] for d in shares}
    total = sum(w.values())
    return {d: v / total for d, v in w.items()}


class MixtureSampler:
    """Infinite stream of training examples.

    Within a domain, examples are visited in a shuffled order. When a domain
    runs out, it is reshuffled (a new epoch for that domain). One numpy rng
    drives the domain choice, the shuffles and the augmentation, so the same
    seed always gives the same stream.
    """

    def __init__(self, datasets: dict, shares: dict, mean_tokens: dict, seed: int):
        self.datasets = datasets
        probs = domain_probs(shares, mean_tokens)
        self.domains = list(probs)
        self.probs = np.array([probs[d] for d in self.domains])
        self.rng = np.random.default_rng(seed)
        self.order = {d: np.empty(0, dtype=np.int64) for d in self.domains}
        self.pos = {d: 0 for d in self.domains}
        self.seen = {d: 0 for d in self.domains}  # examples drawn so far

    def _next_index(self, d: str) -> int:
        if self.pos[d] == len(self.order[d]):
            self.order[d] = self.rng.permutation(len(self.datasets[d]))
            self.pos[d] = 0
        i = int(self.order[d][self.pos[d]])
        self.pos[d] += 1
        self.seen[d] += 1
        return i

    def next_example(self) -> dict:
        d = self.domains[self.rng.choice(len(self.domains), p=self.probs)]
        return self.datasets[d].get(self._next_index(d), self.rng)

    def epochs(self) -> dict:
        """How many passes over each domain's training set so far."""
        return {d: self.seen[d] / len(self.datasets[d]) for d in self.domains}
