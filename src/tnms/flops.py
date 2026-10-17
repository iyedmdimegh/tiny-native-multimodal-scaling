"""Parameter and compute counting.

N, the model size used in every scaling fit, is the number of non-embedding
parameters: the transformer blocks plus the final norm. For the late-fusion
variant the encoder blocks are counted too. The token embedding
(which is also the output layer), the position embedding and the two patch
projections are left out, following the usual convention in scaling-law work.
"""


def count_non_embedding_params(model) -> int:
    parts = (model.blocks, model.encoders, model.norm_f)
    return sum(p.numel() for part in parts for p in part.parameters())


def train_flops(n_params: int, n_tokens: int) -> float:
    """Standard estimate C = 6 * N * D (forward plus backward).

    It ignores the attention term, which is small here because sequences are
    at most 128 tokens long.
    """
    return 6.0 * n_params * n_tokens
