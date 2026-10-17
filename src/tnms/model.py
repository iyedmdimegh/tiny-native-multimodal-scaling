"""A small decoder-only transformer that reads text tokens and raw patches.

There is no vision or audio encoder. Each image or spectrogram patch goes
through one linear layer (image_proj or audio_proj) and then into the same
transformer as the text tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tnms.data.sequence import MAX_LEN, PATCH_START
from tnms.data.tokenizer import VOCAB_SIZE

IMAGE_PATCH_DIM = 48  # 4 x 4 x 3
AUDIO_PATCH_DIM = 64  # 8 mel x 8 frames


def attention_mask(lengths, prefix_start, prefix_end, T: int) -> torch.Tensor:
    """Boolean mask (B, T, T), True where query position q may attend to key k.

    - Causal by default: k <= q.
    - Inside the [prefix_start, prefix_end) range (the open token, patches and
      close token) positions also see each other in both directions.
    - Keys at or after the sequence length (padding) are never visible.
    """
    pos = torch.arange(T, device=lengths.device)
    q = pos.view(1, T, 1)
    k = pos.view(1, 1, T)
    start = prefix_start.view(-1, 1, 1)
    end = prefix_end.view(-1, 1, 1)
    causal = k <= q
    both_in_prefix = (q >= start) & (q < end) & (k >= start) & (k < end)
    not_pad = k < lengths.view(-1, 1, 1)
    return (causal | both_in_prefix) & not_pad


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden = 8 * round(8 / 3 * d_model / 8)  # about 8/3 * d, multiple of 8
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    """Early fusion by default. With encoder_layers > 0 it becomes the late-fusion
    variant of experiment C: image and audio patches first pass through their own
    small bidirectional transformer before entering the shared decoder."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, encoder_layers: int = 0):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        self.image_proj = nn.Linear(IMAGE_PATCH_DIM, d_model, bias=False)
        self.audio_proj = nn.Linear(AUDIO_PATCH_DIM, d_model, bias=False)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.encoders = nn.ModuleDict({
            name: nn.ModuleList(Block(d_model, n_heads) for _ in range(encoder_layers))
            for name in ("image", "audio")
        } if encoder_layers else {})
        self.norm_f = nn.RMSNorm(d_model)
        self._init_weights(n_layers)

    def _init_weights(self, n_layers: int):
        """GPT-2 style: every matrix ~ N(0, 0.02). The two projections that write
        into the residual stream get a smaller std, so the residual does not
        grow with depth."""
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
        for stack in (self.blocks, *self.encoders.values()):
            for block in stack:
                for p in (block.attn.proj.weight, block.mlp.down.weight):
                    nn.init.normal_(p, std=0.02 / math.sqrt(2 * len(stack)))

    def embed(self, batch) -> torch.Tensor:
        """Token embeddings, with patch slots replaced by projected patches."""
        x = self.tok_emb(batch["input_ids"])
        for name, proj in (("image", self.image_proj), ("audio", self.audio_proj)):
            rows = batch[f"{name}_rows"]
            if len(rows):
                patches = proj(batch[f"{name}_patches"])
                if self.encoders:
                    patches = self.encode(name, patches)
                x[rows, PATCH_START : PATCH_START + patches.shape[1]] = patches.to(x.dtype)
        T = x.shape[1]
        return x + self.pos_emb(torch.arange(T, device=x.device))

    def encode(self, name: str, patches: torch.Tensor) -> torch.Tensor:
        """Late fusion only: run the patches through the modality's own encoder.

        The encoder needs positions to tell patches apart, so it adds the same
        position embeddings the patch slots get in the decoder. Every patch
        sees every other patch (no mask).
        """
        P = patches.shape[1]
        x = patches + self.pos_emb(torch.arange(PATCH_START, PATCH_START + P, device=patches.device))
        full = torch.ones(1, P, P, dtype=torch.bool, device=patches.device)
        for block in self.encoders[name]:
            x = block(x, full)
        return x

    def forward(self, batch) -> torch.Tensor:
        """Return logits (B, T, vocab). Logit at position t predicts token t + 1."""
        x = self.embed(batch)
        mask = attention_mask(batch["lengths"], batch["prefix_start"], batch["prefix_end"], x.shape[1])
        for block in self.blocks:
            x = block(x, mask)
        # The output layer reuses the token embedding matrix (weight tying).
        return self.norm_f(x) @ self.tok_emb.weight.T
