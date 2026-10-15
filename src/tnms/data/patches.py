"""Cut a 2D signal into non-overlapping patches and put it back together.

Images (32 x 32 x 3) and log-mel spectrograms (64 mel x 96 frames x 1) both go
through the same function. Each patch is flattened to one vector, which the
model later maps to d_model with a single linear layer. That linear layer is
the whole "encoder", which is what encoder-free means here.
"""

import torch


def patchify(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """(H, W, C) -> (num_patches, ph * pw * C).

    Patches are listed in raster order: left to right along the first row of
    patches, then the next row, and so on.
    """
    H, W, C = x.shape
    assert H % ph == 0 and W % pw == 0, "patch size must divide the input"
    x = x.reshape(H // ph, ph, W // pw, pw, C)
    x = x.permute(0, 2, 1, 3, 4)  # (rows, cols, ph, pw, C)
    return x.reshape((H // ph) * (W // pw), ph * pw * C)


def unpatchify(p: torch.Tensor, H: int, W: int, ph: int, pw: int) -> torch.Tensor:
    """Inverse of patchify: (num_patches, ph * pw * C) -> (H, W, C)."""
    C = p.shape[1] // (ph * pw)
    x = p.reshape(H // ph, W // pw, ph, pw, C)
    x = x.permute(0, 2, 1, 3, 4)  # (rows, ph, cols, pw, C)
    return x.reshape(H, W, C)
