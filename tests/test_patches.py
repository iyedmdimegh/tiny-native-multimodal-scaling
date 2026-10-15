import torch

from tnms.data.patches import patchify, unpatchify


def test_image_round_trip():
    x = torch.randn(32, 32, 3)
    p = patchify(x, 4, 4)
    assert p.shape == (64, 48)
    assert torch.equal(unpatchify(p, 32, 32, 4, 4), x)


def test_audio_round_trip():
    x = torch.randn(64, 96, 1)  # mel x frames x 1
    p = patchify(x, 8, 8)
    assert p.shape == (96, 64)
    assert torch.equal(unpatchify(p, 64, 96, 8, 8), x)


def test_raster_order():
    """Patch 1 is the block to the right of patch 0, not the one below it."""
    x = torch.randn(32, 32, 3)
    p = patchify(x, 4, 4)
    assert torch.equal(p[0], x[0:4, 0:4].reshape(-1))
    assert torch.equal(p[1], x[0:4, 4:8].reshape(-1))
    assert torch.equal(p[8], x[4:8, 0:4].reshape(-1))
