import numpy as np
import pytest

from tnms.config import DATA_DIR
from tnms.data.cifar100 import COARSE_NAMES, COARSE_TO_FINE, FINE_NAMES, FINE_TO_COARSE


def test_superclass_map_is_complete_and_balanced():
    assert len(COARSE_TO_FINE) == 20
    assert all(len(fines) == 5 for fines in COARSE_TO_FINE.values())
    all_fine = [f for fines in COARSE_TO_FINE.values() for f in fines]
    assert len(set(all_fine)) == 100


@pytest.mark.skipif(not (DATA_DIR / "cifar100" / "test.npz").exists(), reason="CIFAR-100 not downloaded")
def test_map_agrees_with_coarse_labels_in_the_data():
    """Every image carries both labels. Our hardcoded map, applied to the fine
    label, must give the coarse label stored with the image."""
    data = np.load(DATA_DIR / "cifar100" / "test.npz")
    for fine, coarse in zip(data["fine"], data["coarse"]):
        assert FINE_TO_COARSE[FINE_NAMES[fine]] == COARSE_NAMES[coarse]
