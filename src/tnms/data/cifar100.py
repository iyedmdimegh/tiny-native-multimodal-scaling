"""CIFAR-100 as an image-to-caption task.

Every example is a 32 x 32 image cut into 64 patches of 4 x 4 x 3 = 48 values,
plus a short caption naming its fine class and its coarse superclass.
"""

import io

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from tnms.config import DATA_DIR
from tnms.data.patches import patchify

# The standard 20 CIFAR-100 superclasses. torchvision only exposes the fine
# labels, so the mapping is written out here and checked by a test.
COARSE_TO_FINE = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}
FINE_TO_COARSE = {fine: coarse for coarse, fines in COARSE_TO_FINE.items() for fine in fines}

TEMPLATES = [
    "a photo of a {fine}, a kind of {coarse}",
    "this image shows a {fine}, which is a {coarse}",
    "{fine}, a type of {coarse}",
    "an example of {coarse}: {fine}",
    "it is a {fine} ({coarse})",
]

PATCH = 4
NUM_PATCHES = 64
PATCH_DIM = PATCH * PATCH * 3
VAL_SIZE = 5000
SPLIT_SEED = 0  # fixed, so every run uses the same 45k / 5k split


def caption(fine: str, template: int) -> str:
    coarse = FINE_TO_COARSE[fine]
    return TEMPLATES[template].format(fine=fine.replace("_", " "), coarse=coarse.replace("_", " "))


# Fine and coarse label ids follow alphabetical order of the names, as in
# the original CIFAR-100 release. A test checks this against the data.
FINE_NAMES = sorted(FINE_TO_COARSE)
COARSE_NAMES = sorted(COARSE_TO_FINE)

# The original host is very slow from some networks, so we read the copy that
# the University of Toronto publishes on the Hugging Face Hub, pinned to one
# revision. Same 50k train and 10k test images.
HF_REPO = "uoft-cs/cifar100"
HF_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"


def _load(root, train: bool):
    """Return uint8 images (N, 32, 32, 3), fine labels and coarse labels.

    The first call downloads the parquet file and decodes the PNGs into a
    .npz cache. Later calls only read the cache.
    """
    split = "train" if train else "test"
    cache = root / "cifar100" / f"{split}.npz"
    if not cache.exists():
        path = hf_hub_download(
            HF_REPO, f"cifar100/{split}-00000-of-00001.parquet", repo_type="dataset",
            revision=HF_REVISION, cache_dir=str(root / "hf"),
        )
        df = pd.read_parquet(path)
        images = np.stack([np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) for img in df["img"]])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, images=images, fine=df["fine_label"].to_numpy(), coarse=df["coarse_label"].to_numpy())
    data = np.load(cache)
    return torch.from_numpy(data["images"]), torch.from_numpy(data["fine"]), torch.from_numpy(data["coarse"])


def _train_val_indices(n: int):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED))
    return perm[VAL_SIZE:], perm[:VAL_SIZE]


def _channel_stats(images: torch.Tensor):
    """Per-channel mean and std of uint8 images (N, 32, 32, 3), scaled to [0, 1].

    Summed in chunks in float64 so we never hold a float copy of all images.
    """
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    for chunk in images.split(5000):
        x = chunk.reshape(-1, 3).double() / 255
        total += x.sum(0)
        total_sq += (x * x).sum(0)
    n = images.shape[0] * 32 * 32
    mean = total / n
    std = (total_sq / n - mean**2).sqrt()
    return mean.float(), std.float()


class Cifar100:
    """One split ("train", "val" or "test") of CIFAR-100 captioning.

    `get(i, rng)` with an rng applies train augmentation and picks a random
    caption template. With rng=None it is deterministic: no augmentation and
    template i % 5. Held-out evaluation always uses rng=None.
    """

    domain = "image"
    templates = TEMPLATES

    def __init__(self, split: str, root=DATA_DIR):
        assert split in ("train", "val", "test")
        train_images, train_labels, _ = _load(root, train=True)
        train_idx, val_idx = _train_val_indices(len(train_images))
        # Normalisation always uses the 45k training images only.
        self.mean, self.std = _channel_stats(train_images[train_idx])

        if split == "test":
            self.images, self.labels, _ = _load(root, train=False)
        else:
            idx = train_idx if split == "train" else val_idx
            self.images, self.labels = train_images[idx], train_labels[idx]

    def __len__(self):
        return len(self.images)

    def get(self, i: int, rng: np.random.Generator | None = None) -> dict:
        img = self.images[i].float() / 255  # (32, 32, 3)
        if rng is not None:
            img = _random_crop_flip(img, rng)
        img = (img - self.mean) / self.std
        fine = FINE_NAMES[int(self.labels[i])]
        template = int(rng.integers(len(TEMPLATES))) if rng is not None else i % len(TEMPLATES)
        return {
            "domain": self.domain,
            "patches": patchify(img, PATCH, PATCH),
            "caption": caption(fine, template),
        }


def _random_crop_flip(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Zero-pad by 2 pixels, take a random 32 x 32 crop, flip left-right half the time."""
    padded = torch.zeros(36, 36, 3)
    padded[2:34, 2:34] = img
    y, x = rng.integers(0, 5, size=2)
    img = padded[y : y + 32, x : x + 32]
    if rng.random() < 0.5:
        img = img.flip(1)
    return img
