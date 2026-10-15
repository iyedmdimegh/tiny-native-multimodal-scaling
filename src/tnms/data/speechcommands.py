"""Speech Commands v2 as an audio-to-caption task.

Each clip is turned into a 64 mel x 96 frame log-mel spectrogram, cut into
8 x 8 patches (96 patches of 64 values), and paired with a caption that names
the spoken word.

Computing spectrograms is the slow part, so `prepare()` computes them once and
caches them on disk as float16. Training then only reads slices of the cache.
"""

import json

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from tqdm import tqdm

from tnms.config import DATA_DIR
from tnms.data.patches import patchify

WORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]
TEMPLATES = [
    "the spoken word is {word}",
    "someone says {word}",
    "the word {word}",
    "audio of the word {word}",
]

SAMPLE_RATE = 16000
N_FFT, HOP, N_MELS = 400, 160, 64
FRAMES = 96
PATCH = 8
NUM_PATCHES = (N_MELS // PATCH) * (FRAMES // PATCH)  # 96
PATCH_DIM = PATCH * PATCH  # 64

# The random time shift of up to 100 ms is applied on the cached spectrogram.
# We cache the spectrogram of the clip with 100 ms of silence on each side
# (121 frames), then a shift is just a choice of which 96 frames to read.
# One frame is 10 ms, so this equals a waveform shift rounded to 10 ms.
MAX_SHIFT = 10  # frames
PAD_SAMPLES = MAX_SHIFT * HOP

CACHE_DIR = "speechcommands_cache"


def _base_dir(root):
    return root / "SpeechCommands" / "speech_commands_v0.02"


def _split_files(root) -> dict:
    """Official split: files named in validation_list.txt and testing_list.txt, the rest is train."""
    base = _base_dir(root)
    val = set((base / "validation_list.txt").read_text().split())
    test = set((base / "testing_list.txt").read_text().split())
    files = sorted(
        p.relative_to(base).as_posix()
        for p in base.glob("*/*.wav")
        if p.parent.name != "_background_noise_"
    )
    return {
        "train": [f for f in files if f not in val and f not in test],
        "val": [f for f in files if f in val],
        "test": [f for f in files if f in test],
    }


def _read_padded(path) -> torch.Tensor:
    """Read a clip, pad or crop it to exactly 1 s, then add the 100 ms margins."""
    sr, wave = wavfile.read(path)
    assert sr == SAMPLE_RATE and wave.dtype == np.int16
    wave = wave[:SAMPLE_RATE].astype(np.float32) / 32768
    out = np.zeros(SAMPLE_RATE + 2 * PAD_SAMPLES, dtype=np.float32)
    out[PAD_SAMPLES : PAD_SAMPLES + len(wave)] = wave
    return torch.from_numpy(out)


def prepare(root=DATA_DIR, batch_size=512):
    """Download Speech Commands v2 and build the spectrogram cache for all splits."""
    out_dir = root / CACHE_DIR
    if (out_dir / "stats.json").exists():
        return
    torchaudio.datasets.SPEECHCOMMANDS(root, download=True)  # only used to download
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    ).to(device)
    base = _base_dir(root)

    for split, files in _split_files(root).items():
        labels = np.array([WORDS.index(f.split("/")[0]) for f in files], dtype=np.int64)
        n_frames = (SAMPLE_RATE + 2 * PAD_SAMPLES) // HOP + 1  # 121
        spec = np.lib.format.open_memmap(
            out_dir / f"{split}_spec.npy", mode="w+", dtype=np.float16,
            shape=(len(files), N_MELS, n_frames),
        )
        for start in tqdm(range(0, len(files), batch_size), desc=f"spectrograms {split}"):
            waves = torch.stack([_read_padded(base / f) for f in files[start : start + batch_size]])
            logmel = torch.log(mel(waves.to(device)) + 1e-6)
            spec[start : start + len(waves)] = logmel.cpu().numpy().astype(np.float16)
        spec.flush()
        np.save(out_dir / f"{split}_labels.npy", labels)

    # Normalisation statistics from the unshifted train crops only.
    spec = np.load(out_dir / "train_spec.npy", mmap_mode="r")
    total, total_sq, count = 0.0, 0.0, 0
    for start in range(0, len(spec), 4096):
        x = np.asarray(spec[start : start + 4096, :, MAX_SHIFT : MAX_SHIFT + FRAMES], dtype=np.float64)
        total += x.sum()
        total_sq += (x * x).sum()
        count += x.size
    mean = total / count
    std = (total_sq / count - mean**2) ** 0.5
    (out_dir / "stats.json").write_text(json.dumps({"mean": mean, "std": std}))


class SpeechCommands:
    """One split ("train", "val" or "test") of Speech Commands captioning.

    Same rng convention as Cifar100: with an rng we apply a random time shift
    and a random template, with rng=None everything is deterministic.
    """

    domain = "audio"
    templates = TEMPLATES

    def __init__(self, split: str, root=DATA_DIR):
        assert split in ("train", "val", "test")
        d = root / CACHE_DIR
        self.spec = np.load(d / f"{split}_spec.npy", mmap_mode="r")
        self.labels = np.load(d / f"{split}_labels.npy")
        stats = json.loads((d / "stats.json").read_text())
        self.mean, self.std = stats["mean"], stats["std"]

    def __len__(self):
        return len(self.labels)

    def get(self, i: int, rng: np.random.Generator | None = None) -> dict:
        # shift > 0 moves the sound later, so the 96-frame window starts earlier.
        shift = int(rng.integers(-MAX_SHIFT, MAX_SHIFT + 1)) if rng is not None else 0
        start = MAX_SHIFT - shift
        x = np.asarray(self.spec[i, :, start : start + FRAMES], dtype=np.float32)
        x = (torch.from_numpy(x) - self.mean) / self.std  # (64 mel, 96 frames)
        word = WORDS[int(self.labels[i])]
        template = int(rng.integers(len(TEMPLATES))) if rng is not None else i % len(TEMPLATES)
        return {
            "domain": self.domain,
            "patches": patchify(x.unsqueeze(-1), PATCH, PATCH),
            "caption": TEMPLATES[template].format(word=word),
        }
