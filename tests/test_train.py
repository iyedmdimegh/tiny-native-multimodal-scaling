"""End-to-end runs on synthetic data: seeding and output files."""

import csv
import json

from tnms.config import model_shape
from tnms.train import run


def _cfg(size="S1", tokens=2000, seed=0):
    return {
        "size": size, **model_shape(size), "tokens": tokens, "lr": 1e-3, "batch_size": 4,
        "shares": {"image": 1 / 3, "audio": 1 / 3, "text": 1 / 3}, "seed": seed,
        "eval_examples": 8, "log_every": 1, "device": "cpu",
    }


def _losses(run_dir):
    with open(run_dir / "metrics.csv") as f:
        return [row["train_loss"] for row in csv.DictReader(f)]


def test_same_seed_gives_identical_losses(fake_datasets, tmp_path):
    run(_cfg(), tmp_path / "a", fake_datasets)
    run(_cfg(), tmp_path / "b", fake_datasets)
    a, b = _losses(tmp_path / "a"), _losses(tmp_path / "b")
    assert len(a) >= 5 and a == b


def test_smoke_runs_write_expected_files(fake_datasets, tmp_path):
    for size in ("S1", "S2"):
        run_dir = tmp_path / size
        run(_cfg(size), run_dir, fake_datasets)
        for name in ("metrics.csv", "summary.json", "model.pt"):
            assert (run_dir / name).exists()
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["D"] >= 2000
        assert set(summary["test"]) == {"image", "audio", "text", "mean"}
