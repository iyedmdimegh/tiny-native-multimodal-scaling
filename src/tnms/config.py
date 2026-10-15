"""Paths and config loading.

Configs are plain YAML files in configs/ and are loaded as plain dicts. There
are only a handful of keys, and a dict is easy to print, save into
summary.json and compare between runs.
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
TOKENIZER_PATH = REPO_ROOT / "tokenizer.json"

# Downloads and run outputs are several GB. These two variables let them live
# outside the repo, for example outside a cloud-synced folder.
DATA_DIR = Path(os.environ.get("TNMS_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = Path(os.environ.get("TNMS_RUNS_DIR", REPO_ROOT / "runs"))


def load_yaml(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def model_shape(size: str) -> dict:
    """Return d_model, n_layers and n_heads for a size name like "S3"."""
    return load_yaml("sizes")[size]
