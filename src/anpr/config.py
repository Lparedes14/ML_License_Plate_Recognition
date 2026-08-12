"""Configuration loading and reproducibility control.

Every tunable number lives in `config/default.yaml`, not scattered through
the code. This module reads that file and provides the one function that
makes runs reproducible: `set_seeds()`.

Typical use at the top of any script or notebook:

    from anpr.config import load_config, set_seeds
    cfg = load_config()
    set_seeds(cfg["seed"])
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------
# Resolved from THIS file's location, never from the current working
# directory. That is what lets a notebook in notebooks/, a script in
# scripts/, and pytest in tests/ all find config/ and artifacts/ without
# any of them caring where they were launched from.
#
#   config.py -> anpr/ -> src/ -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"              # EMNIST CSVs live here (gitignored)
GENERATED_DATA_DIR = DATA_DIR / "generated"  # synthetic plates (gitignored)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
PROVENANCE_DIR = ARTIFACTS_DIR / "provenance"


# --------------------------------------------------------------------------
# The label space
# --------------------------------------------------------------------------
# 36 classes: the digits then the uppercase letters, in that order. The order
# matters - it is the contract between the model's output index and the
# character it means. If this string is ever reordered, every saved model
# silently starts producing wrong characters, so it is saved alongside the
# weights (see models/train.py -> save_class_map).
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

IDX2CHAR: dict[int, str] = {i: c for i, c in enumerate(CHARS)}
CHAR2IDX: dict[str, int] = {c: i for i, c in enumerate(CHARS)}

# EMNIST ByClass raw label order, BEFORE any case merge: 62 classes, being
# digits (0-9), uppercase (10-35), then lowercase (36-61). Used by the
# orientation guard and the cross-route check, which both work on raw labels.
RAW_LABEL = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
RAW_CLASSES = 62

# Glyph pairs that are visually ambiguous and therefore expensive in
# production: a 0/O confusion bills the wrong customer just as surely as a
# wild guess does, but it is far more likely. These drive the confusion
# analysis (evaluate/confusion.py, ML-50) and are worth naming in the demo -
# many jurisdictions omit I, O and Q from issued plates for exactly this
# reason, which is a cheap recommendation we can make to the COO.
CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("0", "O"), ("1", "I"), ("5", "S"), ("8", "B"), ("2", "Z"),
    ("6", "G"), ("7", "T"), ("4", "A"), ("D", "O"), ("U", "V"),
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and sanity-check the parts other modules rely on.

    Args:
        path: Config file to read. Defaults to `config/default.yaml`. Pass an
            override file to run an experiment without editing the default -
            that keeps experiments as config diffs rather than code diffs.

    Returns:
        A plain nested dict. Deliberately not a custom class: it has to
        serialise straight into the provenance JSON so that every reported
        metric can be traced back to the exact settings that produced it.

    Raises:
        FileNotFoundError: if the config file is missing.
        ValueError: if a value other modules depend on is inconsistent.
    """
    path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            f"  expected it at: {CONFIG_DIR / 'default.yaml'}\n"
            "  if you moved the repo, re-run `pip install -e .` from the root."
        )

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Fail here, loudly, rather than three hours into a training run.
    if cfg["data"]["n_classes"] != len(CHARS):
        raise ValueError(
            f"config says n_classes={cfg['data']['n_classes']} but CHARS has "
            f"{len(CHARS)} entries. These must agree - the model's output "
            "layer width is derived from one and its meaning from the other."
        )
    if not 0.0 < cfg["data"]["val_fraction"] < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {cfg['data']['val_fraction']}")
    if cfg["data"]["case_strategy"] not in ("merge", "drop"):
        raise ValueError(
            f"case_strategy must be 'merge' or 'drop', got {cfg['data']['case_strategy']!r}"
        )
    if cfg["data"]["imbalance_strategy"] not in ("none", "weighted", "resampled"):
        raise ValueError(
            f"imbalance_strategy must be one of none/weighted/resampled, "
            f"got {cfg['data']['imbalance_strategy']!r}"
        )

    cfg["_config_path"] = str(path)  # recorded in provenance so results reproduce
    return cfg


def set_seeds(seed: int) -> None:
    """Seed every RNG that can affect a result.

    Covers four independent sources of randomness, all of which have to be
    pinned for two runs to agree: Python's hash randomisation, the `random`
    module, NumPy, and TensorFlow (weight init, dropout, shuffling).

    This does NOT make GPU training bit-for-bit deterministic - cuDNN picks
    non-deterministic kernels for speed. It makes splits and initialisation
    reproducible, which is what the brief asks for. If you need full
    determinism for a specific claim, also set
    `tf.config.experimental.enable_op_determinism()` and expect it to be slow.

    Args:
        seed: The seed, from `cfg["seed"]`. Record it with any result.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Imported lazily: this module is imported by tests and by the plate
    # generator, neither of which should pay TensorFlow's multi-second import
    # cost just to read a config value.
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Create the artifact directories if they do not exist.

    Called by the scripts before they write anything. Cheap, idempotent, and
    it means a clean clone does not fail on a missing output folder.
    """
    for d in (RAW_DATA_DIR, GENERATED_DATA_DIR, MODELS_DIR,
              METRICS_DIR, FIGURES_DIR, PROVENANCE_DIR):
        d.mkdir(parents=True, exist_ok=True)
