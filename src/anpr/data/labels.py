"""Label-space decisions: 62 raw classes -> 36 plate classes, and imbalance.

This module holds the two decisions §3 of the brief explicitly asks us to
make and justify rather than assume. Both are configured in
`config/default.yaml` so the report can cite them, and both are reversible so
we can run each arm and report the comparison.

DECISION 1 - CASE (cfg.data.case_strategy)
    EMNIST ByClass has 62 classes: 0-9, A-Z, a-z. Plates are uppercase-only,
    so 62 must become 36. Two defensible routes:

      "merge"  fold lowercase into uppercase ('a' -> class 'A'). Keeps all
               the data. But a handwritten 'a' genuinely does not look like a
               printed plate 'A' - we are teaching the model that two
               different shapes mean the same thing, which costs accuracy on
               the shape we actually care about.

      "drop"   discard the 26 lowercase classes. Cleaner class definitions,
               roughly 40% less data.

    We do not know which wins without measuring. Run both, report both. That
    comparison is worth more marks than either number alone.

DECISION 2 - IMBALANCE (cfg.data.imbalance_strategy)
    ByClass preserves natural English letter frequency. Our own measurement:
    'Q' is ~0.80% of the data while 'N' is ~2.82% - a 3.5x spread, and far
    worse against the digits. Plate characters are close to UNIFORM.

    So the training distribution does not match the deployment distribution.
    Left alone, the model learns "when unsure, guess a common letter", which
    is optimal for EMNIST and wrong for plates.

Ported/extended from ML_Project_Group_8.ipynb cell 17 (ML-39).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anpr.config import CHARS, IDX2CHAR, RAW_CLASSES


def merge_case(y: np.ndarray, n_classes: int = 36) -> np.ndarray:
    """Fold lowercase labels onto their uppercase class. 62 -> 36.

    Raw ByClass ordering is digits 0-9, uppercase 10-35, lowercase 36-61, so
    a lowercase label maps to its uppercase partner by subtracting 26.

    Idempotent by design: if the labels are already merged (max < 36) it
    returns them untouched. That matters because notebooks get re-run out of
    order, and a second merge would silently corrupt the digits.

    Args:
        y: Labels, raw (0-61) or already merged (0-35).
        n_classes: Size of the merged space. 36 here.

    Returns:
        int32 labels in [0, 36).
    """
    y = np.asarray(y)
    if y.max() < n_classes:
        return y.astype(np.int32)          # already merged - do nothing
    return np.where(y >= 36, y - 26, y).astype(np.int32)


def drop_lowercase(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discard the 26 lowercase classes. 62 -> 36 by deletion.

    The alternative to merge_case. Keeps only digits and uppercase, so every
    remaining class contains exactly one glyph shape.

    Args:
        X: Images.
        y: RAW labels 0-61. Passing merged labels here is a bug - after a
            merge there are no lowercase labels left to drop.

    Returns:
        (X, y) filtered to the 36 uppercase/digit classes, labels unchanged
        in [0, 36).
    """
    if y.max() < 36:
        raise ValueError(
            "drop_lowercase() received labels already in [0,36) - they have "
            "been merged. Choose one strategy, not both."
        )
    keep = y < 36
    return X[keep], y[keep].astype(np.int32)


def apply_case_strategy(
    X: np.ndarray, y: np.ndarray, strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to merge_case or drop_lowercase per the config.

    Args:
        X, y: Images and RAW labels 0-61.
        strategy: "merge" or "drop", from cfg["data"]["case_strategy"].

    Returns:
        (X, y) with labels in [0, 36).
    """
    if strategy == "merge":
        return X, merge_case(y)
    if strategy == "drop":
        return drop_lowercase(X, y)
    raise ValueError(f"unknown case_strategy {strategy!r}, expected 'merge' or 'drop'")


def compute_class_weights(y: np.ndarray, n_classes: int = 36) -> dict[int, float]:
    """Weights that make a rare class cost as much as a common one.

    Keras multiplies each sample's loss by its class weight. Setting the
    weight inversely proportional to class frequency means the gradient from
    the ~500 'Q' samples carries the same total pull as the ~6000 'N'
    samples, which is what we want when the deployment distribution is
    uniform but the training distribution is not.

    Normalised so the mean weight is 1.0. Without that, the effective
    learning rate changes whenever the class distribution does, and two runs
    stop being comparable.

    Args:
        y: Merged labels in [0, 36).
        n_classes: Number of classes.

    Returns:
        {class_index: weight}, ready to pass to `model.fit(class_weight=...)`.

    Raises:
        ValueError: if a class has zero samples - weighting cannot rescue a
            class the model has never seen, and the silent alternative is a
            division by zero producing inf.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)

    if (counts == 0).any():
        missing = [IDX2CHAR[i] for i in np.where(counts == 0)[0]]
        raise ValueError(
            f"classes with zero samples: {missing}. Class weighting cannot "
            "fix an absent class - either the subsample is too small or the "
            "case strategy removed them."
        )

    weights = counts.sum() / (n_classes * counts)   # inverse frequency
    weights = weights / weights.mean()              # normalise: mean weight 1.0
    return {int(i): float(w) for i, w in enumerate(weights)}


def resample_to_median(
    y: np.ndarray, seed: int, n_classes: int = 36
) -> np.ndarray:
    """Indices that cap every class at the median class count.

    The blunter alternative to weighting: throw away surplus samples from
    over-represented classes so the training set itself is near-uniform.
    Loses data, but produces a genuinely balanced set rather than a
    reweighted imbalanced one - worth measuring against "weighted".

    Args:
        y: Merged labels in [0, 36).
        seed: RNG seed, so the subsample is reproducible.
        n_classes: Number of classes.

    Returns:
        Sorted index array to apply to both X and y.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=n_classes)
    cap = int(np.median(counts[counts > 0]))

    keep: list[np.ndarray] = []
    for k in range(n_classes):
        idx = np.where(y == k)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)

    return np.sort(np.concatenate(keep))


def class_distribution_report(y: np.ndarray, n_classes: int = 36) -> dict:
    """Summarise the imbalance, in the terms the report needs.

    Produces the numbers that turn "we noticed the imbalance" into "the
    imbalance is 3.5x and here is the evidence" - which is what §3 asks for.

    Args:
        y: Merged labels in [0, 36).
        n_classes: Number of classes.

    Returns:
        Dict with per-class shares, the rarest and most common characters,
        and the imbalance ratio between them.
    """
    counts = np.bincount(y, minlength=n_classes)
    shares = counts / counts.sum()

    rarest = int(np.argmin(np.where(counts > 0, counts, counts.max() + 1)))
    commonest = int(np.argmax(counts))

    return {
        "n_samples": int(counts.sum()),
        "per_class_count": {IDX2CHAR[i]: int(c) for i, c in enumerate(counts)},
        "per_class_share": {IDX2CHAR[i]: round(float(s), 5) for i, s in enumerate(shares)},
        "rarest_char": IDX2CHAR[rarest],
        "rarest_share": round(float(shares[rarest]), 5),
        "commonest_char": IDX2CHAR[commonest],
        "commonest_share": round(float(shares[commonest]), 5),
        "imbalance_ratio": round(float(counts[commonest] / max(counts[rarest], 1)), 2),
        # A uniform plate alphabet would give every class this share. The gap
        # between this and the numbers above IS the domain shift.
        "uniform_share_would_be": round(1.0 / n_classes, 5),
    }


def save_class_map(path: str | Path, case_strategy: str) -> None:
    """Write the index -> character map next to the model weights.

    Non-negotiable (ML-46). The model outputs an integer; only this file says
    what that integer MEANS. Load weights without the matching map and the
    system confidently reads every plate wrong, with no error anywhere.

    Args:
        path: Destination JSON path.
        case_strategy: Recorded so a future reader knows how 62 became 36.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "chars": CHARS,
            "n_classes": len(CHARS),
            "idx2char": {str(i): c for i, c in IDX2CHAR.items()},
            "case_strategy": case_strategy,
            "raw_classes": RAW_CLASSES,
        }, fh, indent=2)


def load_class_map(path: str | Path) -> dict[int, str]:
    """Read a class map and verify it matches the code's current CHARS.

    Guards against the scenario where someone reorders CHARS after a model
    was trained. The weights would still load; the predictions would be
    garbage. This turns that into a loud failure at load time.

    Args:
        path: The JSON written by `save_class_map`.

    Returns:
        {index: character}.

    Raises:
        ValueError: if the saved charset disagrees with the current one.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if payload["chars"] != CHARS:
        raise ValueError(
            "CLASS MAP MISMATCH - refusing to load.\n"
            f"  saved with model : {payload['chars']}\n"
            f"  current in code  : {CHARS}\n"
            "The charset was changed after this model was trained. Its outputs "
            "would decode to the wrong characters. Retrain, or restore the "
            "original ordering in anpr/config.py."
        )

    return {int(i): c for i, c in payload["idx2char"].items()}
