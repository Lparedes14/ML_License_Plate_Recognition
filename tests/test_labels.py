"""Case handling, class weighting, and the class map contract.

    pytest tests/test_labels.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from anpr.config import CHARS
from anpr.data.labels import (
    class_distribution_report,
    compute_class_weights,
    drop_lowercase,
    load_class_map,
    merge_case,
    resample_to_median,
    save_class_map,
)


# --------------------------------------------------------------------------
# Case merging
# --------------------------------------------------------------------------
def test_merge_maps_lowercase_onto_uppercase():
    """'a' is raw class 36 and must land on 'A' at class 10."""
    raw = np.array([0, 9, 10, 35, 36, 61], np.int32)
    assert merge_case(raw).tolist() == [0, 9, 10, 35, 10, 35]


def test_merge_is_idempotent():
    """Notebooks get re-run out of order. A second merge must be a no-op.

    Without the guard clause this would corrupt the digits on a second call.
    """
    raw = np.array([0, 36, 61], np.int32)
    once = merge_case(raw)
    assert merge_case(once).tolist() == once.tolist()


def test_merge_produces_valid_class_range():
    raw = np.arange(62, dtype=np.int32)
    merged = merge_case(raw)
    assert merged.min() >= 0 and merged.max() < len(CHARS)


def test_drop_lowercase_keeps_only_the_36():
    X = np.zeros((62, 28, 28, 1), np.uint8)
    y = np.arange(62, dtype=np.int32)
    Xk, yk = drop_lowercase(X, y)
    assert len(yk) == 36
    assert yk.max() < 36


def test_drop_rejects_already_merged_labels():
    """Merging then dropping is a bug - catch it rather than silently no-op."""
    X = np.zeros((10, 28, 28, 1), np.uint8)
    with pytest.raises(ValueError, match="already"):
        drop_lowercase(X, np.arange(10, dtype=np.int32))


# --------------------------------------------------------------------------
# Imbalance handling
# --------------------------------------------------------------------------
def test_class_weights_favour_rare_classes():
    # class 0 appears 100x, every other class 10x
    y = np.concatenate([np.zeros(100, np.int32),
                        np.repeat(np.arange(1, 36, dtype=np.int32), 10)])
    w = compute_class_weights(y)
    assert w[0] < w[1], "the common class must get the smaller weight"


def test_class_weights_average_to_one():
    """Normalisation keeps the effective learning rate stable across runs."""
    y = np.repeat(np.arange(36, dtype=np.int32), 20)
    w = compute_class_weights(y)
    assert abs(np.mean(list(w.values())) - 1.0) < 1e-6


def test_class_weights_reject_missing_class():
    """Weighting cannot rescue a class the model never sees. Say so loudly."""
    y = np.repeat(np.arange(35, dtype=np.int32), 10)      # class 35 absent
    with pytest.raises(ValueError, match="zero samples"):
        compute_class_weights(y)


def test_resample_caps_at_median():
    y = np.concatenate([np.zeros(500, np.int32),
                        np.repeat(np.arange(1, 36, dtype=np.int32), 20)])
    keep = resample_to_median(y, seed=42)
    counts = np.bincount(y[keep], minlength=36)
    assert counts[0] <= 20, "over-represented class was not capped"


def test_distribution_report_measures_the_imbalance():
    y = np.concatenate([np.zeros(1000, np.int32),
                        np.repeat(np.arange(1, 36, dtype=np.int32), 10)])
    rep = class_distribution_report(y)
    assert rep["commonest_char"] == "0"
    assert rep["imbalance_ratio"] == 100.0
    assert rep["uniform_share_would_be"] == round(1 / 36, 5)


# --------------------------------------------------------------------------
# The class map - ML-46
# --------------------------------------------------------------------------
def test_class_map_round_trips(tmp_path):
    path = tmp_path / "cm.json"
    save_class_map(path, case_strategy="merge")
    loaded = load_class_map(path)
    assert loaded[0] == "0" and loaded[10] == "A" and loaded[35] == "Z"


def test_class_map_detects_a_reordered_charset(tmp_path):
    """A model loaded against a changed charset decodes to wrong characters.

    The weights would load without complaint, so this check is the only thing
    standing between a reordered CHARS and a confidently wrong system.
    """
    import json

    path = tmp_path / "cm.json"
    save_class_map(path, case_strategy="merge")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chars"] = CHARS[::-1]                     # someone reordered it
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="CLASS MAP MISMATCH"):
        load_class_map(path)
