"""Inference-layer guards: confidence aggregation and the two ways to fail loudly.

None of this needs a trained model or TensorFlow. `load_reader()`'s TF import is
deferred until AFTER the class-map check (see the fix in read_plate.py), which
is what makes CA-G7 testable on a machine with no TensorFlow installed at all -
exactly this one.

    pytest tests/test_inference.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from anpr.inference.read_plate import aggregate_confidence, load_reader, read_plate


# --------------------------------------------------------------------------
# CA-E1 - plate confidence is the MINIMUM over characters, not the mean
# --------------------------------------------------------------------------
def test_aggregate_confidence_is_the_minimum():
    """RN-06: a plate is only as trustworthy as its weakest character.

    Six characters at 0.99 and one at 0.30 must report 0.30, not the ~0.89
    mean - the mean would sail past a sensible auto-accept threshold and
    bill the wrong customer on the one character that was actually wrong.
    """
    confidences = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.30]
    assert aggregate_confidence(confidences) == pytest.approx(0.30)


def test_aggregate_confidence_empty_read_is_zero():
    """Nothing found means nothing to trust - not an average of nothing."""
    assert aggregate_confidence([]) == 0.0


def test_aggregate_confidence_single_character():
    assert aggregate_confidence([0.73]) == pytest.approx(0.73)


# --------------------------------------------------------------------------
# CA-G5 - an unreadable image path fails loudly, not as an obscure shape error
# --------------------------------------------------------------------------
def test_read_plate_rejects_nonexistent_path(tmp_path):
    missing = tmp_path / "does_not_exist.png"

    with pytest.raises(FileNotFoundError, match="could not read"):
        # model/idx2char are never reached: cv2.imread fails before either
        # is touched, so passing placeholders is safe here.
        read_plate(missing, model=None, idx2char={})


def test_read_plate_rejects_corrupt_image_file(tmp_path):
    """A path that exists but is not a decodable image - cv2.imread returns
    None for this exactly as it does for a missing path, so both cases are
    caught by the same check."""
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not image data")

    with pytest.raises(FileNotFoundError, match="could not read"):
        read_plate(corrupt, model=None, idx2char={})


# --------------------------------------------------------------------------
# CA-G7 - a model without its class map is refused, not silently guessed
# --------------------------------------------------------------------------
def test_load_reader_refuses_missing_classmap(tmp_path):
    """ML-46: the model's outputs are integers; only the class map says which
    character each integer means. Loading one without the other would decode
    every plate confidently and wrongly, with no error raised anywhere.
    """
    model_path = tmp_path / "plate_cnn.keras"
    model_path.write_bytes(b"")          # never read - the check fails first

    with pytest.raises(FileNotFoundError, match="class map not found"):
        load_reader(model_path)


def test_load_reader_accepts_explicit_classmap_path_argument(tmp_path):
    """The missing-classmap check must also fire when a path is passed
    explicitly, not only when relying on the default sibling-file naming."""
    model_path = tmp_path / "plate_cnn.keras"
    model_path.write_bytes(b"")
    explicit_missing = tmp_path / "elsewhere" / "classmap.json"

    with pytest.raises(FileNotFoundError, match="class map not found"):
        load_reader(model_path, classmap_path=explicit_missing)
