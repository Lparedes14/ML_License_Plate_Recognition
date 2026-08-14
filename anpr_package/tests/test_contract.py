"""The input contract must be impossible to violate quietly.

These are the notebook's inline self-tests (cell 32) promoted to pytest, so
they run on every change instead of only when someone re-executes a cell.

    pytest tests/test_contract.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from anpr.data.contract import assert_input_contract, normalize, to_canonical_uint8


def _u8(*shape) -> np.ndarray:
    """A random uint8 image stack of the given shape."""
    return (np.random.rand(*shape) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# to_canonical_uint8 - accept every input shape the demo might receive
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape,label", [
    ((3, 784), "flat CSV rows"),
    ((4, 64, 64, 3), "RGB photograph"),
    ((2, 14, 14), "small grayscale stack"),
    ((5, 28, 28, 1), "already canonical"),
    ((28, 28), "single image"),
])
def test_canonicalises_any_input_shape(shape, label):
    out = to_canonical_uint8(_u8(*shape))
    assert out.shape[1:] == (28, 28, 1), f"{label} produced {out.shape}"
    assert out.dtype == np.uint8


def test_rejects_uninterpretable_shape():
    with pytest.raises(ValueError):
        to_canonical_uint8(np.zeros((2, 3, 4, 5, 6), np.uint8))


def test_rejects_two_channel_input():
    """1 or 3 channels only. Two channels means something is wrong upstream."""
    with pytest.raises(ValueError):
        to_canonical_uint8(_u8(3, 28, 28, 2))


# --------------------------------------------------------------------------
# normalize - the double-normalisation block
# --------------------------------------------------------------------------
def test_normalize_scales_to_unit_range():
    out = normalize(_u8(8, 28, 28, 1))
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_normalize_refuses_float_input():
    """The single most valuable check in the repo.

    Dividing by 255 twice leaves every pixel below 0.004. Training still
    "works" and validation accuracy still looks plausible, but the model is
    useless on real images. §4 names this the most common cause of exactly
    that failure, so we make it raise instead of hoping nobody does it.
    """
    already_normalised = normalize(_u8(4, 28, 28, 1))
    with pytest.raises(TypeError):
        normalize(already_normalised)

    with pytest.raises(TypeError):
        normalize(_u8(4, 28, 28, 1).astype(np.float32))   # float in [0,255]


# --------------------------------------------------------------------------
# assert_input_contract - the guard itself must fire
# --------------------------------------------------------------------------
def test_contract_accepts_a_valid_batch():
    stats = assert_input_contract(normalize(_u8(8, 28, 28, 1)), "valid")
    assert stats["shape"] == [8, 28, 28, 1]
    assert stats["dtype"] == "float32"


def test_contract_rejects_double_normalised_batch():
    """A check that never fails proves nothing. This proves it fails."""
    with pytest.raises(AssertionError, match="double-normalised"):
        assert_input_contract(normalize(_u8(8, 28, 28, 1)) / 255.0, "double")


def test_contract_rejects_wrong_shape():
    with pytest.raises(AssertionError):
        assert_input_contract(np.zeros((8, 32, 32, 1), np.float32), "wrong shape")


def test_contract_rejects_out_of_range_values():
    with pytest.raises(AssertionError):
        assert_input_contract(np.full((8, 28, 28, 1), 5.0, np.float32), "out of range")
