"""The orientation guard must reject sideways glyphs.

Uses synthetic glyphs rather than real EMNIST so the test runs in
milliseconds with no data download - which means it runs on every change,
which is the only way a guard stays working.

    pytest tests/test_orientation.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from anpr.data.guards import assert_upright, ink_profile, prove_guard_fires


def make_synthetic_glyphs(n_per_class: int = 60, seed: int = 0):
    """Build fake '1', 'T' and 'L' glyphs with the right ink geometry.

    We only need shapes that satisfy the three geometric claims - a vertical
    bar, a top bar, and a bottom foot. Real strokes are unnecessary.

    Args:
        n_per_class: Samples per character.
        seed: For the jitter that keeps the glyphs from being identical.

    Returns:
        (X uint8 (N,28,28,1), y_raw int32) with RAW ByClass label positions:
        '1' = 1, 'T' = 29, 'L' = 21.
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []

    specs = {
        1: "one",                                   # '1'
        10 + (ord("T") - ord("A")): "tee",          # 'T' -> 29
        10 + (ord("L") - ord("A")): "ell",          # 'L' -> 21
    }

    for label, kind in specs.items():
        for _ in range(n_per_class):
            img = np.zeros((28, 28), np.uint8)
            jitter = int(rng.integers(-1, 2))       # +/-1 px, so glyphs vary

            if kind == "one":
                img[4:24, 13 + jitter:15 + jitter] = 255          # vertical bar
            elif kind == "tee":
                img[4:7, 6:22] = 255                              # top bar
                img[4:24, 13 + jitter:15 + jitter] = 255          # stem
            else:  # ell
                img[4:24, 8 + jitter:10 + jitter] = 255           # spine
                img[21:24, 8:22] = 255                            # bottom foot

            images.append(img)
            labels.append(label)

    X = np.asarray(images, np.uint8)[..., None]
    return X, np.asarray(labels, np.int32)


@pytest.fixture(scope="module")
def glyphs():
    return make_synthetic_glyphs()


def test_accepts_upright_glyphs(glyphs):
    X, y = glyphs
    report = assert_upright(X, y, verbose=False)
    assert report["1"]["aspect"] > 1.25
    assert report["T"]["top"] > report["T"]["bottom"]
    assert report["L"]["bottom"] > report["L"]["top"]


def test_rejects_transposed_glyphs(glyphs):
    """The whole point. Transposing breaks all three claims at once."""
    X, y = glyphs
    with pytest.raises(AssertionError, match="ORIENTATION CHECK FAILED"):
        assert_upright(np.transpose(X, (0, 2, 1, 3)), y, verbose=False)


def test_prove_guard_fires_passes_on_good_data(glyphs):
    """The self-test helper used by scripts/prepare_data.py."""
    X, y = glyphs
    prove_guard_fires(X, y)          # raises RuntimeError if the guard is broken


def test_refuses_to_run_on_too_few_samples():
    """Better to refuse than to pass on a sample too small to mean anything."""
    X, y = make_synthetic_glyphs(n_per_class=5)
    with pytest.raises(AssertionError, match=">=30 samples"):
        assert_upright(X, y, verbose=False)


def test_ink_profile_is_scale_invariant():
    """Profiles are normalised, so stroke thickness must not change them much."""
    thin = np.zeros((10, 28, 28), np.uint8)
    thin[:, 4:24, 13:15] = 255
    thick = np.zeros((10, 28, 28), np.uint8)
    thick[:, 4:24, 12:16] = 255

    a, b = ink_profile(thin), ink_profile(thick)
    assert abs(a["spread_y"] - b["spread_y"]) < 1.0
