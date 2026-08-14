"""Grayscale plate image -> clean black-and-white.

Ported from `segment_plate()` in `ML_Draft1_Project.ipynb` (Thenmani). The
thresholding recipe is hers, unchanged - extracted into its own function so
the binary mask can be inspected on its own when a segmentation failure
needs explaining.

Ticket: ML-43.

CONVENTION - THIS IS THE PART THAT BITES
    Output is WHITE characters on a BLACK background (ink = 255), matching
    EMNIST where the glyph is bright and the background is 0. Real plates are
    dark-text-on-light, so `THRESH_BINARY_INV` does the inversion. Getting
    this backwards produces a model that sees photographic negatives and
    predicts confident nonsense.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """RGB or grayscale in, single-channel uint8 out.

    Args:
        image: uint8 (H, W), (H, W, 1) or (H, W, 3).

    Returns:
        uint8 (H, W).
    """
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 3:
        # BT.601 luma - the same weights used in data/contract.py. Using a
        # different formula in the two places would mean training-time and
        # demo-time images differ subtly, which is the mismatch §4 warns of.
        return (image * np.array([0.299, 0.587, 0.114])).sum(-1).astype(np.uint8)
    return image.squeeze().astype(np.uint8)


def binarize(image: np.ndarray) -> np.ndarray:
    """Threshold to a binary image, white ink on black background.

    Blur first (3x3 Gaussian) to stop sensor noise fragmenting glyph strokes,
    then Otsu - which picks the threshold from the image's own histogram
    rather than a fixed value, so it adapts to how light or dark the plate
    is. A 2x2 morphological close then rejoins strokes that thresholding
    broke apart.

    Args:
        image: uint8 (H, W) grayscale plate image.

    Returns:
        uint8 (H, W) with values in {0, 255}, characters at 255.
    """
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
