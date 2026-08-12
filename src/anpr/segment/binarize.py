"""Grayscale plate image -> clean black-and-white.

STATUS: NOT IMPLEMENTED. Owner: Pipeline. Ticket: ML-43.

Everything downstream depends on this step. A bad threshold merges adjacent
characters into one blob (segmentation finds 5 where there are 7) or breaks
one character into two (finds 8). Both are segmentation failures, and neither
is fixable by a better classifier.

WHY ADAPTIVE, NOT A FIXED THRESHOLD
    A single global threshold works on tier A and fails on tier C, where one
    end of the plate is in shade. Adaptive thresholding computes a local
    threshold per neighbourhood, which handles uneven lighting - exactly the
    condition we are deliberately generating in tier C.

    Try Otsu first (global, parameter-free, a good baseline), then adaptive
    Gaussian. Report which you chose and what it cost at each tier: that
    comparison is a legitimate result, not just an implementation detail.

CONVENTION - GET THIS RIGHT
    Output must be WHITE characters on a BLACK background (ink = 255).
    That matches EMNIST, where the glyph is bright and the background is 0.
    Invert if the source is dark-on-light - most real plates are - and assert
    the convention rather than assuming it. Getting this backwards produces a
    model that sees photographic negatives and predicts confident nonsense.
"""

from __future__ import annotations

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
        # BT.601 luma, the same weights used in data/contract.py. Using a
        # different formula in the two places would mean training-time and
        # demo-time images differ subtly - precisely the mismatch §4 warns of.
        return (image * np.array([0.299, 0.587, 0.114])).sum(-1).astype(np.uint8)
    return image.squeeze().astype(np.uint8)


def binarize(
    image: np.ndarray, method: str = "adaptive", block_size: int = 31, C: int = 5
) -> np.ndarray:
    """Threshold to a binary image, white ink on black background.

    Args:
        image: uint8 plate image, grayscale or RGB.
        method: "otsu" (global, parameter-free) or "adaptive" (per-region,
            survives uneven lighting).
        block_size: Adaptive neighbourhood size. Must be ODD. Roughly the
            character height works well as a starting point.
        C: Constant subtracted from the local mean. Larger = more aggressive.

    Returns:
        uint8 (H, W) with values in {0, 255}, characters at 255.
    """
    raise NotImplementedError(
        "ML-43: implement binarize(). Use cv2.threshold with THRESH_OTSU or "
        "cv2.adaptiveThreshold. Then ASSERT the output convention - more "
        "white pixels near the image centre than at the border means ink is "
        "white, which is what we want."
    )
