"""Plate image -> individual character crops.

STATUS: NOT IMPLEMENTED. Owner: Pipeline. Tickets: ML-43, ML-13.

THE RULE THAT SHAPES THIS WHOLE MODULE (§4)
    "Segmentation failures and recognition failures are different bugs. If
    your system reads a 6-character plate as 5 characters, that is not a
    recognition error and no amount of retraining fixes it. Count them
    separately from day one."

    So `segment_characters` returns a SegmentationResult carrying how many
    boxes it found, not just the crops. The evaluation harness compares that
    count to the ground truth and reports segmentation success rate as its
    own number (§5 requires it reported separately).

PIPELINE
    1. binarize        grayscale -> black/white           [binarize.py]
    2. connected components on the binary image           [here]
    3. filter blobs    drop noise, bolts, plate border    [here]
    4. sort left-to-right - reading order is not detection order
    5. normalise each crop to the input contract          [here]

THE STEP THAT WILL COST YOU A DAY IF YOU GET IT WRONG
    Step 5. EMNIST glyphs are centred by CENTRE OF MASS in a 20x20 box inside
    a 28x28 field - that is how the original MNIST was built. If our crops are
    instead stretched to fill 28x28, every glyph arrives at a different scale
    and position from anything the model saw in training, and accuracy
    collapses for a reason that looks like a model problem but is not.

    §4 names this as the single most common cause of "57% validation accuracy,
    unusable on real images". Implement `normalize_crop` to match EMNIST's
    convention exactly, and write a test that asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentationResult:
    """Everything segmentation knows, including how well it thinks it did.

    Attributes:
        crops: uint8 (N, 28, 28, 1), contract-compliant, in READING order.
        boxes: (x, y, w, h) per crop, same order. Kept for the demo overlay -
            drawing the boxes on the input image is the single most
            convincing thing to show a COO.
        n_found: Number of characters detected.
        n_expected: Expected count, when known (ground truth, or the
            jurisdiction's plate length). None if genuinely unknown.
        rejected: Blobs discarded by filtering, with the reason. Do not throw
            these away - "we dropped it as noise" is the explanation for a
            failure you will otherwise be unable to account for live.
    """

    crops: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    n_found: int = 0
    n_expected: int | None = None
    rejected: list[dict] = field(default_factory=list)

    @property
    def count_matches(self) -> bool | None:
        """True if we found the number of characters we expected.

        This is the segmentation success criterion. A plate where this is
        False is a SEGMENTATION failure and must never be counted as a
        recognition error.
        """
        if self.n_expected is None:
            return None
        return self.n_found == self.n_expected


def find_character_boxes(
    binary: np.ndarray,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.25,
    min_aspect: float = 0.15,
    max_aspect: float = 3.0,
) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    """Connected components, filtered down to plausible characters.

    Use `cv2.connectedComponentsWithStats`. The filters exist because a real
    plate image contains more than characters - mounting bolts, the plate
    border, screw shadows, state names, registration stickers.

    Every threshold below is a JUDGEMENT CALL that will need tuning against
    your own generated plates. Tune them on tier A, then check they still hold
    at tier C, and record what you chose. "We tuned min_area on tier B and it
    cost us at tier C" is a good sentence for the results document.

    Args:
        binary: uint8 (H, W), 0 or 255, characters white on black.
        min_area_ratio: Blobs smaller than this fraction of the image are
            noise (specks, sensor dust).
        max_area_ratio: Blobs larger than this are the plate border or
            background bleed, not a character.
        min_aspect: w/h below this is a vertical line - the plate edge.
            Careful: '1' and 'I' are genuinely narrow, so setting this too
            high silently deletes them from every plate. That is a bug that
            looks like a recognition problem.
        max_aspect: w/h above this is a horizontal rule.

    Returns:
        (boxes accepted, rejected blobs with a reason each).
    """
    raise NotImplementedError("ML-43: implement find_character_boxes().")


def sort_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[int]:
    """Order boxes left-to-right.

    Connected-component labelling returns blobs in an arbitrary order, so
    without this the plate reads as a scramble of the right characters -
    100% character accuracy and 0% plate accuracy, which is a confusing
    result to debug at 2am.

    Sorting by x alone is fine for a single-row plate. If you later support
    two-row plates, cluster by y first, then sort by x within each row.

    Args:
        boxes: Unordered (x, y, w, h).

    Returns:
        Indices that put `boxes` into reading order.
    """
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0])


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """One character crop -> a 28x28 glyph matching EMNIST's convention.

    THE EMNIST/MNIST PROCEDURE, which we must reproduce exactly:
      1. tight-crop to the ink's bounding box
      2. scale the longer side to 20 pixels, preserving aspect ratio
         (preserving aspect is what stops a '1' being stretched into a '0')
      3. paste into a 28x28 black field
      4. shift so the ink's CENTRE OF MASS sits at the field centre -
         centre of mass, NOT the bounding-box centre. They differ for
         asymmetric glyphs like 'J' and 'L', and the model was trained on
         centre-of-mass positioning.

    Args:
        crop: uint8 (h, w) character region cut from the binary image.

    Returns:
        uint8 (28, 28, 1), ready for `contract.normalize`.
    """
    raise NotImplementedError(
        "ML-44: implement normalize_crop() to EMNIST convention - 20x20 "
        "aspect-preserved, centred by centre of mass in a 28x28 field. "
        "Add a test asserting the centre of mass lands within one pixel of "
        "(14, 14); this is the highest-value test in the repo."
    )


def segment_characters(
    image: np.ndarray, n_expected: int | None = None, **kwargs
) -> SegmentationResult:
    """Full segmentation: plate image in, contract-compliant crops out.

    Args:
        image: uint8 grayscale or RGB plate image, any size.
        n_expected: Expected character count, if known.
        **kwargs: Threshold overrides passed to `find_character_boxes`.

    Returns:
        A SegmentationResult.
    """
    raise NotImplementedError("ML-43: implement segment_characters().")
