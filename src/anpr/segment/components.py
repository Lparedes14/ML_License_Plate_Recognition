"""Plate image -> individual character crops.

Ported from `segment_plate()` and `to_mnist_format()` in
`ML_Draft1_Project.ipynb` (Thenmani). Filters, thresholds and the
normalisation recipe are hers, unchanged; the structural change is that the
result is returned as a `SegmentationResult` rather than a bare tuple, so
the character COUNT travels with the crops and failures can be counted
separately (§4).

Ticket: ML-43, ML-44.

THE RULE THAT SHAPES THIS MODULE (§4)
    "Segmentation failures and recognition failures are different bugs. If
     your system reads a 6-character plate as 5 characters, that is not a
     recognition error and no amount of retraining fixes it. Count them
     separately from day one."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from anpr.segment.binarize import binarize, to_grayscale


@dataclass
class SegmentationResult:
    """Everything segmentation knows, including how well it thinks it did.

    Attributes:
        crops: uint8 (N, 28, 28, 1), contract-compliant, in READING order.
        boxes: (x, y, w, h) per crop, same order. Kept for the demo overlay -
            drawing the boxes on the input image is the single most
            convincing thing to show a COO.
        n_found: Number of characters detected.
        n_expected: Expected count, when known. None if genuinely unknown.
    """

    crops: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    n_found: int = 0
    n_expected: int | None = None

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


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """One character crop -> a 28x28 glyph matching EMNIST's convention.

    THE PROCEDURE, which must match how the training data was built:
      1. Otsu-threshold, then tight-crop to the ink's bounding box.
      2. Scale the longer side to 20 pixels, preserving aspect ratio -
         preserving aspect is what stops a '1' being stretched into a '0'.
      3. Paste into a 28x28 black field.
      4. Shift so the ink's CENTRE OF MASS sits at the field centre -
         centre of mass, NOT the bounding-box centre. They differ for
         asymmetric glyphs like 'J' and 'L', and EMNIST uses centre of mass.

    §4 names mismatched preprocessing as the single most common cause of
    "57% validation accuracy, unusable on real images". This function is
    called by BOTH the segmentation path and the printed-glyph renderer in
    `data/plates.py`, so there is only one recipe to keep in sync.

    Args:
        crop: uint8 (h, w) character region. White ink on black.

    Returns:
        uint8 (28, 28).
    """
    g = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    ys, xs = np.where(g > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)

    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    s = 20.0 / max(h, w)
    g = cv2.resize(
        g, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((28, 28), np.uint8)
    h, w = g.shape
    canvas[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = g

    m = cv2.moments(canvas)
    if m["m00"] > 0:
        dx, dy = 14 - m["m10"] / m["m00"], 14 - m["m01"] / m["m00"]
        canvas = cv2.warpAffine(
            canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28)
        )
    return canvas


def find_character_boxes(
    binary: np.ndarray, plate_shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    """Contours, filtered down to plausible characters, in reading order.

    The filters exist because a real plate image contains more than
    characters - mounting bolts, the plate border, screw shadows. Every
    threshold below is a judgement call tuned against our generated plates;
    they are what makes segmentation work at all on the degraded tier (the
    spike measured 100% exact with filtering vs 0% without).

    Args:
        binary: uint8 (H, W) binary mask, white ink on black.
        plate_shape: (h, w) of the source plate, for relative thresholds.

    Returns:
        (x, y, w, h) boxes sorted left to right. Detection order from
        `findContours` is arbitrary; without the sort a perfectly-read plate
        comes back scrambled.
    """
    h, w = plate_shape
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 0.35 * h or ch > 0.98 * h:
            continue                              # too small / whole-frame noise
        if cw > 0.30 * w or cw < 0.008 * w:
            continue                              # merged blobs / specks
        if ch / max(cw, 1) < 0.8:
            continue                              # glyphs are taller than wide
        boxes.append((x, y, cw, ch))

    boxes.sort(key=lambda b: b[0])
    return boxes


def segment_characters(
    image: np.ndarray, n_expected: int | None = None, pad: int = 2
) -> SegmentationResult:
    """Full segmentation: plate image in, contract-compliant crops out.

    Args:
        image: uint8 grayscale or RGB plate image.
        n_expected: Expected character count, when known. Recorded on the
            result so `count_matches` can flag a segmentation failure.

            NOTE: this is recorded, NOT enforced. The notebook version
            optionally truncated over-segmented results down to `expected`
            by keeping the tallest boxes - convenient for a demo, but it
            hides over-segmentation from the accuracy numbers. Measurement
            has to see the raw blob count, so that truncation is deliberately
            not carried over here.
        pad: Pixels of padding around each box before normalisation.

    Returns:
        A SegmentationResult. `crops` is an empty (0, 28, 28, 1) array when
        nothing was found - never a fabricated character.
    """
    gray = to_grayscale(image)
    h, w = gray.shape

    binary = binarize(gray)
    boxes = find_character_boxes(binary, (h, w))

    crops = []
    for x, y, cw, ch in boxes:
        crop = gray[max(0, y - pad):min(h, y + ch + pad),
                    max(0, x - pad):min(w, x + cw + pad)]
        # 255 - crop inverts dark-glyph-on-light-plate to the white-on-black
        # convention normalize_crop expects. SAME recipe as the training
        # data - this is the integration trap §4 warns about.
        crops.append(normalize_crop(255 - crop))

    crops_arr = (
        np.array(crops)[..., None] if crops
        else np.zeros((0, 28, 28, 1), np.uint8)
    )

    return SegmentationResult(
        crops=crops_arr,
        boxes=boxes,
        n_found=len(boxes),
        n_expected=n_expected,
    )
