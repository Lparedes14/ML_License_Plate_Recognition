"""Segmentation - plate image to character crops.

Owner: Pipeline role (§7). Tickets: ML-43, ML-13.

    binarize.py    grayscale -> clean black/white     [NOT IMPLEMENTED]
    components.py  connected components -> crops      [NOT IMPLEMENTED]

REMEMBER: a segmentation failure is NOT a recognition failure. Count them
separately from day one (§4). `SegmentationResult.count_matches` is how.
"""

from anpr.segment.binarize import binarize, to_grayscale
from anpr.segment.components import (
    SegmentationResult,
    find_character_boxes,
    normalize_crop,
    segment_characters,
    sort_reading_order,
)

__all__ = [
    "to_grayscale", "binarize",
    "SegmentationResult", "find_character_boxes", "sort_reading_order",
    "normalize_crop", "segment_characters",
]
