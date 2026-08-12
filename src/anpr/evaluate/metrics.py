"""The three numbers the brief requires, each with its conditions attached.

Owner: QA role. Tickets: ML-49, ML-38, ML-15.

§5 REQUIRES ALL THREE, SEPARATELY
    character-level accuracy   >= 75% is a reasonable prototype
    plate-level accuracy       no threshold; report it and explain why it is
                               much lower than character accuracy
    segmentation success rate  reported separately from recognition

AND EVERY NUMBER MUST STATE ITS CONDITIONS. Reporting accuracy without saying
what image quality it was measured on is an explicit -5 deduction. That is
why `EvalResult` carries `tier` and `n_plates` and why `to_sentence()` refuses
to produce a bare percentage.

WHY PLATE ACCURACY IS SO MUCH LOWER - the arithmetic to have ready in the demo
    A plate is correct only when EVERY character is. At 95% per character on
    a 7-character plate, the best case is 0.95^7 = 70%. At 90% it is 48%. So
    a 5-point gain in character accuracy is worth ~20 points of plate
    accuracy. Errors are correlated in practice (a blurry plate is blurry for
    all seven characters), so the real number is usually better than this
    bound - which is worth saying out loud rather than being surprised by.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class EvalResult:
    """Measured performance under ONE stated set of conditions.

    Attributes:
        tier: Image quality tier ("A", "B", "C"). Never None - a number
            without conditions is not reportable.
        n_plates: Sample size. A 92% on 12 plates is not a result.
        char_accuracy: Correct characters / total compared characters.
            Computed only over plates that segmented correctly, so it
            measures the CLASSIFIER and nothing else.
        plate_accuracy: Fraction of plates read entirely correctly. Computed
            over ALL plates, including segmentation failures, because from
            Meridian's point of view a plate that failed to segment is just
            as wrong as one misread.
        segmentation_rate: Fraction where the character count was right.
        n_segmentation_failures: Absolute count, for the error breakdown.
        notes: Anything qualifying the number. Read aloud in the demo.
    """

    tier: str
    n_plates: int
    char_accuracy: float = 0.0
    plate_accuracy: float = 0.0
    segmentation_rate: float = 0.0
    n_segmentation_failures: int = 0
    notes: list[str] = field(default_factory=list)

    def to_sentence(self) -> str:
        """One reportable sentence, with conditions baked in.

        Deliberately impossible to produce a bare "we got 78%" from this
        object. If a number appears in the report or on a slide, it should
        have come through here.
        """
        return (
            f"Tier {self.tier} ({self.n_plates} plates): "
            f"character accuracy {self.char_accuracy:.1%}, "
            f"plate accuracy {self.plate_accuracy:.1%}, "
            f"segmentation success {self.segmentation_rate:.1%} "
            f"({self.n_segmentation_failures} segmentation failures)."
        )


def character_accuracy(predicted: str, truth: str) -> tuple[int, int]:
    """Correct characters in one plate, comparing position by position.

    Only defined when the lengths match - which is why segmentation must be
    checked first. If they differ, the comparison is meaningless and the
    plate belongs in the segmentation-failure bucket, not this one.

    Args:
        predicted: The predicted string.
        truth: The ground-truth string.

    Returns:
        (n_correct, n_compared). (0, 0) when lengths differ, so a
        segmentation failure contributes nothing to character accuracy
        rather than silently dragging it down.
    """
    if len(predicted) != len(truth):
        return 0, 0
    return sum(p == t for p, t in zip(predicted, truth)), len(truth)


def evaluate_reads(
    reads: Sequence, truths: Sequence[str], tier: str, notes: list[str] | None = None
) -> EvalResult:
    """Aggregate a batch of PlateRead results into one reportable EvalResult.

    THE COUNTING RULE, which is the whole point of this function (§4):
      - lengths disagree  -> SEGMENTATION failure. Excluded from character
                             accuracy, counted as wrong for plate accuracy.
      - lengths agree     -> a fair test of the classifier. Contributes to
                             character accuracy.

    Args:
        reads: PlateRead objects from `inference.read_plate`.
        truths: Ground-truth strings, same order and length as `reads`.
        tier: The quality tier these were measured at.
        notes: Extra qualifications for the report.

    Returns:
        An EvalResult.

    Raises:
        ValueError: if the two sequences differ in length - that means the
            evaluation set and the predictions are misaligned, and every
            number derived from them would be wrong.
    """
    if len(reads) != len(truths):
        raise ValueError(
            f"{len(reads)} reads but {len(truths)} ground-truth strings. "
            "These must correspond one-to-one."
        )

    n_correct_chars = n_compared_chars = 0
    n_correct_plates = n_seg_ok = 0

    for read, truth in zip(reads, truths):
        # Segmentation is judged on the character count alone.
        seg_ok = len(read.text) == len(truth)
        n_seg_ok += seg_ok

        if seg_ok:
            correct, compared = character_accuracy(read.text, truth)
            n_correct_chars += correct
            n_compared_chars += compared

        # Plate accuracy is unforgiving on purpose: a segmentation failure
        # bills the wrong customer exactly as surely as a misread does.
        n_correct_plates += (read.text == truth)

    n = len(reads)
    return EvalResult(
        tier=tier,
        n_plates=n,
        char_accuracy=(n_correct_chars / n_compared_chars) if n_compared_chars else 0.0,
        plate_accuracy=(n_correct_plates / n) if n else 0.0,
        segmentation_rate=(n_seg_ok / n) if n else 0.0,
        n_segmentation_failures=n - n_seg_ok,
        notes=notes or [],
    )
