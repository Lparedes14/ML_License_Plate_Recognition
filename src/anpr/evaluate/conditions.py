"""Accuracy as a function of image quality - the strongest result we can get.

Ported from `evaluate_tier()` in `ML_Draft1_Project.ipynb` (Thenmani). The
counting rules are hers, unchanged. One structural change: this reads the
persisted test set from `data/generated/manifest.csv` rather than generating
plates in memory, so every reported number traces to image files that still
exist and can be re-examined.

Tickets: ML-41, ML-49.

WHY A CURVE BEATS A NUMBER
    "We get 78%" invites one question: 78% of what? "91% on clean images,
    78% on realistic capture, 54% on motion-blurred night shots" answers the
    COO's actual question, which is not "is it accurate" but "will it work at
    MY sites". Some of Meridian's 34 sites are airports with good lighting;
    some are parks at night.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TierResult:
    """Measured performance at ONE stated quality tier.

    Attributes:
        tier: "clean", "normal" or "hard". Never None - a number without
            conditions is not reportable (§5).
        n_plates: Sample size.
        segmentation_rate: Fraction where the character count was right.
        char_accuracy: Correct characters / compared characters, over
            correctly-segmented plates ONLY, so it measures the classifier.
        plate_accuracy: Fraction read entirely correctly, over ALL plates
            including segmentation failures - from Meridian's point of view a
            plate that failed to segment bills the wrong customer just as
            surely as one misread.
        n_segmentation_failures: Absolute count, for the error breakdown.
        example_failures: (truth, predicted, n_boxes) for the first few
            failures - kept so a failure can be explained, not just counted.
        confidences: Per-plate aggregate confidence, for the trust threshold.
        correct_flags: Per-plate correctness, aligned with `confidences`.
    """

    tier: str
    n_plates: int
    segmentation_rate: float = 0.0
    char_accuracy: float = 0.0
    plate_accuracy: float = 0.0
    n_segmentation_failures: int = 0
    example_failures: list = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    correct_flags: list[bool] = field(default_factory=list)

    def to_sentence(self) -> str:
        """One reportable sentence, with conditions baked in.

        Deliberately impossible to produce a bare "we got 78%" from this
        object. If a number appears in a document or on a slide, it should
        have come through here.
        """
        return (
            f"Tier {self.tier} ({self.n_plates} plates): "
            f"character accuracy {self.char_accuracy:.1%}, "
            f"plate accuracy {self.plate_accuracy:.1%}, "
            f"segmentation success {self.segmentation_rate:.1%} "
            f"({self.n_segmentation_failures} segmentation failures)."
        )


def evaluate_tier(samples, reader_fn, tier: str, max_examples: int = 5) -> TierResult:
    """Score one tier's plates.

    THE COUNTING RULE, which is the whole point of this function (§4):
      - lengths disagree  -> SEGMENTATION failure. Excluded from character
                             accuracy, counted wrong for plate accuracy.
      - lengths agree     -> a fair test of the classifier.

    Args:
        samples: PlateSample list for this tier.
        reader_fn: Callable taking an image and returning a PlateRead.
        tier: The tier label these were measured at.
        max_examples: How many failure examples to retain.

    Returns:
        A TierResult.
    """
    n_seg_ok = n_char_correct = n_char_total = n_plate_correct = 0
    confidences, correct_flags, failures = [], [], []

    for s in samples:
        read = reader_fn(s.image)
        pred, truth = read.text, s.text

        seg_ok = len(pred) == len(truth)
        n_seg_ok += seg_ok

        if seg_ok:
            n_char_correct += sum(p == t for p, t in zip(pred, truth))
            n_char_total += len(truth)
        elif len(failures) < max_examples:
            failures.append((truth, pred, read.n_chars_found))

        is_correct = (pred == truth)
        n_plate_correct += is_correct
        confidences.append(read.plate_confidence)
        correct_flags.append(bool(is_correct))

    n = len(samples)
    return TierResult(
        tier=tier,
        n_plates=n,
        segmentation_rate=(n_seg_ok / n) if n else 0.0,
        char_accuracy=(n_char_correct / n_char_total) if n_char_total else 0.0,
        plate_accuracy=(n_plate_correct / n) if n else 0.0,
        n_segmentation_failures=n - n_seg_ok,
        example_failures=failures,
        confidences=confidences,
        correct_flags=correct_flags,
    )


def evaluate_all_tiers(reader_fn, data_dir: str | Path = "data/generated") -> dict:
    """Run the full evaluation across every tier in the persisted test set.

    Args:
        reader_fn: Callable taking an image, returning a PlateRead.
        data_dir: Directory holding `manifest.csv` and the tier folders.

    Returns:
        {tier: TierResult}.
    """
    from anpr.data.plates import load_test_set

    samples = load_test_set(data_dir)
    by_tier: dict[str, list] = {}
    for s in samples:
        by_tier.setdefault(s.tier, []).append(s)

    results = {}
    for tier, tier_samples in by_tier.items():
        results[tier] = evaluate_tier(tier_samples, reader_fn, tier)
        print("  " + results[tier].to_sentence())
    return results


def save_results(results: dict, path: str | Path) -> Path:
    """Write tier results to JSON so the report quotes measurements, not memory.

    This file is committed to git (see .gitignore) because it is evidence.
    Anyone should be able to open it and see exactly what was measured, on
    how many plates, under what conditions.

    Args:
        results: Output of `evaluate_all_tiers`.
        path: Destination JSON.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        tier: {
            "tier": r.tier,
            "n_plates": r.n_plates,
            "segmentation_rate": round(r.segmentation_rate, 4),
            "char_accuracy": round(r.char_accuracy, 4),
            "plate_accuracy": round(r.plate_accuracy, 4),
            "n_segmentation_failures": r.n_segmentation_failures,
            "example_failures": r.example_failures,
            "confidences": r.confidences,
            "correct_flags": r.correct_flags,
            "sentence": r.to_sentence(),
        }
        for tier, r in results.items()
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
