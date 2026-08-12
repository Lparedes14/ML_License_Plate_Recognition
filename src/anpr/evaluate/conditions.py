"""Accuracy as a function of image quality - the strongest result we can get.

Owner: QA role. Tickets: ML-41, ML-49.

§5: "you can dial image quality up and down to report accuracy as a function
of conditions. That single move gives you a far better result section than a
handful of hand-labelled photos."

WHY A CURVE BEATS A NUMBER
    "We get 78%" invites one question: 78% of what?
    "91% on clean images, 78% on realistic camera capture, 54% on motion-
    blurred night shots" answers the COO's actual question, which is not
    "is it accurate" but "will it work at MY sites". Some of Meridian's 34
    sites are airports with good lighting; some are parks at night. The
    curve tells her which ones to start with - and "deploy at the 12 sites
    that look like tier A first" is a recommendation, not just a result.
"""

from __future__ import annotations

import json
from pathlib import Path

from anpr.evaluate.metrics import EvalResult


def evaluate_all_tiers(reader_fn, cfg: dict, seed: int) -> dict[str, EvalResult]:
    """Run the full evaluation at every quality tier.

    Args:
        reader_fn: Callable taking a plate image and returning a PlateRead.
            Passed in rather than constructed here so the same harness can
            evaluate the handwriting-only model and the fine-tuned one.
        cfg: Full config; reads cfg["plates"].
        seed: Base seed for plate generation.

    Returns:
        {tier: EvalResult}.
    """
    from anpr.data.plates import generate_test_set
    from anpr.evaluate.metrics import evaluate_reads

    results: dict[str, EvalResult] = {}

    for tier in cfg["plates"]["tiers"]:
        # Same seed per tier on purpose: the tiers then differ ONLY in image
        # quality, not in which plate strings were drawn. Otherwise a tier
        # could look worse simply for having drawn more 'Q's.
        samples = generate_test_set(cfg["plates"]["n_eval_plates"], tier, cfg, seed)

        reads = [reader_fn(s.image) for s in samples]
        truths = [s.text for s in samples]

        results[tier] = evaluate_reads(
            reads, truths, tier=tier,
            notes=[f"synthetic plates, {cfg['plates']['plate_len']} chars, "
                   f"tier params {cfg['plates']['tiers'][tier]}"],
        )
        print("  " + results[tier].to_sentence())

    return results


def save_results(results: dict[str, EvalResult], path: str | Path) -> Path:
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
            "char_accuracy": round(r.char_accuracy, 4),
            "plate_accuracy": round(r.plate_accuracy, 4),
            "segmentation_rate": round(r.segmentation_rate, 4),
            "n_segmentation_failures": r.n_segmentation_failures,
            "notes": r.notes,
            "sentence": r.to_sentence(),
        }
        for tier, r in results.items()
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
