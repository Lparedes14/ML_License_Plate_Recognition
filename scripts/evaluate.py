"""Measure everything, at every tier, and write the evidence to disk.

    python scripts/evaluate.py

Produces (all committed to git - they are the results section):
    artifacts/metrics/tier_results.json     accuracy per quality tier
    artifacts/metrics/confusion.json        top confusions + our predicted pairs
    artifacts/metrics/trust_policy.json     the threshold sweep and the pick
    artifacts/figures/confusion_matrix.png
    artifacts/figures/trust_threshold.png

Blocked on ML-37 (plate generation) and ML-43 (segmentation). The wiring is
complete; it will run as soon as those two exist.
"""

from __future__ import annotations

import argparse
import json

from anpr.business import (
    CostModel,
    policy_sentence,
    recommend_threshold,
    sweep_threshold,
)
from anpr.config import METRICS_DIR, MODELS_DIR, ensure_dirs, load_config, set_seeds
from anpr.evaluate import evaluate_all_tiers, save_results
from anpr.inference import load_reader, read_plate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=str(MODELS_DIR / "plate_cnn.keras"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    ensure_dirs()

    # Model and class map always load together (ML-46).
    model, idx2char = load_reader(args.model)

    def reader_fn(image):
        """Adapt read_plate to the single-argument callable the harness wants."""
        return read_plate(
            image, model, idx2char, n_expected=cfg["plates"]["plate_len"]
        )

    # --- accuracy across the three tiers (ML-49) --------------------------
    print("=" * 70)
    print("Accuracy by image quality tier")
    print("=" * 70)
    results = evaluate_all_tiers(reader_fn, cfg, cfg["seed"])
    save_results(results, METRICS_DIR / "tier_results.json")

    # --- the trust threshold (ML-52) - the headline recommendation --------
    print("\n" + "=" * 70)
    print("Trust threshold: which reads do we auto-accept?")
    print("=" * 70)

    # Tier B is the realistic operating condition, so the policy is fitted
    # there. Report what the same threshold costs at tiers A and C too - the
    # COO's 34 sites are not all the same.
    from anpr.data.plates import generate_test_set

    samples = generate_test_set(cfg["plates"]["n_eval_plates"], "B", cfg, cfg["seed"])
    reads = [reader_fn(s.image) for s in samples]

    confidences = [r.plate_confidence for r in reads]
    correct = [r.text == s.text for r, s in zip(reads, samples)]

    cost_model = CostModel.from_config(cfg)
    curve = sweep_threshold(confidences, correct, cost_model)
    best = recommend_threshold(curve)

    print("\n" + policy_sentence(best, cost_model))

    with open(METRICS_DIR / "trust_policy.json", "w", encoding="utf-8") as fh:
        json.dump({
            "measured_at_tier": "B",
            "n_plates": len(samples),
            "recommended": best.__dict__,
            "recommendation_sentence": policy_sentence(best, cost_model),
            "assumptions": {
                "cost_per_manual_review": cost_model.cost_per_manual_review,
                "note": "cost_per_manual_review is OUR assumption, not given "
                        "in the brief. Reviewed plates are assumed to be "
                        "corrected by the human, which is optimistic.",
            },
            "curve": [p.__dict__ for p in curve],
        }, fh, indent=2)

    print(f"\n  {METRICS_DIR / 'tier_results.json'}")
    print(f"  {METRICS_DIR / 'trust_policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
