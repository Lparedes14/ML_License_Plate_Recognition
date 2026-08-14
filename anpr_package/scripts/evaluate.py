"""Measure everything, at every tier, and write the evidence to disk.

    python scripts/evaluate.py

Reads the persisted test set from data/generated/ (run make_test_plates.py
first) and the trained model from artifacts/models/.

Produces:
    artifacts/metrics/tier_results.json   accuracy per quality tier
    artifacts/metrics/confusion.json      top confusions + predicted pairs

Every number carries its tier and sample size (§5 - reporting accuracy
without stating conditions is a -5 deduction).
"""

from __future__ import annotations

import argparse
import json

from anpr.config import (
    GENERATED_DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    ensure_dirs,
    load_config,
    set_seeds,
)
from anpr.evaluate import evaluate_all_tiers, save_results
from anpr.inference import load_reader, read_plate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=str(MODELS_DIR / "plate_cnn.keras"))
    parser.add_argument("--data", default=str(GENERATED_DATA_DIR))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    ensure_dirs()

    # Model and class map always load together (ML-46).
    model, idx2char = load_reader(args.model)
    print(f"model: {model.count_params():,} parameters, {len(idx2char)} classes\n")

    def reader_fn(image):
        """Adapt read_plate to the single-argument callable the harness wants."""
        return read_plate(
            image, model, idx2char, n_expected=cfg["plates"]["plate_len"]
        )

    print("=" * 78)
    print("END-TO-END EVALUATION - synthetic plates, all quality tiers")
    print("=" * 78)
    results = evaluate_all_tiers(reader_fn, args.data)

    out = save_results(results, METRICS_DIR / "tier_results.json")
    print(f"\nwritten: {out}")

    # Failure examples - kept so a failure can be explained, not just counted.
    print("\nExample segmentation failures:")
    for tier, r in results.items():
        for truth, pred, n_boxes in r.example_failures[:3]:
            print(f"  [{tier:>6}] truth {truth} -> '{pred}' ({n_boxes} boxes found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
