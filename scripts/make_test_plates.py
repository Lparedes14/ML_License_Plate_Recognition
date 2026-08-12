"""Generate the synthetic plate test sets, one per quality tier.

    python scripts/make_test_plates.py

Writes to data/generated/<tier>/ plus a labels.json holding the ground truth
and the per-character boxes.

Blocked on ML-37 (anpr.data.plates). This is the critical-path ticket - it
unblocks the fine-tuning data AND the entire results section. Build it first.
"""

from __future__ import annotations

import argparse
import json

from anpr.config import GENERATED_DATA_DIR, ensure_dirs, load_config, set_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=None,
                        help="plates per tier; defaults to cfg.plates.n_eval_plates")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    ensure_dirs()

    from anpr.data.plates import generate_test_set

    n = args.n or cfg["plates"]["n_eval_plates"]

    for tier in cfg["plates"]["tiers"]:
        out_dir = GENERATED_DATA_DIR / tier
        out_dir.mkdir(parents=True, exist_ok=True)

        # Same seed across tiers, so the tiers differ only in image quality
        # and not in which plate strings were drawn. Without this, a tier
        # could score worse purely for having drawn harder characters.
        samples = generate_test_set(n, tier, cfg, cfg["seed"])

        import cv2
        labels = []
        for i, s in enumerate(samples):
            fname = f"plate_{i:04d}.png"
            cv2.imwrite(str(out_dir / fname), s.image)
            labels.append({"file": fname, "text": s.text,
                           "char_boxes": s.char_boxes, "tier": s.tier})

        with open(out_dir / "labels.json", "w", encoding="utf-8") as fh:
            json.dump({"tier": tier, "n": len(labels),
                       "tier_params": cfg["plates"]["tiers"][tier],
                       "seed": cfg["seed"], "plates": labels}, fh, indent=2)

        print(f"  tier {tier}: {len(labels)} plates -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
