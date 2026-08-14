"""Generate the synthetic plate test set - 400 plates per quality tier.

    python scripts/make_test_plates.py
    python scripts/make_test_plates.py --n 100        # smaller, for a quick check

Writes to data/generated/:
    manifest.csv          filename, text, tier - the ground truth
    clean/plate_NNNN.png  400 plates, no degradation
    normal/plate_NNNN.png 400 plates, mild blur/skew/noise
    hard/plate_NNNN.png   400 plates, heavy blur/skew/noise

Ground truth is free because we choose the text (ML-37). The images are
gitignored - they regenerate from this script - but manifest.csv is
committed as evidence of what was measured.
"""

from __future__ import annotations

import argparse

from anpr.config import GENERATED_DATA_DIR, ensure_dirs, load_config, set_seeds
from anpr.data.plates import build_persisted_test_set, monospace_font


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=None,
                        help="plates per tier; defaults to cfg.plates.n_eval_plates")
    parser.add_argument("--out", default=str(GENERATED_DATA_DIR))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    ensure_dirs()

    n = args.n or cfg["plates"]["n_eval_plates"]
    font = monospace_font()

    print(f"font        : {font}")
    print(f"plates/tier : {n}")
    print(f"seed        : {cfg['seed']}")
    print()

    manifest = build_persisted_test_set(
        n_per_tier=n,
        out_dir=args.out,
        font_path=font,
        length=cfg["plates"]["plate_len"],
    )

    print(f"\nmanifest -> {manifest}")
    print("Next: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
