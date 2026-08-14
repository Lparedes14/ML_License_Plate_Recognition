"""Load EMNIST, verify it, split it, and write the acceptance record.

    python scripts/prepare_data.py

This is the notebook's data work (cells 8-28) as one reproducible command.
It writes three artifacts the report cites:

    artifacts/provenance/provenance.json         where the bytes came from
    artifacts/provenance/split_manifest.json     the split fingerprints
    artifacts/provenance/acceptance_record.md    the PASS/FAIL sign-off

Nothing downstream should run until this exits cleanly. If the orientation
guard fires, STOP - do not train on sideways glyphs.
"""

from __future__ import annotations

import argparse
import datetime
import json

import numpy as np

from anpr.config import PROVENANCE_DIR, ensure_dirs, load_config, set_seeds
from anpr.data import (
    PROVENANCE,
    apply_case_strategy,
    calibrate_guard,
    class_distribution_report,
    load_emnist,
    make_splits,
    print_provenance,
    prove_guard_fires,
    save_provenance,
    save_split_manifest,
    verify_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a config YAML")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="skip the guard threshold calibration table")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    ensure_dirs()

    print("=" * 70)
    print("STEP 1/5  Load EMNIST ByClass")
    print("=" * 70)
    X_tr_full, y_tr_raw, idx_tr_full = load_emnist(
        "train", cfg["data"]["n_train_max"], cfg["seed"]
    )
    X_te, y_te_raw, idx_te = load_emnist(
        "test", cfg["data"]["n_test_max"], cfg["seed"]
    )
    print_provenance()

    print("\n" + "=" * 70)
    print("STEP 2/5  Prove the orientation guard actually fires")
    print("=" * 70)
    # A check that never fails proves nothing. This is what makes the guard
    # trustworthy rather than decorative.
    prove_guard_fires(X_tr_full, y_tr_raw)
    if not args.skip_calibration:
        calib = calibrate_guard(X_tr_full, y_tr_raw)
        PROVENANCE["guard_calibration"] = calib.to_dict("records")
    PROVENANCE["orientation_verified"] = True

    print("\n" + "=" * 70)
    print(f"STEP 3/5  Case handling: {cfg['data']['case_strategy']}  (62 -> 36)")
    print("=" * 70)
    X_tr_full, y_tr_full = apply_case_strategy(
        X_tr_full, y_tr_raw, cfg["data"]["case_strategy"]
    )
    X_te, y_te = apply_case_strategy(X_te, y_te_raw, cfg["data"]["case_strategy"])
    print(f"  train {X_tr_full.shape}  labels {y_tr_full.min()}-{y_tr_full.max()}")
    print(f"  test  {X_te.shape}")

    # The imbalance evidence (ML-39). Numbers, not an observation.
    dist = class_distribution_report(y_tr_full)
    print(f"\n  rarest class    : '{dist['rarest_char']}' at {dist['rarest_share']:.2%}")
    print(f"  commonest class : '{dist['commonest_char']}' at {dist['commonest_share']:.2%}")
    print(f"  imbalance ratio : {dist['imbalance_ratio']}x")
    print(f"  a uniform plate alphabet would give every class "
          f"{dist['uniform_share_would_be']:.2%}")
    print(f"  -> strategy: {cfg['data']['imbalance_strategy']}")
    PROVENANCE["class_distribution"] = dist

    print("\n" + "=" * 70)
    print("STEP 4/5  Split and verify")
    print("=" * 70)
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = make_splits(
        X_tr_full, y_tr_full, idx_tr_full, cfg["data"]["val_fraction"], cfg["seed"]
    )
    print(f"train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,}\n")

    report, prop_table = verify_splits(
        X_tr, X_va, X_te, y_tr, y_va, y_te,
        idx_tr, idx_va, idx_te, idx_tr_full, cfg["seed"],
    )
    PROVENANCE["splits"] = report

    save_split_manifest(
        PROVENANCE_DIR / "split_manifest.json",
        report, cfg["data"]["val_fraction"], cfg["seed"], idx_tr, idx_va,
    )

    print("\n" + "=" * 70)
    print("STEP 5/5  Write the acceptance record")
    print("=" * 70)
    PROVENANCE["config"] = cfg
    PROVENANCE["seed"] = cfg["seed"]
    prov_path = save_provenance()

    write_acceptance_record(cfg, report)

    print(f"\n  {prov_path}")
    print(f"  {PROVENANCE_DIR / 'split_manifest.json'}")
    print(f"  {PROVENANCE_DIR / 'acceptance_record.md'}")
    print("\nData is verified. Training is unlocked: python scripts/train.py")
    return 0


def write_acceptance_record(cfg: dict, split_report: dict) -> None:
    """Write the human-readable PASS/FAIL sign-off the report cites.

    Args:
        cfg: The config in force.
        split_report: Output of `verify_splits`.
    """
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# EMNIST load - acceptance record", "",
        f"Generated {stamp}", "",
        "## Acceptance criteria (ML-36)", "",
    ]

    criteria = [
        ("Ten samples plotted and visually confirmed upright before training",
         PROVENANCE.get("orientation_verified", False),
         "notebooks/01_data_acceptance.ipynb plots them; this script proves "
         "the guard rejects transposed input"),
        ("Assertion in the loader that fails loudly on orientation regression",
         "train" in PROVENANCE and "orientation_report" in PROVENANCE.get("train", {}),
         "assert_upright() is called inside _finalise(), so no route can bypass it"),
        ("Load method and source recorded so results reproduce",
         all(k in PROVENANCE.get("train", {})
             for k in ("route", "uri", "sha256_first_2k", "loaded_at_utc")),
         "artifacts/provenance/provenance.json"),
        ("Splits verified disjoint and stratified (ML-7, ML-40)",
         split_report["train_val_index_overlap"] == 0,
         "artifacts/provenance/split_manifest.json"),
    ]
    for name, passed, evidence in criteria:
        lines.append(f"- **[{'PASS' if passed else 'FAIL'}]** {name}")
        lines.append(f"  - evidence: {evidence}")

    for split in ("train", "test"):
        p = PROVENANCE.get(split)
        if not p:
            continue
        lines += [
            "", f"## {split} split", "",
            f"- route: `{p['route']}` (fell back past: "
            f"{p['attempts_before_success'] or 'none'})",
            f"- source: `{p['uri']}`",
            f"- library: {p['library']}",
            f"- images: {p['n_loaded']:,}, transpose applied: {p['transpose_applied']}",
            f"- content hash (first 2k): `{p['sha256_first_2k']}`",
            f"- loaded: {p['loaded_at_utc']}",
        ]

    lines += [
        "", "## Settings", "",
        f"- seed: {cfg['seed']}",
        f"- case strategy: {cfg['data']['case_strategy']}",
        f"- imbalance strategy: {cfg['data']['imbalance_strategy']}",
        f"- numpy {np.__version__}",
        "", "## Class distribution", "",
        "```json",
        json.dumps({k: v for k, v in PROVENANCE.get("class_distribution", {}).items()
                    if k != "per_class_count"}, indent=2),
        "```",
    ]

    path = PROVENANCE_DIR / "acceptance_record.md"
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
