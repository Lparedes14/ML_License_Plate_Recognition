"""Train the character classifier.

    python scripts/train.py                # the CNN
    python scripts/train.py --model mlp    # the dense baseline, for comparison
    python scripts/train.py --both         # both, so the comparison is real

Writes to artifacts/models/:
    <name>.keras            weights + architecture
    <name>.classmap.json    index -> character (ML-46: never separate these)
    <name>.history.json     per-epoch curves for the results document

Run scripts/prepare_data.py first - it verifies the data this trains on.
"""

from __future__ import annotations

import argparse

import numpy as np

from anpr.config import load_config, set_seeds, ensure_dirs
from anpr.data import (
    apply_case_strategy,
    load_emnist,
    make_splits,
    resample_to_median,
)
from anpr.data.pipeline import make_dataset
from anpr.models import build_baseline_mlp, build_cnn, compile_model, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--both", action="store_true",
                        help="train both, so the CNN's number has a control")
    parser.add_argument("--no-augment", action="store_true",
                        help="disable augmentation, to measure what it buys")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    ensure_dirs()

    # --- data -------------------------------------------------------------
    print("Loading EMNIST ...")
    X_full, y_raw, idx_full = load_emnist(
        "train", cfg["data"]["n_train_max"], cfg["seed"]
    )
    X_full, y_full = apply_case_strategy(X_full, y_raw, cfg["data"]["case_strategy"])

    # Resampling changes the ARRAYS, so it happens before the split. Class
    # weighting changes the LOSS, so it happens at fit time in train_model().
    if cfg["data"]["imbalance_strategy"] == "resampled":
        keep = resample_to_median(y_full, cfg["seed"])
        print(f"  resampled to median class count: {len(X_full):,} -> {len(keep):,}")
        X_full, y_full, idx_full = X_full[keep], y_full[keep], idx_full[keep]

    X_tr, X_va, y_tr, y_va, _, _ = make_splits(
        X_full, y_full, idx_full, cfg["data"]["val_fraction"], cfg["seed"]
    )
    print(f"  train {len(X_tr):,} | val {len(X_va):,}")

    train_ds = make_dataset(
        X_tr, y_tr, training=True, batch_size=cfg["training"]["batch_size"],
        seed=cfg["seed"], augment=not args.no_augment,
    )
    val_ds = make_dataset(
        X_va, y_va, training=False, batch_size=cfg["training"]["batch_size"],
        seed=cfg["seed"],
    )

    # --- train ------------------------------------------------------------
    targets = ["mlp", "cnn"] if args.both else [args.model]
    for kind in targets:
        print("\n" + "=" * 70)
        print(f"Training: {kind}")
        print("=" * 70)

        if kind == "cnn":
            model = build_cnn(cfg["data"]["n_classes"])
            epochs = cfg["training"]["epochs_cnn"]
        else:
            model = build_baseline_mlp(cfg["data"]["n_classes"])
            epochs = cfg["training"]["epochs_mlp"]

        compile_model(model, cfg["training"]["lr"])
        model.summary()          # print it: the demo will ask about layer count

        train_model(
            model, train_ds, val_ds,
            epochs=epochs,
            y_train=y_tr,
            imbalance_strategy=cfg["data"]["imbalance_strategy"],
            patience=cfg["training"]["early_stopping_patience"],
            name=f"plate_{kind}",
            case_strategy=cfg["data"]["case_strategy"],
        )

    print("\nNext: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
