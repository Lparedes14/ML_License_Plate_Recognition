"""Train / validation / test splitting, and proof that they do not leak.

WHERE EACH SPLIT COMES FROM
    train + validation : carved from the EMNIST TRAIN file
    test               : the EMNIST TEST file, a physically separate download

    That is a stronger arrangement than a three-way cut of one file, and it
    means the test metric is not measuring memorisation of the training file.

WHY THE VERIFICATION IS NOT OPTIONAL
    A leaked split inflates accuracy and cannot be detected by looking at the
    number - a leaked 94% looks exactly like an honest 94%. §1 of the brief:
    "a team claiming 99% they cannot reproduce" scores worse than an honest
    61%. So we check rather than assert, in four ways:

      1. index disjointness   train vs val share a source file, so row
                              indices are directly comparable
      2. conservation         nothing invented, nothing silently dropped
      3. content duplicates   train/val vs test come from different files, so
                              indices mean nothing - compare image BYTES
      4. proportion drift     stratification actually preserved the classes

Ported from ML_Project_Group_8.ipynb cells 25-26 (ML-7, ML-40).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anpr.config import IDX2CHAR


def make_splits(X, y, orig_idx, val_fraction: float, seed: int):
    """Carve a stratified validation set out of the training data.

    Args:
        X, y, orig_idx: The EMNIST train split and its source row positions.
        val_fraction: Share held out for validation, e.g. 0.10.
        seed: Fixed seed. Record it with any result - changing it changes
            every number downstream.

    Returns:
        (X_tr, X_va, y_tr, y_va, idx_tr, idx_va)
    """
    return train_test_split(
        X, y, orig_idx,
        test_size=val_fraction,
        random_state=seed,   # fixed seed -> identical splits across reruns
        stratify=y,          # preserves class proportions, which matters a
                             # great deal when 'Q' is 0.8% of the data
    )


def _content_keys(X: np.ndarray) -> np.ndarray:
    """One opaque key per image, for exact-duplicate detection.

    Views each flattened image as a single void-typed scalar so NumPy's fast
    set operations (`intersect1d`) can compare whole images at once. Far
    quicker than hashing each image individually, and exact - no collisions,
    because it compares the actual bytes.

    Args:
        X: Image stack.

    Returns:
        1-D array of per-image keys.
    """
    a = np.ascontiguousarray(X.reshape(len(X), -1))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def verify_splits(
    X_tr, X_va, X_te, y_tr, y_va, y_te,
    idx_tr, idx_va, idx_te, idx_tr_full,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run every leakage and stratification check. Raise on anything fatal.

    Args:
        X_tr, X_va, X_te: The three image splits.
        y_tr, y_va, y_te: Their labels.
        idx_tr, idx_va, idx_te: Source row positions.
        idx_tr_full: Row positions of the full train split, for conservation.
        seed: Recorded in the report.

    Returns:
        (report dict, proportion table). The report goes into the provenance
        JSON; the table drives the split-proportions figure.

    Raises:
        AssertionError: on a genuine leak or failed stratification. Note that
            train/test duplicates are WARNED about rather than raised - those
            are a property of the EMNIST source files, not of our split code,
            and hard-failing on someone else's bug would just block us.
    """
    report: dict[str, Any] = {}

    # --- 1. index disjointness (the real test for train vs val) -----------
    inter = np.intersect1d(idx_tr, idx_va)
    report["train_val_index_overlap"] = int(len(inter))
    if len(inter):
        raise AssertionError(
            f"SPLIT LEAK: {len(inter)} indices appear in BOTH train and "
            f"validation. Examples: {inter[:10].tolist()}"
        )

    # --- 2. conservation --------------------------------------------------
    assert len(idx_tr) + len(idx_va) == len(idx_tr_full), "train+val != source rows"
    assert len(np.union1d(idx_tr, idx_va)) == len(idx_tr_full), "union != source rows"
    report["conservation"] = "ok"

    # --- 3. content duplicates across files -------------------------------
    # Index numbers are not comparable between the train and test FILES, so
    # compare the images themselves. This is the stronger check.
    ktr, kva, kte = _content_keys(X_tr), _content_keys(X_va), _content_keys(X_te)
    dup_tr_te = int(len(np.intersect1d(ktr, kte)))
    dup_va_te = int(len(np.intersect1d(kva, kte)))
    dup_tr_va = int(len(np.intersect1d(ktr, kva)))
    report.update(dup_train_test=dup_tr_te, dup_val_test=dup_va_te,
                  dup_train_val=dup_tr_va)

    # --- 4. class proportions preserved -----------------------------------
    def props(y):
        return pd.Series(y).value_counts(normalize=True).sort_index()

    tab = pd.DataFrame({"train": props(y_tr), "val": props(y_va), "test": props(y_te)})
    tab.index = [IDX2CHAR.get(i, i) for i in tab.index]
    tab["val-train (pp)"] = (tab["val"] - tab["train"]) * 100
    tab["test-train (pp)"] = (tab["test"] - tab["train"]) * 100

    max_val_dev = float(tab["val-train (pp)"].abs().max())
    max_test_dev = float(tab["test-train (pp)"].abs().max())
    report["max_val_deviation_pp"] = round(max_val_dev, 4)
    report["max_test_deviation_pp"] = round(max_test_dev, 4)

    # --- 5. reproducibility fingerprints ----------------------------------
    report["seed"] = seed
    for nm, arr in (("train", idx_tr), ("val", idx_va), ("test", idx_te)):
        report[f"sha256_{nm}_idx"] = hashlib.sha256(np.sort(arr).tobytes()).hexdigest()[:16]
        report[f"n_{nm}"] = int(len(arr))

    # --- output -----------------------------------------------------------
    print("SPLIT VERIFICATION")
    print(f"  train/val index overlap : {report['train_val_index_overlap']}  (must be 0)")
    print(f"  conservation            : {report['conservation']}")
    print(f"  max class-proportion drift, val  vs train : {max_val_dev:.3f} pp")
    print(f"  max class-proportion drift, test vs train : {max_test_dev:.3f} pp")
    print(f"  exact duplicate images  : train~val {dup_tr_va} | "
          f"train~test {dup_tr_te} | val~test {dup_va_te}")
    print("\n  split fingerprints (identical across reruns => identical splits):")
    for nm in ("train", "val", "test"):
        print(f"    {nm:<5} n={report[f'n_{nm}']:>7,}  sha256={report[f'sha256_{nm}_idx']}")

    if max_val_dev > 0.5:
        raise AssertionError(
            f"stratification failed: val drifts {max_val_dev:.2f} pp from train"
        )
    if dup_tr_va:
        raise AssertionError(f"{dup_tr_va} identical images in both train and val")

    if dup_tr_te or dup_va_te:
        print(f"\n  NOTE: {dup_tr_te + dup_va_te} images are byte-identical between "
              "the EMNIST train and test files.")
        print("  That is inherited from the dataset, not caused by our split logic.")
        print("  Report it, and if the count is large, deduplicate before trusting")
        print("  the test metric.")
    else:
        print("\n  No leakage detected between any pair of splits.")

    return report, tab


def save_split_manifest(
    path: str | Path, report: dict, val_fraction: float, seed: int,
    idx_tr, idx_va,
) -> Path:
    """Persist the split fingerprints so a result can be reproduced exactly.

    Args:
        path: Destination JSON.
        report: Output of `verify_splits`.
        val_fraction, seed: The settings that produced the split.
        idx_tr, idx_va: Index arrays; a short sample is stored for eyeballing.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": seed,
            "val_fraction": val_fraction,
            "stratified": True,
            **report,
            "train_idx_sample": np.asarray(idx_tr)[:20].tolist(),
            "val_idx_sample": np.asarray(idx_va)[:20].tolist(),
        }, fh, indent=2)
    return path
