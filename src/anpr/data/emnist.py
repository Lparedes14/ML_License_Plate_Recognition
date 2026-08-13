"""EMNIST ByClass loader with three independent routes and full provenance.

WHY THREE ROUTES
    The NIST host refuses downloads often enough that a single-route loader
    will fail on demo day. So we try, in order:

      1. kaggle_csv   local CSV or .csv.zip on disk. No network, fastest,
                      fully reproducible - the route to prefer for a demo.
      2. torchvision  downloads from the NIST mirror.
      3. tfds         tensorflow_datasets.

    Whichever succeeds is recorded, so a result can always be traced to the
    bytes that produced it.

WHY PROVENANCE
    Reported accuracy is meaningless without knowing what it was measured on.
    Every load records: the route, the source URI, the library version, the
    row count, whether a transpose was applied, and a SHA-256 of the first
    2000 images. Two runs with the same hash saw the same data. Written to
    `artifacts/provenance/provenance.json` and cited in the report.

A REAL FAILURE THIS CAUGHT
    Our first Colab run reported 129,461 training rows where ByClass has
    697,932 - an upload that had silently stopped partway. The row-count
    check caught it and the loader fell through to torchvision rather than
    quietly training on 18% of the data. Keep that check.

Ported from ML_Project_Group_8.ipynb cells 8, 10, 11 (ML-36, ML-58).
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anpr.config import PROVENANCE_DIR, RAW_CLASSES, RAW_DATA_DIR
from anpr.data.contract import to_canonical_uint8
from anpr.data.guards import assert_upright

# Row counts of the official ByClass split. Used to detect truncated files.
EXPECTED_N = {"train": 697_932, "test": 116_323}

# Orientation policy.
#   "auto"     try transposed and untransposed, keep whichever passes the guard
#   True/False pin it (the guard still has to pass)
#
# "auto" rather than a hard-coded transpose because the routes are maintained
# by different people: a library could start correcting the layout upstream,
# and we would rather detect that than be broken by it. EXPECTED_TRANSPOSE is
# what we believe is true - a mismatch warns loudly and is recorded, never
# silently accepted.
ORIENTATION_MODE: str | bool = "auto"
EXPECTED_TRANSPOSE = True

# Module-level provenance accumulator. Written to disk by `save_provenance()`.
PROVENANCE: dict[str, Any] = {"orientation_verified": False}


# ==========================================================================
# Route 1/3 - local CSV (preferred: no network, reproducible)
# ==========================================================================
def _find_emnist_csv(split: str, search_dirs: list[Path] | None = None) -> Path:
    """Locate emnist-byclass-{split}.csv (or .csv.zip), listing what IS there.

    Accepts the zipped form directly - pandas reads `.csv.zip` natively - so
    there is no need to unzip a 1.35 GB file just to read it. Unzipping is
    also where the truncation bug came from.

    Args:
        split: "train" or "test".
        search_dirs: Directories to search. Defaults to `data/raw/`, then
            Colab's /content, then the CWD.

    Returns:
        Path to the first match.

    Raises:
        FileNotFoundError: naming every directory searched and what was found
            in the primary one, so the fix is obvious.
    """
    names = [
        f"emnist-byclass-{split}.csv",
        f"emnist-byclass-{split}.csv.zip",
        f"emnist-byclass-{split}.CSV",
    ]
    roots = search_dirs or [RAW_DATA_DIR, Path("/content"), Path(".")]

    seen: set[str] = set()
    for root in roots:
        root = Path(root)
        if str(root) in seen or not root.is_dir():
            continue
        seen.add(str(root))
        for nm in names:
            hits = sorted(glob.glob(str(root / nm))) + \
                   sorted(glob.glob(str(root / "**" / nm), recursive=True))
            if hits:
                return Path(hits[0])

    present = sorted(p.name for p in RAW_DATA_DIR.glob("*"))[:15]
    raise FileNotFoundError(
        f"emnist-byclass-{split}.csv not found.\n"
        f"  searched: {sorted(seen)}\n"
        f"  files in {RAW_DATA_DIR}: {present or '(empty)'}\n"
        "  need: emnist-byclass-train.csv(.zip) and emnist-byclass-test.csv(.zip)\n"
        "  get them from kaggle.com/datasets/crawford/emnist and put them in "
        f"{RAW_DATA_DIR}"
    )


def _load_kaggle_csv(split: str, max_items: int | None):
    """Read the Kaggle CSV, validating the row count rather than trusting it.

    Args:
        split: "train" or "test".
        max_items: Target sample size, used to pre-filter while parsing so we
            never hold the full 1.35 GB train file in memory.

    Returns:
        (pixels uint8 (N,784), labels int32 (N,), detail dict).

    Raises:
        ValueError: on a truncated file, wrong column count, or labels that
            indicate a different EMNIST split.
    """
    path = _find_emnist_csv(split)
    print(f"  reading {path.name} ({path.stat().st_size / 1e6:,.0f} MB)")

    # Kaggle's files have no header, but mirrors vary. Probe rather than assume.
    probe = pd.read_csv(path, header=None, nrows=1)
    has_header = not pd.api.types.is_numeric_dtype(probe.iloc[:, 0])
    if has_header:
        print("  header row detected, skipping it")

    # Sample while parsing. The 1.25 factor over-samples slightly so the
    # stratified subsample downstream still has choices in every class.
    # No dtype=uint8 coercion here: a truncated final row yields NaN, and
    # forcing uint8 turns that into an opaque "Integer column has NA values".
    frac = min(1.0, (max_items or EXPECTED_N[split]) / EXPECTED_N[split] * 1.25)

    parts, rows = [], 0
    for chunk in pd.read_csv(path, header=0 if has_header else None, chunksize=100_000):
        rows += len(chunk)
        parts.append(chunk.sample(frac=frac, random_state=0) if frac < 1.0 else chunk)
    df = pd.concat(parts, ignore_index=True)

    # --- the check that caught our truncated upload -----------------------
    expected = EXPECTED_N[split]
    print(f"  {rows:,} rows read (file should contain {expected:,})")
    if rows < expected * 0.95:
        raise ValueError(
            f"{path.name} has {rows:,} rows but byclass-{split} has {expected:,}. "
            "The file is truncated or is a different split - the download or "
            "upload probably did not finish. Re-fetch it."
        )

    n_na = int(df.isna().any(axis=1).sum())
    if n_na:
        print(f"  !! {n_na} rows contain missing values - dropping them")
        df = df.dropna()

    if df.shape[1] != 785:
        raise ValueError(
            f"expected 785 columns (1 label + 784 pixels), got {df.shape[1]}. "
            "This is not the byclass CSV."
        )

    labels = df.iloc[:, 0].to_numpy().astype(np.int32)
    if labels.max() >= RAW_CLASSES:
        raise ValueError(
            f"labels reach {labels.max()}, expected < {RAW_CLASSES}. "
            "Looks like balanced/letters/digits rather than byclass."
        )

    pixels = np.clip(df.iloc[:, 1:].to_numpy(), 0, 255).astype(np.uint8)
    return pixels, labels, {
        "library": f"pandas {pd.__version__}",
        "uri": f"local CSV (kaggle:crawford/emnist) -> {path}",
        "prefiltered": bool(frac < 1.0),
    }


# ==========================================================================
# Route 2/3 - torchvision download
# ==========================================================================
def _load_torchvision(split: str, max_items: int | None):
    """Download via torchvision. The route that rescued us when the CSV failed."""
    import torchvision
    from torchvision import datasets

    ds = datasets.EMNIST(
        root=str(RAW_DATA_DIR / "emnist_torchvision"),
        split="byclass",
        train=(split == "train"),
        download=True,
    )
    return ds.data.numpy(), ds.targets.numpy(), {
        "library": f"torchvision {torchvision.__version__}",
        "uri": "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip (via torchvision)",
    }


# ==========================================================================
# Route 3/3 - tensorflow_datasets
# ==========================================================================
def _load_tfds(split: str, max_items: int | None, seed: int = 42):
    """Load via TFDS. Last resort - heaviest dependency, slowest cold start."""
    import tensorflow_datasets as tfds

    ds, info = tfds.load(
        "emnist/byclass", split=split, as_supervised=True,
        with_info=True, shuffle_files=True,
    )
    ds = ds.shuffle(100_000, seed=seed, reshuffle_each_iteration=False)
    if max_items:
        ds = ds.take(int(max_items * 1.05))

    xs, ys = [], []
    for xb, yb in tfds.as_numpy(ds.batch(8192)):
        xs.append(xb)
        ys.append(yb)

    return np.concatenate(xs), np.concatenate(ys), {
        "library": f"tensorflow_datasets {tfds.__version__}",
        "uri": f"tfds:emnist/byclass:{info.version}",
    }


# Order matters. Local CSV first (no network, fastest, reproducible for a demo),
# then tfds, then torchvision LAST.
#
# torchvision was our primary route in Week 1 and is now the fallback of last
# resort: it downloads from the NIST mirror, whose TLS certificate has been
# expired since at least August 2026, so it fails with CERTIFICATE_VERIFY_FAILED
# regardless of the local trust store. tfds pulls from a different mirror and is
# the route that actually succeeds today.
#
# Keeping torchvision in the list rather than deleting it: if NIST renews the
# certificate it starts working again, and the provenance record shows which
# route was actually used either way.
ROUTES: list[tuple[str, Callable]] = [
    ("kaggle_csv", _load_kaggle_csv),
    ("tfds", _load_tfds),
    ("torchvision", _load_torchvision),
]


# ==========================================================================
# Shared finalisation - runs on every route
# ==========================================================================
def _finalise(X, y, max_items: int | None, seed: int, mode=None):
    """Canonicalise, subsample, then let the guard decide the orientation.

    Every route funnels through here, so the input contract and the
    orientation guard apply no matter where the bytes came from. That is the
    point: a new route cannot accidentally skip validation.

    Args:
        X, y: Raw route output.
        max_items: Target size, or None for everything.
        seed: For the stratified subsample.
        mode: Orientation mode override; defaults to ORIENTATION_MODE.

    Returns:
        (X uint8 (N,28,28,1), y int32, orig_idx int64, meta dict).

    Raises:
        AssertionError: if no orientation passes the guard.
    """
    mode = ORIENTATION_MODE if mode is None else mode

    X = to_canonical_uint8(X)                 # one definition of "an image"
    y = np.asarray(y, np.int32).ravel()

    assert len(X) == len(y), f"length mismatch {len(X)} vs {len(y)}"
    assert y.min() >= 0 and y.max() < RAW_CLASSES, f"labels outside 0-{RAW_CLASSES - 1}"

    # Original row positions in the source file. Kept so the train/val split
    # can later be PROVEN disjoint by index - without these there is nothing
    # to check, only a promise.
    orig_idx = np.arange(len(X), dtype=np.int64)

    if max_items and len(X) > max_items:
        # Stratified, not random. A plain rng.choice matches class proportions
        # only in expectation, which is not what "preserving class
        # proportions" means - and with Q at 0.8% the variance is not small.
        keep, _ = train_test_split(
            orig_idx, train_size=max_items, stratify=y, random_state=seed
        )
        keep = np.sort(keep)
        X, y, orig_idx = X[keep], y[keep], orig_idx[keep]
        subsample = "stratified"
    else:
        subsample = "none (full split)"

    # --- orientation: try, verify, record ---------------------------------
    candidates = [True, False] if mode == "auto" else [bool(mode)]
    failures: dict[bool, list[str]] = {}

    for t in candidates:
        Xc = np.transpose(X, (0, 2, 1, 3)) if t else X
        try:
            report = assert_upright(Xc, y, verbose=False)
        except AssertionError as exc:
            failures[t] = str(exc).split("\n")[1:4]
            continue

        if t != EXPECTED_TRANSPOSE:
            print(
                f"  !! WARNING: this route needed transpose={t}, expected "
                f"{EXPECTED_TRANSPOSE}. The source's layout convention has "
                "changed. Recorded in provenance - explain it in the report."
            )
        return Xc, y, orig_idx, {
            "transpose_applied": t,
            "orientation_mode": mode,
            "subsample": subsample,
            "orientation_report": report,
        }

    raise AssertionError(
        "EMNIST ORIENTATION CHECK FAILED - refusing to return data.\n"
        f"Tried transpose={candidates} and no option produced upright glyphs.\n"
        + json.dumps(failures, indent=2, default=str)
        + "\n\nThis is not a transpose bug - both orientations failed, so the "
          "data itself is wrong. Check that you loaded emnist-BYCLASS (not "
          "balanced/letters/digits) and that labels are ordered 0-9, A-Z, a-z."
    )


# ==========================================================================
# Public entry point
# ==========================================================================
def load_emnist(
    split: str,
    max_items: int | None = None,
    seed: int = 42,
    prefer: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an EMNIST ByClass split, trying each route until one works.

    Args:
        split: "train" or "test".
        max_items: Stratified subsample size. None loads everything.
        seed: Seed for the subsample.
        prefer: Route order override, e.g. `["torchvision"]` to force one.

    Returns:
        X: uint8 (N, 28, 28, 1), upright, guard-verified.
        y: int32 RAW ByClass labels 0-61 (case merge happens later, in
           `labels.apply_case_strategy` - the guard needs raw labels).
        orig_idx: int64 row positions in the source file, for split checks.

    Raises:
        AssertionError: on orientation failure. Deliberately NOT caught as a
            fallback case - if glyphs are sideways we want to stop, not try
            another route and hope.
        RuntimeError: if every route fails.
    """
    order = prefer or [name for name, _ in ROUTES]
    attempts: list[dict[str, str]] = []

    for name in order:
        fn = dict(ROUTES)[name]
        print(f"[{split}] trying route: {name} ...")
        try:
            t0 = time.time()
            Xr, yr, detail = fn(split, max_items)
            X, y, oidx, ometa = _finalise(Xr, yr, max_items, seed)
            secs = time.time() - t0
            print(f"  OK  {len(X):,} images from '{name}' in {secs:.1f}s")

            PROVENANCE[split] = {
                "route": name,
                **detail,
                "n_loaded": int(len(X)),
                "raw_classes": RAW_CLASSES,
                **ometa,
                "seconds": round(secs, 1),
                "attempts_before_success": attempts,
                # Content fingerprints: two runs with matching hashes saw
                # identical bytes. This is what makes a metric reproducible.
                "sha256_first_2k": hashlib.sha256(X[:2000].tobytes()).hexdigest()[:16],
                "sha256_orig_idx": hashlib.sha256(oidx.tobytes()).hexdigest()[:16],
                "loaded_at_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds"),
            }
            return X, y, oidx

        except AssertionError:
            raise                       # orientation failure is never a fallback
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc)[:120]}"
            print(f"  FAILED ({msg}) -> next route")
            attempts.append({name: msg})

    raise RuntimeError(
        "All EMNIST routes failed.\n" + json.dumps(attempts, indent=2)
        + "\n\nManual fallback: download emnist-byclass-train.csv and "
          f"emnist-byclass-test.csv from kaggle.com/datasets/crawford/emnist "
          f"into {RAW_DATA_DIR}"
    )


def save_provenance(path: str | Path | None = None) -> Path:
    """Write the accumulated provenance to JSON. Cite this file in the report.

    Args:
        path: Destination. Defaults to
            `artifacts/provenance/provenance.json`.

    Returns:
        The path written.
    """
    path = Path(path) if path else PROVENANCE_DIR / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(PROVENANCE, fh, indent=2, default=str)
    return path


def print_provenance() -> None:
    """Print the one-screen summary that belongs in the demo."""
    print("\n" + "=" * 66)
    print("DATA PROVENANCE")
    print("=" * 66)
    for split in ("train", "test"):
        p = PROVENANCE.get(split)
        if not p:
            continue
        print(f"{split:>6}: route={p['route']:<12} n={p['n_loaded']:>7,}  "
              f"transpose={p['transpose_applied']}  hash={p['sha256_first_2k']}")
        print(f"        source: {p['uri']}")
        if p["attempts_before_success"]:
            print(f"        fell back after: {p['attempts_before_success']}")
    print("=" * 66)
