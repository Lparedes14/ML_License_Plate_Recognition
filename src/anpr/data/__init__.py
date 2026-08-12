"""Data layer - loading, validating, splitting and generating images.

Owner: Data role (§7). Tickets: ML-36 (done), ML-37, ML-39, ML-40, ML-41.

MODULE MAP
    contract.py  the ONE definition of a valid model input. Read this first -
                 everything else depends on it.
    guards.py    orientation guard: makes sideways EMNIST impossible
    emnist.py    three-route loader with provenance
    labels.py    62 -> 36 case handling, class imbalance handling
    splits.py    train/val/test split plus leakage proof
    pipeline.py  tf.data pipelines and augmentation
    plates.py    synthetic plate generation  [NOT IMPLEMENTED - ML-37]

TYPICAL USE
    from anpr.config import load_config, set_seeds
    from anpr.data import load_emnist, apply_case_strategy, make_splits

    cfg = load_config()
    set_seeds(cfg["seed"])
    X, y_raw, idx = load_emnist("train", cfg["data"]["n_train_max"], cfg["seed"])
    X, y = apply_case_strategy(X, y_raw, cfg["data"]["case_strategy"])
"""

from anpr.data.contract import (
    INPUT_SPEC,
    assert_input_contract,
    normalize,
    to_canonical_uint8,
)
from anpr.data.emnist import (
    PROVENANCE,
    load_emnist,
    print_provenance,
    save_provenance,
)
from anpr.data.guards import (
    assert_upright,
    calibrate_guard,
    ink_profile,
    prove_guard_fires,
)
from anpr.data.labels import (
    apply_case_strategy,
    class_distribution_report,
    compute_class_weights,
    drop_lowercase,
    load_class_map,
    merge_case,
    resample_to_median,
    save_class_map,
)
from anpr.data.splits import make_splits, save_split_manifest, verify_splits

__all__ = [
    # contract
    "INPUT_SPEC", "to_canonical_uint8", "normalize", "assert_input_contract",
    # guards
    "ink_profile", "assert_upright", "prove_guard_fires", "calibrate_guard",
    # emnist
    "load_emnist", "PROVENANCE", "save_provenance", "print_provenance",
    # labels
    "merge_case", "drop_lowercase", "apply_case_strategy",
    "compute_class_weights", "resample_to_median", "class_distribution_report",
    "save_class_map", "load_class_map",
    # splits
    "make_splits", "verify_splits", "save_split_manifest",
]
