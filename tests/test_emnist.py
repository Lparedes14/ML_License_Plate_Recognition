"""The row-count guard on the local-CSV route.

This is the check that already caught a real bug: our first Colab run read a
train file with 129,461 rows where ByClass has 697,932 - a silently truncated
upload that would otherwise have trained on 18% of the data with no error.
It had no automated test protecting it until now.

    pytest tests/test_emnist.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anpr.data import emnist as emnist_module
from anpr.data.emnist import _load_kaggle_csv


def _write_fake_csv(path, n_rows: int, n_cols: int = 785) -> None:
    """A minimal, structurally valid EMNIST-shaped CSV: label + pixel columns.

    Labels are kept in 0-9 (safely under RAW_CLASSES=62) so this file can also
    pass every check beyond the row-count gate, which is what
    `test_sufficient_rows_pass_the_gate` relies on.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=n_rows)
    pixels = rng.integers(0, 256, size=(n_rows, n_cols - 1))
    pd.DataFrame(np.column_stack([labels, pixels])).to_csv(
        path, header=False, index=False
    )


@pytest.fixture
def small_expected(monkeypatch):
    """Shrink EXPECTED_N so the test can use small files instead of
    generating something on the order of 100k+ real EMNIST-sized rows."""
    monkeypatch.setattr(emnist_module, "EXPECTED_N", {"train": 100, "test": 100})


def test_truncated_file_is_rejected(tmp_path, monkeypatch, small_expected):
    """Below 95% of the expected row count must raise, not train on a
    silent fraction of the data - this is the exact failure mode from
    Week 1 (see docs/spike.md)."""
    bad_csv = tmp_path / "emnist-byclass-test.csv"
    _write_fake_csv(bad_csv, n_rows=40)          # 40 < 100 * 0.95 = 95

    monkeypatch.setattr(emnist_module, "_find_emnist_csv", lambda *a, **k: bad_csv)

    with pytest.raises(ValueError, match="truncated|different split"):
        _load_kaggle_csv("test", max_items=None)


def test_error_names_actual_and_expected_row_counts(tmp_path, monkeypatch, small_expected):
    """The error must be actionable - state what was found and what was
    expected, not just 'something is wrong'."""
    bad_csv = tmp_path / "emnist-byclass-test.csv"
    _write_fake_csv(bad_csv, n_rows=40)

    monkeypatch.setattr(emnist_module, "_find_emnist_csv", lambda *a, **k: bad_csv)

    with pytest.raises(ValueError, match=r"40.*100|has 40 rows"):
        _load_kaggle_csv("test", max_items=None)


def test_sufficient_rows_pass_the_gate(tmp_path, monkeypatch, small_expected):
    """The other side of the same check: a file at/above the 95% threshold
    is NOT rejected by the row-count gate and loads successfully."""
    ok_csv = tmp_path / "emnist-byclass-test.csv"
    _write_fake_csv(ok_csv, n_rows=96)           # 96 >= 100 * 0.95 = 95

    monkeypatch.setattr(emnist_module, "_find_emnist_csv", lambda *a, **k: ok_csv)

    pixels, labels, detail = _load_kaggle_csv("test", max_items=None)

    assert len(labels) == 96
    assert pixels.shape == (96, 784)
    assert detail["library"].startswith("pandas")


def test_exactly_at_threshold_is_accepted(tmp_path, monkeypatch, small_expected):
    """Boundary check: exactly 95 rows out of 100 expected (== 95%) must pass,
    not fail, since the guard is `rows < expected * 0.95`, not `<=`."""
    boundary_csv = tmp_path / "emnist-byclass-test.csv"
    _write_fake_csv(boundary_csv, n_rows=95)

    monkeypatch.setattr(emnist_module, "_find_emnist_csv", lambda *a, **k: boundary_csv)

    pixels, labels, _ = _load_kaggle_csv("test", max_items=None)
    assert len(labels) == 95
