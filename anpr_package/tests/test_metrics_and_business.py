"""Metrics counting rules, and the cost model's arithmetic.

The counting rules matter as much as the model: §4 requires segmentation and
recognition failures to be counted separately, and getting that wrong makes
every reported number wrong in a way no amount of retraining shows up.

    pytest tests/test_metrics_and_business.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from anpr.business import CostModel, recommend_threshold, sweep_threshold
from anpr.evaluate.metrics import character_accuracy, evaluate_reads


@dataclass
class FakeRead:
    """Minimal stand-in for PlateRead - evaluate_reads only reads `.text`."""
    text: str
    plate_confidence: float = 1.0
    char_confidences: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Character accuracy
# --------------------------------------------------------------------------
def test_character_accuracy_counts_positionally():
    assert character_accuracy("ABC1234", "ABC1234") == (7, 7)
    assert character_accuracy("ABC1235", "ABC1234") == (6, 7)


def test_character_accuracy_undefined_on_length_mismatch():
    """A length mismatch is a SEGMENTATION failure, not a recognition score.

    Returning (0, 0) keeps it out of the character-accuracy denominator
    entirely, so character accuracy measures the classifier and nothing else.
    """
    assert character_accuracy("ABC123", "ABC1234") == (0, 0)


# --------------------------------------------------------------------------
# The counting rule that §4 insists on
# --------------------------------------------------------------------------
def test_segmentation_failure_excluded_from_character_accuracy():
    reads = [FakeRead("ABC1234"), FakeRead("ABC123")]   # second dropped a char
    truths = ["ABC1234", "XYZ9876"]

    r = evaluate_reads(reads, truths, tier="B")

    assert r.segmentation_rate == 0.5
    assert r.n_segmentation_failures == 1
    # Only the first plate contributed characters, and it was perfect.
    assert r.char_accuracy == 1.0
    # But plate accuracy counts the failure as wrong - Meridian is billed
    # either way.
    assert r.plate_accuracy == 0.5


def test_plate_accuracy_is_all_or_nothing():
    reads = [FakeRead("ABC1234"), FakeRead("ABC1235")]
    truths = ["ABC1234", "ABC1234"]
    r = evaluate_reads(reads, truths, tier="A")
    assert r.plate_accuracy == 0.5
    assert r.char_accuracy == pytest.approx(13 / 14)     # 6/7 wrong on one char


def test_result_sentence_always_states_conditions():
    """Reporting a number without conditions is a -5 deduction (§5)."""
    r = evaluate_reads([FakeRead("AB")], ["AB"], tier="C")
    sentence = r.to_sentence()
    assert "Tier C" in sentence and "1 plates" in sentence


def test_misaligned_evaluation_raises():
    with pytest.raises(ValueError):
        evaluate_reads([FakeRead("AB")], ["AB", "CD"], tier="A")


# --------------------------------------------------------------------------
# Cost model - check against the brief's own arithmetic
# --------------------------------------------------------------------------
def test_matches_the_brief_30000_wrong_bills():
    """§2: 'At 4,200 vehicles per day, a 2% plate-level error rate is roughly
    30,000 wrong bills a year.' If this drifts, the model is wrong."""
    cm = CostModel()
    assert cm.vehicles_per_year == 4200 * 365
    wrong_bills = cm.vehicles_per_year * 0.02
    assert 29_000 < wrong_bills < 32_000


def test_dispenser_capex_across_all_sites():
    cm = CostModel()
    assert cm.total_dispenser_capex == 14_000 * 34       # $476,000


def test_dispute_cost_scales_with_error_rate():
    cm = CostModel()
    assert cm.annual_dispute_cost(0.04) == pytest.approx(
        cm.annual_dispute_cost(0.02) * 2
    )


# --------------------------------------------------------------------------
# Trust policy
# --------------------------------------------------------------------------
def test_threshold_sweep_finds_a_cheaper_policy_than_accepting_everything():
    """The core claim of the business analysis.

    A model whose confidence is informative should always beat blanket
    auto-acceptance, because routing the doubtful reads to a person costs
    less than paying $8.50 for each of their disputes.
    """
    # 80 confident-and-correct reads, 20 unconfident-and-wrong ones.
    confidences = [0.95] * 80 + [0.40] * 20
    correct = [True] * 80 + [False] * 20

    curve = sweep_threshold(confidences, correct, CostModel())
    best = recommend_threshold(curve)

    accept_all = curve[0]                                # threshold 0.0
    assert best.annual_total_cost < accept_all.annual_total_cost
    # The optimum should sit between the two confidence clusters.
    assert 0.40 < best.threshold <= 0.95


def test_sweep_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        sweep_threshold([0.9, 0.8], [True], CostModel())


def test_sweep_rejects_empty_input():
    with pytest.raises(ValueError):
        sweep_threshold([], [], CostModel())
