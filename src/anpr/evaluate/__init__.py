"""Measurement - honest numbers with their conditions attached.

Owner: QA role (§7). Tickets: ML-38, ML-49, ML-50, ML-51.

    metrics.py     char/plate accuracy, segmentation rate, EvalResult
    confusion.py   confusion matrix, CONFUSABLE_PAIRS check, per-class recall
    conditions.py  accuracy across the three quality tiers

THE RULE: no number leaves this package without its conditions. §5 makes
reporting accuracy without stating conditions a -5 deduction, which is why
`EvalResult.to_sentence()` always includes the tier and the sample size.
"""

from anpr.evaluate.conditions import (
    TierResult,
    evaluate_all_tiers,
    evaluate_tier,
    save_results,
)
from anpr.evaluate.confusion import (
    check_predicted_confusions,
    confusion_matrix,
    per_class_accuracy,
    top_confusions,
)
from anpr.evaluate.metrics import EvalResult, character_accuracy, evaluate_reads

__all__ = [
    "EvalResult", "character_accuracy", "evaluate_reads",
    "confusion_matrix", "top_confusions", "check_predicted_confusions",
    "per_class_accuracy",
    "TierResult", "evaluate_tier", "evaluate_all_tiers", "save_results",
]
