"""Business layer - what the numbers mean in dollars.

Owner: Business role (§7). Tickets: ML-52, ML-53.

    cost_model.py    Meridian's economics from §2 of the brief
    trust_policy.py  the auto-accept threshold sweep - THE central deliverable

This is a first-class package rather than a slide because 35 of the 100
points are business and presentation, and because the COO will push back on
the assumptions in Q&A. Recomputing live beats "we'd have to redo it".

    from anpr.business import CostModel, sweep_threshold, recommend_threshold

    cm = CostModel.from_config(cfg)
    curve = sweep_threshold(confidences, correct, cm)
    best = recommend_threshold(curve)
    print(policy_sentence(best, cm))
"""

from anpr.business.cost_model import CostModel
from anpr.business.trust_policy import (
    PolicyPoint,
    policy_sentence,
    recommend_threshold,
    sweep_threshold,
)

__all__ = [
    "CostModel",
    "PolicyPoint", "sweep_threshold", "recommend_threshold", "policy_sentence",
]
