"""The trust threshold: which reads we auto-accept, which go to a human.

Owner: Business role. Ticket: ML-52. THIS IS THE CENTRAL DELIVERABLE.

§2, in the COO's words:
    "the design question is not 'how accurate can we get' but 'which reads do
    we trust automatically, and which do we send to a human.'"

    So the answer to "is the model good enough?" is not a percentage. It is a
    threshold plus its cost. A model with 78% plate accuracy that knows WHEN
    it is unsure is more valuable to Meridian than an 85% model that is
    confidently wrong 15% of the time - because the first one can route its
    doubts to a person and the second cannot.

HOW IT WORKS
    Every read carries a confidence (`PlateRead.plate_confidence`, the
    minimum over its characters). Pick a threshold t:

        confidence >= t   ->  auto-accept, bill the customer.
                              If wrong: a dispute, at $8.50.
        confidence <  t   ->  route to a human.
                              Costs a review, but produces no wrong bill.

    Sweeping t and costing each point gives a curve with a minimum. That
    minimum, in dollars, is the recommendation - and it is what to lead with
    in the first minute of the demo (§8: "the recommendation, first").

WHAT TO EXPECT
    The optimum is usually NOT the threshold that maximises accuracy, and
    that is the interesting finding. Because a dispute costs several times a
    review, the economics favour routing generously. Being able to say "the
    cost-optimal policy reviews 22% of reads, which is more than felt
    intuitive, and here is the arithmetic" is exactly the understanding §10
    awards 15 points for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from anpr.business.cost_model import CostModel


@dataclass
class PolicyPoint:
    """The consequences of one threshold choice.

    Attributes:
        threshold: The confidence cut-off.
        auto_accept_rate: Share of reads handled without a human.
        manual_review_rate: Share routed to a person. 1 - auto_accept_rate.
        error_rate_among_accepted: Share of AUTO-ACCEPTED reads that are
            wrong. This is the number that generates disputes, and it is the
            one to quote to the COO - overall accuracy is not what she pays
            for.
        annual_dispute_cost: Dollars per year in wrong bills.
        annual_review_cost: Dollars per year in human review.
        annual_total_cost: Their sum. The quantity being minimised.
    """

    threshold: float
    auto_accept_rate: float
    manual_review_rate: float
    error_rate_among_accepted: float
    annual_dispute_cost: float
    annual_review_cost: float
    annual_total_cost: float


def sweep_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    cost_model: CostModel,
    n_steps: int = 101,
) -> list[PolicyPoint]:
    """Cost every threshold from 0 to 1 and return the whole curve.

    Args:
        confidences: `plate_confidence` for each evaluated plate.
        correct: Whether each read was entirely correct. Same order.
        cost_model: Meridian's economics.
        n_steps: Points on the sweep. 101 gives 0.01 resolution.

    Returns:
        A PolicyPoint per threshold, ascending. Plot
        `annual_total_cost` against `threshold` - that chart is the single
        most persuasive slide in the deck.

    Raises:
        ValueError: if the two sequences disagree in length.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    if len(confidences) != len(correct):
        raise ValueError(
            f"{len(confidences)} confidences but {len(correct)} outcomes - "
            "these must correspond one-to-one."
        )
    if len(confidences) == 0:
        raise ValueError("no reads to sweep over")

    n = len(confidences)
    points: list[PolicyPoint] = []

    for t in np.linspace(0.0, 1.0, n_steps):
        accepted = confidences >= t
        n_accepted = int(accepted.sum())

        auto_rate = n_accepted / n
        review_rate = 1.0 - auto_rate

        # Errors among ACCEPTED reads only. Reads sent to a human are assumed
        # corrected there, so they cost a review but never a dispute. That
        # assumption is optimistic - humans misread plates too - and should be
        # stated plainly in the business note.
        if n_accepted:
            err_rate = float((~correct[accepted]).sum() / n_accepted)
        else:
            err_rate = 0.0        # nothing accepted, so nothing billed wrong

        # Scale to the whole population: the dispute rate is per-vehicle, and
        # only the auto-accepted fraction can produce a dispute.
        population_error_rate = err_rate * auto_rate

        dispute = cost_model.annual_dispute_cost(population_error_rate)
        review = cost_model.annual_review_cost(review_rate)

        points.append(PolicyPoint(
            threshold=float(t),
            auto_accept_rate=auto_rate,
            manual_review_rate=review_rate,
            error_rate_among_accepted=err_rate,
            annual_dispute_cost=dispute,
            annual_review_cost=review,
            annual_total_cost=dispute + review,
        ))

    return points


def recommend_threshold(points: list[PolicyPoint]) -> PolicyPoint:
    """The cheapest point on the curve. This is the recommendation.

    Args:
        points: Output of `sweep_threshold`.

    Returns:
        The PolicyPoint with the lowest annual total cost.
    """
    return min(points, key=lambda p: p.annual_total_cost)


def policy_sentence(point: PolicyPoint, cost_model: CostModel) -> str:
    """The recommendation in one sentence a COO can act on.

    §8 says to lead with this. Practise saying it without notes.

    Args:
        point: The recommended PolicyPoint.
        cost_model: For the annual volumes.

    Returns:
        A sentence naming the threshold, the split, and the cost.
    """
    reviews = round(cost_model.vehicles_per_year * point.manual_review_rate)
    wrong = round(
        cost_model.vehicles_per_year * point.error_rate_among_accepted
        * point.auto_accept_rate
    )
    return (
        f"Auto-accept reads above {point.threshold:.2f} confidence "
        f"({point.auto_accept_rate:.0%} of traffic) and route the rest to a "
        f"person. That is about {reviews:,} reviews and {wrong:,} wrong bills "
        f"a year, costing roughly ${point.annual_total_cost:,.0f} annually "
        f"against ${cost_model.total_dispenser_capex:,.0f} of dispenser "
        "hardware avoided."
    )
