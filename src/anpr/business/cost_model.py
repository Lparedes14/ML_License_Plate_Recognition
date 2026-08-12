"""What a misread costs Meridian, in dollars.

Owner: Business role. Tickets: ML-52, ML-53.

§2 gives us the numbers and the framing:
    34 sites, ticket dispensers at ~$14,000 installed each
    lost-ticket disputes cost the call centre ~$8.50 per incident
    4,200 vehicles per day
    "a 2% plate-level error rate is roughly 30,000 wrong bills a year"

    "A misread is not a neutral event. It bills the wrong customer, producing
    a refund, a support call, and an annoyed regular."

WHY THIS IS EXECUTABLE CODE AND NOT A SLIDE
    35 of the 100 points are business and presentation. When the COO pushes
    back in Q&A - "what if disputes cost $12, not $8.50?" - recomputing live
    is a far better answer than "we'd have to redo the analysis". Keep this
    module and a notebook cell ready during the demo.

STATE YOUR ASSUMPTIONS
    `cost_per_manual_review` is OURS, not the brief's. It must be labelled as
    an assumption everywhere it appears. Presenting an assumption as a given
    is the kind of overclaiming §1 warns is "the fastest way to lose points".
"""

from __future__ import annotations

from dataclasses import dataclass

DAYS_PER_YEAR = 365


@dataclass
class CostModel:
    """Meridian's economics. Defaults come from §2 of the brief.

    Attributes:
        vehicles_per_day: Throughput across all sites.
        cost_per_dispute: Call-centre cost of one wrong bill. From the brief.
        cost_per_manual_review: Cost of a human checking one uncertain read.
            OUR ASSUMPTION - not in the brief. Label it as such.
        dispenser_capex: Installed cost of one ticket dispenser. From the brief.
        n_sites: Number of sites.
    """

    vehicles_per_day: int = 4200
    cost_per_dispute: float = 8.50
    cost_per_manual_review: float = 1.20     # ASSUMPTION
    dispenser_capex: float = 14000.0
    n_sites: int = 34

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        """Build from the `business:` block of config/default.yaml."""
        b = cfg["business"]
        return cls(
            vehicles_per_day=b["vehicles_per_day"],
            cost_per_dispute=b["cost_per_dispute"],
            cost_per_manual_review=b["cost_per_manual_review"],
            dispenser_capex=b["dispenser_capex"],
            n_sites=b["n_sites"],
        )

    @property
    def vehicles_per_year(self) -> int:
        """Annual throughput. ~1.53M at the brief's numbers."""
        return self.vehicles_per_day * DAYS_PER_YEAR

    @property
    def total_dispenser_capex(self) -> float:
        """What Meridian would avoid by not installing dispensers: ~$476,000.

        The upper bound on what a perfect system is worth as a one-off saving,
        and the number that makes the whole project worth discussing.
        """
        return self.dispenser_capex * self.n_sites

    def annual_dispute_cost(self, plate_error_rate: float) -> float:
        """Yearly call-centre cost of misreads at a given error rate.

        Sanity check against the brief: at a 2% error rate this gives
        1,533,000 x 0.02 = 30,660 wrong bills a year, which matches §2's
        "roughly 30,000". If your number does not match, something is wrong -
        check it before quoting anything else from this module.

        Args:
            plate_error_rate: Share of plates billed to the wrong customer,
                in [0, 1]. This is PLATE-level, not character-level - one
                wrong character means one wrong bill.

        Returns:
            Annual cost in dollars.
        """
        return self.vehicles_per_year * plate_error_rate * self.cost_per_dispute

    def annual_review_cost(self, manual_review_rate: float) -> float:
        """Yearly cost of the humans reviewing reads the system does not trust.

        Args:
            manual_review_rate: Share of reads routed to a person, in [0, 1].

        Returns:
            Annual cost in dollars, using our ASSUMED per-review cost.
        """
        return self.vehicles_per_year * manual_review_rate * self.cost_per_manual_review

    def annual_total_cost(
        self, plate_error_rate: float, manual_review_rate: float
    ) -> float:
        """The number the trust threshold minimises.

        The whole design tension in one line: pushing the threshold up sends
        more reads to humans (review cost rises) but lets fewer errors through
        (dispute cost falls). Neither term alone identifies the right
        threshold; their sum does.

        Args:
            plate_error_rate: Error rate among AUTO-ACCEPTED reads only.
                Reads sent to a human are assumed to be corrected there, so
                they do not generate disputes.
            manual_review_rate: Share routed to a human.

        Returns:
            Total annual operating cost in dollars.
        """
        return (self.annual_dispute_cost(plate_error_rate)
                + self.annual_review_cost(manual_review_rate))

    def summary(self, plate_error_rate: float, manual_review_rate: float) -> dict:
        """One dict with every number needed for the business note (ML-53)."""
        return {
            "vehicles_per_year": self.vehicles_per_year,
            "plate_error_rate": round(plate_error_rate, 4),
            "manual_review_rate": round(manual_review_rate, 4),
            "wrong_bills_per_year": round(self.vehicles_per_year * plate_error_rate),
            "reviews_per_year": round(self.vehicles_per_year * manual_review_rate),
            "annual_dispute_cost": round(self.annual_dispute_cost(plate_error_rate), 2),
            "annual_review_cost": round(self.annual_review_cost(manual_review_rate), 2),
            "annual_total_cost": round(
                self.annual_total_cost(plate_error_rate, manual_review_rate), 2
            ),
            "dispenser_capex_avoided": round(self.total_dispenser_capex, 2),
            "ASSUMPTION_cost_per_manual_review": self.cost_per_manual_review,
        }
