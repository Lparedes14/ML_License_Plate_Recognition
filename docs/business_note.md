# Business Note

**Ticket ML-53 · 1 page · For the COO**

> **Template — not yet written.** Generate the numbers with
> `python scripts/evaluate.py`, which writes
> `artifacts/metrics/trust_policy.json`.
>
> §9: *"which reads you'd auto-accept vs. route to a human, and roughly what
> that policy costs at 4,200 vehicles/day."*
>
> §8: lead with the recommendation. Do not build up to it.

---

## Recommendation

> *One sentence. Paste from `policy_sentence()` — it is written to be read
> aloud.*

---

## The policy

| | Share of traffic | Volume / year | Cost / year |
|---|---|---|---|
| Auto-accepted | | | $ (disputes) |
| Routed to a human | | | $ (reviews) |
| **Total** | 100% | | **$** |

Against **$476,000** of ticket-dispenser hardware avoided (34 sites ×
$14,000).

---

## Why this threshold and not a higher one

The instinct is to auto-accept only near-certain reads. The arithmetic says
otherwise: a dispute costs $8.50 and a review costs [our assumption], so the
optimum routes more generously than feels comfortable. Show the cost curve
and point at its minimum.

---

## Assumptions — ours, not the brief's

State these plainly. §1: overclaiming is the fastest way to lose points.

1. **Manual review costs $1.20 per read.** Not given in the brief. Our
   estimate. The recommendation is [sensitive / insensitive] to it — at
   $3.00 the optimal threshold moves to […].
2. **A human reviewer always reads the plate correctly.** Optimistic. Real
   reviewers make mistakes, which would raise the true cost of the
   review path.
3. **Every misread produces exactly one dispute at $8.50.** Ignores churn
   from annoyed regulars, which §2 implies but does not price.

---

## What four more weeks would buy

Concrete and prioritised (§8, 1 minute).

1.
2.
3.
