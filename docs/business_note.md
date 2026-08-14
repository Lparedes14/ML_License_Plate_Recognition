# Business Note — Meridian Access Systems

**Ticket ML-53 · 1 page · For the COO**
*All figures measured, not estimated. Volume: 4,200 vehicles/day = 1,533,000 plates/year.*

---

## Recommendation

**Do not roll this out across the estate. Fund a paid pilot at 2–3 sites instead.**

At today's measured accuracy the system would **increase** operating cost by
about **$20,000 a year**, not reduce it. That is the honest answer, and it is
more useful than a favourable number we could not defend — because the reason
it loses money is specific, measured, and fixable.

---

## The policy the model supports today

| | |
|---|---|
| **Auto-accept** | reads where every character clears **0.98** confidence |
| | → **10% of plates**, at **100% precision** on that 10% |
| **Route to a human** | the other 90%, plus any plate where segmentation returns the wrong character count |

At that operating point the system never billed a customer wrongly across 400
test plates. It simply cannot handle much volume yet.

---

## What it costs, at 4,200 vehicles/day

| Line | Annual |
|---|---:|
| Baseline — all manual | $256,011 |
| With the model | $276,410 |
| **Net** | **−$20,399** |
| 3-year NPV @ 10% | **−$230,729** |
| Payback | **never, at current accuracy** |

### Why it loses money — the one number that explains it

Automating 10% of plates avoids **$25,601** of manual handling and human
error. Running the system costs **$46,000** a year in infrastructure.

**The infrastructure bill is larger than the saving.** Nothing subtle is
happening: the automation rate is too low to pay for the platform.

**Break-even is 18% automation** at the current 100% precision — we are at
10%. The system needs to roughly **double its coverage**, not become more
accurate on the reads it already accepts. Those are different engineering
problems, and knowing which one you are solving is worth more than another
point of accuracy.

---

## The assumption to challenge — and what happens when you do

Everything above assumes a misread costs **$14.00** (refund + support call +
goodwill). We tested that across a $3–$60 range:

| Cost per misread | Optimal threshold | Automation | Annual net |
|---:|---:|---:|---:|
| $3 | 0.98 | 10% | −$33,889 |
| $14 *(assumed)* | 0.98 | 10% | −$20,399 |
| $25 | 0.98 | 10% | −$6,908 |
| **$40** | 0.98 | 10% | **+$11,488** |
| $60 | 0.98 | 10% | +$36,016 |

Two things here are worth reading carefully.

**The optimal threshold never moves.** Not at $3, not at $60. That is not a
modelling artefact — it is a real property of this model. Below 0.98
confidence, precision falls to roughly 50%, so automated errors cost
**millions** per year at any plausible misread price and swamp every labour
saving. Until precision holds up at lower confidence, there is no threshold
worth trading down to. The confidence threshold is **not** a useful business
dial today; coverage is the constraint, not the trade-off.

**The business case only turns positive above roughly $40 per misread.** If
Meridian's true cost of a wrong bill — including churn from an annoyed
regular, not just the refund — is nearer $40 than $14, this becomes
marginally profitable at today's accuracy. **That number is worth
establishing precisely before any rollout decision**, because it moves the
answer more than any modelling choice we made.

---

## What four more weeks would buy

Ordered by expected effect on the number above:

1. **Replace segmentation with a CTC sequence model.** On hard-condition
   plates we measured **351 segmentation failures against 27 recognition
   failures** — segmentation is the bottleneck by a factor of thirteen, and
   it is what caps coverage at 10%. This is the change most likely to push
   automation past the 18% break-even.
2. **Train on printed glyphs, not just handwriting.** The spike measured a
   **0.19 absolute accuracy drop** from handwritten to printed characters.
   Every number in this note inherits that gap.
3. **Jurisdiction-constrained decoding.** If Meridian's plates never contain
   I, O or Q, masking those outputs is free accuracy — our confusion analysis
   puts O→0 (46.5%) and I→1 (33.7%) among the worst error pairs.
4. **Calibrate the confidence scores.** The entire policy assumes the model's
   confidence means something. Softmax outputs usually are not calibrated,
   and we have not verified ours.

---

## What we are not claiming

- Every figure is measured on **synthetically rendered plates**, never
  photographs (§11 — plate numbers are personal data; we used no licensed
  corpus). Real plates add mud, frames, screws, IR illumination and motion
  blur at speed. **Treat all of this as an upper bound.**
- The $46,000/year infrastructure and $180,000 build figures are planning
  assumptions, not quotes.
- 400 plates per condition is enough to rank the options, not to certify a
  production SLA.
