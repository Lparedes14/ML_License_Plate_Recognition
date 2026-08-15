# Business Note — Meridian Access Systems

**Ticket ML-53 · For the COO**
*Measured on 400 synthetic plates per condition. Volume: 4,200 vehicles/day
= 1,533,000 plates/year.*

---

## Recommendation

**Do not roll this out across the estate. Fund a paid pilot at 2–3 sites.**

At today's accuracy the system would **increase** operating cost by about
**$20,000 a year**, not reduce it. That is more useful than a favourable
number we could not defend, because the reason it loses money is specific,
measured and fixable.

---

## The policy the model supports today

| | |
|---|---|
| **Auto-accept** | every character clears **0.98** confidence → **10% of plates**, at **100% precision** |
| **Route to a human** | the other 90%, plus any plate where segmentation returns the wrong character count |

Across 400 test plates the system never billed a customer wrongly. It simply
cannot handle much volume yet.

---

## What it costs

| Line | Annual |
|---|---:|
| Baseline — all manual | $256,011 |
| With the model | $276,410 |
| **Net** | **−$20,399** |
| 3-year NPV @ 10% | **−$230,729** |

Automating 10% of plates avoids **$25,601** of manual handling. Running the
platform costs **$46,000**. The infrastructure bill is larger than the
saving — nothing subtle is happening.

**Break-even is 18% automation; we are at 10%.** The system needs to roughly
double its *coverage*, not become more accurate on the reads it already
accepts. Those are different engineering problems, and knowing which one you
are solving is worth more than another point of accuracy.

---

## The assumption to challenge

Everything above assumes a misread costs **$14**. We tested $3–$60:

| Cost per misread | Threshold | Automation | Annual net |
|---:|---:|---:|---:|
| $3 | 0.98 | 10% | −$33,889 |
| $14 *(assumed)* | 0.98 | 10% | −$20,399 |
| $25 | 0.98 | 10% | −$6,908 |
| **$40** | 0.98 | 10% | **+$11,488** |
| $60 | 0.98 | 10% | +$36,016 |

**The optimal threshold never moves.** Below 0.98 confidence precision falls
to roughly 50%, so automated errors cost millions at any plausible price and
swamp every labour saving. The confidence threshold is **not** a useful
business dial today — coverage is the constraint.

**The case turns positive above roughly $40 per misread.** If Meridian's true
cost of a wrong bill — including churn from an annoyed regular, not just the
refund — is nearer $40 than $14, this is marginally profitable at today's
accuracy. **Worth establishing precisely before any rollout decision**: it
moves the answer more than any modelling choice we made.

---

## What four more weeks would buy

1. **Replace segmentation with a CTC sequence model.** On hard-condition
   plates we measured **351 segmentation failures against 27 recognition
   failures**. Segmentation is what caps coverage at 10%, and this is the
   change most likely to clear the 18% break-even.
2. **Train on printed glyphs, not just handwriting.** The spike measured a
   **0.194** accuracy drop from handwritten to printed. Every number here
   inherits it.
3. **Jurisdiction-constrained decoding.** If Meridian's plates exclude I, O
   and Q, masking those outputs is free accuracy — O→0 (46.5%) and I→1
   (33.7%) are among our worst error pairs.
4. **Calibrate the confidence scores.** The entire policy assumes the model's
   confidence means something. Softmax outputs usually are not calibrated,
   and we have not verified ours.

---

## What we are not claiming

Every figure is measured on **synthetically rendered plates**, never
photographs (Section 11 — plate numbers are personal data). Real plates add mud,
frames, screws, IR illumination and motion blur at speed: **treat all of this
as an upper bound.** The $46,000/year infrastructure and $180,000 build
figures are planning assumptions, not quotes. 400 plates per condition is
enough to rank the options, not to certify a production SLA.
