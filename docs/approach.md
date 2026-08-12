# Approach Document

**Group 8 · ISM 6642 · Meridian Access Systems ANPR Prototype**

> **Template — not yet written.** Tickets ML-26 … ML-32. Worth **30 of 100
> points**, the single largest component. Section headings below are taken
> verbatim from §3 of the brief; do not rename them.
>
> Target length 3–4 pages. §10: *"decisions justified rather than asserted;
> scope stated explicitly including what's out; risks named before they
> materialize."*

---

## 1. Problem framing
*What Meridian actually needs; what a misread costs.* — ML-26

Lead with the money, not the model. A 2% plate-level error rate at 4,200
vehicles/day is ~30,000 wrong bills a year at $8.50 each. Contrast with
$14,000 × 34 sites of dispenser hardware avoided. Numbers come from
`anpr.business.CostModel` — quote the code's output, not a hand calculation.

State the real design question in the COO's terms: not *how accurate*, but
*which reads do we trust automatically*.

---

## 2. Pipeline
*Diagram + one paragraph per stage.* — ML-27

Diagram the six stages (see the README). One paragraph each: what it does,
how we implement it, what it can fail at.

Name the two integration traps §4 warns about and say how the design avoids
them:
- preprocessing identical between training and inference → one
  `data.contract` module, used by both paths
- segmentation vs recognition failures counted separately →
  `SegmentationResult.count_matches`

---

## 3. Data
*Source, class handling, imbalance decision, test set with known ground
truth.* — ML-28

- **Source and provenance.** Cite `artifacts/provenance/provenance.json` —
  route, URI, row count, content hash.
- **The transpose.** EMNIST ships column-major. We assert rather than
  remember; the guard has a self-test proving it fires.
- **Case handling.** merge vs drop — state which, and the measured cost of
  the other.
- **Imbalance.** Q at ~0.80% vs N at ~2.82% against a near-uniform plate
  alphabet. State the strategy and show both arms.
- **Test set.** Programmatically rendered plates at three quality tiers.
  Ground truth is free; §11 (consent) is satisfied by construction.

---

## 4. Modeling
*What you'll try first and why; what your fallback is.* — ML-29

CNN as primary, MLP as control. Say what the fallback is if the CNN
underperforms — fewer classes (drop I/O/Q), more augmentation, or accepting
lower accuracy and leaning harder on the trust threshold.

Be ready for the live questions (§8): layer count, learning rate,
train/val/test sizes. All in `config/default.yaml` and `model.summary()`.

---

## 5. Evaluation
*The metrics you'll report and the conditions you'll report them under.* —
ML-30

Character accuracy · plate accuracy · segmentation success rate, each at
tiers A/B/C, each with its sample size. Explain in advance why plate accuracy
will be far lower: 0.95⁷ = 70%.

Define what counts as a correct read **before** measuring (ML-38). Spacing?
Case? Leading zeros? Agree it in writing or the number is arguable.

---

## 6. Spike
*What you tested, what you found, what changed.* — see `spike.md`

One paragraph summary here; detail in the spike document.

---

## 7. Scope
*What you will build — and explicitly what you are not attempting.* — ML-31

§3 is blunt about this: *"A team that says 'we are not attempting plate
localization; we assume a cropped plate' has scoped honestly. A team that
stays silent and then fails to localize has not."*

**We will build:** …
**We are explicitly NOT attempting:** …

Candidates for the second list: plate localisation within a full vehicle
photo; multi-row plates; non-US plate formats; closing the
handwriting→printed domain gap.

---

## 8. Risks
*Top three, with mitigations.* — ML-31

Named before they materialise, not after.

| Risk | Impact | Mitigation |
|---|---|---|
| | | |

---

## Appendix: AI assistance
See [`ai_disclosure.md`](ai_disclosure.md). Required by §6.
