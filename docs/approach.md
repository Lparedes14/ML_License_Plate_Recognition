# Approach Document

**Group 8 · ISM 6642 · Meridian Access Systems ANPR Prototype**

*Every figure in this document comes from the executed outputs of
`ML_FinalProject_Group_8.ipynb`. Section references (§2, §4, §11…) point to
the project brief.*

> **Before submission:** download the notebook's `provenance.json`,
> `results_summary.md` and `plate_char_model.keras` and commit them, so every
> number here is backed by a file in `artifacts/`. The artifacts currently
> committed are from the parallel `anpr_package/` run and carry **different**
> tier numbers (that implementation uses a single monospace font and the
> pre-split segmenter). Both runs are real; only one should be cited, and
> this document cites the notebook.

---

## 1. Problem framing

Meridian runs gated parking at 34 sites on ticket dispensers — ~$14,000
installed each, jamming often enough that lost-ticket disputes cost the call
centre ~$8.50 to resolve. The COO's question is whether plate recognition can
replace the ticket: the plate becomes the account, the gate opens on entry,
the customer is billed on exit.

**Our answer, up front: not yet, and we can say exactly why.**

At today's measured accuracy the system automates **10% of plates at 100%
precision** and, at an assumed $14 cost per misread, would **increase**
operating cost by **$20,399 a year** — a 3-year NPV of **−$230,729**. We
recommend a paid pilot at 2–3 sites, not an estate-wide rollout.

That is a more useful result than a flattering one, because the reason is
specific and fixable. Automating 10% of plates avoids **$25,601** of manual
handling and human error, while the platform costs **$46,000** a year to run.
**Break-even is 18% automation.** The constraint is *coverage* — how many
plates the system dares to accept — not accuracy on the plates it already
accepts. Those are different engineering problems, and §5 shows which one to
attack.

The framing that matters for the COO is therefore not *"how accurate is it"*
but *"which reads do we trust automatically, and which do we send to a
human"* — and at 4,200 vehicles/day, a 2% plate-level error rate would mean
roughly 30,000 wrong bills a year, each one a refund, a support call, and an
annoyed regular.

---

## 2. Pipeline

```
plate image → binarise → segment → normalise crop → CNN → assemble → trust policy
```

| Stage | What it does | Where it fails |
|---|---|---|
| **Binarise** | Otsu threshold, inverted (plates are dark-on-light, EMNIST is light-on-dark) | Uneven lighting merges or erases strokes |
| **Segment** | Contours → filter by glyph geometry → **split merged glyphs** → order left-to-right | Touching or broken characters. **The dominant failure — see §5** |
| **Normalise** | 20px longest side, centred by *centre of mass*, into 28×28 | Silent killer if it drifts from the training recipe |
| **Classify** | Each crop → character + softmax confidence | Confusable glyph shapes (O/0, L/1, I/1) |
| **Assemble** | Join characters; plate confidence = **minimum** across them | — |
| **Trust policy** | Above threshold → auto-accept; below → human | — |

Two design decisions we would defend:

**One preprocessing function, used by both paths.** `to_mnist_format()` runs
at training time *and* at inference time. §4 of the brief names mismatched
preprocessing as the single most common cause of "high validation accuracy,
unusable on real images"; sharing one function makes that drift structurally
impossible rather than merely unlikely.

**Plate confidence is the minimum, not the mean.** Six characters at 0.99 and
one at 0.30 average to 0.89 — which sails past any sensible threshold and
bills the wrong customer on the one character that was actually wrong. A
plate is only as trustworthy as its weakest character.

---

## 3. Data

**Source.** EMNIST ByClass — 220,000 training and 60,000 test images, loaded
through a three-route loader (local CSV → TFDS → torchvision) with content
hashes recorded per split. ByClass rather than Balanced because it preserves
natural letter frequency and covers digits plus both cases in one dataset.

**The transpose.** EMNIST ships column-major — sideways relative to what
every downstream tool assumes. We do not rely on remembering to fix it:
`assert_upright()` makes three falsifiable geometric claims ('1' is taller
than wide, 'T' is top-heavy, 'L' is bottom-heavy) and refuses to return data
if they fail. The guard has a **self-test that feeds it deliberately
transposed data and fails if it is accepted** — a check that can never fail
is not a check.

**Case handling.** 62 raw classes fold to the 36 a plate needs by merging
lowercase into uppercase. This keeps ~40% more data at a real cost: a
handwritten 'a' does not look like a printed plate 'A', and m/M, o/O, u/U are
genuinely inseparable at 28×28.

**Imbalance.** ByClass follows English letter frequency; plates are close to
uniform. Measured: rarest class **'K' at 0.725%**, commonest **'1' at 5.52%**
— a **7.61× spread** against a uniform expectation of 2.78%. We correct with
inverse-frequency class weighting (up to 3.91×) rather than resampling, which
uses every sample instead of discarding the majority classes.

**Test set.** Plates are rendered programmatically at three degradation
tiers, so the ground-truth string is known by construction — labelling costs
nothing, and §11's consent constraint is satisfied by never touching a
photograph.

---

## 4. Modeling

**Primary: a 3-block CNN.** 32→64→128 filters, each block two 3×3
convolutions with batch normalisation, then pooling and dropout.
**592,964 parameters**, Adam at 1e-3, class-weighted for the imbalance above.

**Control: a dense MLP.** Same data, same splits, no spatial structure. Its
only job is to make the CNN's number mean something:

| Model | Test accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|
| Baseline MLP | 0.841 | — | 542,500 |
| **CNN** | **0.901** | **0.898** | 592,964 |

Six points of accuracy for 9% more parameters is the value of spatial
structure, not of scale.

**Sample sizes** (§8 promises these will be asked live): train **198,000**,
validation **22,000**, test **60,000** — the test split from a physically
separate EMNIST file, opened once.

**Fallback, had the CNN underperformed:** drop the confusable I/O/Q from the
plate alphabet entirely; increase augmentation rather than model capacity
(the domain gap in §6 is a data problem, not a capacity problem); or accept
lower accuracy and lean harder on the trust threshold.

**A constraint we worked under:** Tesseract, EasyOCR, PaddleOCR and cloud OCR
APIs are not permitted as the system's classifier (§6 — using one costs 20
points). Every character prediction here comes from the CNN above, which we
trained. Running an off-the-shelf engine *alongside* ours as a published
comparison would have earned credit; we did not have the time budget and are
saying so rather than implying we chose not to.

---

## 5. Evaluation

**Definition of a correct read**, agreed in writing before any evaluation ran:
strict exact match on the uppercase character sequence. A wrong character
*count* is a **segmentation** failure, never a near-miss recognition error.

**400 plates per tier.** Segmentation, character and plate accuracy reported
separately, each with its conditions:

| Condition | Segmentation | Character | Plate |
|---|---:|---:|---:|
| clean | 0.955 | 0.843 | 0.502 |
| normal | 0.730 | 0.624 | 0.340 |
| hard | 0.122 | 0.109 | 0.055 |

### The finding that shaped everything downstream

On 400 hard plates: **351 segmentation failures against 27 recognition
failures.** Segmentation is the bottleneck by a factor of thirteen — the
pipeline often never gets far enough to ask the classifier. More training
data would barely move the hard-tier number.

This is *only* visible because the two are counted separately. A single
blended "accuracy" figure would have hidden it, and we would have spent the
remaining time on the wrong problem.

**Errors compound.** Plate accuracy ≈ character accuracy⁷. At our measured
0.901 the ceiling is **0.483**, and reaching a 95% plate-level target would
need **0.993** per character. That arithmetic — not model capacity — is why
plate reading is hard.

**One measured improvement, reported with its before/after.** The first
segmentation pass discarded any contour flatter than a single upright glyph,
which is exactly what two touching characters look like. Splitting merged
glyphs at their vertical-projection valley:

| Condition | v1 | v2 | Gain |
|---|---:|---:|---:|
| clean | 80.3% | **94.7%** | +14.3pp |
| normal | 54.7% | **74.0%** | +19.3pp |
| hard | 9.3% | **13.7%** | +4.3pp |

**A caveat we state rather than bury:** hard-tier character accuracy is
computed over only the ~12% of plates that segmented at all — a small,
self-selected sample of the easiest hard plates. It is not a reliable
estimate of classifier performance under those conditions.

---

## 6. Spike

Two risky assumptions tested cheaply; both changed the plan. Detail in
[`spike.md`](spike.md).

**Will a handwriting-trained model read printed plates?** Measured on a
2-epoch throwaway CNN: **0.816 handwritten vs 0.623 printed — a +0.194
domain gap.** §3 permits not closing this in two weeks, provided it is
measured rather than assumed away. Every plate-level number above inherits it.

**Can connected components segment reliably?** Viable, but fragile to
character spacing in a way the original tier design never tracked. That
finding is what led directly to the v2 splitting fix in §5.

---

## 7. Scope

**Built:** EMNIST-trained CNN classifier (36 classes, case-merged,
class-weighted); connected-components segmentation with merged-glyph
splitting; a shared preprocessing contract used identically at training and
inference; end-to-end read with per-character and aggregate confidence; a
synthetic plate generator at three quality tiers; a cost-based trust
threshold; and a live demo accepting any uploaded image.

**Explicitly not attempted:**

- **Plate localisation** within a road scene — we assume a cropped plate
- **Closing the handwriting→printed domain gap** — measured (§6), not fixed
- **Multi-line or non-Latin plates**
- **Real photographic validation** — §11 makes non-consensual plate imagery
  an automatic zero, so every number here is synthetic and should be read as
  an upper bound
- **Hyperparameter search** — one architecture, justified by reasoning rather
  than a sweep we had no time budget for

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Segmentation collapses under degradation** — measured 95.5% → 12.2% clean to hard, and it caps automation at 10% | The business case stays negative; a rollout cannot be justified | Report segmentation and recognition separately so the limitation is precise; pilot at sites resembling clean/normal first; a CTC sequence model is the named fix |
| **Handwriting-trained classifier on printed plates** — +0.194 measured gap | Every accuracy figure is an upper bound on real-plate performance | Stated wherever accuracy is quoted; fine-tuning on rendered glyphs is the first "four more weeks" item |
| **Live demo fails on an unseen image** — our own clean-tier segmentation is 95.5%, not 100% | Reads as an unrehearsed failure rather than a known limitation | Show a chosen failure ourselves before being asked, name which failure mode it is, and keep a backup recording (§8) |

---

## Appendix: AI assistance

See [`ai_disclosure.md`](ai_disclosure.md). Required by §6 — we own every line
submitted and expect to explain any part of it from memory.
