# QA Acceptance Criteria — ANPR Prototype

| | |
|---|---|
| **Project** | ML — Group 8 Final Project (ISM 6642) |
| **Feature** | Automated Number Plate Recognition prototype for Meridian Access Systems |
| **Version** | 1.0 |
| **Date** | 12 August 2026 |
| **Author** | Luis Paredes (QA owner) |
| **Repository** | `github.com/Lparedes14/ML_License_Plate_Recognition` |
| **Source PRD** | `MachineLearning-FinalProject.docx` — the assignment brief. Section references below (§2, §4, §5…) point to it |
| **Epics covered** | ML-19 Approach · ML-20 Spike · ML-21 Data and Test Set · ML-22 Model and Pipeline · ML-23 Measurement · ML-24 Business and Communication |
| **Open bugs** | None on the board at time of writing |

> **On `anpr_package/` references below.** The team consolidated on the
> notebook as the single graded deliverable; the parallel `anpr_package/`
> implementation referenced in a few criteria below has since been removed.
> This document is supporting material, not a graded deliverable.

**What this document is.** It defines what "successful" means *before* anything is
measured. It is not the results summary — `docs/results.md` records what we actually got.
This one records what we agreed to check.

**Why it is written now.** §1 of the brief: *"A team reporting 61% character accuracy with
a clear-eyed analysis of why will score higher than a team claiming 99% they cannot
reproduce."* Agreeing the success criteria after seeing results is how a team accidentally
picks the definition that flatters it.

---

## Gate types

Every criterion carries one of two gate types. This split is deliberate.

| Gate | Meaning |
|---|---|
| **HARD** | Must pass. A failure blocks submission or costs a stated deduction. These are things we control completely — leakage, reproducibility, compliance |
| **REPORT** | Records a measurement. Cannot "fail". §5 is explicit: *"If you come in under 75%, you have not failed the project. You have failed the project if you cannot say why"* |

---

## Feature context

**The problem.** Meridian Access Systems runs gated parking at 34 sites using ticket
dispensers (~$14,000 installed each). Dispensers jam, and lost-ticket disputes cost the
call centre ~$8.50 per incident. The COO wants to know whether plate recognition could
replace tickets — the plate becomes the account, the gate opens on entry, the customer is
billed on exit.

**Why accuracy alone is the wrong question.** A misread is not neutral: it bills the wrong
customer, producing a refund, a support call and an annoyed regular. At 4,200 vehicles/day,
a 2% plate-level error rate is roughly 30,000 wrong bills a year. The real design question
(§2) is *which reads do we trust automatically, and which do we send to a human.*

### End-to-end flow

| # | Stage | Module |
|---|---|---|
| 1 | A plate image is supplied (file path or array) | `anpr_package/scripts/demo.py` |
| 2 | Converted to grayscale, thresholded to black/white | `segment/binarize.py` |
| 3 | Connected components → N character bounding boxes | `segment/components.py` |
| 4 | Boxes sorted into reading order (left to right) | `segment/components.py` |
| 5 | Each crop normalised to 28×28 EMNIST convention | `segment/components.py` |
| 6 | Crops validated against the input contract | `data/contract.py` |
| 7 | Each crop classified → character + confidence | `models/` (CNN) |
| 8 | String assembled, confidence aggregated | `inference/read_plate.py` |
| 9 | Auto-accept, or route to a human | `business/trust_policy.py` |

### Actors

| Actor | Role |
|---|---|
| **Driver** | Arrives at the gate. Never interacts with the system directly; experiences it as a correct or incorrect bill |
| **ANPR system** | Reads the plate and emits a string plus a confidence |
| **Human reviewer** | Meridian call-centre staff who resolve reads the system does not trust |
| **COO** | Decides whether to replace dispensers. The audience for the recommendation |
| **Instructor** | Supplies an unseen image at demo time and asks the team to explain any part of the code |

### Glossary

| Term | Meaning |
|---|---|
| **EMNIST ByClass** | Handwritten character dataset, 62 classes (0-9, A-Z, a-z), 697,932 train / 116,323 test |
| **Character accuracy** | Correct characters ÷ compared characters, over correctly segmented plates only |
| **Plate accuracy** | Fraction of plates read entirely correctly, over **all** plates |
| **Segmentation success rate** | Fraction of plates where the character count matched ground truth |
| **Tier A / B / C** | Image quality tiers: clean / realistic camera capture / degraded |
| **Trust threshold** | Confidence cut-off above which a read is auto-accepted |
| **Calibration half** | Plates used to *fit* the threshold. Never used for reported numbers |
| **Reporting half** | Held-out plates. All published numbers come from here |
| **Domain gap** | EMNIST is handwritten; plates are printed. Named in §3, not closed by us |

---

## Cross-cutting business rules

Referenced throughout rather than repeated per block.

### RN-01 — Definition of a correct plate read · **HARD**

A read is correct **only** when the predicted string matches the ground truth exactly:

- Comparison is **uppercase**. Plates are uppercase-only, so case never differs in practice
- **No whitespace or formatting** is compared — the raw character sequence only
- **Length must match exactly.** A 6-character read of a 7-character plate is wrong, and is
  classified as a segmentation failure (RN-02), not as a near-miss

No partial credit, no edit-distance tolerance. This mirrors the business reality: a single
wrong character bills the wrong customer, and Meridian does not care that six of seven were
right. *Agreed before any measurement was taken (ML-38).*

### RN-02 — Segmentation and recognition failures are counted separately · **HARD**

§4: *"If your system reads a 6-character plate as 5 characters, that is not a recognition
error and no amount of retraining fixes it. Count them separately from day one."*

| Condition | Classification | Character accuracy | Plate accuracy |
|---|---|---|---|
| Predicted length ≠ truth length | **Segmentation failure** | Excluded from the denominator | Counted as wrong |
| Length matches, some characters wrong | **Recognition error** | Contributes to the denominator | Counted as wrong |
| Length matches, all characters right | Correct | Contributes | Counted as correct |

Merging these makes both numbers meaningless.

### RN-03 — Every reported number states its conditions · **HARD**

Any accuracy figure in any document, slide or spoken sentence carries: the dataset, the
quality tier, and the sample size. Reporting accuracy without stating conditions is an
explicit **−5 deduction** (§10).

### RN-04 — Four datasets, with strict access rules · **HARD**

| Dataset | Size | May be used to… | May **not** be used to… |
|---|---|---|---|
| Train | 198,000 | Fit weights | — |
| Validation | 22,000 | Choose anything (stopping, model, strategy) | Be reported as final accuracy |
| Character test | 60,000 | Report classifier accuracy, **once** | Choose anything |
| Plate test | 3 × 800 | Report system accuracy | Fit the threshold on its reporting half |

If the character test set is evaluated twice and the better run is kept, it has become a
second validation set and the number is no longer honest.

### RN-05 — The trust threshold is fitted and reported on different data · **HARD**

Each tier generates 800 plates, split 400 calibration / 400 reporting. The threshold is
swept and chosen on the calibration half; every published cost and accuracy figure comes
from the reporting half. Fitting and reporting on the same plates makes the analysis
circular.

### RN-06 — Plate confidence is the minimum over its characters · **REPORT**

A plate is correct only if every character is, so it is exactly as trustworthy as its
weakest character. The mean would hide the problem: six characters at 0.99 and one at 0.30
averages to 0.89 and would sail past any sensible threshold.

### RN-07 — A model and its class map are inseparable · **HARD**

The model outputs an integer; only the class map says which character that integer means.
Loading weights without the matching map produces a system that reads every plate
confidently and wrongly, with no error raised anywhere.

### RN-08 — Assumptions are labelled as assumptions · **HARD**

Any figure not given in the brief is marked as ours wherever it appears — in particular the
$1.20 per-manual-review cost, and the assumption that a human reviewer always reads
correctly. §1 names overclaiming as the fastest way to lose points.

---

## BLOCK A — Data integrity and input contract

*Covers: ML-36, ML-6, ML-7, ML-40, ML-58 · Owner: Thenmani*

**CA-A1 · HARD** — Orientation guard rejects transposed data
```gherkin
Given a set of EMNIST images loaded through any of the three routes
When the images are transposed relative to the correct orientation
Then assert_upright() raises AssertionError
And the loader refuses to return the data
```
*Mapped to: ML-36 AC2*

**CA-A2 · HARD** — The guard is proven to fire
```gherkin
Given a set of correctly oriented EMNIST images
When prove_guard_fires() feeds the guard deliberately transposed copies
Then the guard raises AssertionError
And if it does not, the run aborts with "GUARD IS BROKEN"
```
> A check that never fails proves nothing. This is what makes CA-A1 trustworthy.

**CA-A3 · HARD** — A truncated source file is rejected
```gherkin
Given a local EMNIST CSV containing fewer than 95% of the expected rows
When the loader parses it
Then it raises ValueError naming the actual and expected row counts
And it falls through to the next route rather than training on partial data
```
*This criterion exists because it already caught a real failure: a 129,461-row upload where
ByClass has 697,932. See the spike write-up.*

**CA-A4 · HARD** — Double normalisation is impossible
```gherkin
Given an image batch that has already been normalised to float32 in [0,1]
When normalize() is called on it a second time
Then it raises TypeError
```

**CA-A5 · HARD** — Any reasonable input reaches canonical form
```gherkin
Given image data as flat 784 rows, (H,W), (N,H,W), 1-channel or 3-channel RGB
When to_canonical_uint8() processes it
Then the output is uint8 with shape (N, 28, 28, 1)
```

**CA-A6 · HARD** — Train and validation share no source rows
```gherkin
Given the stratified split of the EMNIST train file with seed 42
When the index arrays are intersected
Then the overlap is exactly 0
And a non-zero overlap raises AssertionError naming the offending indices
```
*Mapped to: ML-7, ML-40*

**CA-A7 · HARD** — Stratification is preserved
```gherkin
Given the train, validation and test splits
When per-class proportions are compared
Then no class deviates from the training distribution by more than 0.5 percentage points
```

**CA-A8 · REPORT** — Byte-identical duplicates across splits are counted
```gherkin
Given the three splits
When images are compared by content rather than by index
Then the count of byte-identical duplicates between each pair is recorded
And any duplicates inherited from the EMNIST source files are reported, not silently kept
```
> REPORT rather than HARD: duplication between the EMNIST train and test *files* is a
> property of the source, not of our split logic.

**CA-A9 · HARD** — Provenance is recorded for every load
```gherkin
Given a completed data load
When provenance.json is inspected
Then it records route, source URI, library version, row count, transpose applied,
     load timestamp and a SHA-256 content hash for each split
```

**CA-A10 · HARD** — Splits are reproducible
```gherkin
Given the same seed and configuration
When the split is regenerated on a different machine
Then the split fingerprints (SHA-256 of the sorted index arrays) are identical
```

---

## BLOCK B — Character classification

*Covers: ML-42, ML-39, ML-8 · Owner: Thenmani*

**CA-B1 · HARD** — Training produces three inseparable artifacts
```gherkin
Given a completed training run named <name>
When artifacts/models/ is inspected
Then <name>.keras, <name>.classmap.json and <name>.history.json all exist
```
*Mapped to: ML-46, RN-07*

**CA-B2 · REPORT** — Character accuracy on the held-out EMNIST test set
```gherkin
Given the trained CNN and the 60,000-image EMNIST test split
When the model is evaluated exactly once
Then character accuracy is recorded with its sample size
And the result is not used to select any model or hyperparameter
```
*Reference point: ≥75% is described as a reasonable prototype (§5). Report it whatever it is.*

**CA-B3 · REPORT** — Per-class recall is reported
```gherkin
Given the confusion matrix from the character test set
When per-class recall is computed
Then a value is recorded for all 36 classes
And rare classes are called out explicitly, in particular the rarest measured class
    (see acceptance_record.md for the current figure — this shifts slightly between
    EMNIST loads, so cite the committed run rather than a fixed number here)
```

**CA-B4 · REPORT** — The CNN is compared against a baseline
```gherkin
Given both the CNN and the dense MLP control trained on the same data and splits
When both are evaluated on validation data
Then both accuracies and both parameter counts are recorded
```
> Without a control, the CNN's number means nothing.

**CA-B5 · REPORT** — Augmentation is measured, not assumed
```gherkin
Given training runs with and without augmentation
When both are evaluated on the same validation set
Then the difference is recorded
```
*Mapped to: ML-8*

**CA-B6 · REPORT** — The imbalance decision is evidenced
```gherkin
Given the chosen imbalance strategy and the "none" baseline
When both arms are trained and evaluated
Then per-class counts before and after handling are recorded
And per-class recall is compared between the two arms
```
*Mapped to: ML-39 AC1. The comparison is the deliverable, not the setting.*

**CA-B7 · HARD** — Model selection never touches the test data
```gherkin
Given the training script
When model.fit() is called
Then only train and validation datasets are passed
And the EMNIST test split is not loaded during training
```

---

## BLOCK C — Segmentation

*Covers: ML-43, ML-44 · Owner: Luis*

**CA-C1 · HARD** — Binarisation output convention is asserted, not assumed
```gherkin
Given any plate image, light-on-dark or dark-on-light
When binarize() produces a binary image
Then characters are white (255) on a black (0) background
And the convention is asserted rather than trusted
```
> Backwards means the model sees photographic negatives and predicts confident nonsense.

**CA-C2 · REPORT** — Character count matches ground truth
```gherkin
Given a generated plate of known length
When segment_characters() processes it
Then the number of returned crops is recorded
And count_matches indicates whether it equals the expected length
```

**CA-C3 · HARD** — Crops are returned in reading order
```gherkin
Given a segmented plate with N character boxes
When the crops are returned
Then they are ordered left to right by x-coordinate
```
> Connected-component labelling returns blobs arbitrarily. Without this you get high
> character accuracy and near-zero plate accuracy — a confusing bug to chase.

**CA-C4 · HARD** — Crops match the EMNIST centring convention
```gherkin
Given a character crop cut from a binarised plate
When normalize_crop() processes it
Then the glyph's longer side measures 20 pixels with aspect ratio preserved
And the glyph's centre of mass lies within one pixel of (14, 14) in the 28x28 field
```
*Mapped to: ML-44. §4 names mismatched preprocessing as the single most common cause of
"57% validation accuracy, unusable on real images". Centre of **mass**, not bounding-box
centre — they differ for asymmetric glyphs like J and L.*

**CA-C5 · HARD** — Crops satisfy the same contract as training data
```gherkin
Given crops produced by segmentation
When they pass through to_canonical_uint8() and normalize()
Then assert_input_contract() passes with the same shape, dtype and range as a training batch
```

**CA-C6 · REPORT** — Rejected blobs are recorded with a reason
```gherkin
Given a plate image containing non-character features such as bolts or a border
When blobs are filtered
Then each rejected blob is recorded with the filter that rejected it
```
> "We dropped it as noise" is the explanation for a failure you would otherwise be unable to
> account for live.

**CA-C7 · HARD** — Narrow glyphs survive filtering
```gherkin
Given a plate containing the characters "1" and "I"
When aspect-ratio filtering is applied
Then those characters are retained in the output
```
> Setting the narrow-glyph filter too aggressively silently deletes 1 and I from every
> plate. It presents as a recognition problem and is not one.

**CA-C8 · REPORT** — Segmentation success rate is reported on its own
```gherkin
Given a full evaluation run at one quality tier
When results are recorded
Then segmentation success rate appears as its own figure, separate from recognition accuracy
```
*Mapped to: §5, which requires it reported separately.*

---

## BLOCK D — End-to-end plate read

*Covers: ML-45, ML-38, ML-49 · Owner: Luis*

**CA-D1 · HARD** — Correctness follows RN-01
```gherkin
Given a plate read and its ground truth
When correctness is evaluated
Then the comparison is exact-match on the uppercase character sequence
And a length mismatch is classified as a segmentation failure
```

**CA-D2 · REPORT** — Character accuracy measures the classifier alone
```gherkin
Given a set of evaluated plates
When character accuracy is computed
Then only plates whose predicted length equals the truth length contribute
```

**CA-D3 · REPORT** — Plate accuracy measures the whole system
```gherkin
Given a set of evaluated plates
When plate accuracy is computed
Then every plate contributes, including segmentation failures
```
> A plate that failed to segment bills the wrong customer just as surely as one misread.

**CA-D4 · HARD** — Every read carries a confidence
```gherkin
Given any completed read, successful or otherwise
When the PlateRead is returned
Then plate_confidence is populated per RN-06
And an empty read returns confidence 0.0
```
> Without a confidence the read cannot be routed, and the business layer has nothing to
> threshold on.

**CA-D5 · REPORT** — Measured at all three tiers
```gherkin
Given the trained system
When evaluation runs
Then character accuracy, plate accuracy and segmentation rate are recorded for tiers A, B and C
And each is stated with its sample size and tier parameters
```

**CA-D6 · HARD** — Tiers differ only in image quality
```gherkin
Given the three tier test sets
When the plate strings are compared across tiers
Then the same strings appear in all three
```
> Otherwise a tier scores worse purely for having drawn harder characters, and the
> comparison is meaningless.

**CA-D7 · HARD** — Failure type is named in the output
```gherkin
Given a read where segmentation produced the wrong character count
When the result is displayed
Then it is labelled explicitly as a segmentation failure
And states that retraining would not fix it
```

---

## BLOCK E — Trust threshold and business policy

*Covers: ML-52, ML-53 · Owner: Valentina, shared with Luis*

**CA-E1 · HARD** — Confidence aggregation follows RN-06
```gherkin
Given per-character confidences for a plate
When plate confidence is computed
Then it equals the minimum of the character confidences
```

**CA-E2 · HARD** — The full threshold range is swept
```gherkin
Given a set of reads with confidences and correctness outcomes
When the threshold is swept
Then every value from 0.0 to 1.0 is costed
And both degenerate endpoints appear on the curve
```

**CA-E3 · HARD** — Fitted and reported on different plates (RN-05)
```gherkin
Given 800 plates at a tier, split 400 calibration / 400 reporting
When the threshold is chosen on the calibration half
Then all published figures are computed on the reporting half using that fixed threshold
```

**CA-E4 · HARD** — The cost model reproduces the brief's own arithmetic
```gherkin
Given the default cost model
When annual wrong bills are computed at a 2% plate error rate
Then the result falls between 29,000 and 32,000
```
*§2 states "roughly 30,000". If our model disagrees, our model is wrong. Automated in
`anpr_package/tests/test_metrics_and_business.py`.*

**CA-E5 · REPORT** — The recommended policy beats blanket acceptance
```gherkin
Given the costed threshold curve
When the cost-optimal threshold is selected
Then its annual total cost is compared against accepting every read
And the difference is recorded
```
> If the model's confidence is informative, routing doubtful reads to a person should cost
> less than paying $8.50 per resulting dispute. If it does not, that is a finding worth
> reporting honestly.

**CA-E6 · HARD** — Assumptions are labelled (RN-08)
```gherkin
Given the business note and the trust policy output
When assumptions are inspected
Then the manual review cost is marked as our assumption, not the brief's
And the assumption that reviewers always read correctly is stated
```

**CA-E7 · HARD** — The recommendation is one actionable sentence
```gherkin
Given the selected policy point
When the recommendation is produced
Then it names the threshold, the share of traffic auto-accepted,
     the annual review volume and the annual cost
```
*§8: lead with the recommendation. The COO should be able to act on one sentence.*

---

## BLOCK F — Demo and reproducibility

*Covers: ML-46, ML-47, ML-48, ML-55 · Owner: Luis, with Thenmani*

**CA-F1 · HARD** — The repository runs from a clean clone
```gherkin
Given a fresh clone on a machine that has never run this project
When the documented install command is executed
Then the package installs and "import anpr" succeeds
```
*Failure is a **−8 deduction** (§10).*

**CA-F2 · HARD** — The test suite passes
```gherkin
Given a clean clone with dev dependencies installed
When pytest is run from the repository root
Then all tests pass
```

**CA-F3 · HARD** — The demo accepts an arbitrary file path
```gherkin
Given any readable image file anywhere on disk
When anpr_package/scripts/demo.py is invoked with --image pointing at it
Then the pipeline runs without code changes or file relocation
```
*A demo that only works on pre-selected images is a **−5 deduction** (§10).*

**CA-F4 · HARD** — The demo runs on a genuinely unseen image
```gherkin
Given a plate image no team member has previously tested against
When the demo is run on it before the presentation
Then it completes and produces a read or an explained failure
```

**CA-F5 · HARD** — The demo shows the stages, not only the answer
```gherkin
Given a demo run
When output is displayed
Then it shows the segmented character count, the read, the per-character confidences,
     the aggregate confidence and the auto-accept decision
```

**CA-F6 · HARD** — A deliberate failure is prepared in advance
```gherkin
Given the demo script and a known-bad image chosen before the presentation
When the failure case is run
Then it fails in a way the team can explain
And the failure type is named as segmentation or recognition
```
*Mapped to: ML-48. §8: teams that only show successes will be asked to produce a failure
live, and it will go worse.*

**CA-F7 · HARD** — The model reloads without retraining (RN-07)
```gherkin
Given a saved model and its class map in a fresh runtime with no Drive mount
When the model is reloaded and run on a fixed sample
Then predictions are identical to the original run
And loading without the class map raises rather than guessing
```
*Mapped to: ML-46 AC1*

**CA-F8 · HARD** — No environment-specific paths in shipped code
```gherkin
Given the repository source
When it is searched for hardcoded Colab or personal Drive paths
Then none are found
```
*Mapped to: ML-46 AC2*

**CA-F9 · HARD** — A backup recording exists and plays
```gherkin
Given the completed demo
When the backup recording is opened on a second machine
Then it plays and shows a full end-to-end run
```
*Mapped to: ML-55*

---

## BLOCK G — Edge and negative cases

*QA-identified. Not enumerated in the brief — these are our responsibility.*

**CA-G1 · HARD** — Zero detected characters produces an honest empty read
```gherkin
Given an image where segmentation finds no character-shaped blobs
When read_plate() completes
Then it returns an empty string with confidence 0.0
And no plate string is fabricated
```
> A confident wrong answer is worse for Meridian than an admitted failure, because a wrong
> answer bills someone.

**CA-G2 · REPORT** — Over-segmentation is detected and classified
```gherkin
Given a plate where a broken glyph splits into two blobs
When the system reads it
Then more characters are found than expected
And it is recorded as a segmentation failure, not a recognition error
```

**CA-G3 · REPORT** — Under-segmentation is detected and classified
```gherkin
Given a plate where two touching characters merge into one blob
When the system reads it
Then fewer characters are found than expected
And it is recorded as a segmentation failure
```

**CA-G4 · HARD** — A non-plate image fails gracefully
```gherkin
Given an image that contains no plate at all
When the demo is run on it
Then the system completes without crashing
And returns either an empty read or a low-confidence read routed to a human
```

**CA-G5 · HARD** — An unreadable file produces a clear error
```gherkin
Given a path that does not exist or points to a corrupt file
When the demo is invoked
Then FileNotFoundError is raised naming the path
And the error does not surface as an obscure array-shape failure
```

**CA-G6 · HARD** — A uniform image does not crash the pipeline
```gherkin
Given an image that is entirely black or entirely white
When binarisation and segmentation run
Then the system completes and reports zero characters found
```

**CA-G7 · HARD** — A missing class map is refused
```gherkin
Given a saved model whose class map file is absent
When load_reader() is called
Then FileNotFoundError is raised explaining why the map is required
And the model is not loaded
```

**CA-G8 · HARD** — A mismatched class map is refused
```gherkin
Given a class map whose charset differs from the current code
When load_class_map() is called
Then ValueError is raised showing both charsets
```
> The weights would load fine; every prediction would decode to the wrong character with no
> error anywhere.

**CA-G9 · REPORT** — Pre-registered confusable pairs are measured
```gherkin
Given the confusion matrix and the CONFUSABLE_PAIRS list recorded before training
When confusion analysis runs
Then a confusion rate is reported for each predicted pair, both directions
And pairs we predicted that did not materialise are reported as such
```
> Confirmed or refuted, both are findings. If confusion concentrates on I/1, O/0 and Q/O,
> recommending that Meridian exclude those three characters at issue is a fix that costs
> nothing and beats any amount of retraining.

**CA-G10 · REPORT** — Rare-class performance is surfaced
```gherkin
Given per-class recall from the character test set
When rare classes are inspected
Then recall for Q is reported explicitly alongside its training share
```
> Q is ~0.79% of our EMNIST subsample (`artifacts/provenance/acceptance_record.md`) but
> ~2.78% of a uniform plate alphabet, so poor recall on Q damages plate accuracy roughly
> 3.5x more than its training frequency suggests. This Q-vs-uniform ratio is separate from
> the overall imbalance ratio (rarest-vs-commonest class), which is currently 7.61x — see
> the note on CA-B3.

**CA-G11 · REPORT** — Degenerate policies are visible on the curve
```gherkin
Given the threshold sweep
When the endpoints are inspected
Then threshold 0.0 (accept everything) and 1.0 (review everything) both appear with their costs
```

**CA-G12 · REPORT** — A tier where the policy collapses is reported honestly
```gherkin
Given a quality tier where nearly every read falls below the chosen threshold
When results are recorded
Then the near-100% manual review rate is reported
And stated as a limitation rather than omitted
```

---

## BLOCK H — Reporting and submission compliance

*The Friday checklist. Each row maps to a stated deduction in §10 or a §9 deliverable.*

**CA-H1 · HARD** — No off-the-shelf OCR is used as the classifier
```gherkin
Given the shipped codebase and its dependencies
When they are inspected
Then no OCR engine or cloud OCR API is present as the system's classifier
```
*Violation is a **−20 deduction**. Permitted only as a side-by-side comparison, reported
honestly — which we are not attempting (see scope).*

**CA-H2 · HARD** — Clean clone verified by more than one person
```gherkin
Given the pushed repository
When each team member clones and installs it independently
Then all report a successful install and passing tests
```
*Violation is a **−8 deduction**.*

**CA-H3 · HARD** — No number appears without its conditions (RN-03)
```gherkin
Given every document, slide and spoken claim
When each accuracy figure is checked
Then it states the dataset, the quality tier and the sample size
```
*Violation is a **−5 deduction**.*

**CA-H4 · HARD** — Demo verified on an image supplied at demo time (CA-F3, CA-F4)
*Violation is a **−5 deduction**.*

**CA-H5 · HARD** — No non-consensual plate imagery anywhere
```gherkin
Given every image in the repository, documents and presentation
When their origin is checked
Then all are programmatically generated
And no photograph of a real plate appears in any form
```
*§11 is a hard constraint: any team submitting scraped or covertly captured plate imagery
**receives zero on the project**.*

**CA-H6 · HARD** — AI assistance is disclosed specifically
```gherkin
Given docs/ai_disclosure.md
When it is reviewed
Then it names each area where an AI assistant contributed and who owns the result
And each owner can explain their code without notes
```
*§6: "You own every line you submit — expect to be asked in the demo to explain any part of
your code from memory, and 'the model wrote it' is not an answer."*

**CA-H7 · HARD** — All deliverables exist
```gherkin
Given the submission
When the §9 deliverable list is checked
Then the approach document, git repository, results summary, business note,
     backup recording and three individual contribution statements all exist
```

**CA-H8 · HARD** — Every published number traces to a generated file
```gherkin
Given every figure in every document
When it is traced
Then it originates from a file in artifacts/metrics/ or artifacts/provenance/
And no number has been typed by hand
```
> Verified by someone who did not produce the numbers. An untraceable number is
> overclaiming even when it happens to be right.

---

## Pending bugs

None open at the time of writing. Any bug raised before submission should be added here
with its environment, a short description, and the CA it affects.

---

## Coverage map

Written last, as an audit that nothing is uncovered.

| Ticket | Summary | Covered by |
|---|---|---|
| ML-6 | Image normalisation | CA-A4, CA-A5 |
| ML-7 | Train/val/test split with fixed seed | CA-A6, CA-A7, CA-A10 |
| ML-8 | Augmentation | CA-B5 |
| ML-36 | EMNIST loading with orientation assertion | CA-A1, CA-A2, CA-A3, CA-A9 |
| ML-37 | Plate image generator with ground truth | CA-D6, CA-C2 |
| ML-38 | Definition of a correct read | RN-01, CA-D1 |
| ML-39 | 36-class map and imbalance decision | CA-B3, CA-B6, CA-G10 |
| ML-40 | Stratified split | CA-A6, CA-A7, CA-B7 |
| ML-41 | Test set at three quality tiers | CA-D5, CA-D6 |
| ML-42 | Baseline character classifier | CA-B1, CA-B2, CA-B4 |
| ML-43 | Character segmentation | CA-C1, CA-C2, CA-C3, CA-C6, CA-C7 |
| ML-44 | Preprocessing identical train/inference | CA-C4, CA-C5 |
| ML-45 | Segmentation and recognition chained | CA-D1 – CA-D7 |
| ML-46 | Model and class map saved and reloadable | CA-B1, CA-F7, CA-F8, CA-G7, CA-G8 |
| ML-47 | Demo accepts an image supplied at demo time | CA-F3, CA-F4, CA-F5 |
| ML-48 | Known failure case prepared | CA-F6 |
| ML-49 | Accuracy at three tiers, segmentation separated | CA-C8, CA-D2, CA-D3, CA-D5 |
| ML-50 | Confusion matrix and ambiguous pairs | CA-G9 |
| ML-51 | Results summary written | CA-H3, CA-H8 |
| ML-52 | Cost model and trust threshold | CA-E1 – CA-E5, CA-G11 |
| ML-53 | One-page business note | CA-E6, CA-E7 |
| ML-55 | Backup recording | CA-F9 |
| ML-57 | Contribution statements and ethics confirmation | CA-H5, CA-H6, CA-H7 |

**Totals:** 8 blocks · 8 cross-cutting rules · 68 acceptance criteria (49 HARD, 19 REPORT).

| Block | CAs | | Block | CAs |
|---|---:|---|---|---:|
| A — Data integrity | 10 | | E — Trust threshold | 7 |
| B — Classification | 7 | | F — Demo & reproducibility | 9 |
| C — Segmentation | 8 | | G — Edge & negative | 12 |
| D — End-to-end read | 7 | | H — Compliance | 8 |

**Criteria carrying a stated deduction if failed:** CA-H1 (−20), CA-H2 (−8), CA-H3 (−5),
CA-H4 (−5), CA-H5 (zero on the project).
