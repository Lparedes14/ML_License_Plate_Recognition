# Reading the Road — ANPR Prototype

**ISM 6642 Final Project · Group 8 · Meridian Access Systems**

An automated number plate reader built from scratch: an EMNIST-trained
character classifier, our own segmentation pipeline, and a cost model that
answers the question the COO actually asked — *which reads do we trust
automatically, and which do we send to a human?*

---

## Run the demo

```bash
pip install -e ".[download,demo,dev]"
python scripts/demo.py --image path/to/plate.png
```

That is the one command. It accepts **any** image path, including one handed
over at demo time.

---

## Full pipeline, from a clean clone

```bash
git clone https://github.com/Lparedes14/ML_License_Plate_Recognition.git
cd ML_License_Plate_Recognition
pip install -e ".[download,demo,dev]"

pytest                              # 53 tests, ~4s
python scripts/prepare_data.py      # load + verify EMNIST, split, sign off
python scripts/make_test_plates.py  # 1,200 synthetic plates + manifest
python scripts/evaluate.py          # measure at 3 quality tiers
python scripts/demo.py --image <path>
```

Training itself runs in **Google Colab** (that's where the GPU is) —
see [Trained models](#trained-models) below.

---

## Results

Measured on 1,200 persisted synthetic plates (400 per tier,
`data/generated/`), run end to end through the committed pipeline:

| Tier | n | Segmentation success | Character accuracy | Plate accuracy |
|---|---:|---:|---:|---:|
| Clean | 400 | 93.0% | 95.3% | 65.8% |
| Normal | 400 | 67.5% | 96.5% | 51.7% |
| Hard | 400 | 10.0% | 96.8% | 7.8% |

Character classifier on held-out EMNIST: **CNN 90.2%** vs **MLP baseline
84.2%**.

**The headline finding:** segmentation, not the classifier, is what breaks
under degradation. Character accuracy barely moves across tiers (95–97%, if
anything slightly *higher* under degradation); segmentation success
collapses from 93% to 10%, and that alone explains why plate accuracy falls
to 7.8%. This is exactly why §4 requires the two failure types to be counted
separately — a single blended "accuracy" number would have hidden it
entirely.

**A secondary finding:** an earlier measurement using randomly-selected
system fonts (several italic) scored 82.2% / 53.2% / 9.8% segmentation at
the same three tiers. Switching to one consistent monospace font (ML-37)
improved segmentation by +11–14pp at Clean/Normal — font consistency isn't
just a reproducibility fix, it measurably affects accuracy. Hard tier barely
moved either way; past a certain degradation level the image itself is the
limiting factor, not the typeface.

⚠️ Hard-tier character accuracy (96.8%) is computed over only the 40 plates
(10.0% of 400) that segmented correctly — a small, self-selected easy
subset, not a reliable estimate. Stated rather than reported bare.

⚠️ The three tiers currently draw independently random plate text rather
than the same 400 strings at three quality levels — a methodology gap
against the original design intent, noted in `docs/approach.md` §5 rather
than silently left uncorrected.

---

## How the code works

**The one rule: logic lives in `src/anpr/`. Notebooks and scripts import it.**
A notebook cell reads `from anpr.models import build_cnn` — it never contains
the model code itself. That's what lets three people work in parallel and
what makes the repo run from a clean clone.

### The pipeline

```
plate image
  → segment.binarize        Otsu threshold → white ink on black
  → segment.components      contours → filtered boxes → 28×28 crops
  → data.contract           normalise to float32 [0,1]  ← SAME as training
  → models (CNN)            each crop → character + confidence
  → inference.read_plate    assemble string, aggregate confidence
  → business.trust_policy   auto-accept, or route to a human?
```

The **same** `data.contract` functions run at training time and at inference
time. §4 names mismatched preprocessing as the single most common cause of
"57% validation accuracy, unusable on real images" — one definition used in
both places makes that mismatch impossible rather than unlikely.

### Package map

| Package | Owner (§7) | What's in it |
|---|---|---|
| `anpr.config` | everyone | settings, RNG seeding, the 36-class charset, paths |
| `anpr.data` | Data | EMNIST loading, guards, splits, plate generation |
| `anpr.models` | Model | CNN + MLP architectures, training loop |
| `anpr.segment` | Pipeline | plate image → ordered character crops |
| `anpr.inference` | Pipeline | crops → plate string + confidence |
| `anpr.evaluate` | QA | accuracy, confusion, per-tier results |
| `anpr.business` | Business | cost model, trust threshold |

### Module by module

**`data/contract.py`** — the single definition of a valid model input.
`to_canonical_uint8()` accepts anything image-shaped; `normalize()` is the
only division by 255 in the codebase and **raises on float input**, so
double-normalisation is blocked structurally. Read this file first.

**`data/guards.py`** — EMNIST ships column-major (sideways). `assert_upright()`
makes three falsifiable geometric claims ('1' is tall, 'T' is top-heavy, 'L'
is bottom-heavy) and refuses to return data if they fail. `prove_guard_fires()`
feeds it transposed data and fails if accepted — a check that can never fail
is not a check.

**`data/emnist.py`** — three fallback routes (local CSV → tfds → torchvision)
with full provenance: route, URI, row count, SHA-256. Also validates row
counts, which already caught a real truncated-file bug.

**`data/labels.py`** — 62 → 36 case handling, inverse-frequency class weights
(the rarest class gets 3.91×), and the class-map save/load that keeps model
outputs meaningful.

**`data/plates.py`** — synthetic plate generator. Renders text in a monospace
font, applies tier-specific degradation (skew → blur → lighting → noise, in
that physical order), and persists a manifest CSV pairing every image with
its ground truth.

**`segment/`** — `binarize.py` does Otsu thresholding with inversion (plates
are dark-on-light, EMNIST is light-on-dark). `components.py` finds contours,
filters by glyph geometry, sorts left-to-right, and normalises each crop to
EMNIST's convention: 20px longest side, centred by **centre of mass** (not
bounding-box centre) in a 28×28 field.

**`inference/read_plate.py`** — the end-to-end read. Accepts a path or an
array. Returns text, per-character confidences, and an aggregate confidence
that is the **minimum** over characters, because a plate is only as
trustworthy as its weakest character.

**`evaluate/`** — the counting rules. A wrong character *count* is a
segmentation failure: excluded from character accuracy, counted wrong for
plate accuracy. `TierResult.to_sentence()` makes a bare percentage hard to
produce by accident.

**`business/`** — Meridian's economics ($8.50/dispute, 4,200 vehicles/day,
$14k × 34 sites) and the confidence-threshold sweep that turns an accuracy
number into an operational policy.

---

## Trained models

Committed under `artifacts/models/`:

| File | Test accuracy | Parameters |
|---|---:|---:|
| `plate_cnn.keras` | 90.2% | 592,964 |
| `plate_mlp.keras` (control) | 84.2% | 542,500 |
| `plate_cnn.classmap.json` | — | index → character |

```python
from anpr.inference import load_reader, read_plate

model, idx2char = load_reader("artifacts/models/plate_cnn.keras")
result = read_plate("some_plate.png", model, idx2char, n_expected=7)
print(result.text, result.plate_confidence)
```

`load_reader()` refuses to load a model without its class map — the model
outputs integers, and only the map says what they mean. See
[`CLAUDE.md`](CLAUDE.md) for the full usage guide.

---

## Documents — what's in each

### Deliverables

| File | What you'll find in it |
|---|---|
| [`docs/approach.md`](docs/approach.md) | **The main deliverable (30 pts).** Eight sections: the business problem and what a misread costs; the pipeline diagrammed stage by stage; the data decisions (EMNIST ByClass, case merging, the 7.61× class imbalance and how we handle it); the modeling choices and fallbacks; the evaluation metrics with real measured numbers; a spike summary; explicit scope including what we are *not* attempting; and the top three risks with mitigations. |
| [`docs/spike.md`](docs/spike.md) | **The spike write-up (10 pts).** Two experiments, each in "we assumed X, tested it, found Y, changed Z" form. Spike 1: can connected components segment reliably? (Breaks at tight kerning — 100% → 0%.) Spike 2: will a handwriting-trained model read printed text? (23% relative accuracy drop.) Plus a bonus finding on a truncated data file. |
| [`docs/results.md`](docs/results.md) | Results summary — the measured numbers with their conditions, confusion analysis, failure modes. |
| [`docs/business_note.md`](docs/business_note.md) | One page for the COO: which reads to auto-accept vs route to a human, and what that policy costs at 4,200 vehicles/day. |
| [`docs/ai_disclosure.md`](docs/ai_disclosure.md) | Where AI assistance was used (§6 requires this). |
| `docs/contributions/` | Individual contribution statements, submitted privately. |

### QA

Two documents with distinct jobs — the first says what "successful" means and
was written *before* anything was measured; the second records what actually
happened when each criterion was run.

| File | What you'll find in it |
|---|---|
| [`docs/qa/qa_acceptance_criteria.md`](docs/qa/qa_acceptance_criteria.md) | **68 acceptance criteria in Gherkin**, across 8 blocks (data integrity, classification, segmentation, end-to-end read, trust threshold, demo/reproducibility, edge cases, submission compliance), each traced to a Jira ticket. Includes 8 cross-cutting rules — the definition of a correct read, the segmentation-vs-recognition counting rule, the four-dataset discipline. Block H doubles as the pre-submission checklist, with each stated deduction attached. |
| [`docs/qa/qa_execution_log.md`](docs/qa/qa_execution_log.md) | **When each criterion was actually run, the result, and the evidence file.** Status per criterion (PASS / PENDING / BLOCKED / FAIL), test-suite run history, findings raised during QA, and a sign-off table. This is the link between "what we promised to check" and "what we measured". |

### Notebooks

| File | What you'll find in it |
|---|---|
| [`notebooks/spike_segmentation.py`](notebooks/spike_segmentation.py) | The segmentation spike — deliberately throwaway, a script rather than a notebook because it is not meant to be kept or built upon. |
| [`notebooks/README.md`](notebooks/README.md) | How notebooks relate to `src/` and how Colab training works. |

---

## Where the numbers come from

Never retype a number — every figure in every document traces to a file:

| File | Contains |
|---|---|
| `artifacts/provenance/provenance.json` | data routes, source URIs, content hashes |
| `artifacts/provenance/split_manifest.json` | split fingerprints, leakage checks |
| `artifacts/provenance/acceptance_record.md` | PASS/FAIL data sign-off |
| `artifacts/metrics/training_summary.json` | model accuracy, parameters, hyperparameters |
| `artifacts/metrics/tier_results.json` | per-tier accuracy + per-plate confidences |
| `data/generated/manifest.csv` | ground truth for all 1,200 test plates |

The plate **images** are gitignored (regenerable via
`scripts/make_test_plates.py`), but `manifest.csv` is committed — without it,
the accuracy numbers cannot be audited.

---

## Configuration

Every tunable number is in [`config/default.yaml`](config/default.yaml) —
learning rate, layer sizes, quality tiers, Meridian's cost figures. Point at
it when asked about hyperparameters; don't quote from memory.

---

## Constraints we work under

- **No off-the-shelf OCR as the classifier.** Tesseract, EasyOCR, PaddleOCR
  and cloud OCR APIs are −20 points if used as our system's classifier.
  Permitted only as a side-by-side comparison, reported honestly.
- **No non-consensual plate photography.** Plate numbers are personal data
  under GDPR and several US state statutes. We use programmatically generated
  plates only — scraped or covertly captured imagery is an automatic zero.
- **Every accuracy number states its conditions.** Reporting accuracy without
  the quality tier it was measured on is −5.
- **The repo must run from a clean clone.** Failing that check is −8.
