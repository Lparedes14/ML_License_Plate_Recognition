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

python scripts/prepare_data.py       # load + verify EMNIST, split, sign off
python scripts/make_test_plates.py   # generate the synthetic test set
python scripts/train.py --both       # train the CNN and the MLP baseline
python scripts/evaluate.py           # measure at 3 tiers, fit trust threshold
python scripts/demo.py --image <path>
```

`prepare_data.py` downloads EMNIST automatically. To use local files instead,
drop `emnist-byclass-train.csv` and `emnist-byclass-test.csv` (`.zip` is fine,
no need to extract) into `data/raw/`.

Run the tests with `pytest`.

---

## How the code is organised

**The one rule: logic lives in `src/anpr/`. Notebooks and scripts import it.**
A notebook cell reads `from anpr.models import build_cnn` — it never contains
the model code itself. That is what lets five people work in parallel and what
makes this repo run from a clean clone.

| Package | Owner (§7) | What it does |
|---|---|---|
| `anpr.config` | everyone | settings, seeds, the 36-class charset |
| `anpr.data` | Data | EMNIST loading, guards, splits, plate generation |
| `anpr.models` | Model | architectures, training, fine-tuning |
| `anpr.segment` | Pipeline | plate image → character crops |
| `anpr.inference` | Pipeline | crops → plate string + confidence |
| `anpr.evaluate` | QA | accuracy, confusion, quality tiers |
| `anpr.business` | Business | cost model, trust threshold |

```
plate image
  → segment.binarize          threshold to black/white
  → segment.components        connected components → N character boxes
  → data.contract             each box → canonical 28×28 float32 [0,1]
  → models (CNN)              each crop → class + confidence
  → inference.read_plate      assemble string, aggregate confidence
  → business.trust_policy     auto-accept, or route to a human?
```

The **same** `data.contract` functions run at training time and at inference
time. That is deliberate — §4 names mismatched preprocessing as the single
most common cause of *"57% validation accuracy, unusable on real images."*

---

## Configuration

Every tunable number is in [`config/default.yaml`](config/default.yaml) —
learning rate, layer sizes, quality tiers, Meridian's cost figures. Point at
it when asked about hyperparameters; don't quote from memory.

```bash
python scripts/train.py --config config/experiment_dropped_case.yaml
```

---

## What's built and what isn't

| Component | Status | Ticket |
|---|---|---|
| EMNIST loader, 3 routes, provenance | done | ML-36 |
| Orientation guard + self-test | done | ML-36 |
| Input contract (resize, normalise) | done | ML-6, ML-44 |
| Stratified split + leakage proof | done | ML-7, ML-40 |
| tf.data pipeline + augmentation | done | ML-8 |
| Case handling + class weighting | done | ML-39 |
| CNN + MLP baseline, training loop | done | ML-42 |
| Metrics, confusion, cost model, trust policy | done | ML-49, ML-50, ML-52 |
| **Synthetic plate generation** | **TODO** | **ML-37, ML-41** |
| **Segmentation (binarize, components)** | **TODO** | **ML-43** |
| **EMNIST-convention crop normalisation** | **TODO** | **ML-44** |
| Fine-tune on printed glyphs | blocked on ML-37 | — |

Unimplemented functions raise `NotImplementedError` with the ticket number and
an implementation sketch in the docstring. Nothing fails silently.

**Start with ML-37** (`src/anpr/data/plates.py`). It unblocks both the
fine-tuning data and the entire results section.

---

## Constraints we work under

- **No off-the-shelf OCR as the classifier.** Tesseract, EasyOCR, PaddleOCR
  and cloud OCR APIs are −20 points if used as our system's classifier. They
  are permitted only as a side-by-side comparison, reported honestly.
- **No non-consensual plate photography.** Plate numbers are personal data
  under GDPR and several US state statutes. We use programmatically generated
  plates only. Scraped or covertly captured imagery is an automatic zero.
- **Every accuracy number states its conditions.** Reporting accuracy without
  the image-quality tier it was measured on is −5.

---

## Documents

| File | Deliverable |
|---|---|
| [`docs/approach.md`](docs/approach.md) | Approach document (30 pts) |
| [`docs/spike.md`](docs/spike.md) | Spike write-up (10 pts) |
| [`docs/results.md`](docs/results.md) | Results summary |
| [`docs/business_note.md`](docs/business_note.md) | Business note |
| [`docs/ai_disclosure.md`](docs/ai_disclosure.md) | AI usage appendix (§6) |
| `docs/contributions/` | Individual statements (private) |
