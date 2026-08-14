# CLAUDE.md

Guidance for working in this repository.

---

## What this project is

An Automated Number Plate Recognition prototype for ISM 6642 (Group 8).
A CNN trained on EMNIST handwritten characters reads synthetic license
plates end to end: image → binarize → segment → classify → plate string →
auto-accept-or-route-to-human.

**The brief is `MachineLearning-FinalProject.docx`** in the parent folder.
Section references throughout the code (§4, §5, §11...) point to it. It
defines the grading rubric and several hard constraints — read it before
making design decisions.

---

## How this repository is organised

**The notebook at the repo root is the source of truth.**
`ML_FinalProject_Group_8.ipynb` is the graded deliverable and the thing that
runs in the demo — self-contained, one click to open in Colab.

`anpr_package/` holds the same pipeline factored into an installable Python
package with 53 tests. It is **not** imported by the notebook; the two are
parallel implementations that share `artifacts/` and `data/`.

That is a deliberate trade, and worth understanding before changing either:

- The notebook wins on *demonstrability* — everything runs in one place, in
  Colab, with a GPU and a public Gradio link.
- The package wins on *verifiability* — automated tests, runtime guards that
  are proven to fire, structural blocks on double-normalisation and class-map
  mismatches.

**The cost of that trade is drift.** The two already differ (font choice,
segmentation version — see `anpr_package/README.md`). When you change
behaviour in one, decide explicitly whether the other should follow, rather
than letting them diverge silently.

---

## Using the trained models

Two models are committed under `artifacts/models/`:

| File | What | Test accuracy |
|---|---|---|
| `plate_cnn.keras` | The real classifier, 3 conv blocks, 592,964 params | **90.2%** |
| `plate_mlp.keras` | Dense baseline / control, 542,500 params | 84.2% |
| `plate_cnn.classmap.json` | index → character map | — |

### Load a model — always with its class map

```python
from anpr.inference import load_reader

model, idx2char = load_reader("artifacts/models/plate_cnn.keras")
```

`load_reader()` finds `plate_cnn.classmap.json` automatically (same name,
`.classmap.json` suffix) and **refuses to load without it**. This is
deliberate: the model outputs integers, and only the class map says which
character each integer means. Loading weights without the matching map
produces a system that reads every plate confidently and wrongly, with no
error raised anywhere (ML-46).

It also verifies the saved charset matches `anpr.config.CHARS` and raises if
they differ — protecting against someone reordering `CHARS` after training.

### Read a plate

```python
from anpr.inference import load_reader, read_plate

model, idx2char = load_reader("artifacts/models/plate_cnn.keras")
result = read_plate("data/generated/clean/plate_0000.png", model, idx2char,
                    n_expected=7)

result.text               # 'AM32ULG'
result.plate_confidence   # 0.97  - the MINIMUM over characters, not the mean
result.char_confidences   # [0.99, 0.98, 0.97, ...]
result.n_chars_found      # 7
result.segmentation_ok    # False if the character count was wrong
result.boxes              # [(x, y, w, h), ...] for drawing the overlay
```

`read_plate()` accepts either a **file path or a numpy array** — the path
form matters because §8 requires the demo to run on an image supplied at
demo time, and "demo only works on pre-selected images" is −5.

### The two things to understand about the output

**`plate_confidence` is the minimum, not the mean.** A plate is only as
trustworthy as its weakest character. Six characters at 0.99 and one at 0.30
average to ~0.89, which would sail past any sensible auto-accept threshold
and bill the wrong customer on the one character that was actually wrong.

**`segmentation_ok=False` means the character count was wrong** — that is a
*segmentation* failure, not a recognition error, and no amount of retraining
fixes it (§4). Never blend the two when reporting.

### Retraining

Training happens **in Google Colab** (that's where the GPU is). The workflow:

1. Open `ML_FinalProject_Group_8.ipynb` in Colab (badge in the root README)
2. `Runtime → Run all`
3. Download the artifacts it writes, and unzip **into** `artifacts/` — not
   the repo root: `unzip anpr_artifacts.zip -d artifacts/`

`anpr_package/src/anpr/models/` holds an equivalent architecture and training
loop so the code is reviewable and testable, but the model actually committed
under `artifacts/models/` comes from the notebook run.

---

## Repository map

```
anpr_package/src/anpr/
├── config.py          settings, seeds, the 36-class charset, paths
├── data/
│   ├── contract.py    THE input contract - read this first
│   ├── guards.py      orientation guard (EMNIST ships sideways)
│   ├── emnist.py      3-route loader with provenance
│   ├── labels.py      62→36 case handling, class weights
│   ├── splits.py      train/val/test + leakage proof
│   ├── pipeline.py    tf.data + augmentation
│   └── plates.py      synthetic plate generator (ML-37)
├── models/            architectures + training loop
├── segment/           binarize → contours → 28×28 crops
├── inference/         read_plate: image → string + confidence
├── evaluate/          metrics, confusion, per-tier results
└── business/          cost model, trust threshold
```

**Read `data/contract.py` before anything else.** It defines the single
input contract used by *both* the training path and the inference path.
§4 names mismatched preprocessing between those two paths as the most common
cause of "high validation accuracy, useless on real images", and the entire
design of this repo is arranged to make that mismatch structurally
impossible rather than merely unlikely.

---

## Running things

**The demo** — open the root notebook in Colab and run all. That is the
graded path (§9's "one command").

**The package** — everything below runs from inside `anpr_package/`, not the
repo root, because that is where `pyproject.toml` and `config/` live:

```bash
cd anpr_package
pip install -e ".[download,demo,dev]"   # editable install; TensorFlow included

pytest                                  # 53 tests, ~4s, no GPU needed
python scripts/prepare_data.py          # load + verify EMNIST, split, sign off
python scripts/make_test_plates.py      # 1,200 synthetic plates + manifest
python scripts/evaluate.py              # measure at all 3 tiers
python scripts/demo.py --image <path>
```

Paths still resolve to the shared `artifacts/` and `data/` at the **repo
root** (`PROJECT_ROOT = parents[3]` in `config.py`), so the package and the
notebook read the same model and the same test set.

**Note:** the tests are written to run *without* TensorFlow installed — all
TF imports are deferred inside functions. That's why `pytest` passes on a
machine with no TF, but it also means `models/`, `data/pipeline.py` and the
training scripts are **not covered by any executed test**. Don't mistake a
green suite for full coverage.

---

## Conventions that are load-bearing

**Every accuracy number states its conditions.** Tier and sample size, always.
Reporting accuracy without conditions is an explicit −5 (§10). The
`TierResult.to_sentence()` helper exists so a bare percentage is hard to
produce by accident.

**Segmentation and recognition failures are counted separately.** A wrong
character *count* is excluded from character accuracy and counted wrong for
plate accuracy. Merging them makes both numbers meaningless.

**Guards are proven to fire.** `prove_guard_fires()` feeds the orientation
guard deliberately transposed data and fails if it's accepted. A check that
can never fail is not a check.

**No off-the-shelf OCR as the classifier.** Tesseract, EasyOCR, PaddleOCR and
cloud OCR APIs are −20 points if used as our system's classifier. Permitted
only as a side-by-side comparison, reported honestly.

**No non-consensual plate photography.** §11 is a hard constraint — plate
numbers are personal data. All plates are programmatically generated. Any
scraped or covertly captured plate imagery is an automatic zero on the
project.

---

## Where the numbers live

Never retype a number into a document — cite the file:

| File | Contains |
|---|---|
| `artifacts/provenance/provenance.json` | data routes, hashes, row counts |
| `artifacts/provenance/split_manifest.json` | split fingerprints, leakage checks |
| `artifacts/provenance/acceptance_record.md` | PASS/FAIL data sign-off |
| `artifacts/metrics/training_summary.json` | model accuracy, params, hyperparams |
| `artifacts/metrics/tier_results.json` | per-tier accuracy + confidences |
| `data/generated/manifest.csv` | ground truth for all 1,200 test plates |
