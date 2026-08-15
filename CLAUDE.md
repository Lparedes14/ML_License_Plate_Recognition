# CLAUDE.md

Guidance for working in this repository.

---

## What this project is

An Automated Number Plate Recognition prototype for ISM 6642 (Group 8).
A CNN trained on EMNIST handwritten characters reads synthetic license
plates end to end: image → binarize → segment → classify → plate string →
auto-accept-or-route-to-human.

**The brief is `MachineLearning-FinalProject.docx`** in the parent folder.
Section references throughout the docs (Section 4, Section 5, Section 11...)
point to it. It defines the grading rubric and several hard constraints —
read it before making design decisions.

---

## How this repository is organised

**`ML_FinalProject_Group_8.ipynb` is the whole project.** Data loading,
training, segmentation, evaluation and the live Gradio demo all live in that
one notebook, self-contained and one click to open in Colab. There is no
separate package — everything runs top to bottom in the notebook, in order.

`docs/` holds the written deliverables required by Section 9 of the brief.
`artifacts/` and `data/` hold what the notebook writes and reads.

---

## Using the trained models

Two models are committed under `artifacts/models/`:

| File | What | Test accuracy |
|---|---|---|
| `plate_cnn.keras` | The real classifier, 3 conv blocks, 592,964 params | **90.2%** |
| `plate_mlp.keras` | Dense baseline / control, 542,500 params | 84.2% |

The notebook loads these directly with `keras.models.load_model(...)` in the
demo cell — there is no separate class-map file to load alongside them,
because `CHARS` (the index → character mapping) is defined earlier in the
same notebook and is already in memory by the time the demo cell runs. See
ML-46 in the Jira board for why that is a deliberate, accepted design choice
rather than an oversight.

### Reading a plate

The notebook's `read_plate()` function is the whole pipeline in one call:

```python
text, confidences, chars = read_plate(plate_img, model, expected=CFG["plate_len"])
```

- `text` — the predicted string, e.g. `"AM32ULG"`
- `confidences` — a NumPy array, one softmax confidence per character
- `chars` — the segmented 28×28 crops the model actually saw

There is no `segmentation_ok` flag on the return value — segmentation success
is inferred by comparing `len(text)` to the expected character count, which
is exactly what `evaluate_plates()` does a few cells later.

### The two things to understand about the output

**Plate confidence should be read as the minimum across `confidences`, not
the mean.** A plate is only as trustworthy as its weakest character — six
characters at 0.99 and one at 0.30 average to ~0.89, which would sail past
any sensible auto-accept threshold and bill the wrong customer on the one
character that was actually wrong. This is why the business note's trust
policy is built on the minimum.

**A character-count mismatch is a segmentation failure, not a recognition
error** (Section 4), and no amount of retraining fixes it. Never blend the
two when reporting — see `evaluate_plates()`, which tracks `seg_ok` and
`char_acc` as separate columns for exactly this reason.

### Retraining

Training happens **in Google Colab** (that's where the GPU is):

1. Open `ML_FinalProject_Group_8.ipynb` in Colab (badge in the root README)
2. `Runtime → Run all`
3. Download the artifacts it writes, and unzip **into** `artifacts/` — not
   the repo root: `unzip anpr_artifacts.zip -d artifacts/`

---

## Conventions that are load-bearing

**Every accuracy number states its conditions.** Tier and sample size,
always. Reporting accuracy without conditions is an explicit −5 (Section 10).

**Segmentation and recognition failures are counted separately.** A wrong
character *count* is excluded from character accuracy and counted wrong for
plate accuracy. Merging them makes both numbers meaningless.

**No off-the-shelf OCR as the classifier.** Tesseract, EasyOCR, PaddleOCR and
cloud OCR APIs are −20 points if used as our system's classifier. Permitted
only as a side-by-side comparison, reported honestly.

**No non-consensual plate photography.** Section 11 is a hard constraint —
plate numbers are personal data. All plates are programmatically generated.
Any scraped or covertly captured plate imagery is an automatic zero on the
project.

---

## Where the numbers live

Never retype a number into a document — cite the file:

| File | Contains |
|---|---|
| `artifacts/provenance/provenance.json` | data routes, hashes, row counts |
| `artifacts/provenance/split_manifest.json` | split fingerprints, leakage checks |
| `artifacts/metrics/training_summary.json` | model accuracy, params, hyperparams |
| `data/generated/manifest.csv` | ground truth for the persisted test plates |

**Note:** `artifacts/metrics/tier_results.json` predates this repository
structure and holds numbers from an earlier parallel implementation that has
since been removed — it does not match the notebook's tier numbers quoted in
`docs/results.md` and `docs/approach.md`. Regenerate it from the notebook
before citing it, or don't cite it at all.
