# Reading the Road — ANPR Prototype

**ISM 6642 Final Project · Group 8 · Meridian Access Systems**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lparedes14/ML_License_Plate_Recognition/blob/main/ML_FinalProject_Group_8.ipynb)

An automated number plate reader built from scratch: a CNN character
classifier trained on EMNIST, our own segmentation stage, and a cost model
that answers the question the COO actually asked — *which reads do we trust
automatically, and which do we send to a human?*

---

## Run it — one click

**[▶ Open the notebook in Colab](https://colab.research.google.com/github/Lparedes14/ML_License_Plate_Recognition/blob/main/ML_FinalProject_Group_8.ipynb)**

`ML_FinalProject_Group_8.ipynb` is the complete project: data loading through
to a live Gradio demo that accepts any uploaded plate image. `Runtime → Run
all`, then use the public Gradio link it prints at the end.

The demo accepts an image supplied **at demo time** — not only prepared ones.

---

## Results

Character classifier, held-out EMNIST (handwritten):

| Model | Test accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|
| Baseline MLP (control) | 84.1% | — | 542,500 |
| **CNN** | **90.1%** | 0.898 | 592,964 |

End-to-end pipeline on synthetic plates, 400 per condition:

| Condition | Segmentation | Character | Plate |
|---|---:|---:|---:|
| clean | 95.5% | 84.3% | 50.2% |
| normal | 73.0% | 62.4% | 34.0% |
| hard | 12.2% | 10.9% | 5.5% |

**The headline finding: segmentation, not the classifier, is what breaks.**
On 400 hard plates there were **351 segmentation failures vs 27 recognition
failures** — different bugs, different fixes, which is exactly why they are
counted separately.

**The business answer is uncomfortable, and we report it anyway.** At the
assumed $14 cost per misread, the optimal policy automates only 10% of reads
and the system **does not pay for itself**: −$20,399/year, 3-year NPV
−$230,729. The recommendation is a paid pilot at 2–3 sites, not an
estate-wide rollout. See [`docs/business_note.md`](docs/business_note.md).

---

## Repository layout

```
├── ML_FinalProject_Group_8.ipynb   ← THE deliverable. Everything runs here.
├── docs/                            ← the written deliverables
├── artifacts/                       ← trained model, metrics, provenance, figures
├── data/                            ← EMNIST (gitignored) + generated plates
└── anpr_package/                    ← the same pipeline as an installable
                                        package, with 53 tests
```

**The notebook is the source of truth.** `anpr_package/` is the Sprint-1
modular implementation, kept because it carries the automated test suite and
the runtime guards — evidence the notebook cannot show on its own. Both read
and write the *same* `artifacts/` and `data/`, so they never drift on the
model or the test set. See
[`anpr_package/README.md`](anpr_package/README.md), which also documents
where the two implementations currently differ.

---

## The pipeline

```
plate image
  → binarise            Otsu threshold, inverted (plates are dark-on-light)
  → segment             contours → filter → SPLIT merged glyphs → order L-to-R
  → normalise crop      20px longest side, centred by centre of mass, 28×28
  → CNN                 each crop → character + confidence
  → assemble            plate string; confidence = MINIMUM over characters
  → trust policy        auto-accept, or route to a human
```

Two design decisions worth defending:

**The same `to_mnist_format()` runs at training time and inference time.**
§4 names mismatched preprocessing as the most common cause of "high
validation accuracy, useless on real images"; one shared function makes that
drift structurally impossible.

**Plate confidence is the minimum, not the mean.** Six characters at 0.99 and
one at 0.30 average to 0.89 — which would sail past any sensible threshold and
bill the wrong customer on the one character that was actually wrong.

---

## Documents — what's in each

| File | What you'll find in it |
|---|---|
| [`docs/approach.md`](docs/approach.md) | **Main deliverable (30 pts).** Problem framing, pipeline decomposition, data decisions (EMNIST ByClass, case merge, 7.61× class imbalance), modeling choices and fallbacks, evaluation with measured numbers, spike summary, explicit scope, top-three risks |
| [`docs/spike.md`](docs/spike.md) | **Spike (10 pts).** Two experiments in "assumed X, tested it, found Y, changed Z" form — segmentation viability under tight kerning, and the handwriting→printed domain gap (+0.19) |
| [`docs/results.md`](docs/results.md) | Measured numbers with their conditions, confusion analysis, five named failure modes, explicit out-of-scope list. **The notebook generates a `results_summary.md` with the same content — reconcile the two before submitting rather than shipping both** |
| [`docs/business_note.md`](docs/business_note.md) | One page for the COO: the auto-accept policy, what it costs at 4,200 vehicles/day, the assumption that moves the answer most, and what four more weeks would buy |
| [`docs/ai_disclosure.md`](docs/ai_disclosure.md) | Where AI assistance was used (§6) |
| `docs/contributions/` | Individual contribution statements (private) |

### QA

| File | What you'll find in it |
|---|---|
| [`docs/qa/qa_acceptance_criteria.md`](docs/qa/qa_acceptance_criteria.md) | 68 acceptance criteria in Gherkin across 8 blocks, each traced to a Jira ticket, written **before** anything was measured. Block H doubles as the pre-submission checklist with each stated deduction attached |
| [`docs/qa/qa_execution_log.md`](docs/qa/qa_execution_log.md) | When each criterion was actually run, the result, the evidence file — including the ones that **failed** |

---

## Where the numbers come from

Never retype a number — cite the file:

| File | Contains |
|---|---|
| `artifacts/provenance/provenance.json` | data routes, source URIs, content hashes, results block |
| `artifacts/metrics/training_summary.json` | model accuracy, parameters, hyperparameters |
| `artifacts/metrics/tier_results.json` | per-tier accuracy + per-plate confidences |
| `data/generated/manifest.csv` | ground truth for 1,200 synthetic test plates |

---

## Constraints we work under

- **No off-the-shelf OCR as the classifier** — Tesseract, EasyOCR, PaddleOCR
  and cloud OCR APIs are −20 if used as the system's classifier. Ours is a
  CNN we trained.
- **No non-consensual plate photography** — §11 is a hard constraint; plate
  numbers are personal data. Every plate here is programmatically generated.
- **Every accuracy number states its conditions** — reporting accuracy
  without the quality tier is −5.
- **The repo runs from a clean clone** — one Colab click, or
  `pip install -e ./anpr_package` locally. Failing this is −8.
