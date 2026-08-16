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

**[▶ Try the live demo](https://bd9be50f40d7c88f77.gradio.live/)** — the
Gradio link from the most recent run. Gradio share links expire after the
Colab session ends (usually within a few hours), so if it's down, re-run the
notebook and swap in the fresh link it prints.

### Or run the demo locally — no share link, nothing to expire

[`demo_app.py`](demo_app.py) serves the same interface from the trained model
in `artifacts/models/`, on `http://127.0.0.1:7860`:

```bash
python demo_app.py
```

Needs `tensorflow==2.20.0`, `gradio`, `opencv-python` and `pillow`. There is
also a headless mode for a quick check without a browser:

```bash
python demo_app.py --image "TEST PLATE.png"
```

`demo_app.py` is a convenience runner, **not** a second implementation: it
copies the pipeline functions verbatim from the notebook and deliberately
computes no accuracy, tier or business numbers. Every figure in `docs/` comes
from executing the notebook.

> **Windows: installing TensorFlow can fail with `OSError [Errno 2]`.** This
> repo sits under a long path, and Windows caps full paths at 260 characters
> unless `LongPathsEnabled` is set. TensorFlow's own nested paths then push
> past the limit. Either enable long paths (registry, needs admin), or create
> the virtualenv somewhere short — e.g. `python -m venv C:\Users\<you>\.venvs\anpr`
> — and install into that.

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
└── data/                            ← EMNIST (gitignored) + generated plates
```

**The notebook is the whole project.** There is no separate package —
data loading, training, segmentation, evaluation and the demo all run in one
place, top to bottom, in Colab.

---

## The pipeline

```
plate image
  → binarise            auto-thresholded, inverted (plates are dark-on-light)
  → segment             contours → filter → SPLIT merged glyphs → order L-to-R
  → normalise crop      20px longest side, centred by centre of mass, 28×28
  → CNN                 each crop → character + confidence
  → assemble            plate string; confidence = MINIMUM over characters
  → trust policy        auto-accept, or route to a human
```

Two design decisions worth defending:

**The same `to_mnist_format()` runs at training time and inference time.**
Section 4 names mismatched preprocessing as the most common cause of "high
validation accuracy, useless on real images"; one shared function makes that
drift structurally impossible.

**Plate confidence is the minimum, not the mean.** Six characters at 0.99 and
one at 0.30 average to 0.89 — which would sail past any sensible threshold and
bill the wrong customer on the one character that was actually wrong.

---

## Deliverables (Section 9 of the brief)

| # | File | What's in it |
|---|---|---|
| 1 | [`docs/approach.md`](docs/approach.md) | **Main deliverable (30 pts), 3–4 pp.** Problem framing, pipeline decomposition, data decisions (EMNIST ByClass, case merge, 7.61× imbalance), modeling and fallbacks, evaluation, **the spike (Section 6, 10 pts)**, explicit scope, risks, and the Section 6 AI-assistance appendix |
| 2 | This repo + the Colab badge above | Runs from a clean clone; one command is "open the notebook, Run all" |
| 3 | [`docs/results.md`](docs/results.md) | **2 pp.** Measured numbers with their conditions, confusion analysis, five named failure modes, explicit out-of-scope list |
| 4 | [`docs/business_note.md`](docs/business_note.md) | **1 p for the COO.** The auto-accept policy, what it costs at 4,200 vehicles/day, the assumption that moves the answer most, what four more weeks would buy |
| 5 | `ML_FinalProject_Group_8.ipynb` + backup recording | Live demo, 10 min. Failure cases in [`demo_images/`](demo_images/) |
| 6 | `docs/contributions/` | Individual statements, ½ page each, **submitted privately** |

The spike is **Section 6 of the approach document**, not a separate file — Section 3 of the
brief lists it as a row of that document.

### Supporting material — not graded deliverables

Kept because it is evidence of how the work was done, not because Section 9 asks for it.

| File | What's in it |
|---|---|
| [`docs/qa/qa_acceptance_criteria.md`](docs/qa/qa_acceptance_criteria.md) | 68 acceptance criteria in Gherkin across 8 blocks, each traced to a Jira ticket, written **before** anything was measured |
| [`docs/qa/qa_execution_log.md`](docs/qa/qa_execution_log.md) | When each criterion was run, the result, the evidence file — including the ones that **failed** |
| [`docs/ai_disclosure.md`](docs/ai_disclosure.md) | Full per-item attribution and the from-memory drill sheet. The graded disclosure is the appendix in `approach.md` |

---

## Constraints we work under

- **No off-the-shelf OCR as the classifier** — Tesseract, EasyOCR, PaddleOCR
  and cloud OCR APIs are not used as the system's classifier. Ours is a CNN
  we trained.
- **No non-consensual plate photography** — Section 11 is a hard constraint; plate
  numbers are personal data. Every plate here is programmatically generated.
- **Every accuracy number states its conditions** — always reported with the
  quality tier it was measured under.
- **The repo runs from a clean clone** — one Colab click opens and runs the
  entire project.
