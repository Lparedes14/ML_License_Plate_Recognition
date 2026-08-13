# Notebooks

**The rule: notebooks import from `src/anpr/`. They do not contain logic.**

A cell should read:

```python
from anpr.data import load_emnist, prove_guard_fires
X, y, idx = load_emnist("train", 220_000, seed=42)
```

…not 200 lines of loader code. If you find yourself writing a function in a
notebook, it belongs in `src/anpr/` — move it there and import it. Two people
editing the same notebook produces merge conflicts that are effectively
unresolvable; two people editing different modules does not.

## Notebooks

| Notebook | Purpose | Ticket | Status |
|---|---|---|---|
| `spike_segmentation.py` | Spike: can connected components segment plates? Throwaway by design — a script, not a notebook, because it is not meant to be kept | ML-34 | **built** |
| `01_data_acceptance.ipynb` | Ten upright samples, class balance, split proportions | ML-36, ML-39 | to build |
| `03_results.ipynb` | Every figure in `docs/results.md` | ML-51 | to build |
| `04_demo.ipynb` | Colab demo — accepts an uploaded image | ML-47 | to build |

### Training happens in Colab, independently of this repo's notebook set

Thenmani trains the model directly in a Google Colab kernel and downloads the resulting
`.keras` file. That file (plus its `.classmap.json`) is what lands in `artifacts/models/`
and is what `src/anpr/inference/load_reader()` picks up locally — everything after training
(segmentation, evaluation, the demo) runs in VS Code against the downloaded model.

There is deliberately no `02_training.ipynb` in this repo: an earlier version of the plan
called for one, but the team's actual workflow doesn't route through it, so it was dropped
rather than kept as a file nobody runs.

## Running in Colab

The repo installs into Colab like any package, so the demo notebook stays thin:

```python
!git clone https://github.com/Lparedes14/ML_License_Plate_Recognition.git
%cd ML_License_Plate_Recognition
!pip install -q -e ".[download,demo]"

from anpr.inference import load_reader, read_plate
model, idx2char = load_reader("artifacts/models/plate_cnn.keras")
result = read_plate("uploaded_plate.png", model, idx2char, n_expected=7)
print(result.text, result.plate_confidence)
```

This satisfies both requirements at once: §9 wants a repo that runs from a
clean clone, and ML-47 wants a Colab demo. Same code, one source of truth.

## Before committing a notebook

Clear the outputs (`Kernel → Restart & Clear Output`). Notebook outputs are
enormous binary diffs — the Week-1 notebook was 310 KB, almost all of it
embedded images. Figures worth keeping belong in `artifacts/figures/`.

## The Week 1 notebook

`ML_Project_Group_8.ipynb` (in the parent folder) is the original Colab work.
Its data-layer code now lives in `src/anpr/data/`, module by module. Keep it
as a reference for what was run in Week 1; do not develop in it further.
