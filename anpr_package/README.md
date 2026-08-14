# `anpr_package/` — the modular Python implementation

**This is not the primary deliverable.** The graded artifact is the notebook
at the repository root:
[`ML_FinalProject_Group_8.ipynb`](../ML_FinalProject_Group_8.ipynb).

## What this folder is

The same pipeline as the notebook, factored into an installable Python
package with a test suite. It was built during Sprint 1 when the team was
working in VS Code; the team later consolidated on a single Colab notebook
so that everything runs in one place for the demo.

It is kept — not deleted — because it carries evidence the notebook cannot:

- **53 automated tests** covering the input contract, the orientation guard,
  label handling, metric counting rules and the cost model
- **Guards proven to fire**, not merely written (`prove_guard_fires()` feeds
  the orientation guard deliberately transposed data and fails if accepted)
- **Runtime assertions** that block double-normalisation and class-map
  mismatches structurally rather than by convention

## Running it

```bash
cd anpr_package
pip install -e ".[download,demo,dev]"
pytest                                   # 53 tests, ~4s, no GPU needed
```

Scripts, from inside this folder:

```bash
python scripts/prepare_data.py     # load + verify EMNIST, split, sign off
python scripts/make_test_plates.py # regenerate the synthetic plate test set
python scripts/evaluate.py         # measure at three quality tiers
python scripts/demo.py --image <path>
```

## How it shares state with the notebook

`config/` lives here (it describes how *this* code runs), but `data/` and
`artifacts/` deliberately resolve to the **repository root**, shared with the
notebook:

```
<repo root>/
├── ML_FinalProject_Group_8.ipynb   ← notebook writes artifacts/ + reads data/
├── artifacts/                       ← SHARED: model, metrics, provenance
├── data/                            ← SHARED: EMNIST + generated plates
└── anpr_package/
    ├── config/                      ← package-local
    └── src/anpr/config.py           ← PROJECT_ROOT = parents[3]
```

Both therefore read and write the *same* trained model and the *same*
manifest, rather than each keeping a private copy that silently drifts apart.

## Known divergence from the notebook

Worth stating plainly rather than discovering during a demo question — the
two implementations are **not** currently identical:

| | This package | Notebook |
|---|---|---|
| Plate font | single monospace (`LiberationMono-Bold`) | `random.choice(FONTS)` — mixed, some italic |
| Segmentation | v1 — no merged-glyph splitting | **v2** — splits merged glyphs at projection valleys |
| Clean segmentation | 93.0% | **95.5%** |
| Clean character accuracy | **95.3%** | 84.3% |

Each is better at one thing: the notebook's v2 segmentation is a genuine
improvement, and this package's consistent font is a genuine improvement.
Neither currently has both. Combining them (v2 splitting **and** a fixed
monospace font) is the single highest-value change available and is noted in
the project's open actions.
