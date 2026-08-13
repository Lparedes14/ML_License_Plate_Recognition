# QA Execution Log — ANPR Prototype

| | |
|---|---|
| **Project** | ML — Group 8 Final Project (ISM 6642) |
| **Criteria document** | [`qa_acceptance_criteria.md`](qa_acceptance_criteria.md) — 68 criteria across 8 blocks |
| **QA owner** | Luis Paredes |
| **Opened** | 12 August 2026 |
| **Submission deadline** | Friday 14 August 2026 |

**What this document is.** The criteria document says what we agreed to verify. The
artifacts in `artifacts/` hold what we measured. This log is the link between them: it
records *when* each criterion was actually run, *what happened*, and *where the evidence
lives*.

Without it the criteria document reads as aspirational. At the demo the chain has to hold:
criteria agreed in advance → log showing they were run → raw numbers → the narrative in
`docs/results.md`.

---

## How to use this

Fill a row the moment a criterion is exercised. Do not batch it up on Friday — the point is
the date, and a date written from memory is not evidence.

| Status | Meaning |
|---|---|
| **PASS** | Verified **in this repository**, evidence file named |
| **FAIL** | Verified and failed. Record what happened; do not delete the row |
| **PENDING** | Not yet run. Nothing blocking it |
| **BLOCKED** | Cannot run — the code it tests does not exist yet. Blocker named |
| **N/A** | Does not apply. Justify |

**One rule that matters:** evidence produced by the Week-1 Colab notebook does **not** count
as PASS. The modules were refactored during the repo restructure (globals became
parameters), and `scripts/prepare_data.py` has never been executed. Those criteria are
PENDING with the Week-1 observation recorded as *expected* result — which makes any
disagreement on first run visible rather than invisible.

---

## Status summary

Updated at each sync point.

| Block | Total | PASS | PENDING | BLOCKED | FAIL |
|---|---:|---:|---:|---:|---:|
| A — Data integrity & input contract | 10 | 4 | 6 | 0 | 0 |
| B — Character classification | 7 | 1 | 6 | 0 | 0 |
| C — Segmentation | 8 | 0 | 0 | 8 | 0 |
| D — End-to-end plate read | 7 | 3 | 0 | 4 | 0 |
| E — Trust threshold & business policy | 7 | 3 | 3 | 1 | 0 |
| F — Demo & reproducibility | 9 | 3 | 1 | 5 | 0 |
| G — Edge & negative cases | 12 | 1 | 1 | 10 | 0 |
| H — Reporting & submission compliance | 8 | 1 | 7 | 0 | 0 |
| **Total** | **68** | **16** | **24** | **28** | **0** |

**Read this as: 16 verified, 28 waiting on code that does not exist yet.** The blocked
count falls sharply once ML-37 (plate generator) and ML-43 (segmentation) land — those two
tickets gate 28 of the 68 criteria between them.

---

## Test suite runs

| Date | Command | Result | Notes |
|---|---|---|---|
| 12 Aug 2026 | `pytest` | **42 passed** | Clean run, no TensorFlow installed locally — TF-dependent paths untested |
| | | | |

> TensorFlow is not installed on the QA machine. Every test in the suite avoids it via lazy
> imports, so the suite passes — but `models/`, `data/pipeline.py` and the training scripts
> are **not covered by any executed test**. Recorded here so it is not mistaken for coverage.

---

## BLOCK A — Data integrity and input contract

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-A1 | HARD | **PASS** | 12 Aug | `tests/test_orientation.py::test_rejects_transposed_glyphs` | Synthetic glyphs; guard raises as specified |
| CA-A2 | HARD | **PASS** | 12 Aug | `tests/test_orientation.py::test_prove_guard_fires_passes_on_good_data` | The guard is proven to fire |
| CA-A3 | HARD | PENDING | | | Week-1 expected: rejects a 129,461-row file against 697,932. **No automated test — write one** |
| CA-A4 | HARD | **PASS** | 12 Aug | `tests/test_contract.py::test_normalize_refuses_float_input` | Double-normalisation blocked |
| CA-A5 | HARD | **PASS** | 12 Aug | `tests/test_contract.py::test_canonicalises_any_input_shape` | 5 input shapes covered |
| CA-A6 | HARD | PENDING | | | Week-1 expected: overlap 0. Needs `prepare_data.py` |
| CA-A7 | HARD | PENDING | | | Week-1 expected: max drift 0.003 pp val, 0.149 pp test |
| CA-A8 | REPORT | PENDING | | | Week-1 expected: 0 duplicates across all pairs |
| CA-A9 | HARD | PENDING | | | Needs a run to produce `provenance.json` |
| CA-A10 | HARD | PENDING | | | Requires two runs on different machines to compare fingerprints |

**Action:** CA-A3 has no automated test despite having caught a real failure in Week 1. It
is the cheapest test in the project to add and the one with a proven track record.

---

## BLOCK B — Character classification

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-B1 | HARD | PENDING | | | Class-map round-trip passes (`test_labels.py`); the three-artifact check needs a training run |
| CA-B2 | REPORT | PENDING | | | Blocked on ML-42. **Run exactly once** |
| CA-B3 | REPORT | PENDING | | | Needs the confusion matrix |
| CA-B4 | REPORT | PENDING | | | Needs both CNN and MLP trained |
| CA-B5 | REPORT | PENDING | | | Needs `--no-augment` comparison run |
| CA-B6 | REPORT | PENDING | | | Needs both imbalance arms |
| CA-B7 | HARD | **PASS** | 12 Aug | Code inspection — `scripts/train.py` | Only train and validation passed to `fit()`; test split never loaded |

---

## BLOCK C — Segmentation

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-C1 | HARD | BLOCKED | | | ML-43 — `binarize()` not implemented |
| CA-C2 | REPORT | BLOCKED | | | ML-43 |
| CA-C3 | HARD | BLOCKED | | | ML-43 — `sort_reading_order()` written, untested |
| CA-C4 | HARD | BLOCKED | | | ML-44 — **highest-risk criterion in the project** |
| CA-C5 | HARD | BLOCKED | | | ML-43 |
| CA-C6 | REPORT | BLOCKED | | | ML-43 |
| CA-C7 | HARD | BLOCKED | | | ML-43 |
| CA-C8 | REPORT | BLOCKED | | | ML-43 + ML-37 |

**All 8 blocked on one ticket.** CA-C4 (centre-of-mass crop normalisation) needs its test
written *at the same time as the code*, not after — the brief names this exact mismatch as
the most common cause of "high validation accuracy, useless on real images".

---

## BLOCK D — End-to-end plate read

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-D1 | HARD | **PASS** | 12 Aug | RN-01 in the criteria document; ML-38 closed | Definition agreed before any evaluation ran |
| CA-D2 | REPORT | **PASS** | 12 Aug | `tests/test_metrics_and_business.py::test_segmentation_failure_excluded_from_character_accuracy` | Counting rule verified |
| CA-D3 | REPORT | **PASS** | 12 Aug | `tests/test_metrics_and_business.py::test_plate_accuracy_is_all_or_nothing` | Counting rule verified |
| CA-D4 | HARD | BLOCKED | | | ML-45 |
| CA-D5 | REPORT | BLOCKED | | | ML-37 + ML-43 + ML-45 |
| CA-D6 | HARD | BLOCKED | | | ML-37 — same seed across tiers |
| CA-D7 | HARD | BLOCKED | | | ML-45 |

> CA-D2 and CA-D3 are PASS on the *counting logic* — the rule is verified in isolation. The
> numbers they will eventually carry are not yet measured. Do not read these as "accuracy
> verified".

---

## BLOCK E — Trust threshold and business policy

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-E1 | HARD | PENDING | | | `aggregate_confidence()` implemented, **no direct test** — add one |
| CA-E2 | HARD | **PASS** | 12 Aug | `tests/test_metrics_and_business.py::test_threshold_sweep_finds_a_cheaper_policy...` | Full 0.00–1.00 sweep, 101 points |
| CA-E3 | HARD | BLOCKED | | | ML-37 — needs the 400/400 calibration/reporting split |
| CA-E4 | HARD | **PASS** | 12 Aug | `tests/test_metrics_and_business.py::test_matches_the_brief_30000_wrong_bills` | 30,660 vs the brief's "roughly 30,000" |
| CA-E5 | REPORT | **PASS** | 12 Aug | Same test file — mechanism verified on synthetic confidences | Real data still required |
| CA-E6 | HARD | PENDING | | | Assumption labels present in code; needs the written business note |
| CA-E7 | HARD | PENDING | | | `policy_sentence()` implemented; needs real numbers |

---

## BLOCK F — Demo and reproducibility

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-F1 | HARD | **PASS** | 12 Aug | `pip install -e .` succeeded; `import anpr` returns 0.1.0 | **Verified on one machine only — see CA-H2** |
| CA-F2 | HARD | **PASS** | 12 Aug | 42 passed | Caveat in the test-runs section above |
| CA-F3 | HARD | PENDING | | | `--image` accepts any path; untestable until the pipeline runs |
| CA-F4 | HARD | BLOCKED | | | Needs a working pipeline and an unseen image |
| CA-F5 | HARD | BLOCKED | | | ML-45 |
| CA-F6 | HARD | BLOCKED | | | ML-48 — choose the failure case Friday, not Saturday |
| CA-F7 | HARD | BLOCKED | | | ML-42 must produce a model first |
| CA-F8 | HARD | **PASS** | 12 Aug | `grep -rn "google.colab\|/content\|MyDrive\|drive.mount" src/ scripts/ tests/ config/` | Two hits, both accepted — **see the finding below** |
| CA-F9 | HARD | BLOCKED | | | ML-55 — recording made Friday |

### Finding QA-01 — `/content` appears in the shipped source (CA-F8, accepted)

Recorded here so that nobody is surprised by it, including an instructor who greps for it.

The check found two hits, both in `src/anpr/data/emnist.py`:

- line 82 — a docstring mentioning Colab's `/content`
- line 96 — `roots = search_dirs or [RAW_DATA_DIR, Path("/content"), Path(".")]`

**Assessed as PASS, deliberately.** ML-46's acceptance criterion names `/content/drive/` —
the personal Drive mount a grader cannot open. This is not that:

1. It is one of three fallback search roots, not a hardcoded destination.
2. `_find_emnist_csv()` calls `root.is_dir()` before using any root, so on a non-Colab
   machine it is skipped silently. It cannot break a clean clone.
3. It is there on purpose: the same loader has to work when the team uploads EMNIST into a
   Colab session, which is how ML-47's demo notebook will run.

**If challenged in the demo, the answer is the three points above.** If the team would
rather remove the ambiguity entirely, move `/content` out of the default list and into the
Colab notebook as an explicit `search_dirs` argument — a five-minute change. Not doing it
is a judgement call, recorded here rather than left implicit.

---

## BLOCK G — Edge and negative cases

| CA | Gate | Status | Date | Evidence | Notes |
|---|---|---|---|---|---|
| CA-G1 | HARD | BLOCKED | | | ML-43 — must return empty, never fabricate |
| CA-G2 | REPORT | BLOCKED | | | ML-43 |
| CA-G3 | REPORT | BLOCKED | | | ML-43 |
| CA-G4 | HARD | BLOCKED | | | Needs a non-plate test image — prepare one Friday |
| CA-G5 | HARD | BLOCKED | | | `read_plate()` raises `FileNotFoundError`; untested |
| CA-G6 | HARD | BLOCKED | | | Needs an all-black / all-white test image |
| CA-G7 | HARD | PENDING | | | `load_reader()` raises on a missing map — **no test yet** |
| CA-G8 | HARD | **PASS** | 12 Aug | `tests/test_labels.py::test_class_map_detects_a_reordered_charset` | Reordered charset refused |
| CA-G9 | REPORT | BLOCKED | | | ML-50 |
| CA-G10 | REPORT | BLOCKED | | | ML-42 |
| CA-G11 | REPORT | BLOCKED | | | ML-52 with real data |
| CA-G12 | REPORT | BLOCKED | | | ML-49 tier C |

**Three cheap tests missing:** CA-G5, CA-G6 and CA-G7 need no new production code — only a
test each. Worth 30 minutes total, and they are exactly the cases an instructor probes when
handed the keyboard.

---

## BLOCK H — Reporting and submission compliance

**This is the Friday evening checklist. Run it top to bottom before submitting.**

| CA | Gate | Penalty | Status | Date | Verified by | Notes |
|---|---|---|---|---|---|---|
| CA-H1 | HARD | **−20** | **PASS** | 12 Aug | Luis | No OCR engine in `pyproject.toml` or any import |
| CA-H2 | HARD | **−8** | PENDING | | | Verified on **one** machine. Needs Thenmani and Valentina to confirm independently |
| CA-H3 | HARD | **−5** | PENDING | | | Check every number in every document carries dataset, tier and sample size |
| CA-H4 | HARD | **−5** | PENDING | | | Depends on CA-F3, CA-F4 |
| CA-H5 | HARD | **ZERO** | PENDING | | | Confirm every image is programmatically generated. **Check the slide deck too** |
| CA-H6 | HARD | — | PENDING | | | AI disclosure specific, and each owner can explain their code |
| CA-H7 | HARD | — | PENDING | | | All six §9 deliverables exist |
| CA-H8 | HARD | — | PENDING | | | Every number traces to a file. **Verified by Valentina, not by whoever measured** |

---

## Open QA actions

Ranked. Items 1–4 need no production code and can be done today.

| # | Action | CA | Effort | Owner |
|---|---|---|---|---|
| ~~1~~ | ~~Grep for hardcoded Colab/Drive paths~~ | CA-F8 | done | Luis — see Finding QA-01 |
| 2 | Add a test for a missing class map | CA-G7 | 10 min | Luis |
| 3 | Add tests for unreadable file and uniform image | CA-G5, CA-G6 | 20 min | Luis |
| 4 | Add a test for the truncated-file guard | CA-A3 | 20 min | Luis |
| 5 | Add a direct test for `aggregate_confidence()` | CA-E1 | 10 min | Luis |
| 6 | Run `prepare_data.py` and clear the six PENDING Block A rows | CA-A6 – CA-A10 | 30 min | Thenmani |
| 7 | Confirm clean clone on the other two machines | CA-H2 | 15 min | All three |
| 8 | Write the CA-C4 centre-of-mass test **alongside** the code | CA-C4 | — | Luis |
| 9 | Prepare hostile demo images (non-plate, all-black, corrupt) | CA-G4, G5, G6 | 15 min | Luis |

---

## Sign-off

Completed at the Friday evening sync. Nothing is submitted until all three rows are signed.

| Role | Name | Statement | Signed |
|---|---|---|---|
| QA owner | Luis Paredes | All HARD criteria are PASS or have a documented, accepted exception | |
| Data/Model owner | Thenmani Sayebaba | Every number I produced traces to a file in `artifacts/` | |
| Final read-through | Valentina Valdez | No figure in any document is untraceable to a generated artifact | |

**Exceptions accepted at submission** — any HARD criterion not passing, with the reason and
who accepted it. An empty table means everything passed; a documented exception is
acceptable, an undocumented gap is not.

| CA | Why it did not pass | Accepted by | Impact stated in |
|---|---|---|---|
| | | | |
