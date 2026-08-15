# Individual Contribution Statement — Thenmani Sayebaba

**ISM 6642 · Group 8 · ANPR Prototype · ML-57**
*½ page. Submitted privately.*

> Replace every bracketed prompt with your own words, then delete this line
> and the prompts. The ticket list below is from the Jira board — check it
> before you sign it.

---

## What I owned

Data, the model, and end-to-end integration. Tickets assigned to me:

| Ticket | What it was |
|---|---|
| ML-39 | 36-class map and the class-imbalance decision |
| ML-42 | Baseline CNN plus the MLP control |
| ML-44 | Preprocessing parity between training and inference |
| ML-45 | Segmentation and recognition chained into an end-to-end read |
| ML-55 | Backup demo recording |

Also: `ML_FinalProject_Group_8.ipynb` itself — the graded deliverable — and
the Week-1 EMNIST loading and orientation work.

## What I actually built

[Be specific enough that it can be checked against the notebook and the board.
Name cells or functions, not areas. You own the numbers that get asked about
live: CNN 0.901 test / MLP 0.841 control / 592,964 parameters / lr 1e-3 /
train 198,000, val 22,000, test 60,000.]

## The decision I would defend

[Pick one and explain the reasoning. Candidates: EMNIST ByClass over Balanced
and what that cost in imbalance (7.61× spread, corrected by class weighting
up to 3.91×); merging 62 classes to 36 knowing m/M, o/O and u/U are
inseparable at 28×28; using one shared preprocessing function so training and
inference cannot drift.]

## What I got wrong

[Optional but it reads well. Real candidate: the first EMNIST load silently
truncated at 129,461 rows against ByClass's 697,932 — worth describing how it
was caught and what check now makes it impossible.]

## Where I used AI assistance

[Be specific — see `../ai_disclosure.md`. The notebook was AI-scaffolded and
AI-debugged; the experimental design and the runs were yours. §6 is explicit
that "the model wrote it" is not an answer, so claim only what you can
explain from memory.]

## What I would do differently with another week

[Concrete and prioritised. The measured case: the domain gap is +0.194
handwritten→printed, so fine-tuning on rendered glyphs is the first move.]
