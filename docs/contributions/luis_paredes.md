# Individual Contribution Statement — Luis Paredes

**ISM 6642 · Group 8 · ANPR Prototype · ML-57**
*½ page. Submitted privately.*

> Replace every bracketed prompt with your own words, then delete this line
> and the prompts. The ticket list below is from the Jira board — check it
> before you sign it.

---

## What I owned

Segmentation, QA/measurement, and the demo. Tickets assigned to me:

| Ticket | What it was |
|---|---|
| ML-43 | Character segmentation — contours, glyph filters, reading order |
| ML-47 | Demo accepting an image supplied at demo time |
| ML-48 | The deliberate failure cases |
| ML-49 | Accuracy at three tiers, segmentation vs recognition separated |
| ML-50 | Confusion matrix and the confusable-pair analysis |
| ML-51 | Results summary |
| ML-38 | Written definition of a correct read, agreed before measuring |

Also: repository structure, the 53-test suite, QA acceptance criteria and
execution log, and Jira administration.

## What I actually built

[Be specific enough that it can be checked against the commit history and the
board. Name files, not areas. The strongest single item is the v1→v2
segmentation fix — you can quote the before/after: clean 80.3%→94.7%,
normal 54.7%→74.0%, hard 9.3%→13.7%.]

## The decision I would defend

[Pick one and explain the reasoning, not just the outcome. Candidates: why a
wrong character *count* is excluded from character accuracy; why plate
confidence is the minimum and not the mean; why the orientation guard has a
self-test that feeds it deliberately transposed data.]

## What I got wrong

[Optional but it reads well. Real candidates: CA-D6 — the three tiers draw
independent random plate strings rather than the same 400 at three quality
levels, which weakens the tier comparison and was caught in QA rather than in
design. Or the `load_reader()` TensorFlow import ordering bug.]

## Where I used AI assistance

[Be specific — see `../ai_disclosure.md`. Over-disclosing costs nothing;
under-disclosing is an integrity problem. Note that the test suite and the QA
documents were AI-drafted, and that you must be able to explain them live.]

## What I would do differently with another week

[Concrete and prioritised. The honest answer from the measurement work:
segmentation is the bottleneck by 13×, so a CTC sequence model beats any
amount of additional classifier training.]
