# `demo_images/`

Plate images for the live demo (§8). The notebook generates these — download
them from the Colab session and commit them here so the demo does not depend
on a live runtime.

## What belongs here

**One happy path**, to show the pipeline working end to end.

**Three failures, each a *different* failure mode.** One bad image proves
nothing; showing that the system fails in distinguishable, diagnosable ways
is the point — and §8 warns that teams which only show successes will be
asked to produce a failure live, "and it will go worse."

| File | Intended failure mode | What the system does |
|---|---|---|
| `normal_*.png` | none — happy path | reads the plate, or flags low confidence |
| `fail_1_segmentation_hard.png` | **segmentation** — hard-tier degradation | finds 0 characters → `REVIEW` |
| `fail_2_recognition_L1.png` | **recognition** — L/1 confusion on a clean plate | reads `IIB1178` for `LLB1178` → `REVIEW` on low confidence |
| `fail_3_not_a_plate.png` | **out of scope** — not a plate at all | declines rather than inventing a read |

## Why the third one matters

The honest answer to *"what if I upload something random?"* is `REVIEW`, not
a confident wrong read. Proving the system **declines** is a stronger
demonstration than proving it succeeds — it is the behaviour that protects
Meridian from billing the wrong customer.

## Note on the current failure images

All three currently return `REVIEW`, which is the correct outcome — but two
of them get there via *low confidence* rather than via the mechanism named in
their filename. `fail_3_not_a_plate.png` reads `PARKIGO` at 0.56 confidence
rather than finding no characters. Worth either renaming the file to match
what actually happens, or picking an image that fails the way the name
claims, before the demo.
