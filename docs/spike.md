# Spike

**Tickets ML-33, ML-34, ML-35 · Worth 10 of 100 points · Owner: Luis Paredes**

> §3: *"The spike is graded on whether it changed your plan, not on whether
> it succeeded."*

We identified two candidate risky assumptions from §3's list — both cheap
enough to test in the time available — and ran both rather than choosing one.
Each changed the plan.

---

## Spike 1 — Can we even segment characters reliably?

**The assumption:** threshold a plate image, run connected components, count
the blobs — one per character?

**Why this one.** Everything downstream assumes segmentation works.
If connected components cannot reliably return one blob per character, no
amount of classifier accuracy saves the system — a 7-character plate read as
6 characters is wrong before the model is even consulted.

### What we did

A throwaway script (`anpr_package/notebooks/spike_segmentation.py`, deliberately not
production code): render synthetic plates with a system font, threshold with
Otsu, run `cv2.connectedComponentsWithStats`, filter blobs by size and
aspect ratio, count what survives. 40 plates per condition, three baseline
"quality levels," then two stress tests targeting the parameters most likely
to break it.

### What we found

The baseline levels (clean / realistic / degraded blur+noise) all scored
**100% exact** — which was itself informative: it meant the baseline test
was too easy and wasn't stressing the thing that actually matters.

Two stress tests found the real breaking points:

| Character spacing (kerning) | Exact match | Failure mode |
|---:|---:|---|
| 1.00 (comfortable) | 100% | — |
| 0.85 | 90% | under-segmentation |
| 0.75 | 70% | under-segmentation |
| 0.55 | **0%** | glyphs fully merged |

| Blob-sized speckle / plate | Exact match |
|---:|---:|
| 0 | 100% |
| 20 | 70% |
| 60 | **2%** |

And a second, equally important finding: at the "degraded" blur/noise level,
the blob **filters** were doing 100% of the work — the same run with
filtering disabled scored **0%** (1,442 stray blobs per plate). The filters
are not a tuning detail; they are the segmenter.

### What changed in the plan

1. **Kerning became a tracked variable.** It wasn't in the original tier
   design (`anpr_package/config/default.yaml`'s tiers vary blur/noise/rotation/contrast —
   none of which broke anything in this spike). It is the one parameter that
   *did* break segmentation, and it wasn't being measured at all.
2. **The plate generator's font/spacing choices matter more than image
   quality.** Communicated to the plate-generator owner before that ticket
   was built, not after.
3. **Tier "hard" needed rethinking.** The original design used pixel-level
   noise, which the area filter removes trivially. The stress test shows the
   real failure mode is *larger* artifacts (touching characters, blob-sized
   debris) — that's what the degraded tier should model.
4. **Confirmed at full scale, later.** Sprint 2's full 1,200-plate
   evaluation (`artifacts/metrics/tier_results.json`) shows segmentation
   success collapsing from 82.2% (clean) to 9.8% (hard) — the spike's warning,
   not an unexpected surprise when it showed up in the real numbers.

---

## Spike 2 — Will a model trained on handwriting read printed plates at all?

**The assumption:** EMNIST is handwritten; plates are printed. Does that gap
actually cost accuracy, or is it a theoretical concern?

**Why this one.** It is the assumption every downstream training decision
rests on, and unlike segmentation it cannot be fixed by better engineering —
only by measuring it and deciding what to do about it.

### What we did

A deliberately small, 2-epoch CNN (throwaway, not the final architecture),
trained on 40,000 EMNIST samples in `ML_Draft1_Project.ipynb`. Evaluated on
two held-out sets: the standard handwritten EMNIST test split, and 612
characters rendered from 17 system fonts and run through the same
EMNIST-convention preprocessing (`to_mnist_format`).

### What we found

| | Accuracy |
|---|---:|
| Handwritten (held-out EMNIST) | 0.814 |
| Printed (rendered fonts) | **0.627** |
| **Domain gap** | **+0.187 (23% relative drop)** |

Worst printed characters: **M, Q, G at 0.000%**; 6 and K at 5.9%.

### What changed in the plan

1. **The domain gap is real and material, not a theoretical footnote.**
   §3 explicitly permits not closing it in two weeks — but only if we
   *measure and state* it rather than assume it away. This spike is that
   measurement.
2. **It reframes what "high accuracy" would mean.** A model scoring 90% on
   EMNIST's test set is not a model scoring 90% on real plates. Every number
   in the results document has to carry this caveat.
3. **It names a concrete Week 3+ direction without committing scope to it
   now:** a short fine-tune pass on rendered printed glyphs
   (`anpr_package/config/default.yaml`'s `epochs_ft`, `lr_ft` — already provisioned for
   this, not yet run). Explicitly listed under "what four more weeks would
   buy," not under "what we will build this sprint."
4. **A secondary, unplanned finding:** M and Q both failing completely
   suggests the font list (17 general-purpose system fonts, several italic)
   may be harder than a real plate typeface — worth a follow-up before
   trusting the 23% figure as *the* domain gap rather than an upper bound
   on it.

---

## Bonus finding — the truncated data file (Sprint 0/1 boundary)

Not a formal spike, but the same shape of result: our first EMNIST load
reported **129,461 training rows where ByClass has 697,932** — a silently
truncated local file. The row-count check in `_load_kaggle_csv` caught it
and the loader fell through to the next route rather than training on 18%
of the data with no error. That check is now permanent
(`anpr_package/tests/test_emnist.py`, 4 tests) and has already prevented the same class of
failure from recurring on a teammate's machine with a different EMNIST
download.

---

## Summary for the approach document (§6)

Both spikes changed the plan before a week of work was built on top of an
untested assumption. Segmentation is viable but fragile to character spacing
in a way the original tier design didn't account for — fixed by adding
kerning to the tracked variables. The handwriting→printed gap is real
(23% relative), named explicitly rather than discovered late, and scoped
into "future work" rather than something we silently assumed away.
