# Results Summary

**Ticket ML-51 · Group 8 · ANPR prototype**

*Every number here comes from the executed outputs of
`ML_FinalProject_Group_8.ipynb`, which regenerates all of them end to end.*

---

## Conditions — read these before any number below

All plate-level figures are measured on **synthetically rendered plates**,
never photographs. No licensed plate corpus was used (Section 11: plate numbers are
personal data). Three degradation levels, reported separately:

| Tier | Skew | Blur | Lighting | Noise |
|---|---|---|---|---|
| **clean** | none | none | uniform | none |
| **normal** | 4.5% perspective | 3px Gaussian (70% of images) | gradient | σ=8 |
| **hard** | 8.5% perspective | 5px Gaussian | stronger gradient | σ=18 |

400 plates per tier, 7 characters each, fixed seed.

---

## 1. Character classifier — held-out EMNIST (handwritten)

| Model | Test accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|
| Baseline MLP (control) | 0.841 | — | 542,500 |
| **CNN** | **0.901** | **0.898** | 592,964 |

The MLP exists to make the CNN's number mean something: 6 points of accuracy
for 9% more parameters is the value of spatial structure, not of scale.

---

## 2. End-to-end pipeline, by condition

| Condition | Segmentation | Character | Plate |
|---|---:|---:|---:|
| clean | 0.955 | 0.843 | 0.502 |
| normal | 0.730 | 0.624 | 0.340 |
| hard | 0.122 | 0.109 | 0.055 |

**Segmentation and recognition are reported separately because they are
different bugs with different fixes.** Blending them into one "accuracy"
number would hide the finding below.

### The headline finding

On 400 hard-condition plates:

| Failure type | Count |
|---|---:|
| **Segmentation** (wrong character count) | **351** |
| **Recognition** (right count, wrong text) | **27** |

**Segmentation is the bottleneck by a factor of thirteen.** Better training
data would barely move the hard-tier number; the pipeline never gets far
enough to ask the classifier. The fix is architectural — a CTC sequence
model that skips explicit segmentation — not more epochs.

### Why plate accuracy is so much lower than character accuracy

Errors compound: plate accuracy ≈ character accuracy ^ 7.

At our measured 0.901 character accuracy, the predicted plate ceiling is
**0.483** — and to reach a 95% plate-level target you would need **0.993**
per character. That is the arithmetic that makes plate reading hard, and it
is why the trust threshold matters more than another point of accuracy.

---

## 3. The spike — the domain gap

| | Accuracy |
|---|---:|
| Handwritten (held-out EMNIST) | 0.816 |
| Printed (17 rendered fonts, 612 glyphs) | 0.623 |
| **Domain gap** | **+0.194** |

EMNIST is handwritten; plates are printed. Section 3 permits not closing this gap in
two weeks, provided it is measured and stated rather than assumed away. This
is that measurement, and every plate-level number above inherits it.

---

## 4. Confusion analysis

Top error pairs on the held-out EMNIST test set:

| True → Predicted | Error rate | n |
|---|---:|---:|
| O → 0 | 46.5% | 1,108 |
| L → 1 | 42.4% | 732 |
| I → 1 | 33.7% | 430 |
| Q → 9 | 19.5% | 92 |
| 0 → O | 18.8% | 560 |
| L → I | 12.7% | 220 |
| 1 → L | 10.2% | 332 |
| 1 → I | 9.5% | 309 |

**Weakest classes by recall:** L (0.433), O (0.511), I (0.574) — all members
of the O/0/L/1/I cluster, which is a *shape* problem, not a training-data
problem.

### We pre-registered our predictions, then checked them

`CONFUSABLE_PAIRS` was written down **before** training. Six of ten were
confirmed at >1% error rate:

| Predicted pair | Confirmed? |
|---|---|
| 0 ↔ O | ✅ |
| 1 ↔ I | ✅ |
| 5 ↔ S | ✅ |
| 2 ↔ Z | ✅ |
| 6 ↔ G | ✅ |
| D ↔ O | ✅ |
| U ↔ V | ✅ |
| 8 ↔ B | ❌ not observed |
| 7 ↔ T | ❌ not observed |
| 4 ↔ A | ❌ not observed |

The three misses are as informative as the hits: the vertical-stroke cluster
(0/O/1/I/L) dominates, while the pairs we expected to confuse on *curvature*
(8/B, 4/A) did not. **A practical consequence:** many jurisdictions already
exclude I, O and Q from issued plates. If Meridian's do, masking those three
outputs is free accuracy.

---

## 5. Named failure modes

1. **Domain gap.** EMNIST is handwritten, plates are printed. Largest single
   source of error, measured at +0.194.
2. **Segmentation on touching or broken glyphs.** Connected components cannot
   split merged characters — 351 of 378 hard-tier failures.
3. **Compounding.** Plate accuracy is character accuracy to the 7th power.
4. **Case-ambiguous glyphs.** m/M, o/O, u/U are inseparable at 28×28; merged
   into 36 classes, which caps achievable accuracy on those characters.
5. **Synthetic evaluation.** Real plates add mud, screws, frames, IR
   illumination and motion blur at speed. Everything here is an upper bound.

---

## 6. Explicitly not attempted

- Plate localisation within a road scene (we assume a cropped plate)
- Multi-line and non-Latin plates
- Real photographic validation
- Closing the handwriting→printed domain gap
