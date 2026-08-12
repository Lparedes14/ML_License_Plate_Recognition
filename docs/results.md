# Results Summary

**Ticket ML-51 · 2 pages · Metrics, conditions, confusion analysis, failure modes**

> **Template — not yet written.** Fill from
> `artifacts/metrics/tier_results.json` and `artifacts/metrics/confusion.json`.
> Quote the generated files; do not retype numbers by hand.
>
> §1: *"A team reporting 61% character accuracy with a clear-eyed analysis of
> why will score higher than a team claiming 99% they cannot reproduce on a
> new image."*

---

## Headline numbers

Every row states its conditions. A number without a tier and a sample size is
a −5 deduction.

| Tier | Conditions | n | Character acc. | Plate acc. | Segmentation |
|---|---|---|---|---|---|
| A | clean, straight-on, high contrast | | | | |
| B | realistic camera capture | | | | |
| C | motion blur, low light, steep angle | | | | |

**Model:** … **Trained on:** … samples · **Validated on:** … · **Seed:** 42

---

## Why plate accuracy is so much lower than character accuracy

A plate is correct only when every character is. At *p* per character over 7
characters the ceiling is *p*⁷ — 0.95⁷ = 70%, 0.90⁷ = 48%. Our measured plate
accuracy is (higher / lower) than that bound because errors are (correlated /
independent) — a blurry plate is blurry for all seven characters.

---

## Segmentation vs recognition

Counted separately from day one, as §4 requires.

| Failure type | Count | Share |
|---|---|---|
| Segmentation (wrong character count) | | |
| Recognition (right count, wrong character) | | |

---

## Confusion analysis — ML-50

We pre-registered `CONFUSABLE_PAIRS` in `anpr/config.py` before training. How
did the prediction hold up?

| Pair | Predicted? | Confusions | Rate |
|---|---|---|---|

**Per-class recall:** watch Q, which is ~0.8% of EMNIST but ~2.8% of a
uniform plate alphabet.

**Recommendation:** if confusion concentrates on I/1, O/0 and Q/O, ask
Meridian to exclude those three from issued plates — it costs nothing and
beats any amount of retraining.

---

## Failure modes

Named without being asked (§10).

1.
2.
3.

---

## Baseline comparison

| Model | Character acc. (tier B) | Parameters |
|---|---|---|
| MLP (control) | | |
| CNN | | |

---

## Reproducibility

- Seed: 42 · config: `config/default.yaml`
- Data provenance: `artifacts/provenance/provenance.json`
- Split fingerprints: `artifacts/provenance/split_manifest.json`
- Every number here regenerates with `python scripts/evaluate.py`
