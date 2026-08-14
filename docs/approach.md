# Approach Document

**Group 8 · ISM 6642 · Meridian Access Systems ANPR Prototype**

> **Status: fully drafted, pending review + trim (ML-32).** All eight
> sections now have real content grounded in committed code, data and
> measured results — no section is still a bare template. Two different
> confidence levels, though:
>
> - **§3 (Data), §5 (Evaluation), §6 (Spike)** were written and are owned by
>   Luis directly — Jira ML-28/30/35 closed against these.
> - **§1, §2, §4, §7, §8** (Business/Pipeline/Model/Scrum Lead sections)
>   were drafted by Luis as a starting point, using the real committed
>   numbers throughout, but are **marked `> Draft —` inline** and need each
>   actual owner's review before this stops being a first draft.
>
> **Known issue: this draft runs ~5–6 pages against the brief's 3–4 page
> target** (`wc -w` ≈ 3,080 words). Content over length is a cheaper problem
> to have than missing content, but it needs a real trim pass, not a
> word-count exercise — cut the sections least central to the grading
> rubric first, not evenly across all eight. That trim is ML-32.
>
> Worth **30 of 100 points**, the single largest component. Section
> headings below are taken verbatim from §3 of the brief; do not rename
> them. §10: *"decisions justified rather than asserted; scope stated
> explicitly including what's out; risks named before they materialize."*

---

## 1. Problem framing
*What Meridian actually needs; what a misread costs.* — ML-26

> *Draft — Business owner's section. Numbers are verified against
> `anpr.business.CostModel`; the framing and emphasis should be reviewed
> and adjusted by whoever presents this to the class.*

Meridian Access Systems runs gated parking at 34 sites — airports,
residential communities, parks — using ticket dispensers that cost roughly
$14,000 installed and jam often enough that lost-ticket disputes cost the
call centre about $8.50 to resolve. The COO's question is whether plate
recognition can replace the ticket entirely: the plate becomes the account,
the gate opens on entry, the customer is billed on exit.

The number that makes this a hard problem rather than an easy one: at 4,200
vehicles/day (1,533,000/year), a plate-level error rate of just 2% is
**~30,660 wrong bills a year** — each one a refund, a support call, and an
annoyed regular, not a neutral statistic. Every wrong bill also risks the
one thing a subscription business cannot afford to lose repeatedly: the
customer's trust that they'll be charged correctly. Against that, full
automation would avoid **$476,000** in dispenser hardware across all 34
sites (`CostModel.total_dispenser_capex`).

**The question this project answers is therefore not "how accurate can we
get."** It's "which reads do we trust automatically, and which do we send to
a human" — because a system that knows when it's unsure is worth more to
Meridian than one that is simply more often right. Where that line should
sit, worked out in dollars against the measured confidence data, is the
subject of the business note (`docs/business_note.md`), built directly on
the accuracy numbers in §5 below.

---

## 2. Pipeline
*Diagram + one paragraph per stage.* — ML-27

> *Draft — Pipeline owner's section. Technical description matches the
> shipped code; please correct anything that doesn't match your intent for
> how this is presented.*

```
 plate image
     │
     ▼
 ┌─────────────────┐   threshold to black/white, white ink on black
 │   binarize      │   background (inverted if the source is dark-on-light)
 └─────────────────┘
     │
     ▼
 ┌─────────────────┐   connected components → N bounding boxes, filtered
 │  segment.        │   by size/aspect, sorted left to right
 │  components      │
 └─────────────────┘
     │
     ▼
 ┌─────────────────┐   each crop → 28×28 float32 [0,1], centred by centre
 │  data.contract   │   of mass — the SAME function used at training time
 └─────────────────┘
     │
     ▼
 ┌─────────────────┐   each crop → predicted class + softmax confidence
 │  CNN classifier  │
 └─────────────────┘
     │
     ▼
 ┌─────────────────┐   assemble the string; aggregate confidence as the
 │  inference.      │   MINIMUM over characters, not the mean
 │  read_plate      │
 └─────────────────┘
     │
     ▼
 ┌─────────────────┐   confidence ≥ threshold → auto-accept
 │  business.       │   confidence <  threshold → route to a human
 │  trust_policy    │
 └─────────────────┘
```

**Binarize.** Otsu thresholding converts the plate image to a clean
black/white mask. The convention — white ink on a black background — matches
EMNIST, so a plate that arrives dark-text-on-light-background (the normal
case for a real plate) is inverted first; getting this backwards means the
classifier sees photographic negatives and predicts confident nonsense. This
step is where lighting and contrast problems first become visible — a
washed-out or backlit plate produces a threshold that either merges
characters together or drops them into the background.

**Segmentation.** `cv2.connectedComponentsWithStats` finds candidate blobs;
size and aspect-ratio filters discard the plate border, mounting bolts, and
sensor noise; the survivors are sorted left to right into reading order,
since detection order is arbitrary and would otherwise scramble a
perfectly-read plate. This is the pipeline's most fragile stage: our own
segmentation spike found exact-match segmentation collapsing from 100% to
0% as character spacing tightened, and the full evaluation confirms it in
practice — segmentation success falls from 82.2% (clean images) to 9.8%
(hard conditions), which is the dominant reason plate-level accuracy drops
under degradation, well before the classifier is even consulted.

**Input contract.** Every crop — whether it came from a training image or a
character just cut out of a live plate photo — passes through the exact same
function: resize to 20px on the longer side preserving aspect ratio, paste
into a 28×28 field, shift so the ink's centre of mass sits at the centre.
§4 names mismatched preprocessing between training and inference as the
single most common cause of high validation accuracy that is useless on real
images; using one function for both paths makes that mismatch structurally
impossible rather than merely unlikely.

**Classification.** A CNN (3 convolutional blocks, 592,964 parameters — see
§4) predicts a class and a softmax confidence for each crop independently.
It has no knowledge of neighbouring characters or that it is looking at a
plate at all; that context is added in the next stage.

**Assembly and confidence.** The predicted characters are joined into a
string, and the plate's overall confidence is the **minimum** across its
characters, not the mean — a plate is only as trustworthy as its weakest
character, and a mean would let one badly-misread digit hide behind six
confident ones.

**Trust policy.** The single confidence number is compared against a
threshold, fitted separately (`docs/business_note.md`) by minimising total
annual cost across auto-accept disputes and manual-review labour. This is
the stage that turns a measured accuracy number into an operational decision
Meridian can actually act on.

**The two integration traps named in §4, and how this design avoids them:**
- *Preprocessing mismatch* — solved structurally, not by discipline: one
  `data.contract` module is imported by both the training pipeline and
  `inference.read_plate`. There is no second implementation to drift out of
  sync.
- *Segmentation vs. recognition failures counted together* — solved by
  `SegmentationResult.count_matches`, which is checked *before* any
  character comparison happens. A plate segmented into the wrong number of
  pieces is recorded as a segmentation failure and excluded from the
  character-accuracy denominator entirely, never blamed on the classifier.

---

## 3. Data
*Source, class handling, imbalance decision, test set with known ground
truth.* — ML-28

**Source and provenance.** EMNIST ByClass, loaded via the local Kaggle CSV
route on both splits (train 220,000 / test 60,000 images), no fallback
needed. Content hashes `fcbe50e6825a945e` (train) and `bda03cba5be30971`
(test) fix exactly which bytes produced every number in this document —
see `artifacts/provenance/provenance.json`. The loader tries three routes
in order (local CSV, TensorFlow Datasets, torchvision) precisely because a
demo cannot depend on one network path staying up; torchvision's route
depends on a NIST mirror whose certificate has been expired since at least
August 2026, and the route order was set to try it last for that reason.

**The transpose.** EMNIST ships column-major, sideways relative to the MNIST
convention every downstream tool assumes. We do not rely on remembering to
fix it: `assert_upright()` makes three falsifiable geometric claims about
specific characters (a '1' is taller than wide, a 'T' is top-heavy, an 'L'
is bottom-heavy) and refuses to return data if they fail. The guard has a
self-test that feeds it deliberately transposed data and asserts it is
rejected — a check that can never fail is not a check.

**Case handling.** We merge lowercase into uppercase (`a`→`A`) rather than
discarding it, trading a real cost — a handwritten `a` does not look like a
printed plate `A` — for roughly 40% more training data. This is a decision
we can defend but have not yet closed the loop on: the `drop` arm has not
been measured against it. That comparison remains open (ML-39).

**Imbalance.** ByClass preserves natural English letter frequency; plate
characters are close to uniform. Measured on the committed load: rarest
class **'K' at 0.725%**, commonest **'1' at 5.52%**, an imbalance ratio of
**7.61×** against a uniform-alphabet expectation of 2.78% per class
(`artifacts/provenance/acceptance_record.md`). We address this with
inverse-frequency class weighting at training time (up to 3.91× for 'K'),
not resampling — weighting uses every available sample rather than
discarding the majority classes down to the minority count. The unweighted
arm has not yet been measured for comparison (ML-39 AC1).

**Test set.** Plates are rendered programmatically (PIL + system fonts) at
three quality tiers — clean, normal (mild blur/skew/noise), hard (heavier
of each) — with the ground-truth string known by construction, so labelling
costs nothing and §11's consent constraint is satisfied by never touching a
real photograph. Two gaps remain open against ML-37's full acceptance
criteria: no manifest CSV is persisted (plates are generated in memory,
evaluated, then discarded), and the font list is drawn from whatever is
installed on the Colab runtime rather than bundled with the repo — so the
exact test set is not currently reproducible byte-for-byte across machines,
only reproducible in distribution (same seed, same generation logic).

---

## 4. Modeling
*What you'll try first and why; what your fallback is.* — ML-29

> *Draft — Model owner's section. Architecture and numbers match the
> trained model committed to `artifacts/models/`; please review the
> reasoning and correct anything that doesn't reflect your actual intent.*

**Primary: a small CNN, three convolutional blocks.** 32 → 64 → 128 filters,
each block two 3×3 convolutions with batch normalisation and ReLU, followed
by max-pooling and dropout; a global-average-pooled dense head into 36
classes. **592,964 parameters.** Trained with Adam at an initial learning
rate of 1e-3, halved once by `ReduceLROnPlateau` (triggered at epoch 15),
batch size 256, for the full 18 configured epochs — early stopping
(patience 5 on validation accuracy) never triggered, meaning the model was
still improving at the final epoch rather than overfitting. Class weights
applied inversely to frequency (up to 3.91× for the rarest class) to correct
for EMNIST's natural letter-frequency imbalance against a near-uniform plate
alphabet (§3).

**Control: a dense baseline (MLP), not a candidate.** Two hidden layers,
flatten-based, no spatial structure — 542,500 parameters, same training
data and splits. Its only job is to make the CNN's number mean something:

| Model | Test accuracy | Parameters |
|---|---:|---:|
| Baseline MLP | 84.2% | 542,500 |
| **CNN** | **90.2%** | 592,964 |

A CNN scoring 90% in isolation is unremarkable; against an MLP at 84%
trained identically, it demonstrates that the spatial structure the
convolutions capture is doing real work, not just that the classes happen
to be separable.

**Sample sizes** — one of the questions §8 promises will be asked live:
train 198,000 · validation 22,000 · test 60,000 (EMNIST ByClass, a
physically separate file from train/validation, opened once).

**Fallback if the CNN had underperformed** (not needed — 90.2% comfortably
clears the 75% reference point in §5, but stated here because §10 rewards
naming the fallback rather than discovering the need for one live):
1. Drop the visually confusable I/O/Q from the plate alphabet entirely
   (many real jurisdictions already do this for the same reason) rather
   than trying to train the confusion away.
2. Increase augmentation strength rather than architecture complexity —
   the domain gap measured in the spike (§6) is a data problem, not a
   capacity problem, and a bigger model does not fix a distribution
   mismatch.
3. Accept a lower character-accuracy number and lean harder on the trust
   threshold (§1) — routing more reads to a human is always available as a
   release valve, and the business note quantifies exactly what that costs.

---

## 5. Evaluation
*The metrics you'll report and the conditions you'll report them under.* —
ML-30

**Definition of a correct read**, agreed in writing before any evaluation
code ran (ML-38): strict exact match on the uppercase 7-character sequence.
No case tolerance is needed (plates are uppercase-only by construction), no
whitespace is compared, and a wrong character *count* is classified as a
segmentation failure, never as a near-miss recognition error.

**Three metrics, three tiers, every number conditioned.** Measured on 400
persisted synthetic plates per tier (`data/generated/`, one consistent
monospace font, manifest-verified ground truth), run through the actual
committed pipeline code — not a notebook approximation:

| Tier | n | Segmentation success | Character accuracy | Plate accuracy |
|---|---:|---:|---:|---:|
| Clean | 400 | 93.0% | 95.3% | 65.8% |
| Normal | 400 | 67.5% | 96.5% | 51.7% |
| Hard | 400 | 10.0% | 96.8% | 7.8% |

**A methodology limitation, stated rather than hidden:** the three tiers
draw *independently random* plate text rather than the same 400 strings
re-rendered at three quality levels. The original design intent — same
characters, only image quality differing, so no tier can score worse purely
for having drawn harder characters — is not fully met by the current
persisted test set. The finding below is still credible (character accuracy
is stable across tiers regardless, which is the opposite of what a
draw-difficulty confound would produce), but a stricter version of this
experiment would fix the same 400 strings across all three tiers.

**A finding worth reporting on its own: font consistency measurably helps.**
An earlier run using randomly-selected system fonts (several italic)
measured 82.2% / 53.2% / 9.8% segmentation and 92.8% / 94.0% / 92.3%
character accuracy at the same three tiers. Switching to one consistent
monospace font — closer to a real plate typeface, and the fix required by
ML-37 — improved segmentation by +11pp (Clean) and +14pp (Normal), and
character accuracy by +2 to +5pp across the board. Hard tier barely moved
(9.8% → 10.0%): at that level of blur and noise, font choice stops
mattering — the image itself is the limiting factor, not the typeface.

**Why plate accuracy is so much lower than character accuracy — and why the
gap widens with degradation.** If segmentation were perfect, plate accuracy
would be bounded above by character-accuracy⁷: roughly 71% at Clean, 78% at
Normal, 80% at Hard. At Clean, measured plate accuracy (65.8%) sits
reasonably close to that bound — segmentation is mostly working (93.0%), so
the character-level errors are most of what's limiting the plate-level
number. At Hard, the story is completely different: the bound predicts
~80%, but measured plate accuracy is 7.8% — a gap almost entirely explained
by segmentation collapsing to 10.0%. **The classifier barely degrades at
all under harsh conditions (95.3% → 96.8%, if anything slightly higher);
segmentation is what breaks.** That is the single most important sentence
in this section, and it is the direct result of separating the two failure
modes rather than reporting one blended "accuracy" number.

**A caveat worth stating plainly:** the Hard tier's 96.8% character accuracy
is computed over only the 40 plates (10.0% of 400) that happened to segment
correctly despite the degradation — a small, self-selected sample of the
*easiest* hard-tier plates. It is not a reliable estimate of classifier
performance under hard conditions, and we say so rather than report it bare.

---

## 6. Spike
*What you tested, what you found, what changed.* — see `spike.md`

We tested two of the brief's candidate risky assumptions rather than one,
since both were cheap. Segmentation via connected components works, but
breaks under tight character kerning and blob-sized artifacts — a
constraint the original tier design didn't track, and now does. A model
trained purely on handwritten EMNIST loses 23% relative accuracy on printed
characters — a real, measured domain gap we are scoping out of this
sprint's work rather than assuming away. Full detail, numbers, and what each
finding changed in the plan: `docs/spike.md`.

---

## 7. Scope
*What you will build — and explicitly what you are not attempting.* — ML-31

> *Draft — Scrum Lead's section. Reflects what's actually implemented as of
> this sprint; please adjust framing/emphasis as appropriate.*

§3 is blunt about this: *"A team that says 'we are not attempting plate
localization; we assume a cropped plate' has scoped honestly. A team that
stays silent and then fails to localize has not."*

**We will build, and have built:** an EMNIST-trained CNN character
classifier (36 classes, case-merged, class-weighted for imbalance); a
connected-components segmenter that turns a plate image into ordered
character crops; a shared preprocessing contract used identically at
training and inference time; an end-to-end read from image to string with
per-character and aggregate confidence; a synthetic plate generator at three
quality tiers so accuracy can be reported as a function of conditions; and a
cost-based trust threshold that converts a confidence score into an
auto-accept-or-route-to-human decision.

**We are explicitly NOT attempting:**
- **Plate localisation within a full vehicle photograph.** The pipeline
  takes an already-cropped plate image as input. Finding the plate within a
  wider scene is a separate computer-vision problem we have not built any
  part of.
- **Closing the handwriting→printed domain gap.** Measured at 23% relative
  (§6). §3 explicitly permits leaving this open for two weeks provided it is
  named rather than assumed away — which this document does.
- **Multi-row plates or non-US plate formats.** The generator and the
  segmenter both assume a single row of fixed-length characters.
- **Hyperparameter search.** One architecture, chosen and justified by
  reasoning (§4) rather than a tuning sweep — there was no time budget for
  one in a two-week sprint, and a justified single architecture is what §10
  rewards over an unexplained best-of-many.
- **A fully reproducible byte-identical test set.** The plate generator
  works and is seeded, but currently draws fonts from whatever is installed
  on the runtime rather than a bundled set, and does not yet persist a
  filename→ground-truth manifest. Noted honestly in §3 rather than silently
  left for someone to discover.

---

## 8. Risks
*Top three, with mitigations.* — ML-31

Named before they materialise — the first two already have, and are
reported rather than hidden.

| Risk | Impact | Mitigation |
|---|---|---|
| **Segmentation degrades sharply as image quality drops.** Measured, not hypothetical: 82.2% success on clean images, 9.8% on hard. It is the dominant driver of low plate accuracy, well ahead of any classifier weakness. | Plate-level accuracy on degraded images could be too low to make a credible business case for the harder sites in Meridian's 34-site portfolio. | Report segmentation and recognition failures separately (§5) so the limitation is precise rather than vague; recommend Meridian pilot at sites resembling our "clean"/"normal" tiers first; name specific tuning directions (kerning-aware filters, per §6's spike) as future work rather than claiming the problem is solved. |
| **Handwriting-trained classifier underperforms on printed plate typefaces.** Measured at 23% relative accuracy drop (§6). | Every character-accuracy number in this document is an upper bound on real-plate performance, not the real-plate number itself. | State the gap explicitly wherever accuracy is quoted; scope a short fine-tune pass on rendered printed glyphs as the first concrete "what four more weeks would buy" item rather than attempting it under sprint time pressure. |
| **The live demo depends on an image we do not control, and our own clean-tier segmentation success is only 82%.** A meaningful chance the instructor's unseen image fails to segment correctly, live. | Could look like an unrehearsed failure rather than a demonstrated, understood limitation — the difference between those two outcomes is worth real points (§8 of the brief: teams that only show successes are asked to produce a failure live, "and it will go worse"). | Deliberately show a chosen failure case ourselves, before being asked, and narrate which failure mode it is (segmentation vs. recognition) and why; rehearse the explanation; record a backup video per §8's requirement so a live technical failure costs nothing. |

---

## Appendix: AI assistance
See [`ai_disclosure.md`](ai_disclosure.md). Required by §6.
