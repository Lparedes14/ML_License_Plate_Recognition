# Spike

**Tickets ML-33, ML-34, ML-35 · Worth 10 of 100 points**

> **Template — not yet written.**
>
> §3: *"The spike is graded on whether it changed your plan, not on whether
> it succeeded. 'We assumed X, tested it, found Y, and here's how the plan
> changed' is the ideal result."*
>
> A spike that succeeded and changed nothing scores worse than one that
> failed and redirected the project. Do not pick a safe question.

---

## The riskiest assumption

*What would hurt most if it were wrong?* — ML-33

Candidates from §3:
- "Can we even segment characters reliably?" — threshold a plate image, run
  connected components, count the blobs. One per character?
- "Will a model trained on handwriting read printed plates at all?" — train
  a quick classifier on EMNIST, feed it rendered font characters.
- "Is class imbalance going to sink us?" — how many Q and X samples does
  ByClass actually contain?

**Our question:**

**Why this one:**

---

## What we did

*Cheaply.* — ML-34. A spike is throwaway; timebox it.

---

## What we found

Numbers and images, not impressions.

---

## What changed in the plan

*The part that is actually graded.* — ML-35

| Before the spike | After the spike | Why |
|---|---|---|
| | | |

If nothing changed, say so plainly and explain what that tells you.

---

## Note

One spike result is already banked and worth writing up: our first EMNIST
load reported **129,461 training rows where ByClass has 697,932** — a
truncated upload. The row-count check in `_load_kaggle_csv` caught it and the
loader fell through to torchvision rather than silently training on 18% of
the data. "We assumed the data file was complete; it was not; here is the
check that now makes that impossible" is a legitimate spike finding.
