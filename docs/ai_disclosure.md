# AI Assistance Disclosure

**Required by §6 of the brief**

> *"AI assistants are permitted and expected. Disclose where you used them in
> a short appendix. You own every line you submit — expect to be asked in the
> demo to explain any part of your code from memory, and 'the model wrote it'
> is not an answer."*

---

## How to use this document

Add a row whenever an AI assistant materially contributed. Be specific: the
file, what it did, and who reviewed it. Over-disclosing costs nothing;
under-disclosing is an integrity problem.

| Area | Tool | What it did | Reviewed & owned by |
|---|---|---|---|
| Repository structure, module scaffolding, docstrings | Claude | Proposed the `anpr_package/src/anpr/` layout; ported the Week-1 notebook data code into modules; wrote docstrings and the test suite for the contract, guards, labels, metrics and cost model | |
| | | | |

---

## Before the demo

Each owner should be able to explain, from memory, without notes:

- **Data** — why EMNIST needs a transpose, and how the guard proves it fired
- **Model** — layer count, learning rate, why `use_bias=False` before
  BatchNorm, train/val/test sizes
- **Pipeline** — why crops are centred by centre of mass in a 20×20 box
  inside a 28×28 field
- **QA** — why a length mismatch is excluded from character accuracy
- **Business** — why the aggregate confidence is the minimum and not the mean

If you cannot explain a function you own, rewrite it until you can. That is
the point of the exercise, and §8 says it will be tested.
