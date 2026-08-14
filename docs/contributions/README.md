# Individual contribution statements

**Ticket ML-57 · §9 requires one per person, ½ page, submitted privately.**

Three files needed — one per team member:

- `luis_paredes.md`
- `thenmani_sayebaba.md`
- `valentina_valdez.md`

## What each should cover

- **What you owned** — which epic, which tickets, which parts of the notebook
- **What you actually built**, specifically enough to be checked against the
  commit history and the Jira board
- **What you would do differently** with another week

## Why specificity matters here

§6 is blunt: *"You own every line you submit — expect to be asked in the demo
to explain any part of your code from memory, and 'the model wrote it' is not
an answer."*

A statement claiming work you cannot explain live is worse than one that
scopes honestly. If you used AI assistance, say so here and in
[`../ai_disclosure.md`](../ai_disclosure.md) — over-disclosing costs nothing;
under-disclosing is an integrity problem.

## Before the demo, each owner should be able to explain without notes

| Area | Be ready for |
|---|---|
| **Data** | why EMNIST needs a transpose, and how the guard *proves* it fired |
| **Model** | layer count, learning rate, why `use_bias=False` before BatchNorm, train/val/test sizes |
| **Pipeline** | why crops are centred by centre of mass in a 20×20 box inside 28×28 |
| **QA** | why a wrong character count is excluded from character accuracy |
| **Business** | why plate confidence is the minimum and not the mean |
