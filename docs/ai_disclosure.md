# AI assistance — detailed attribution

> **Not a deliverable.** The graded disclosure Section 6 asks for is the appendix in
> [`approach.md`](approach.md). This file is the team's working detail behind
> it, and the drill sheet for the ML-56 rehearsal.

---

## Who owns what

The **Owner** column names who should be able to defend each item live — not
who has confirmed they can. That confirmation is the point of the rehearsal.

| Area | AI's role | Owner |
|---|---|---|
| `ML_FinalProject_Group_8.ipynb` | Code scaffolding, debugging and drafting throughout, including the two spike experiments (Section 6). The experimental design, the runs, and the decisions about what to try were the team's | Thenmani |
| QA documents | Drafted by AI; the pass/fail determinations came from running the code | Luis |
| `approach.md`, `results.md`, `business_note.md` | Drafted by AI from measured outputs. **Every number comes from executed code** | Valentina / Luis |
| Bugs found by AI review | The business note contradicting its own sensitivity table | Luis |
| Jira administration | Ticket transitions and evidence comments | Luis |

---

## The from-memory drill (ML-56)

Section 8 promises these get asked. Answer without notes.

| Area | Question |
|---|---|
| **Data** | Why does EMNIST need a transpose, and how does the guard *prove* it fired rather than just claiming to check? |
| **Model** | Layer count, learning rate, why `use_bias=False` before BatchNorm, and the train/validation/test sizes |
| **Pipeline** | Why are crops centred by *centre of mass* in a 20×20 box inside 28×28, rather than by bounding-box centre? |
| **Segmentation** | What does the v2 splitting step do, and why did v1 discard merged glyphs instead of splitting them? |
| **QA** | Why is a wrong character *count* excluded from character accuracy? |
| **Business** | Why is plate confidence the minimum and not the mean — and why does the optimal threshold not move when the misread cost changes? |

The last one is worth rehearsing specifically: below 0.98 confidence precision
falls to ~50%, so automated error costs swamp any labour saving at any
plausible misread price. That is a real finding, and explaining it unprompted
is the difference between reporting a number and understanding it.

**A section you cannot defend live is worth less than a shorter one you wrote
yourself.** If that applies to anything above, rewrite it before Saturday.
