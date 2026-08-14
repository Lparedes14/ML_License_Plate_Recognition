"""Confusion analysis, focused on the pairs that actually cost money.

Owner: QA role. Tickets: ML-50, ML-12.

A 36x36 confusion matrix has 1,260 off-diagonal cells and tells a COO
nothing. What she needs is the short list: which specific character pairs the
model confuses, how often, and what it would cost to stop.

That is why `CONFUSABLE_PAIRS` exists in `anpr.config`. It was written down
before any model was trained - 0/O, 1/I, 5/S, 8/B, 2/Z and so on - as a
prediction about where the errors would land. Checking that prediction
against measured results is a genuine finding either way:

    confirmed  -> "we predicted these and here is the rate"
    refuted    -> "we expected 0/O and actually got D/O; here is why"

AND THERE IS A CHEAP RECOMMENDATION IN IT
    Many jurisdictions omit I, O and Q from issued plates precisely because
    they collide with 1, 0 and O. If our measured confusion is concentrated
    there, "ask Meridian to exclude three characters at issue" is a fix that
    costs nothing and beats any amount of extra training.
"""

from __future__ import annotations

import numpy as np

from anpr.config import CHARS, CHAR2IDX, CONFUSABLE_PAIRS, IDX2CHAR


def confusion_matrix(y_true, y_pred, n_classes: int = 36) -> np.ndarray:
    """Build the raw confusion matrix.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        n_classes: Number of classes.

    Returns:
        (n_classes, n_classes) int array; rows are truth, columns prediction.
        So `cm[i, j]` = times a true 'i' was called 'j'.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # np.add.at handles repeated indices correctly; cm[t, p] += 1 would not.
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def top_confusions(cm: np.ndarray, k: int = 15) -> list[dict]:
    """The k most frequent mistakes, as a readable list.

    Args:
        cm: A confusion matrix.
        k: How many to return.

    Returns:
        Dicts with true/predicted characters, the count, and the rate as a
        share of that true class - the rate matters more than the count,
        since a common class produces more errors simply by being common.
    """
    errors = cm.copy()
    np.fill_diagonal(errors, 0)              # correct predictions are not confusions

    flat = np.argsort(errors, axis=None)[::-1][:k]
    row_totals = cm.sum(axis=1)

    out = []
    for pos in flat:
        i, j = np.unravel_index(pos, errors.shape)
        if errors[i, j] == 0:
            break
        out.append({
            "true": IDX2CHAR[int(i)],
            "predicted": IDX2CHAR[int(j)],
            "count": int(errors[i, j]),
            "rate_of_true_class": round(float(errors[i, j] / max(row_totals[i], 1)), 4),
        })
    return out


def check_predicted_confusions(cm: np.ndarray) -> list[dict]:
    """Score our pre-registered CONFUSABLE_PAIRS against what happened.

    Checks both directions of each pair (0 read as O, and O read as 0) and
    reports the combined rate.

    Args:
        cm: A confusion matrix.

    Returns:
        One dict per pair, sorted worst first. Paste straight into the
        results document.
    """
    row_totals = cm.sum(axis=1)
    out = []

    for a, b in CONFUSABLE_PAIRS:
        if a not in CHAR2IDX or b not in CHAR2IDX:
            continue
        ia, ib = CHAR2IDX[a], CHAR2IDX[b]

        a_as_b = int(cm[ia, ib])
        b_as_a = int(cm[ib, ia])
        opportunities = int(row_totals[ia] + row_totals[ib])

        out.append({
            "pair": f"{a}/{b}",
            f"{a}_read_as_{b}": a_as_b,
            f"{b}_read_as_{a}": b_as_a,
            "total_confusions": a_as_b + b_as_a,
            "rate": round((a_as_b + b_as_a) / max(opportunities, 1), 4),
            "predicted_in_advance": True,
        })

    return sorted(out, key=lambda d: d["rate"], reverse=True)


def per_class_accuracy(cm: np.ndarray) -> dict[str, float]:
    """Recall per character - where the rare classes show their damage.

    Watch 'Q' in particular. It is ~0.8% of EMNIST ByClass but ~2.8% of a
    uniform plate alphabet, so a poor recall on Q hurts plate accuracy about
    three times more than its training frequency suggests. This dict is the
    direct evidence for whether the class-weighting decision (ML-39) worked.

    Args:
        cm: A confusion matrix.

    Returns:
        {character: recall}. NaN-free: classes with no samples report 0.0.
    """
    totals = cm.sum(axis=1)
    return {
        IDX2CHAR[i]: (float(cm[i, i] / totals[i]) if totals[i] else 0.0)
        for i in range(len(CHARS))
    }
