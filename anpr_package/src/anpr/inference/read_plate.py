"""End-to-end read: a plate image goes in, a string and a confidence come out.

Owner: Pipeline role. Tickets: ML-45, ML-14, ML-47.

This module is the integration point, and §4 warns it will be humbling: the
first end-to-end run on a real image almost always reads far worse than the
character accuracy suggests. That is expected and is itself a finding.

THE WIRING IS COMPLETE HERE; the pieces it calls are not yet. Once
`segment.binarize`, `segment.segment_characters` and `segment.normalize_crop`
exist, this works with no changes.

WHY CONFIDENCE IS RETURNED, NOT JUST THE STRING
    §2: "the design question is not 'how accurate can we get' but 'which
    reads do we trust automatically, and which do we send to a human'."
    A read without a confidence cannot be routed, so the business layer
    (`business.trust_policy`) would have nothing to threshold on. The
    confidence is the product, as much as the string is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from anpr.data.contract import assert_input_contract, normalize, to_canonical_uint8


@dataclass
class PlateRead:
    """The result of one end-to-end read.

    Attributes:
        text: The predicted plate string.
        char_confidences: Softmax probability of each chosen character.
        plate_confidence: Aggregate confidence for the whole plate - see
            `aggregate_confidence` for why it is the minimum.
        n_chars_found: How many characters segmentation produced.
        n_chars_expected: How many were expected, if known.
        segmentation_ok: False when the counts disagree. When this is False
            the text is unreliable for a reason that has nothing to do with
            the classifier, and it must be reported as a SEGMENTATION failure.
        crops: The normalised crops fed to the model - keep them, they are
            what you show when explaining a failure live.
        boxes: Character boxes in the source image, for the demo overlay.
    """

    text: str
    char_confidences: list[float] = field(default_factory=list)
    plate_confidence: float = 0.0
    n_chars_found: int = 0
    n_chars_expected: int | None = None
    segmentation_ok: bool = True
    crops: np.ndarray | None = None
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def aggregate_confidence(char_confidences: list[float]) -> float:
    """Collapse per-character confidences into one number for the plate.

    We use the MINIMUM, deliberately, and this is a decision worth defending
    in the demo.

    A plate is correct only if EVERY character is correct, so the plate is
    exactly as trustworthy as its weakest character. The mean would hide the
    problem: six characters at 0.99 and one at 0.30 averages to 0.89, which
    would sail past any sensible auto-accept threshold and bill the wrong
    customer. The minimum reports 0.30 and routes it to a human, which is
    the correct outcome.

    The product of the probabilities is the other defensible choice - it is
    the actual probability all characters are right under an independence
    assumption. It is also brutally pessimistic on 7-character plates
    (0.95^7 = 0.70), which pushes almost everything to manual review. Try
    both against the cost model and pick with evidence rather than taste.

    Args:
        char_confidences: Per-character softmax probabilities.

    Returns:
        Aggregate confidence in [0, 1]. Zero for an empty read - nothing
        found means nothing to trust.
    """
    if not char_confidences:
        return 0.0
    return float(min(char_confidences))


def read_plate(
    image: np.ndarray | str | Path,
    model,
    idx2char: dict[int, str],
    n_expected: int | None = None,
) -> PlateRead:
    """Read a plate, end to end.

    Args:
        image: A plate image, or a path to one. Accepting a path matters:
            §8 requires the demo to run on an image supplied at demo time,
            and "demo only works on pre-selected images" is a -5 deduction.
        model: A loaded Keras model.
        idx2char: From `labels.load_class_map` - NOT from the code's CHARS
            constant, so a model/charset mismatch is caught rather than
            silently decoding to the wrong letters.
        n_expected: Expected character count, when known.

    Returns:
        A PlateRead.
    """
    # --- accept a path, so the demo can take any file ---------------------
    if isinstance(image, (str, Path)):
        import cv2
        path = str(image)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read an image from {path!r}")
        image = image[..., ::-1]                # OpenCV gives BGR; we use RGB

    # --- stage 1: segmentation --------------------------------------------
    from anpr.segment import segment_characters
    seg = segment_characters(image, n_expected=n_expected)

    if seg.n_found == 0:
        # An honest empty read. Do NOT fabricate a string - a confident wrong
        # answer is worse for Meridian than an admitted failure, because a
        # wrong answer bills someone.
        return PlateRead(
            text="", n_chars_found=0, n_chars_expected=n_expected,
            segmentation_ok=False, plate_confidence=0.0,
        )

    # --- stage 2: the input contract, same as at training time ------------
    crops = to_canonical_uint8(seg.crops)
    batch = normalize(crops)
    assert_input_contract(batch, "inference crops")   # cheap, catches drift

    # --- stage 3: classify ------------------------------------------------
    probs = model.predict(batch, verbose=0)           # (N, 36)
    indices = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    text = "".join(idx2char[int(i)] for i in indices)

    return PlateRead(
        text=text,
        char_confidences=[float(c) for c in confidences],
        plate_confidence=aggregate_confidence([float(c) for c in confidences]),
        n_chars_found=seg.n_found,
        n_chars_expected=n_expected,
        segmentation_ok=(seg.count_matches is not False),
        crops=crops,
        boxes=seg.boxes,
    )


def load_reader(model_path: str | Path, classmap_path: str | Path | None = None):
    """Load a model and its class map together. Never load one without the other.

    Args:
        model_path: Path to the .keras file.
        classmap_path: Path to the class map. Defaults to the sibling
            `<model>.classmap.json` written by `train_model`.

    Returns:
        (model, idx2char) ready to pass to `read_plate`.

    Raises:
        FileNotFoundError: if the class map is missing. Checked BEFORE
            TensorFlow is imported - the classmap-existence check is a cheap
            path lookup with no reason to depend on an ML runtime being
            installed at all, and it means this failure mode is testable
            (and reachable) on a machine that has no TensorFlow.
    """
    from anpr.data.labels import load_class_map

    model_path = Path(model_path)
    classmap_path = Path(classmap_path) if classmap_path else \
        model_path.with_suffix("").with_suffix(".classmap.json")

    if not classmap_path.exists():
        raise FileNotFoundError(
            f"class map not found at {classmap_path}. Refusing to load the "
            "model without it: the outputs are integers, and only the class "
            "map says which character each integer means. See ML-46."
        )

    from tensorflow import keras   # deferred: only needed once we know the
                                   # class map exists and there is a real
                                   # model to load

    return keras.models.load_model(model_path), load_class_map(classmap_path)
