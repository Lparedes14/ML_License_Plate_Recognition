"""Synthetic plate generation - our test set with free ground truth.

STATUS: NOT IMPLEMENTED. Owner: Data / Business. Tickets: ML-37, ML-41.
        This is the critical path - it unblocks both the fine-tuning data and
        the entire results section. Build it first.

WHY WE GENERATE PLATES RATHER THAN PHOTOGRAPH THEM
    1. Ground truth is free. We choose the text, so we know the answer. No
       hand-labelling, no label noise.
    2. We control image quality, so accuracy can be reported AS A FUNCTION OF
       CONDITIONS (§5). "78% at tier B" is a far stronger claim than "78%",
       and reporting accuracy without stating conditions is a -5 deduction.
    3. §11 is a hard constraint: photographing plates belonging to people who
       have not consented is an automatic ZERO. Generated plates sidestep the
       problem entirely. This is the safe route, not just the convenient one.

THE THREE TIERS (config/default.yaml -> plates.tiers)
    A  clean      straight-on, high contrast, no noise    -> optimistic ceiling
    B  realistic  mild blur, slight angle                 -> the headline number
    C  degraded   motion blur, low light, steep angle     -> where we break

    Reporting all three tells the COO what the system does on a good day, a
    normal day, and a bad day - which is the information she actually needs.

IMPLEMENTATION SKETCH
    render_plate(text, tier):
      1. blank canvas, plate-coloured background
      2. draw `text` with PIL.ImageDraw using a fixed-width font, evenly spaced
      3. record each character's bounding box AS WE DRAW IT - this is the
         segmentation ground truth, and getting it free is the whole point.
         Never re-derive boxes afterwards by thresholding: then a segmentation
         bug becomes invisible because both sides share it.
      4. apply the tier's degradations in physical order:
         rotate -> blur -> contrast -> noise
         (that order matters: noise added before blur gets smoothed away and
          stops being a realistic sensor model)
      5. return image + PlateSample(text, char_boxes, tier)

FONT
    Ships with the repo under `assets/fonts/` so a clean clone renders
    identically on every machine. Do NOT reach for a system font - Windows,
    macOS and Colab resolve different files and the test set stops being
    comparable between teammates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlateSample:
    """One generated plate and everything known to be true about it.

    Attributes:
        image: uint8 (H, W), the rendered plate.
        text: The ground-truth string, e.g. "ABC1234".
        char_boxes: One (x, y, w, h) per character, in draw order. Compare
            against segmentation output to measure segmentation success
            SEPARATELY from recognition - §4 insists these are different bugs
            and must be counted separately from day one.
        tier: "A", "B" or "C". Every metric derived from this sample must
            carry the tier with it.
    """

    image: np.ndarray
    text: str
    char_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    tier: str = "A"


def random_plate_text(
    length: int, rng: np.random.Generator, excluded_chars: str = ""
) -> str:
    """Draw a uniformly random plate string.

    Uniform on purpose: real plates are close to uniform over their alphabet,
    which is exactly the mismatch with EMNIST's English-letter-frequency
    distribution that `labels.compute_class_weights` corrects for. Sampling
    the test set uniformly is what makes that mismatch visible in the results.

    Args:
        length: Characters per plate, from cfg.plates.plate_len.
        rng: Seeded generator, so the test set is reproducible.
        excluded_chars: Characters to omit, e.g. "IOQ". Many jurisdictions
            drop these because they collide with 1, 0 and O - measuring the
            difference is a cheap, concrete recommendation for the COO.

    Returns:
        A string of `length` characters.
    """
    from anpr.config import CHARS

    alphabet = [c for c in CHARS if c not in excluded_chars]
    return "".join(rng.choice(alphabet, size=length))


def render_plate(text: str, tier_cfg: dict, seed: int) -> PlateSample:
    """Render one plate at the given quality tier.

    Args:
        text: The plate string to draw.
        tier_cfg: One entry from cfg["plates"]["tiers"], with keys
            blur_sigma, noise_std, rotation_deg, contrast.
        seed: Per-plate seed, so an individual failing sample can be
            regenerated exactly for debugging.

    Returns:
        A PlateSample carrying the image and its ground truth.
    """
    raise NotImplementedError(
        "ML-37: implement render_plate(). See the sketch in this module's "
        "docstring. Use PIL.ImageDraw with a repo-bundled fixed-width font, "
        "and capture each character's bounding box as you draw it."
    )


def generate_test_set(n: int, tier: str, cfg: dict, seed: int) -> list[PlateSample]:
    """Generate `n` plates at one tier - the evaluation set for that tier.

    Args:
        n: Number of plates, from cfg.plates.n_eval_plates.
        tier: "A", "B" or "C".
        cfg: The full config dict.
        seed: Base seed; each plate derives its own from this plus its index.

    Returns:
        A list of PlateSample.
    """
    raise NotImplementedError("ML-41: implement generate_test_set().")


def render_font_glyphs(chars: str, cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Render isolated PRINTED characters, for the domain-adaptation pass.

    The gap §3 warns about: a classifier trained purely on handwriting
    underperforms on plate typefaces. We are not required to close it, but a
    short fine-tune on rendered font glyphs (cfg.training.epochs_ft, lr_ft) is
    the cheapest meaningful attempt, and measuring before/after is a strong
    result either way.

    Output must satisfy the SAME input contract as EMNIST - run it through
    `contract.to_canonical_uint8` before returning. If these glyphs are
    centred differently from the EMNIST ones, fine-tuning will make the model
    worse, and it will not be obvious why.

    Args:
        chars: Characters to render, normally anpr.config.CHARS.
        cfg: Full config.
        seed: For font/jitter variation.

    Returns:
        (X uint8 (N, 28, 28, 1), y int32 (N,)).
    """
    raise NotImplementedError("ML-37: implement render_font_glyphs().")
