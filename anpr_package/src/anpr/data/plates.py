"""Synthetic plate generation - our test set with known ground truth.

Ported verbatim from `ML_Draft1_Project.ipynb` (Thenmani). The rendering,
degradation and manifest logic is hers, unchanged; the only differences are
structural: notebook globals (`CFG`, `FONTS`, `SEED`) become function
arguments, and font discovery searches Windows paths as well as Linux ones
so the same code runs on a teammate's laptop and in Colab.

Tickets: ML-37 (generator + manifest), ML-41 (three quality tiers).

WHY WE GENERATE PLATES RATHER THAN PHOTOGRAPH THEM
    1. Ground truth is free. We choose the text, so we know the answer.
    2. We control image quality, so accuracy can be reported AS A FUNCTION OF
       CONDITIONS (§5) rather than as one unqualified number.
    3. §11 is a hard constraint: photographing plates belonging to people who
       have not consented is an automatic ZERO on the project. Generated
       plates sidestep it entirely.
"""

from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Degradation parameters per tier: (skew, blur_kernel, light_range, noise_std).
# These are the values used to produce the committed tier results - changing
# them invalidates `artifacts/metrics/tier_results.json`.
TIER_PARAMS = {
    "clean":  (0.000, 0, (1.00, 1.00), 0),
    "normal": (0.045, 3, (0.75, 1.00), 8),
    "hard":   (0.085, 5, (0.55, 1.00), 18),
}

# Font search roots. The notebook globbed /usr/share/fonts (Colab/Linux only);
# the Windows path is added so the same call works on a teammate's laptop.
_FONT_ROOTS = [
    "/usr/share/fonts/**/*.ttf",
    "C:/Windows/Fonts/*.ttf",
]

# Fonts excluded from the general list - none of these render Latin plate
# characters usefully.
_FONT_BLOCKLIST = (
    "emoji", "symbol", "music", "kacst", "lohit", "samyak", "sinhala", "japanese",
)


@dataclass
class PlateSample:
    """One generated plate and everything known to be true about it.

    Attributes:
        image: uint8 (H, W) grayscale, degraded to its tier.
        text: The ground-truth string.
        tier: "clean", "normal" or "hard". Every metric derived from this
            sample must carry the tier with it (§5).
        char_boxes: Reserved for per-character ground-truth boxes. The
            notebook generator does not emit these - segmentation is scored
            by character COUNT against `text`, not by box overlap.
    """

    image: np.ndarray
    text: str
    tier: str = "normal"
    char_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def available_fonts() -> list[str]:
    """Every usable TrueType font on this machine, blocklist applied.

    Returns:
        Sorted list of font paths. Sorted (the notebook's glob order was
        filesystem-dependent) so the same machine picks the same fallback
        font across runs.
    """
    found: list[str] = []
    for pattern in _FONT_ROOTS:
        found.extend(glob.glob(pattern, recursive=True))
    return sorted(
        f for f in found
        if not any(b in f.lower() for b in _FONT_BLOCKLIST)
    )


def monospace_font() -> str:
    """The single bold monospace font used for the persisted test set.

    One consistent font, not a random pick per plate: a real plate typeface
    is monospaced and upright, and mixing in italic/serif system fonts makes
    the test set measure "unusual font vs handwriting" rather than
    "printed vs handwriting".

    Returns:
        Path to a bold monospace font, falling back to the first available
        font if none matches.

    Raises:
        RuntimeError: if no usable font exists at all.
    """
    for pattern in ("/usr/share/fonts/**/*Mono*Bold*.ttf",
                    "C:/Windows/Fonts/*onsolab.ttf",      # consolab.ttf
                    "C:/Windows/Fonts/*ourbd.ttf"):       # courbd.ttf
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]

    fonts = available_fonts()
    if not fonts:
        raise RuntimeError(
            "No usable TrueType font found. Plate generation needs at least "
            f"one. Searched: {_FONT_ROOTS}"
        )
    return fonts[0]


def random_plate_text(length: int = 7, rng: random.Random | None = None) -> str:
    """Draw a random plate string in the LL-DD-LLL format.

    I, O and Q are omitted from the letter positions: many jurisdictions
    never issue them, precisely because they are confusable with 1, 0 and O.

    Args:
        length: Characters per plate.
        rng: Seeded generator. Defaults to the module `random` state.

    Returns:
        A string of `length` characters.
    """
    r = rng or random
    return "".join(
        r.choice("ABCDEFGHJKLMNPRSTUVWXYZ" if i < 2 or i > 3 else "0123456789")
        for i in range(length)
    )


def render_plate(
    text: str,
    font_path: str | None = None,
    W: int = 560,
    H: int = 140,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Render `text` as dark glyphs on a light plate background.

    Args:
        text: The plate string to draw.
        font_path: Font to use. Defaults to a random available font, which is
            what the exploratory runs used; pass `monospace_font()` for the
            reproducible test set.
        W, H: Plate dimensions in pixels.
        rng: Seeded generator for the background/ink/size jitter.

    Returns:
        uint8 (H, W) grayscale image.
    """
    r = rng or random
    fonts = available_fonts()
    font_path = font_path or r.choice(fonts)

    img = Image.new("L", (W, H), color=r.randint(200, 255))     # light plate
    draw = ImageDraw.Draw(img)
    size = int(H * r.uniform(0.62, 0.76))

    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        font = ImageFont.truetype(fonts[0], size)

    b = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((W - (b[2] - b[0])) // 2 - b[0], (H - (b[3] - b[1])) // 2 - b[1]),
        text, fill=r.randint(0, 55), font=font,                 # dark glyphs
    )
    return np.array(img)


def degrade(
    img: np.ndarray, level: str = "normal", rng: random.Random | None = None
) -> np.ndarray:
    """Apply camera-realistic degradation for one quality tier.

    Transforms are applied in physical order - skew, blur, lighting, noise.
    Noise last, because noise added before blur gets smoothed away and stops
    being a realistic sensor model.

    Args:
        img: uint8 (H, W) rendered plate.
        level: "clean", "normal" or "hard".
        rng: Seeded generator.

    Returns:
        uint8 (H, W), degraded.
    """
    r = rng or random
    skew, blur_k, light, noise = TIER_PARAMS[level]
    h, w = img.shape

    if skew > 0:
        d = w * skew
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [r.uniform(0, d), r.uniform(0, d)],
            [w - r.uniform(0, d), r.uniform(0, d)],
            [w - r.uniform(0, d), h - r.uniform(0, d)],
            [r.uniform(0, d), h - r.uniform(0, d)],
        ])
        img = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(src, dst), (w, h),
            borderValue=int(img[0, 0]),
        )

    if blur_k >= 3 and r.random() < 0.7:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    grad = np.linspace(r.uniform(*light), r.uniform(*light), w)[None, :]
    img = np.clip(img.astype(np.float32) * grad, 0, 255)

    if noise:
        img = np.clip(img + np.random.normal(0, noise, img.shape), 0, 255)

    return img.astype(np.uint8)


def make_plate(
    text: str | None = None,
    level: str = "normal",
    font_path: str | None = None,
    length: int = 7,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, str]:
    """Render and degrade one plate in a single call.

    Args:
        text: Plate string. Random if omitted.
        level: Quality tier.
        font_path: Font override.
        length: Characters, when generating random text.
        rng: Seeded generator.

    Returns:
        (degraded image, ground-truth text).
    """
    text = text or random_plate_text(length, rng)
    return degrade(render_plate(text, font_path, rng=rng), level, rng), text


def build_persisted_test_set(
    n_per_tier: int,
    out_dir: str | Path = "data/generated",
    font_path: str | None = None,
    length: int = 7,
) -> Path:
    """Render, save and manifest N plates per tier.

    Ground truth is free - we chose the text - so the manifest costs nothing
    beyond bookkeeping, and it is what makes the test set auditable rather
    than regenerated-and-hoped-identical (ML-37 AC).

    Args:
        n_per_tier: Plates per tier.
        out_dir: Destination. Written as `<out_dir>/<tier>/plate_NNNN.png`
            plus `<out_dir>/manifest.csv`.
        font_path: Font. Defaults to `monospace_font()` - the reproducible
            choice, not a random pick.
        length: Characters per plate.

    Returns:
        Path to the manifest CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_path or monospace_font()
    manifest_path = out_dir / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "text", "tier"])

        for tier in TIER_PARAMS:
            (out_dir / tier).mkdir(parents=True, exist_ok=True)
            for i in range(n_per_tier):
                text = random_plate_text(length)
                img = degrade(render_plate(text, font_path=font_path), level=tier)
                filename = f"{tier}/plate_{i:04d}.png"
                cv2.imwrite(str(out_dir / filename), img)
                writer.writerow([filename, text, tier])

    print(f"wrote {n_per_tier * len(TIER_PARAMS)} plates + manifest -> {manifest_path}")
    return manifest_path


def load_test_set(out_dir: str | Path = "data/generated") -> list[PlateSample]:
    """Read a persisted test set back from disk via its manifest.

    Args:
        out_dir: Directory containing `manifest.csv`.

    Returns:
        One PlateSample per manifest row, image loaded grayscale.

    Raises:
        FileNotFoundError: if the manifest is missing - regenerate with
            `build_persisted_test_set()`.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}. Generate the test set first: "
            "python scripts/make_test_plates.py"
        )

    samples = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            img = cv2.imread(str(out_dir / row["filename"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(
                    f"manifest lists {row['filename']} but the file is missing"
                )
            samples.append(PlateSample(image=img, text=row["text"], tier=row["tier"]))
    return samples


def render_glyph(ch: str, font_path: str, size: int = 64) -> np.ndarray | None:
    """Draw one character white-on-black, for the domain-gap spike.

    Args:
        ch: The character.
        font_path: Font to render with.
        size: Canvas size before normalisation.

    Returns:
        uint8 (size, size), or None if the font cannot render this character
        (empty or near-empty bounding box).
    """
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype(font_path, int(size * 0.7))
    except Exception:
        return None

    b = d.textbbox((0, 0), ch, font=f)
    if b[2] - b[0] < 3 or b[3] - b[1] < 3:
        return None

    d.text(
        ((size - (b[2] - b[0])) // 2 - b[0], (size - (b[3] - b[1])) // 2 - b[1]),
        ch, fill=255, font=f,
    )
    return np.array(img)


def render_font_glyphs(
    chars: str, n_fonts: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Render isolated PRINTED characters - the domain-gap spike's test set.

    Output goes through `segment.normalize_crop`, the same EMNIST-convention
    normalisation used on real segmentation crops, so the comparison measures
    the handwriting/printed gap and not a preprocessing difference.

    Args:
        chars: Characters to render, normally `anpr.config.CHARS`.
        n_fonts: How many fonts to use.

    Returns:
        (X uint8 (N, 28, 28, 1), y int32 (N,)).
    """
    from anpr.segment.components import normalize_crop

    fonts = available_fonts()[:n_fonts]
    X, y = [], []
    for k, ch in enumerate(chars):
        for fp in fonts:
            g = render_glyph(ch, fp)
            if g is not None:
                X.append(normalize_crop(g))
                y.append(k)

    return np.array(X)[..., None], np.array(y, np.int32)
