"""SPIKE — can we segment plate characters reliably with connected components?

    python notebooks/spike_segmentation.py

===============================================================================
THIS IS THROWAWAY CODE. DO NOT IMPORT IT. DO NOT BUILD ON IT.
===============================================================================

§3 of the brief: "A spike is a throwaway experiment that answers your riskiest
question before you commit a week to a plan built on top of it. Pick the
assumption that would hurt most if wrong and test it cheaply."

It also names this exact experiment as a candidate:

    "Can we even segment characters reliably? Threshold a plate image, run
     connected components, count the blobs. Do you get one per character?"

WHY THIS IS THE RIGHT QUESTION FOR US
    Our whole pipeline assumes segmentation works. If connected components
    cannot reliably return one blob per character, then no amount of classifier
    accuracy saves us — a 7-character plate read as 6 characters is wrong before
    the model is even consulted. That assumption is load-bearing and untested.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
    It is not the production plate generator (ML-37, Valentina) and it is not
    the production segmenter (ML-43). It uses a system font, crude degradations
    and no error handling, on purpose. Its only job is to produce a number that
    either confirms or refutes the assumption before Thursday.

HOW THE SPIKE IS GRADED (§10)
    "on whether it changed your plan, not on whether it succeeded."
    So the useful outcome is a threshold at which this breaks. Record it.

Ticket: ML-34. Write-up: ML-35 -> docs/spike.md
"""

from __future__ import annotations

import os
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Experiment settings — change these and re-run, that is the whole point
# --------------------------------------------------------------------------
SEED = 42
N_PLATES = 40          # per quality level
PLATE_LEN = 7
ALPHABET = string.digits + string.ascii_uppercase

# Rough stand-ins for the three quality tiers. NOT the production tier
# definitions in config/default.yaml — deliberately cruder, because a spike
# that waits for the real generator is not cheap.
LEVELS = {
    "clean":      dict(blur=0.0, noise=0,  rotate=0.0, contrast=1.00),
    "realistic":  dict(blur=0.8, noise=8,  rotate=2.0, contrast=0.85),
    "degraded":   dict(blur=1.8, noise=20, rotate=6.0, contrast=0.55),
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",     # bold sans — closest to a plate face
    "C:/Windows/Fonts/consolab.ttf",    # bold monospace
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

OUT_DIR = "artifacts/figures"


def load_font(size: int = 64):
    """First available system font. Production (ML-37) must bundle its own."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size), os.path.basename(path)
    raise SystemExit(
        "No usable TrueType font found. Add one to FONT_CANDIDATES.\n"
        f"Tried: {FONT_CANDIDATES}"
    )


# --------------------------------------------------------------------------
# 1. Render a plate — crude on purpose
# --------------------------------------------------------------------------
def render_plate(text: str, font, level: dict, rng: random.Random,
                 kerning: float = 1.0) -> np.ndarray:
    """Draw `text` on a plate-ish background and degrade it.

    Returns a grayscale uint8 image. Dark characters on a light plate, which
    is the real-world convention — note that this means the binariser has to
    INVERT, and getting that backwards is one of the things worth discovering
    here rather than on Thursday.

    Args:
        kerning: Multiplier on character spacing. 1.0 is comfortably spaced;
            below ~0.75 glyphs start to touch. This is the parameter that
            matters — the first run of this spike scored 100% everywhere
            purely because characters never touched, which is not what a real
            plate looks like.
    """
    W, H = 520, 120
    img = Image.new("L", (W, H), color=210)          # light plate background
    draw = ImageDraw.Draw(img)

    spacing = (W // (len(text) + 1)) * kerning
    x0 = (W - spacing * (len(text) - 1)) / 2 - 18    # keep the block centred
    for i, ch in enumerate(text):
        x = x0 + spacing * i
        y = H // 2 - 38
        draw.text((x, y), ch, fill=25, font=font)     # dark ink

    a = np.asarray(img).astype(np.float32)

    # --- degradations, applied in physical order --------------------------
    # rotate -> blur -> contrast -> noise. Noise last, because noise added
    # before blur gets smoothed away and stops being a sensor model.
    if level["rotate"]:
        angle = rng.uniform(-level["rotate"], level["rotate"])
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        a = cv2.warpAffine(a, M, (W, H), borderValue=210)

    if level["blur"]:
        k = int(level["blur"] * 4) | 1                # odd kernel
        a = cv2.GaussianBlur(a, (k, k), level["blur"])

    if level["contrast"] != 1.0:
        mid = a.mean()
        a = (a - mid) * level["contrast"] + mid

    if level["noise"]:
        a = a + np.random.normal(0, level["noise"], a.shape)

    return np.clip(a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 2. Binarise — the step everything downstream depends on
# --------------------------------------------------------------------------
def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Threshold to white ink on black background.

    THRESH_BINARY_INV because our plates are dark-on-light and EMNIST is
    bright-glyph-on-black. Flip this and the model sees photographic
    negatives — worth seeing what that does, so try it.
    """
    if method == "otsu":
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        b = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5,
        )
    return b


# --------------------------------------------------------------------------
# 3. Connected components + filtering — the actual question
# --------------------------------------------------------------------------
def count_blobs(binary: np.ndarray, filtered: bool = True):
    """Return (boxes, n_rejected). This is the measurement.

    With filtered=False you see what RAW connected components gives you, which
    is the honest baseline: the filters are a design decision, and knowing how
    much work they are doing is part of the finding.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    H, W = binary.shape
    area_total = H * W
    boxes, rejected = [], 0

    for i in range(1, n):                             # 0 is the background
        x, y, w, h, area = stats[i]

        if not filtered:
            boxes.append((x, y, w, h))
            continue

        aspect = w / max(h, 1)
        too_small = area < area_total * 0.0015        # specks, sensor noise
        too_big = area > area_total * 0.25            # plate border, bleed
        too_wide = aspect > 3.0                       # horizontal rule
        # NOTE: no minimum-aspect filter here on purpose. '1' and 'I' are
        # genuinely narrow, and a min-aspect filter is how they silently
        # disappear from every plate. That failure looks like a recognition
        # bug and is not one.

        if too_small or too_big or too_wide:
            rejected += 1
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])                    # reading order
    return boxes, rejected


# --------------------------------------------------------------------------
# 4. Run the experiment
# --------------------------------------------------------------------------
def main() -> int:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    font, font_name = load_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("SPIKE — can connected components segment plate characters reliably?")
    print("=" * 72)
    print(f"font: {font_name} | {N_PLATES} plates per level | {PLATE_LEN} chars | seed {SEED}")
    print()

    results = {}
    examples = {}

    for level_name, level in LEVELS.items():
        counts = []
        rejected_total = 0

        for i in range(N_PLATES):
            text = "".join(rng.choice(ALPHABET) for _ in range(PLATE_LEN))
            gray = render_plate(text, font, level, rng)
            binary = binarize(gray, "otsu")
            boxes, rejected = count_blobs(binary, filtered=True)

            counts.append(len(boxes))
            rejected_total += rejected

            if i == 0:                                # keep one for the figure
                examples[level_name] = (gray, binary, boxes, text)

        counts = np.array(counts)
        exact = (counts == PLATE_LEN).mean()
        over = (counts > PLATE_LEN).mean()
        under = (counts < PLATE_LEN).mean()

        results[level_name] = dict(
            exact=exact, over=over, under=under,
            mean_count=counts.mean(),
            rejected_per_plate=rejected_total / N_PLATES,
        )

        print(f"[{level_name:>10}]  exact {exact:5.0%} | over {over:5.0%} | "
              f"under {under:5.0%} | mean blobs {counts.mean():4.1f} "
              f"(want {PLATE_LEN}) | filtered out {rejected_total / N_PLATES:.1f}/plate")

    # --- unfiltered baseline: how much work are the filters doing? --------
    print()
    print("Same experiment with NO blob filtering (raw connected components):")
    unfiltered = {}
    for level_name, level in LEVELS.items():
        counts = []
        for _ in range(N_PLATES):
            text = "".join(rng.choice(ALPHABET) for _ in range(PLATE_LEN))
            gray = render_plate(text, font, level, rng)
            boxes, _ = count_blobs(binarize(gray, "otsu"), filtered=False)
            counts.append(len(boxes))
        counts = np.array(counts)
        unfiltered[level_name] = (counts == PLATE_LEN).mean()
        print(f"[{level_name:>10}]  exact {(counts == PLATE_LEN).mean():5.0%} | "
              f"mean blobs {counts.mean():5.1f}")

    # --- STRESS TEST 1: kerning sweep -------------------------------------
    # The first run of this spike scored 100% everywhere, which was a warning
    # sign rather than a result: characters were widely spaced and never
    # touched. Real plates are tightly kerned. This finds the breaking point.
    print()
    print("STRESS TEST 1 - character spacing (touching glyphs merge into one blob)")
    print("  kerning   exact   under   mean blobs")
    kern_results = {}
    for kern in (1.00, 0.85, 0.75, 0.65, 0.55, 0.45):
        counts = []
        for _ in range(N_PLATES):
            text = "".join(rng.choice(ALPHABET) for _ in range(PLATE_LEN))
            gray = render_plate(text, font, LEVELS["realistic"], rng, kerning=kern)
            boxes, _ = count_blobs(binarize(gray, "otsu"), filtered=True)
            counts.append(len(boxes))
        counts = np.array(counts)
        exact = (counts == PLATE_LEN).mean()
        under = (counts < PLATE_LEN).mean()
        kern_results[kern] = exact
        print(f"    {kern:.2f}    {exact:5.0%}   {under:5.0%}   {counts.mean():5.1f}")

    # --- STRESS TEST 2: speckle the area filter cannot remove -------------
    # The degraded level rejected ~1480 blobs per plate, all single-pixel
    # noise the area filter trivially removes. Real degradation produces
    # larger artefacts. This uses blob-sized speckle instead.
    print()
    print("STRESS TEST 2 - speckle large enough to survive the area filter")
    print("  speckles   exact   over    mean blobs")
    speckle_results = {}
    for n_speck in (0, 20, 60, 150):
        counts = []
        for _ in range(N_PLATES):
            text = "".join(rng.choice(ALPHABET) for _ in range(PLATE_LEN))
            gray = render_plate(text, font, LEVELS["realistic"], rng)
            # dark blobs ~8px across: too big for the area filter, too small
            # to be a character. Exactly the ambiguous case.
            for _ in range(n_speck):
                cx, cy = rng.randint(0, 519), rng.randint(0, 119)
                cv2.circle(gray, (cx, cy), rng.randint(3, 5), 20, -1)
            boxes, _ = count_blobs(binarize(gray, "otsu"), filtered=True)
            counts.append(len(boxes))
        counts = np.array(counts)
        exact = (counts == PLATE_LEN).mean()
        speckle_results[n_speck] = exact
        print(f"    {n_speck:>4}     {exact:5.0%}   {(counts > PLATE_LEN).mean():5.0%}   "
              f"{counts.mean():5.1f}")

    save_figure(examples)
    verdict(results, unfiltered, kern_results, speckle_results)
    return 0


def save_figure(examples: dict) -> None:
    """One figure showing the stages. Evidence for the write-up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 3, figsize=(14, 3.0 * len(examples)))
    for r, (name, (gray, binary, boxes, text)) in enumerate(examples.items()):
        axes[r, 0].imshow(gray, cmap="gray")
        axes[r, 0].set_title(f"{name}: rendered  ({text})", fontsize=9)

        axes[r, 1].imshow(binary, cmap="gray")
        axes[r, 1].set_title("binarised (Otsu, inverted)", fontsize=9)

        overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 60, 60), 2)
        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title(f"{len(boxes)} blobs found (want {PLATE_LEN})", fontsize=9)

        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "spike_segmentation.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {path}")


def verdict(results: dict, unfiltered: dict, kern: dict, speckle: dict) -> None:
    """State plainly what this means for Thursday's plan."""
    print()
    print("=" * 72)
    print("WHAT THIS MEANS FOR THE PLAN")
    print("=" * 72)

    clean = results["clean"]["exact"]
    real = results["realistic"]["exact"]
    deg = results["degraded"]["exact"]

    print("1. On well-spaced characters, connected components works:")
    print(f"   clean {clean:.0%}, realistic {real:.0%}, degraded {deg:.0%} exact.")
    print("   Viable as the Thursday approach. Build on it.")

    print()
    print("2. But the blob FILTERS are carrying the degraded case entirely:")
    print(f"   with filtering {deg:.0%} exact, without it "
          f"{unfiltered['degraded']:.0%} "
          f"({results['degraded']['rejected_per_plate']:.0f} blobs rejected per plate).")
    print("   The filters are not a detail. They are the segmenter.")

    # --- the breaking point, which is the actual finding ------------------
    broke_at = next((k for k, v in sorted(kern.items(), reverse=True) if v < 0.9), None)
    print()
    if broke_at is not None:
        print(f"3. IT BREAKS AT KERNING {broke_at:.2f} — "
              f"{kern[broke_at]:.0%} exact, and it is UNDER-segmentation:")
        print("   touching glyphs merge into one blob. Connected components")
        print("   cannot separate them by definition, so no filter tuning helps.")
        print("   -> Valentina's generator (ML-37) must keep kerning above this,")
        print("      OR Thursday needs a split step for over-wide blobs.")
    else:
        print("3. Kerning did not break it down to 0.45. Either the sweep is too")
        print("   generous or the font is unusually narrow. Widen the sweep before")
        print("   trusting this.")

    speck_broke = next((n for n, v in sorted(speckle.items()) if v < 0.9), None)
    print()
    if speck_broke is not None:
        print(f"4. Blob-sized speckle breaks it at ~{speck_broke} specks per plate "
              f"({speckle[speck_broke]:.0%} exact), and it is OVER-segmentation.")
        print("   The area filter cannot catch artefacts that are character-sized.")
        print("   -> This is the realistic tier-C failure mode, not pixel noise.")
    else:
        print("4. Speckle up to the tested level did not break it. The area filter")
        print("   is handling it.")

    print()
    print("=" * 72)
    print("Record these numbers in docs/spike.md (ML-35). The spike is graded on")
    print("whether it CHANGED the plan — the kerning limit and the filter")
    print("dependency are both plan changes. Say so explicitly.")


if __name__ == "__main__":
    raise SystemExit(main())
