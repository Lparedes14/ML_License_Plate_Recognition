"""Local demo runner — the notebook's Gradio interface, without Colab.

Same UX as the demo cell in ``ML_FinalProject_Group_8.ipynb``: upload a cropped
plate image, watch every pipeline stage, get an AUTO-ACCEPT / REVIEW decision.
The only difference is that this one runs on your own machine and does *not*
open a public share link, so there is nothing to expire.

    python demo_app.py

Then open the http://127.0.0.1:7860 URL it prints.

--------------------------------------------------------------------------
ON THE DUPLICATED CODE BELOW
--------------------------------------------------------------------------
``to_mnist_format`` and ``segment_plate`` are copied VERBATIM from the
notebook — cells 37 and 52 respectively — and ``normalize`` from cell 10.
They are duplicated, not imported, because the notebook is a .ipynb and its
pipeline cells are interleaved with training and plotting code that cannot be
imported in isolation.

That duplication is a real cost: if you change segmentation in the notebook,
this file silently keeps the old behaviour. Two rules keep that honest:

  1. The notebook remains the single graded deliverable. This file is a
     convenience runner for the live demo, nothing more.
  2. This file deliberately computes NO accuracy, tier or business numbers.
     Every figure quoted in docs/ comes from executing the notebook. A demo
     runner that also produced metrics could disagree with the notebook, and
     that is exactly the failure this project already removed once.

If you edit the pipeline in the notebook, re-copy those three functions here.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Constants — from notebook cell 6
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "plate_cnn.keras"

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

PLATE_LEN = 7          # CFG["plate_len"]

# The NPV-optimal confidence threshold chosen by the cost model in the
# notebook. See docs/business_note.md — below 0.98 precision on accepted
# reads falls to ~50%, which swamps any labour saving.
DEFAULT_THRESHOLD = 0.98


# --------------------------------------------------------------------------
# Pipeline — copied verbatim from the notebook (see module docstring)
# --------------------------------------------------------------------------

def to_mnist_format(glyph):
    """Match EMNIST preprocessing: 20px longest side, centred by centre of mass
    in a 28x28 field. Training and inference MUST use the same recipe."""
    g = cv2.threshold(glyph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(g > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    g = g[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = g.shape
    s = 20.0 / max(h, w)
    g = cv2.resize(g, (max(1, int(round(w*s))), max(1, int(round(h*s)))),
                   interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    h, w = g.shape
    canvas[(28-h)//2:(28-h)//2+h, (28-w)//2:(28-w)//2+w] = g
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        dx, dy = 14 - m["m10"]/m["m00"], 14 - m["m01"]/m["m00"]
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return canvas


def normalize(x):
    """uint8 -> float32 in [0,1]. The type guard is deliberate: silently
    accepting float input is how a batch gets double-normalised to [0, 0.004]
    and the model predicts confident nonsense."""
    if x.dtype != np.uint8:
        raise TypeError(f"normalize() expects uint8, got {x.dtype}")
    return x.astype(np.float32) / 255.0


def segment_plate(plate_img, expected=None):
    """Binarise, find contours, SPLIT merged glyphs, sort left-to-right.

    Sprint 2 measured only 82.5% segmentation success on CLEAN plates with
    zero degradation - the dominant failure mode was under-segmentation:
    touching or kerned-together characters merging into one wide contour,
    which v1 discarded outright instead of splitting.

    Fix: collect candidates with a looser aspect-ratio bar (0.45 instead of
    0.8, so a merged pair is not thrown away before we get a chance at it),
    estimate a reference single-glyph width from the boxes that already look
    like one character, then recursively split any box much wider than that
    reference at the deepest valley in its vertical ink-projection profile -
    the standard column-projection approach to touching-character
    segmentation. A split only fires if the valley is meaningfully emptier
    than its immediate neighbourhood, which stops it triggering on a single
    wide glyph such as M or W.
    """
    h, w = plate_img.shape
    blur = cv2.GaussianBlur(plate_img, (3, 3), 0)
    bin_ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 0.35*h or ch > 0.98*h:  continue   # too small / whole-frame noise
        if cw < 0.008*w:                continue   # specks
        if ch / max(cw, 1) < 0.45:      continue   # even a 2-glyph merge isn't this flat
        raw.append((x, y, cw, ch))
    raw.sort(key=lambda b: b[0])
    if not raw:
        return [], []

    # Reference single-glyph width, from boxes that already look like ONE
    # character (v1's original, stricter bar - kept here just to define "normal").
    single = [b for b in raw if b[3] / max(b[2], 1) >= 0.8]
    ref_w = float(np.median([b[2] for b in single])) if single \
        else float(np.median([b[2] for b in raw]))

    def split_wide(x, y, cw, ch, depth=0):
        n_glyphs = round(cw / ref_w) if ref_w > 0 else 1
        if depth > 3 or n_glyphs < 2:
            return [(x, y, cw, ch)]
        region = bin_[y:y+ch, x:x+cw]
        col_ink = region.sum(axis=0).astype(np.float32)
        margin = max(2, int(cw * 0.15))
        if cw - 2*margin < 3:
            return [(x, y, cw, ch)]
        window = col_ink[margin: cw - margin]
        v_rel = int(np.argmin(window))
        valley = margin + v_rel
        neigh = np.concatenate([col_ink[max(0, valley-3):valley],
                                col_ink[valley+1:valley+4]])
        if len(neigh) == 0 or neigh.mean() <= 0 or col_ink[valley] > 0.6 * neigh.mean():
            return [(x, y, cw, ch)]           # no real gap - one wide glyph, don't split
        left, right = (x, y, valley, ch), (x+valley, y, cw-valley, ch)
        return split_wide(*left, depth+1) + split_wide(*right, depth+1)

    boxes = []
    for (x, y, cw, ch) in raw:
        if cw > 1.45 * ref_w and ref_w > 0:
            boxes.extend(split_wide(x, y, cw, ch))
        elif cw > 0.55 * w:                    # still absurd, no reliable ref - drop
            continue
        else:
            boxes.append((x, y, cw, ch))
    boxes.sort(key=lambda b: b[0])

    if expected and len(boxes) > expected:      # still over-segmented -> keep tallest
        boxes = sorted(sorted(boxes, key=lambda b: -b[3])[:expected], key=lambda b: b[0])

    pad = 2
    chars = []
    for x, y, cw, ch in boxes:
        crop = plate_img[max(0, y-pad):min(h, y+ch+pad), max(0, x-pad):min(w, x+cw+pad)]
        # SAME recipe as training data - see the spike. This is the integration trap.
        chars.append(to_mnist_format(255 - crop))
    return chars, boxes


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

_MODEL = None


def load_model(path=DEFAULT_MODEL):
    """Load the trained CNN once and cache it.

    Imported lazily so that ``--help`` and a missing-model error do not pay
    TensorFlow's multi-second import cost.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = Path(path)
    if not path.exists():
        sys.exit(
            f"Model not found: {path}\n"
            "Run ML_FinalProject_Group_8.ipynb to train it, or point --model at "
            "an existing .keras file."
        )

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF's startup banner

    # Import TensorFlow explicitly, not just Keras. Keras 3 is a front end and
    # installs happily on its own, so `import keras` can succeed while there is
    # no backend to run on -- which then fails much later with an error that
    # does not name the real cause. A half-finished `pip install tensorflow`
    # leaves exactly that state.
    try:
        import tensorflow                                        # noqa: F401
        import keras
    except ImportError as exc:
        sys.exit(
            f"Cannot import TensorFlow/Keras in this interpreter ({exc}).\n"
            f"    {sys.executable}\n\n"
            "Install them, or run this script with an interpreter that has them:\n"
            "    pip install tensorflow==2.20.0 gradio\n\n"
            "On Windows, note that installing TensorFlow into a venv under a long "
            "path fails unless LongPathsEnabled is on -- see the README."
        )

    print(f"loading {path.name} ...")
    _MODEL = keras.models.load_model(path)
    return _MODEL


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def prepare_image(image, invert=False):
    """Accept colour or grayscale, any size. Returns the normalised uint8 plate."""
    img = np.array(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if invert:
        img = 255 - img                       # light-on-dark plates
    if img.shape[0] > 200:                    # normalise plate height
        s = 140 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*s), 140), interpolation=cv2.INTER_AREA)
    return img


def read_plate(plate_img, model, expected=PLATE_LEN):
    """Segment then classify. Returns (text, per-character confidences, crops, boxes)."""
    chars, boxes = segment_plate(plate_img, expected=expected)
    if not chars:
        return "", np.array([]), [], []
    batch = normalize(np.stack(chars)[..., None].astype(np.uint8))
    probs = model.predict(batch, verbose=0)
    text = "".join(IDX2CHAR[i] for i in probs.argmax(1))
    return text, probs, chars, boxes


def decide(text, probs, expected, threshold):
    """The trust policy. Returns (status, reason, plate_confidence).

    Two independent gates, reported separately because they are different
    bugs: a wrong character COUNT is a segmentation failure, a low confidence
    is a recognition failure. Plate confidence is the MINIMUM across
    characters, never the mean - six characters at 0.99 and one at 0.30
    average to 0.89, which would sail past the threshold and bill the wrong
    customer on the one character that was actually wrong.
    """
    if len(probs) == 0:
        return "REVIEW", "segmentation found no characters", 0.0
    conf = float(probs.max(1).min())
    if expected and len(text) != expected:
        return "REVIEW", f"segmented {len(text)} characters, expected {expected}", conf
    if conf < threshold:
        return "REVIEW", f"lowest character confidence {conf:.2f} < {threshold:.2f}", conf
    return "AUTO", None, conf


# --------------------------------------------------------------------------
# Gradio interface — mirrors notebook cell 75
# --------------------------------------------------------------------------

def _stage_images(img, boxes, chars):
    """Return the three stage panels as numpy RGB arrays."""
    s1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)          # stage 1: input
    s2 = s1.copy()                                       # stage 2: segmentation
    for i, (x, y, cw, ch) in enumerate(boxes):
        cv2.rectangle(s2, (x, y), (x+cw, y+ch), (220, 40, 40), 2)
        cv2.putText(s2, str(i+1), (x, max(12, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 40, 40), 1)
    if chars:                                            # stage 3: the 28x28 crops
        strip = np.hstack([np.pad(c, 2, constant_values=0) for c in chars])
        s3 = cv2.cvtColor(
            cv2.resize(strip, (strip.shape[1]*4, strip.shape[0]*4),
                       interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2RGB)
    else:
        s3 = np.zeros((60, 200, 3), np.uint8)
    return s1, s2, s3


# Figures the notebook writes to its working directory. They are matplotlib
# charts, not plates, so they must never be offered as demo examples.
_NOT_PLATES = {"segmentation_demo.png", "plate_conditions.png"}


def _example_images():
    """Any plate images sitting in the repo, for the one-click examples row."""
    found = sorted(glob.glob(str(PROJECT_ROOT / "demo_images" / "*.png")))
    if not found:
        found = [f for f in sorted(glob.glob(str(PROJECT_ROOT / "*.png")))
                 if os.path.basename(f) not in _NOT_PLATES]
    return [[f] for f in found]


def analyse(image, expected_len, threshold, invert):
    """Full pipeline for the UI. Returns panels + a markdown verdict + table."""
    if image is None:
        return None, None, None, "Upload a plate image to begin.", None

    model = load_model()
    img = prepare_image(image, invert=invert)
    expected = int(expected_len) if expected_len else None

    text, probs, chars, boxes = read_plate(img, model, expected=expected)

    if not chars:
        s1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return (s1, s1, None,
                "### REVIEW — segmentation found no characters\n\n"
                "Nothing passed the glyph-geometry filters. If this is a full car photo "
                "rather than a cropped plate, that is expected: plate localization is "
                "out of scope for this prototype. Try the invert toggle if the plate is "
                "light-on-dark.", None)

    status, reason, conf = decide(text, probs, expected, threshold)

    if status == "AUTO":
        verdict = (f"### AUTO-ACCEPT\n\n# `{text}`\n\n"
                   f"Lowest character confidence {conf:.2f} >= {threshold:.2f}. "
                   f"This plate would be billed with no human review.")
    elif "expected" in (reason or ""):
        verdict = (f"### REVIEW — {reason}\n\n**Read anyway:** `{text}`\n\n"
                   f"This is a *segmentation* failure, not a recognition failure. "
                   f"They are different bugs with different fixes.")
    else:
        verdict = (f"### REVIEW — {reason}\n\n**Read:** `{text}`\n\n"
                   f"Routed to a human. At our cost assumptions this is cheaper than "
                   f"risking a wrong bill.")

    rows = [
        [i + 1,
         IDX2CHAR[int(p.argmax())],
         f"{p.max():.3f}",
         IDX2CHAR[int(np.argsort(-p)[1])],
         "pass" if p.max() >= threshold else "FAIL"]
        for i, p in enumerate(probs)
    ]

    s1, s2, s3 = _stage_images(img, boxes, chars)
    return s1, s2, s3, verdict, rows


def _gradio_major():
    """Gradio 6 moved `theme` from the Blocks constructor to launch(); passing
    it to the old place still works but warns on every start. Supporting both
    keeps this runnable on whichever version is already installed."""
    import gradio as gr
    head = gr.__version__.split(".")[0]
    return int(head) if head.isdigit() else 0


def build_ui():
    import gradio as gr

    blocks_kwargs = {"title": "Meridian ANPR Prototype - Group 8"}
    if _gradio_major() < 6:
        blocks_kwargs["theme"] = gr.themes.Soft()

    with gr.Blocks(**blocks_kwargs) as ui:
        gr.Markdown(
            "# Meridian Access Systems — ANPR Prototype\n"
            "**Group 8** · character classifier trained on EMNIST ByClass (36 classes, "
            "case-merged) · segmentation by connected components\n\n"
            "Upload a **cropped plate image**. Plate localization within a full road "
            "scene is explicitly out of scope for this prototype."
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Plate image", height=200)
                exp = gr.Slider(3, 10, value=PLATE_LEN, step=1,
                                label="Expected characters",
                                info="Set to the plate's real length; mismatch = REVIEW")
                thr = gr.Slider(0.50, 0.99, value=DEFAULT_THRESHOLD, step=0.01,
                                label="Auto-accept confidence threshold",
                                info=f"NPV-optimal is {DEFAULT_THRESHOLD:.2f} at our "
                                     f"cost assumptions")
                inv = gr.Checkbox(label="Invert (light characters on a dark plate)")
                btn = gr.Button("Read plate", variant="primary")
                examples = _example_images()
                if examples:
                    gr.Examples(examples=examples, inputs=inp,
                                label="Or try one of these")
            with gr.Column(scale=2):
                out_verdict = gr.Markdown()
                with gr.Row():
                    p1 = gr.Image(label="1 · input (normalised)", height=130)
                    p2 = gr.Image(label="2 · segmentation", height=130)
                p3 = gr.Image(
                    label="3 · what the model sees (28x28, centre-of-mass aligned)",
                    height=130)
                out_tbl = gr.Dataframe(
                    headers=["#", "read", "confidence", "runner-up", "gate"],
                    label="Per-character detail", wrap=True)

        outputs = [p1, p2, p3, out_verdict, out_tbl]
        btn.click(analyse, [inp, exp, thr, inv], outputs)
        inp.change(analyse, [inp, exp, thr, inv], outputs)

        gr.Markdown(
            "---\n"
            "**Measured performance** (synthetic plates, normal degradation): "
            "segmentation 73.0% · character 62.4% · plate 34.0%. "
            "Held-out EMNIST character accuracy 90.1%. "
            "All figures are on generated images, not photographs, and come from the "
            "notebook run documented in `docs/results.md` — this app does not "
            "recompute them."
        )

    return ui


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run the ANPR demo locally (no public share link).")
    ap.add_argument("--image", metavar="PATH",
                    help="read one image and print the result, no web UI")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help=f"path to the .keras model (default: {DEFAULT_MODEL.name})")
    ap.add_argument("--expected", type=int, default=PLATE_LEN,
                    help=f"expected characters per plate (default: {PLATE_LEN})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"auto-accept confidence (default: {DEFAULT_THRESHOLD})")
    ap.add_argument("--invert", action="store_true",
                    help="light characters on a dark plate")
    ap.add_argument("--port", type=int, default=7860, help="web UI port")
    ap.add_argument("--share", action="store_true",
                    help="also open a temporary public gradio.live link")
    args = ap.parse_args()

    model = load_model(args.model)

    if args.image:
        from PIL import Image
        img = prepare_image(Image.open(args.image), invert=args.invert)
        text, probs, chars, _ = read_plate(img, model, expected=args.expected)
        status, reason, conf = decide(text, probs, args.expected, args.threshold)
        print(f"\nREAD     : {text or '(nothing segmented)'}")
        print(f"CHARS    : {len(chars)} (expected {args.expected})")
        print(f"MIN CONF : {conf:.3f}")
        print(f"DECISION : {status}" + (f"  ({reason})" if reason else "  - bill this account"))
        for i, p in enumerate(probs):
            print(f"   {i+1}. {IDX2CHAR[int(p.argmax())]}  {p.max():.3f}"
                  f"  (runner-up {IDX2CHAR[int(np.argsort(-p)[1])]})")
        return

    print(f"\nStarting the demo on http://127.0.0.1:{args.port}  (Ctrl+C to stop)\n")
    launch_kwargs = dict(server_port=args.port, share=args.share, inbrowser=True)
    if _gradio_major() >= 6:
        import gradio as gr
        launch_kwargs["theme"] = gr.themes.Soft()
    build_ui().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
