"""The input contract: the ONE definition of what an image looks like.

WHY THIS MODULE IS THE MOST IMPORTANT ONE IN THE REPO
    §4 of the brief: "Preprocessing must match between training and
    inference. Mismatched preprocessing is the single most common cause of
    '57% validation accuracy, unusable on real images.'"

    The way that bug happens is always the same: someone normalises in the
    training notebook, someone else normalises differently in the demo, and
    nothing complains because both produce a plausible-looking array. So we
    do not rely on everyone remembering. There is exactly one function that
    converts uint8 to float, it REFUSES float input, and there is an
    assertion that any batch can be run through before it reaches a model.

    Every path into the model - EMNIST loading, synthetic plates, a JPEG
    uploaded at demo time - goes through `to_canonical_uint8` then
    `normalize`. No exceptions.

THE CONTRACT
    Stored on disk / in memory : uint8,   shape (N, 28, 28, 1), values 0-255
    Fed to the model           : float32, shape (N, 28, 28, 1), values 0.0-1.0

Ported from ML_Project_Group_8.ipynb cells 10 and 32 (ML-6, ML-44).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# The contract as data, so it can be written into the provenance record and
# cited in the report rather than described in prose.
INPUT_SPEC: dict[str, Any] = {
    "height": 28,
    "width": 28,
    "channels": 1,
    "dtype": "float32",
    "vmin": 0.0,
    "vmax": 1.0,
    "storage_dtype": "uint8",
}


def to_canonical_uint8(X: np.ndarray) -> np.ndarray:
    """Coerce any reasonable image input into (N, 28, 28, 1) uint8.

    Accepts, and this breadth is the point - the demo has to swallow whatever
    image the instructor hands us at demo time (§8):
      - (N, 784)        flat rows, as the EMNIST CSV ships them
      - (H, W)          a single grayscale image
      - (N, H, W)       a stack of grayscale images
      - (N, H, W, 1)    already channelled
      - (N, H, W, 3)    RGB, converted with ITU-R BT.601 luma weights

    Args:
        X: Image data in any of the shapes above, any numeric dtype.

    Returns:
        uint8 array of shape (N, 28, 28, 1).

    Raises:
        ValueError: if the shape cannot be interpreted as images at all.
            Failing here is much cheaper than training on garbage.
    """
    X = np.asarray(X)

    # --- normalise the number of dimensions to 4 --------------------------
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)      # flat CSV rows -> images
    elif X.ndim == 2:
        X = X[None, ...]               # a single (H, W) image -> a stack of 1
    if X.ndim == 3:
        X = X[..., None]               # add the channel axis
    if X.ndim != 4:
        raise ValueError(
            f"cannot interpret shape {X.shape} as images. Expected (N,784), "
            "(H,W), (N,H,W) or (N,H,W,C)."
        )

    # --- collapse colour to a single channel ------------------------------
    if X.shape[-1] == 3:
        # BT.601 luma. Plain .mean(-1) would also "work" but weights green
        # wrongly and shifts contrast, which matters once we threshold a real
        # photograph during segmentation.
        X = (X[..., :3] * np.array([0.299, 0.587, 0.114])).sum(-1, keepdims=True)
    elif X.shape[-1] != 1:
        raise ValueError(f"expected 1 or 3 channels, got {X.shape[-1]}")

    # --- resize to 28x28 --------------------------------------------------
    h, w = X.shape[1], X.shape[2]
    if (h, w) != (28, 28):
        # INTER_AREA for downscaling (averages the pixels being discarded, so
        # thin strokes survive); INTER_CUBIC for upscaling (smoother than
        # nearest, which would produce blocky glyphs the model never saw).
        interp = cv2.INTER_AREA if (h > 28 or w > 28) else cv2.INTER_CUBIC
        X = np.stack([
            cv2.resize(im.squeeze().astype(np.uint8), (28, 28), interpolation=interp)
            for im in X
        ])[..., None]

    # --- clamp to uint8 ---------------------------------------------------
    if X.dtype != np.uint8:
        X = np.clip(X, 0, 255).astype(np.uint8)

    assert X.shape[1:] == (28, 28, 1) and X.dtype == np.uint8
    return X


def normalize(x):
    """uint8 [0,255] -> float32 [0,1]. The ONLY division by 255 in the repo.

    The type check is not defensive programming for its own sake: it makes
    double-normalisation - dividing by 255 twice, leaving every pixel below
    0.004 and the model seeing near-black - impossible rather than merely
    unlikely. A double-normalised batch trains to a plausible-looking
    validation accuracy and then fails completely on real images, which is
    exactly the failure mode §4 warns about.

    Works on both NumPy arrays (eager code, the demo) and TensorFlow tensors
    (inside a tf.data pipeline), because it is called from both.

    Args:
        x: uint8 array or tf.uint8 tensor.

    Returns:
        float32 in [0, 1], same shape, same backend as the input.

    Raises:
        TypeError: if given anything already floating point. If you hit this,
            you are normalising twice - find the first call and delete it.
    """
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            raise TypeError(
                f"normalize() expects uint8, got {x.dtype}. This almost always "
                "means the array was already normalised - dividing again would "
                "leave every pixel below 0.004 and silently ruin training."
            )
        return x.astype(np.float32) / 255.0

    # TensorFlow path. Imported lazily so NumPy-only callers (the plate
    # generator, most tests) do not pay the import cost.
    import tensorflow as tf

    if x.dtype != tf.uint8:
        raise TypeError(f"normalize() expects tf.uint8, got {x.dtype}")
    return tf.cast(x, tf.float32) / 255.0


def assert_input_contract(x, name: str = "batch") -> dict[str, Any]:
    """Verify a batch satisfies the contract, or raise with a full diagnosis.

    Call this on the first batch of anything new: a training pipeline, a
    fine-tuning set, character crops coming out of segmentation at demo time.
    It costs microseconds and catches the class of bug that otherwise shows
    up as an unexplained accuracy collapse.

    Args:
        x: The batch to check - NumPy array or TF tensor.
        name: Label used in the error message, so you know WHICH batch failed.

    Returns:
        A small stats dict (shape, dtype, min, max, mean). Store it in the
        provenance record: it is the evidence that the check actually ran.

    Raises:
        AssertionError: listing every violation at once, not just the first.
    """
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    bad: list[str] = []

    if arr.ndim != 4:
        bad.append(f"ndim {arr.ndim}, expected 4 (N,H,W,C)")
    elif arr.shape[1:] != (28, 28, 1):
        bad.append(f"shape {arr.shape[1:]}, expected (28, 28, 1)")

    if arr.dtype != np.float32:
        bad.append(f"dtype {arr.dtype}, expected float32")

    lo, hi = float(arr.min()), float(arr.max())
    if lo < -1e-6 or hi > 1 + 1e-6:
        bad.append(f"range [{lo:.3f}, {hi:.3f}], expected [0, 1]")

    # A batch of real glyphs always contains some bright ink. If the maximum
    # is essentially zero, the data was normalised twice.
    if hi <= 0.02 and arr.size > 100:
        bad.append(f"max is {hi:.5f} - almost certainly double-normalised")

    if bad:
        raise AssertionError(f"INPUT CONTRACT VIOLATED for '{name}': " + "; ".join(bad))

    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": round(lo, 5),
        "max": round(hi, 5),
        "mean": round(float(arr.mean()), 5),
    }
