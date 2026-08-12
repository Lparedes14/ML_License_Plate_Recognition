"""tf.data input pipelines, and the augmentation that closes the domain gap.

WHY AUGMENT AT ALL
    Two distribution shifts sit between EMNIST and a parking-lot camera:

      1. EMNIST is HANDWRITTEN; plates are PRINTED. Augmentation cannot fix
         this - only training on rendered font glyphs can (see the fine-tune
         pass in models/train.py). §3 of the brief is explicit that we are
         not expected to close this gap, only to notice it and say what we
         would do.

      2. EMNIST glyphs are clean, centred and evenly lit; camera crops are
         rotated, off-centre, blurred and noisy. Augmentation DOES fix this,
         and it is cheap.

    Each transform below is here because it mimics a specific real-world
    effect, not because it is a standard default:

      rotation    ±5%    camera not perfectly square to the plate
      translation ±10%   segmentation bounding boxes are never perfectly tight
      zoom        ±10%   varying vehicle distance from the camera
      contrast    ±25%   sunlight, shade, headlight glare
      blur        30%    of batches: focus and motion blur
      noise       σ=0.06 sensor noise in low light

    Augmentation is applied to TRAINING data only. Validation and test must
    stay clean, or the metric measures our augmentation rather than the model.

Ported from ML_Project_Group_8.ipynb cell 30 (ML-8).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from anpr.data.contract import normalize


def build_augmenter() -> keras.Sequential:
    """The geometric/photometric augmentation stack.

    `fill_value=0.0` matters: 0 is background (black) under our convention,
    so rotating or shifting a glyph exposes background rather than a grey
    smear the model would learn as a feature.

    Returns:
        A Keras Sequential applied with `training=True` inside the pipeline.
    """
    return keras.Sequential([
        layers.RandomRotation(0.05, fill_mode="constant", fill_value=0.0),
        layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0),
        layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0),
        layers.RandomContrast(0.25),
    ], name="augment")


def random_blur(x: tf.Tensor, probability: float = 0.30) -> tf.Tensor:
    """Blur a random subset of each batch with a 3x3 box kernel.

    Applied per-sample rather than per-batch: a fixed fraction of every batch
    is blurred, so the model sees both sharp and soft glyphs in the same
    gradient step and cannot specialise to either.

    Args:
        x: float32 batch, (N, 28, 28, 1).
        probability: Share of samples to blur.

    Returns:
        The batch, some samples blurred.
    """
    kernel = tf.ones((3, 3, 1, 1), tf.float32) / 9.0
    blurred = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")

    # Per-sample 0/1 mask, broadcast over H, W, C.
    mask = tf.cast(
        tf.random.uniform((tf.shape(x)[0], 1, 1, 1)) < probability, tf.float32
    )
    return mask * blurred + (1.0 - mask) * x


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    training: bool,
    batch_size: int,
    seed: int,
    augment: bool = True,
    noise_std: float = 0.06,
    shuffle_buffer: int = 20_000,
) -> tf.data.Dataset:
    """Build a tf.data pipeline for one split.

    Ordering is deliberate: normalise -> shuffle -> batch -> augment. The
    augmentation layers operate on batches, so batching must come first; and
    normalisation comes first of all so augmentation works in a known range.

    Args:
        X: uint8 images (N, 28, 28, 1). uint8 on purpose - `normalize()`
            refuses float input, which is what blocks double-normalisation.
        y: int32 labels.
        training: If True, shuffle and (optionally) augment.
        batch_size: Samples per batch.
        seed: Shuffle seed, so epochs are reproducible.
        augment: Set False to get a shuffled but clean training set - useful
            for measuring how much the augmentation is actually buying.
        noise_std: Gaussian noise sigma, in [0,1] pixel units.
        shuffle_buffer: Shuffle window. Must comfortably exceed batch_size or
            batches end up correlated.

    Returns:
        A prefetching tf.data.Dataset yielding (float32 images, int32 labels).
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    # The single uint8 -> float32 [0,1] conversion. Same function the demo
    # calls at inference time; that is what keeps the two paths identical.
    ds = ds.map(lambda a, b: (normalize(a), b), num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed).batch(batch_size)

        if augment:
            augmenter = build_augmenter()
            ds = ds.map(lambda a, b: (augmenter(a, training=True), b),
                        num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda a, b: (random_blur(a), b),
                        num_parallel_calls=tf.data.AUTOTUNE)
            # Noise last, and clipped: adding it before the geometric
            # transforms would let interpolation smooth it away, and without
            # the clip we would break the [0,1] input contract.
            ds = ds.map(
                lambda a, b: (
                    tf.clip_by_value(
                        a + tf.random.normal(tf.shape(a), 0.0, noise_std), 0.0, 1.0
                    ), b
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    else:
        # No shuffle, no augmentation. Evaluation data stays untouched so the
        # metric describes the model, not the pipeline.
        ds = ds.batch(batch_size)

    return ds.prefetch(tf.data.AUTOTUNE)
