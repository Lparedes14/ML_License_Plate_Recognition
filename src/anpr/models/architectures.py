"""The two classifier architectures, and why each layer is there.

§8 promises the instructor will ask, live: "how many layers does the model
have? what is your learning rate?" Everything needed to answer is in this
file and `config/default.yaml`. Read both before the demo.

WHY TWO MODELS
    The MLP is not a serious candidate - it is the control. It answers "what
    does the simplest thing get us?", which makes the CNN's number mean
    something. A CNN at 88% is unremarkable on its own; a CNN at 88% against
    an MLP at 71% demonstrates that spatial structure is what is doing the
    work. Train both, report both. It costs six epochs.

Owner: Model role. Tickets: ML-42, ML-10.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_baseline_mlp(n_classes: int, input_shape=(28, 28, 1)) -> keras.Model:
    """A dense network. The control, not a candidate.

    Flattening throws away every spatial relationship - the model cannot know
    that two pixels are adjacent. It still reaches a respectable-looking
    accuracy on EMNIST, which is precisely why it is a useful comparison: it
    shows how much of the score comes from the data being easy rather than
    from the architecture being good.

    Args:
        n_classes: Output width, 36.
        input_shape: Per the input contract.

    Returns:
        An uncompiled Keras model.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),                      # spatial structure discarded here
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),                   # 200k+ params on 28x28 overfits fast
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="baseline_mlp")


def build_cnn(n_classes: int, input_shape=(28, 28, 1)) -> keras.Model:
    """The real classifier: 3 convolutional blocks then a dense head.

    ARCHITECTURE, and the reasoning for each choice:

      Block 1  32 filters   learns strokes and edges. 32 is enough for
                            primitives; more would just be slower.
      Block 2  64 filters   learns corners and junctions - the features that
                            separate 'B' from '8'.
      Block 3  128 filters  learns whole-glyph structure.

      Filters double each block while spatial size halves. That keeps the
      compute per block roughly constant, which is the standard trade and
      the reason this shape is everywhere.

      3x3 kernels          two stacked 3x3 layers see the same region as one
                           5x5 but with fewer parameters and an extra
                           non-linearity between them.
      padding="same"       28 -> 14 -> 7 comes only from pooling, so the
                           arithmetic stays easy to explain.
      BatchNormalization   before activation. Stabilises training enough to
                           use lr=1e-3 from epoch 1 without warmup.
      MaxPooling2D         downsample, and buy a little translation
                           invariance - which matters because our
                           segmentation boxes will not be perfectly tight.
      Dropout, rising      0.25 / 0.25 / 0.30 / 0.50. Lightest early (early
                           features are general), heaviest in the dense head
                           (where the parameters and the overfitting are).
      GlobalAveragePooling instead of Flatten. Collapses 7x7x128 to 128 and
                           removes ~300k parameters, which on a class with
                           only ~500 'Q' samples is a real regularisation win.

    Total: 3 conv blocks + 1 dense hidden layer + output. ~250k parameters.
    Call `model.summary()` to confirm before the demo - do not quote from
    memory.

    Args:
        n_classes: Output width, 36.
        input_shape: Per the input contract.

    Returns:
        An uncompiled Keras model.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),

        # --- block 1: strokes and edges (28x28 -> 14x14) ------------------
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # --- block 2: corners and junctions (14x14 -> 7x7) ----------------
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # --- block 3: whole-glyph structure (7x7, no further pooling) -----
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.30),

        # --- head --------------------------------------------------------
        layers.GlobalAveragePooling2D(),       # 7x7x128 -> 128
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.50),
        layers.Dense(n_classes, activation="softmax"),
    ], name="plate_cnn")


# `use_bias=False` above is not a typo: BatchNormalization immediately after a
# Conv2D applies its own learned shift, so the conv's bias term is redundant.
# Standard practice, and one fewer parameter per filter.


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    """Attach optimiser, loss and metrics.

    sparse_categorical_crossentropy because labels are integers, not
    one-hot - it is mathematically identical and avoids materialising a
    36-wide one-hot matrix for every sample.

    Args:
        model: An uncompiled model.
        learning_rate: From cfg.training.lr (or lr_ft when fine-tuning).

    Returns:
        The same model, compiled.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
