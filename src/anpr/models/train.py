"""Training loop, checkpointing, and the fine-tune pass for the domain gap.

Owner: Model role. Tickets: ML-42 (train), ML-46 (save/reload).

WHAT GETS SAVED, AND WHY ALL OF IT MATTERS
    <name>.keras       the weights and architecture
    <name>.classmap.json  index -> character. Without this the model's output
                       integers are meaningless. ML-46 exists because a model
                       that reloads without its class map reads every plate
                       confidently and wrongly, with no error anywhere.
    <name>.history.json  per-epoch loss and accuracy, for the training curve
                       in the results document.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from anpr.config import MODELS_DIR
from anpr.data.labels import compute_class_weights, save_class_map


def build_callbacks(patience: int, checkpoint_path: str | Path) -> list:
    """Early stopping, best-checkpointing and LR reduction.

    `restore_best_weights=True` is the important flag: without it, training
    stops at the epoch AFTER the best one and you keep the worse weights.

    Args:
        patience: Epochs without validation improvement before stopping.
        checkpoint_path: Where to write the best model.

    Returns:
        A list of Keras callbacks.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        # Halve the LR when validation loss plateaus. Cheap, and usually worth
        # a point or two at the end of training.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
    ]


def train_model(
    model: keras.Model,
    train_ds,
    val_ds,
    epochs: int,
    y_train: np.ndarray,
    imbalance_strategy: str,
    patience: int,
    name: str,
    case_strategy: str,
    models_dir: Path | None = None,
) -> tuple[keras.Model, dict]:
    """Fit a model, then save weights, class map and history together.

    Args:
        model: A COMPILED model.
        train_ds, val_ds: tf.data pipelines from `data.pipeline.make_dataset`.
        epochs: Maximum epochs; early stopping usually ends it sooner.
        y_train: Training labels, needed to compute class weights.
        imbalance_strategy: "none" | "weighted" | "resampled". Note that
            "resampled" is applied when building the arrays, not here, so this
            function only acts on "weighted".
        patience: Early-stopping patience.
        name: Model name; determines the output filenames.
        case_strategy: Recorded in the class map for traceability.
        models_dir: Output directory. Defaults to `artifacts/models/`.

    Returns:
        (trained model, history dict).
    """
    models_dir = Path(models_dir or MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / f"{name}.keras"

    # Class weighting is the only imbalance strategy applied at fit time.
    class_weight = None
    if imbalance_strategy == "weighted":
        class_weight = compute_class_weights(y_train)
        rarest = max(class_weight, key=class_weight.get)
        print(f"  class weighting ON - heaviest class index {rarest} "
              f"at weight {class_weight[rarest]:.2f}x")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(patience, ckpt),
        verbose=1,
    )

    # Save all three artifacts together. Splitting them up is how a model and
    # its class map drift apart.
    model.save(ckpt)
    save_class_map(models_dir / f"{name}.classmap.json", case_strategy)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(models_dir / f"{name}.history.json", "w", encoding="utf-8") as fh:
        json.dump(hist, fh, indent=2)

    print(f"  saved -> {ckpt}")
    print(f"  saved -> {models_dir / f'{name}.classmap.json'}")
    return model, hist


def finetune_on_printed(
    model: keras.Model,
    printed_ds,
    val_ds,
    epochs: int,
    lr_ft: float,
    name: str,
    case_strategy: str,
) -> tuple[keras.Model, dict]:
    """Adapt a handwriting-trained model to printed plate typefaces.

    §3: "EMNIST is handwritten; plates are printed. A model trained purely on
    handwriting will underperform on plate typefaces. You do not have to close
    this gap in two weeks. You do have to notice it and say what you would do."

    This is the cheapest meaningful attempt: continue training on rendered
    font glyphs at a LOWER learning rate, so the model adapts rather than
    forgetting what it learned from 220,000 handwritten samples.

    MEASURE BEFORE AND AFTER on the same tier-B plate set. If it makes things
    worse - which happens, usually because the rendered glyphs are centred
    differently from EMNIST's - that is a legitimate and reportable finding,
    not a failure. Say so rather than hiding it.

    Args:
        model: An already-trained model.
        printed_ds: Dataset from `data.plates.render_font_glyphs` [ML-37].
        val_ds: Validation data; keep some handwritten samples in it so
            catastrophic forgetting is visible.
        epochs: cfg.training.epochs_ft.
        lr_ft: cfg.training.lr_ft - deliberately ~5x lower than the initial LR.
        name: Output model name.
        case_strategy: For the class map.

    Returns:
        (fine-tuned model, history dict).
    """
    raise NotImplementedError(
        "Blocked on ML-37 (render_font_glyphs). Once plates.py exists: "
        "recompile the model with lr_ft, fit on printed_ds, and save under a "
        "NEW name so the handwriting-only model survives for comparison."
    )
