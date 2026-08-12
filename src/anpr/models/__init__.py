"""Model architectures and training.

Owner: Model role (§7). Tickets: ML-42, ML-46, ML-11.

    architectures.py  build_baseline_mlp, build_cnn, compile_model
    train.py          train_model, finetune_on_printed, callbacks

    from anpr.models import build_cnn, compile_model, train_model
"""

from anpr.models.architectures import build_baseline_mlp, build_cnn, compile_model
from anpr.models.train import build_callbacks, finetune_on_printed, train_model

__all__ = [
    "build_baseline_mlp", "build_cnn", "compile_model",
    "train_model", "finetune_on_printed", "build_callbacks",
]
