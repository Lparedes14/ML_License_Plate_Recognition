"""End-to-end inference: image -> plate string + confidence.

Owner: Pipeline role (§7). Tickets: ML-45, ML-14, ML-47.

    from anpr.inference import load_reader, read_plate

    model, idx2char = load_reader("artifacts/models/plate_cnn.keras")
    result = read_plate("some_plate.png", model, idx2char, n_expected=7)
    print(result.text, result.plate_confidence)
"""

from anpr.inference.read_plate import (
    PlateRead,
    aggregate_confidence,
    load_reader,
    read_plate,
)

__all__ = ["PlateRead", "read_plate", "load_reader", "aggregate_confidence"]
