# data/raw/

EMNIST goes here. **Nothing in this folder is committed** — it is ~1.4 GB and
fully reproducible from `artifacts/provenance/provenance.json`.

## Option 1 — let the loader download it (default)

```bash
python scripts/prepare_data.py
```

Falls back through three routes automatically: local CSV → torchvision → TFDS.

## Option 2 — local files (faster, no network, best for a demo)

Download from [kaggle.com/datasets/crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)
and place here:

```
data/raw/emnist-byclass-train.csv      (or .csv.zip — no need to extract)
data/raw/emnist-byclass-test.csv
```

The loader reads `.csv.zip` directly.

## Expected row counts

| File | Rows |
|---|---|
| `emnist-byclass-train.csv` | 697,932 |
| `emnist-byclass-test.csv` | 116,323 |

`_load_kaggle_csv` verifies these and **refuses a truncated file**. This is
not paranoia: our first Colab run read 129,461 training rows from an upload
that had silently stopped, and without the check we would have trained on 18%
of the data without noticing.

If you see that error, the download or upload did not finish. Re-fetch it.
