# EMNIST load - acceptance record

Generated 2026-08-13T01:30:37+00:00

## Acceptance criteria (ML-36)

- **[PASS]** Ten samples plotted and visually confirmed upright before training
  - evidence: notebooks/01_data_acceptance.ipynb plots them; this script proves the guard rejects transposed input
- **[PASS]** Assertion in the loader that fails loudly on orientation regression
  - evidence: assert_upright() is called inside _finalise(), so no route can bypass it
- **[PASS]** Load method and source recorded so results reproduce
  - evidence: artifacts/provenance/provenance.json
- **[PASS]** Splits verified disjoint and stratified (ML-7, ML-40)
  - evidence: artifacts/provenance/split_manifest.json

## train split

- route: `kaggle_csv` (fell back past: none)
- source: `local CSV (kaggle:crawford/emnist) -> C:\FIU-MSIS\ML-FinalProject\ML_License_Plate_Recognition\data\raw\emnist-byclass-train.csv`
- library: pandas 3.0.5
- images: 220,000, transpose applied: True
- content hash (first 2k): `fcbe50e6825a945e`
- loaded: 2026-08-13T01:30:28+00:00

## test split

- route: `kaggle_csv` (fell back past: none)
- source: `local CSV (kaggle:crawford/emnist) -> C:\FIU-MSIS\ML-FinalProject\ML_License_Plate_Recognition\data\raw\emnist-byclass-test.csv`
- library: pandas 3.0.5
- images: 60,000, transpose applied: True
- content hash (first 2k): `bda03cba5be30971`
- loaded: 2026-08-13T01:30:34+00:00

## Settings

- seed: 42
- case strategy: merge
- imbalance strategy: weighted
- numpy 2.4.6

## Class distribution

```json
{
  "n_samples": 220000,
  "per_class_share": {
    "0": 0.04995,
    "1": 0.0552,
    "2": 0.049,
    "3": 0.05052,
    "4": 0.04772,
    "5": 0.04468,
    "6": 0.04905,
    "7": 0.051,
    "8": 0.0487,
    "9": 0.04835,
    "A": 0.02366,
    "B": 0.01292,
    "C": 0.01867,
    "D": 0.02142,
    "E": 0.04203,
    "F": 0.01684,
    "G": 0.00887,
    "H": 0.01689,
    "I": 0.02064,
    "J": 0.00812,
    "K": 0.00725,
    "L": 0.0297,
    "M": 0.01659,
    "N": 0.02827,
    "O": 0.0397,
    "P": 0.01538,
    "Q": 0.00794,
    "R": 0.0273,
    "S": 0.0337,
    "T": 0.04059,
    "U": 0.02197,
    "V": 0.01112,
    "W": 0.0106,
    "X": 0.00817,
    "Y": 0.00996,
    "Z": 0.00753
  },
  "rarest_char": "K",
  "rarest_share": 0.00725,
  "commonest_char": "1",
  "commonest_share": 0.0552,
  "imbalance_ratio": 7.61,
  "uniform_share_would_be": 0.02778
}
```