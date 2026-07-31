# Dataset Unification Report

**Date:** 2026-07-31
**Version:** 1.0.0

## 1. Repository Datasets Discovered

Five datasets were discovered in `data/raw/`:

| Dataset | Type | Compatibility | Action |
|---|---|---|---|
| `custom_hindi` | Online Trajectory | ✅ Compatible | Converted |
| `IIIT-HW-Hindi_v1` | Offline Image | ❌ Incompatible | Skipped |
| `Dataset_hindi_character...` | Offline Image | ❌ Incompatible | Skipped |
| `devanagari+handwritten...` | Offline Image | ❌ Incompatible | Skipped |
| `dataset` (Mystery) | Offline Image | ❌ Incompatible | Skipped |

## 2. Research Summary & Compatibility

The repository's models (`TrajectorySample`, `BaselineLSTM`) are strictly designed for **online** handwriting generation. They require temporal sequences of coordinates `(x, y, timestamp, pen_state)`. 

Four of the datasets are **offline** (static 2D images). Because it is mathematically impossible to perfectly reconstruct pen trajectory and timing from a static image, these datasets cannot be natively converted to the canonical schema. They have been left in `data/raw/` untouched for archival purposes.

## 3. Conversion Strategy

The `custom_hindi` dataset was processed through `CustomCollectorConverter` to yield strict Pydantic `TrajectorySample` objects. No coordinates were artificially scaled during conversion to preserve original data fidelity. The data was explicitly tagged as `is_synthetic = True` because `writer_mock` contains mathematically augmented pilot data.

## 4. Canonical Folder Layout

```
data/canonical/
├── train/
├── validation/
├── test/
├── manifests/
└── reports/
```

## 5. Statistics

* **Total Raw Samples Processed:** 410
* **Total Valid Samples Converted:** 410
* **Total Rejected Samples:** 0
* **Duplicates Detected:** 400
* **Characters:** 200
* **Words/Ligatures:** 210

## 6. Writer-Disjoint Splits (Seed=42)

* **Train:** 410 samples (1 writers)
* **Validation:** 0 samples (0 writers)
* **Test:** 0 samples (0 writers)

*(Note: Because only 1 writer currently exists, all data was placed in the train split to prevent empty training sets. Future writers will distribute into val/test deterministically.)*

## 7. Duplicate Detection

Geometric hashing (SHA-256 over all `x,y` coordinate strings) detected **400** exact duplicate trajectories. No data was deleted.

## 8. Recommendations & Future Work

1. **Collect Real Data:** The canonical structure is now ready and strictly validated. Real human writers are required to populate the `validation` and `test` splits.
2. **Offline Data:** Consider moving the 4 offline datasets out of `data/raw/` into `data/archive/` or entirely off-repository to save space (they account for ~300k+ files), as they are completely unusable for the current online modelling pipeline.
