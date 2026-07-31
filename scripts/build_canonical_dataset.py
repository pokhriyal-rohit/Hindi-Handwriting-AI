"""
scripts/build_canonical_dataset.py
===================================
Builds the canonical unified dataset from compatible raw sources.
- Preserves raw data.
- Creates writer-disjoint splits.
- Detects duplicates via geometric hashing.
- Generates manifests and the DATASET_UNIFICATION_REPORT.md.
"""

import os
import sys
import json
import yaml
import shutil
import hashlib
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.converters import CustomCollectorConverter
from src.datasets.structures import TrajectorySample
from src.datasets.validation import validate_sample

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "data", "canonical")

# Offline datasets explicitly marked as incompatible for online trajectory generation
INCOMPATIBLE_DATASETS = [
    "IIIT-HW-Hindi_v1",
    "Dataset_hindi_character-20240412T210635Z-001",
    "devanagari+handwritten+character+dataset",
    "dataset"
]


def geometric_hash(sample: TrajectorySample) -> str:
    """Computes a SHA-256 hash of the exact (x, y) sequences to detect identical drawings."""
    coords = []
    for stroke in sample.strokes:
        for pt in stroke.points:
            coords.append(f"{pt.x:.2f},{pt.y:.2f}")
    return hashlib.sha256("|".join(coords).encode("utf-8")).hexdigest()


def assign_split(writer_id: str, seed: int = 42) -> str:
    """
    Deterministic writer-disjoint split.
    For this initial dataset with only 1 augmented writer ('writer_mock'),
    we forcefully assign to 'train' to ensure training can occur.
    Future writers will be distributed.
    """
    if writer_id == "writer_mock":
        return "train"
        
    random.seed(f"{seed}_{writer_id}")
    r = random.random()
    if r < 0.8:
        return "train"
    elif r < 0.9:
        return "validation"
    else:
        return "test"


def build_dataset():
    print("Building canonical dataset...\n")

    # 1. Setup Directories
    for d in ["train", "validation", "test", "manifests", "reports", "archive"]:
        os.makedirs(os.path.join(CANONICAL_DIR, d), exist_ok=True)

    # 2. Discover and track
    stats = {
        "processed": 0,
        "valid": 0,
        "rejected": 0,
        "duplicates": 0,
        "characters": 0,
        "words": 0,
        "writers": set(),
        "splits": {"train": 0, "validation": 0, "test": 0},
        "writer_splits": {"train": [], "validation": [], "test": []}
    }
    
    seen_hashes = {}  # hash -> filepath
    duplicate_records = []
    rejection_records = []

    converter = CustomCollectorConverter()
    
    # 3. Process Custom Hindi Dataset (Compatible)
    custom_dir = os.path.join(RAW_DIR, "custom_hindi")
    
    if os.path.exists(custom_dir):
        # We look for writer directories
        for writer_dir_name in os.listdir(custom_dir):
            writer_path = os.path.join(custom_dir, writer_dir_name)
            if not os.path.isdir(writer_path) or writer_dir_name == "writer_mock_archive":
                continue
                
            stats["writers"].add(writer_dir_name)
            split = assign_split(writer_dir_name)
            
            if writer_dir_name not in stats["writer_splits"][split]:
                stats["writer_splits"][split].append(writer_dir_name)
                
            # Iterate prompts/words
            for word_dir_name in os.listdir(writer_path):
                word_path = os.path.join(writer_path, word_dir_name)
                if not os.path.isdir(word_path):
                    continue
                    
                for sample_file in os.listdir(word_path):
                    if not sample_file.endswith(".json"):
                        continue
                        
                    stats["processed"] += 1
                    raw_filepath = os.path.join(word_path, sample_file)
                    
                    try:
                        # Validate the raw file BEFORE conversion
                        is_valid, reasons = validate_sample(raw_filepath)
                        if not is_valid:
                            stats["rejected"] += 1
                            rejection_records.append({"file": raw_filepath, "reasons": reasons})
                            continue
                            
                        # Convert to canonical schema
                        sample = converter.from_json(raw_filepath)
                        
                        # Fix metadata for augmented data
                        if writer_dir_name == "writer_mock":
                            sample.metadata.is_synthetic = True
                            
                        # Duplicate Detection
                        h = geometric_hash(sample)
                        if h in seen_hashes:
                            stats["duplicates"] += 1
                            duplicate_records.append({
                                "file": raw_filepath,
                                "duplicate_of": seen_hashes[h]
                            })
                            # Keep it for now as requested
                        seen_hashes[h] = raw_filepath
                        
                        # Serialize Canonical
                        out_dir = os.path.join(CANONICAL_DIR, split, writer_dir_name)
                        os.makedirs(out_dir, exist_ok=True)
                        canonical_filename = f"{word_dir_name}_{sample_file}"
                        out_file = os.path.join(out_dir, canonical_filename)
                        
                        with open(out_file, "w", encoding="utf-8") as f:
                            f.write(sample.model_dump_json(indent=2))
                            
                        stats["valid"] += 1
                        stats["splits"][split] += 1
                        if len(sample.text) == 1:
                            stats["characters"] += 1
                        else:
                            stats["words"] += 1
                            
                    except Exception as e:
                        stats["rejected"] += 1
                        rejection_records.append({"file": raw_filepath, "reasons": [f"Conversion error: {e}"]})

    # 4. Generate Manifests
    manifest_dir = os.path.join(CANONICAL_DIR, "manifests")
    
    writers_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_writers": len(stats["writers"]),
        "writers": list(stats["writers"])
    }
    with open(os.path.join(manifest_dir, "writers.yaml"), "w") as f:
        yaml.dump(writers_manifest, f, sort_keys=False)
        
    splits_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "writer-disjoint",
        "seed": 42,
        "assignments": stats["writer_splits"],
        "counts": stats["splits"]
    }
    with open(os.path.join(manifest_dir, "splits.yaml"), "w") as f:
        yaml.dump(splits_manifest, f, sort_keys=False)
        
    stats_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_processed": stats["processed"],
        "valid_samples": stats["valid"],
        "rejected_samples": stats["rejected"],
        "duplicates_detected": stats["duplicates"],
        "characters": stats["characters"],
        "words": stats["words"]
    }
    with open(os.path.join(manifest_dir, "statistics.json"), "w") as f:
        json.dump(stats_manifest, f, indent=2)
        
    dataset_manifest = {
        "dataset_name": "Hindi-Handwriting-Canonical",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "license": "Research Use Only",
        "total_valid_samples": stats["valid"],
        "total_writers": len(stats["writers"]),
        "language": "hi",
        "script": "devanagari",
        "data_types": ["online_trajectory"]
    }
    with open(os.path.join(manifest_dir, "dataset_manifest.yaml"), "w") as f:
        yaml.dump(dataset_manifest, f, sort_keys=False)
        
    val_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": stats["processed"],
        "valid": stats["valid"],
        "rejected": stats["rejected"],
        "rejected_files": rejection_records
    }
    reports_dir = os.path.join(CANONICAL_DIR, "reports")
    with open(os.path.join(reports_dir, "validation_report.json"), "w") as f:
        json.dump(val_report, f, indent=2)
        
    # 5. Final Report (DATASET_UNIFICATION_REPORT.md)
    report_md = f"""# Dataset Unification Report

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
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

* **Total Raw Samples Processed:** {stats["processed"]}
* **Total Valid Samples Converted:** {stats["valid"]}
* **Total Rejected Samples:** {stats["rejected"]}
* **Duplicates Detected:** {stats["duplicates"]}
* **Characters:** {stats["characters"]}
* **Words/Ligatures:** {stats["words"]}

## 6. Writer-Disjoint Splits (Seed=42)

* **Train:** {stats["splits"]["train"]} samples ({len(stats["writer_splits"]["train"])} writers)
* **Validation:** {stats["splits"]["validation"]} samples ({len(stats["writer_splits"]["validation"])} writers)
* **Test:** {stats["splits"]["test"]} samples ({len(stats["writer_splits"]["test"])} writers)

*(Note: Because only 1 writer currently exists, all data was placed in the train split to prevent empty training sets. Future writers will distribute into val/test deterministically.)*

## 7. Duplicate Detection

Geometric hashing (SHA-256 over all `x,y` coordinate strings) detected **{stats["duplicates"]}** exact duplicate trajectories. No data was deleted.

## 8. Recommendations & Future Work

1. **Collect Real Data:** The canonical structure is now ready and strictly validated. Real human writers are required to populate the `validation` and `test` splits.
2. **Offline Data:** Consider moving the 4 offline datasets out of `data/raw/` into `data/archive/` or entirely off-repository to save space (they account for ~300k+ files), as they are completely unusable for the current online modelling pipeline.
"""
    with open(os.path.join(reports_dir, "DATASET_UNIFICATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"Done. Processed {stats['processed']} samples. Valid: {stats['valid']}.")
    print(f"Report generated at: data/canonical/reports/DATASET_UNIFICATION_REPORT.md")

if __name__ == "__main__":
    build_dataset()
