"""
src/datasets/validation.py
==========================
Core dataset validation logic, importable by both the CLI script
(scripts/validate_dataset.py) and the training pipeline (src/training/train.py).

The script enforces a data quality contract before any sample reaches a model.
Nine checks are applied per sample; a sample must pass all nine to be valid.
"""

import os
import json
import math
from typing import List, Tuple, Dict, Any


# ── Per-sample helpers ────────────────────────────────────────────────────────

def _is_finite(value: Any) -> bool:
    """Returns True if value is a finite, non-NaN float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_sample(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validates a single trajectory JSON file against 9 quality criteria.

    Returns:
        (is_valid, list_of_rejection_reasons)
        is_valid is True only when the reason list is empty.
    """
    reasons: List[str] = []

    # 1. JSON parse
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return False, [f"Corrupt JSON: {e}"]

    strokes = data.get("strokes", [])

    # 2. Non-empty strokes
    if not strokes:
        reasons.append("Empty strokes: no stroke data found")

    # 3. Per-point validation (finite coords, timestamp monotonicity)
    all_points: List[Dict] = []
    last_t = None
    for stroke_idx, stroke in enumerate(strokes):
        if not isinstance(stroke, list) or len(stroke) == 0:
            reasons.append(f"Stroke {stroke_idx}: empty or malformed")
            continue

        for pt_idx, pt in enumerate(stroke):
            label = f"stroke[{stroke_idx}] pt[{pt_idx}]"

            x = pt.get("x")
            y = pt.get("y")
            if not _is_finite(x):
                reasons.append(f"{label}: non-finite x={x}")
            if not _is_finite(y):
                reasons.append(f"{label}: non-finite y={y}")

            t = pt.get("t")
            if t is not None:
                if not _is_finite(t):
                    reasons.append(f"{label}: non-finite timestamp t={t}")
                elif last_t is not None and t < last_t:
                    reasons.append(
                        f"{label}: timestamp not monotonic (t={t} < prev t={last_t})"
                    )
                last_t = t

            all_points.append(pt)

    # 4. Minimum point count
    if len(all_points) < 10:
        reasons.append(f"Too few points: {len(all_points)} (minimum 10 required)")

    # 5 & 6. Duration
    stats = data.get("statistics", {})
    duration = stats.get("duration_ms", 0)
    if duration == 0:
        reasons.append("Zero duration: sample may be corrupted")
    if duration > 30_000:
        reasons.append(f"Suspiciously long duration: {duration}ms (threshold: 30,000ms)")

    # 7. Path length
    path_len = stats.get("path_length_px", 0)
    if path_len < 10:
        reasons.append(f"Path length too short: {path_len:.1f}px (minimum 10px)")

    # 8. Required metadata
    if not data.get("word"):
        reasons.append("Missing 'word' field")

    # 9. Degenerate bounding box
    if all_points:
        xs = [pt.get("x", 0) for pt in all_points if _is_finite(pt.get("x"))]
        ys = [pt.get("y", 0) for pt in all_points if _is_finite(pt.get("y"))]
        if xs and (max(xs) - min(xs)) < 1.0:
            reasons.append(
                f"Degenerate X range: {max(xs)-min(xs):.2f}px (all points on same vertical line)"
            )
        if ys and (max(ys) - min(ys)) < 1.0:
            reasons.append(
                f"Degenerate Y range: {max(ys)-min(ys):.2f}px (all points on same horizontal line)"
            )

    return len(reasons) == 0, reasons


# ── Directory-level scan ──────────────────────────────────────────────────────

def validate_directory(data_dir: str) -> Dict[str, Any]:
    """
    Scans every JSON sample under data_dir (writer/word/*.json hierarchy)
    and returns a structured validation report dict.

    Keys:
        total_samples   int
        valid_samples   int
        rejected_samples int
        pass_rate       float  (0.0–100.0)
        rejection_summary  Dict[str, int]   reason_prefix -> count
        rejected_files     List[Dict]       [{file, reasons}, ...]
    """
    total = 0
    valid_count = 0
    rejected: List[Dict[str, Any]] = []
    rejection_summary: Dict[str, int] = {}

    if not os.path.exists(data_dir):
        return {
            "total_samples": 0,
            "valid_samples": 0,
            "rejected_samples": 0,
            "pass_rate": 0.0,
            "rejection_summary": {},
            "rejected_files": [],
            "error": f"Data directory not found: {data_dir}"
        }

    for writer in sorted(os.listdir(data_dir)):
        writer_dir = os.path.join(data_dir, writer)
        if not os.path.isdir(writer_dir):
            continue
        for word in sorted(os.listdir(writer_dir)):
            word_dir = os.path.join(writer_dir, word)
            if not os.path.isdir(word_dir):
                continue
            for sample_file in sorted(f for f in os.listdir(word_dir) if f.endswith(".json")):
                filepath = os.path.join(word_dir, sample_file)
                total += 1
                is_valid, reasons = validate_sample(filepath)
                if is_valid:
                    valid_count += 1
                else:
                    rel = os.path.relpath(filepath, data_dir)
                    rejected.append({"file": rel, "reasons": reasons})
                    for r in reasons:
                        key = r.split(":")[0].strip()
                        rejection_summary[key] = rejection_summary.get(key, 0) + 1

    return {
        "total_samples": total,
        "valid_samples": valid_count,
        "rejected_samples": len(rejected),
        "pass_rate": round(valid_count / total * 100, 2) if total > 0 else 0.0,
        "rejection_summary": rejection_summary,
        "rejected_files": rejected,
    }


# ── Training gate ─────────────────────────────────────────────────────────────

class DatasetValidationError(RuntimeError):
    """Raised when the training gate detects invalid samples and force=False."""
    pass


def pre_training_gate(data_dir: str, force: bool = False) -> Dict[str, Any]:
    """
    Quality gate: validates the entire dataset before training begins.

    Args:
        data_dir: Path to the raw data directory.
        force:    If True, print a warning but continue even if samples fail.
                  If False (default), raise DatasetValidationError on any rejection.

    Returns:
        The validation report dict (same format as validate_directory).

    Raises:
        DatasetValidationError: if any samples fail and force=False.
    """
    print("Running pre-training dataset validation...")
    report = validate_directory(data_dir)

    total    = report["total_samples"]
    valid    = report["valid_samples"]
    rejected = report["rejected_samples"]

    if total == 0:
        raise DatasetValidationError(
            f"No samples found in {data_dir}. "
            "Collect handwriting data first (see src/tools/dataset_collector/)."
        )

    print(f"  Scanned : {total} samples")
    print(f"  Valid   : {valid}")
    print(f"  Rejected: {rejected}  (pass rate: {report['pass_rate']:.1f}%)")

    if rejected > 0:
        print("\n  Rejection summary:")
        for reason, count in sorted(report["rejection_summary"].items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
        print(f"\n  First failed sample: {report['rejected_files'][0]['file']}")
        for r in report['rejected_files'][0]['reasons']:
            print(f"    * {r}")

        if force:
            print(
                f"\n  [WARNING] {rejected} sample(s) failed validation. "
                "Training is proceeding because --force-train was set. "
                "Invalid samples will be silently skipped by the DataLoader."
            )
        else:
            raise DatasetValidationError(
                f"\n{rejected} sample(s) failed validation in {data_dir}.\n"
                "Fix the issues above and re-run, or pass --force-train to "
                "skip validation and train on the remaining valid samples.\n"
                "Run 'python scripts/validate_dataset.py' for the full report."
            )
    else:
        print(f"  All {total} samples passed validation. [OK]")

    return report
