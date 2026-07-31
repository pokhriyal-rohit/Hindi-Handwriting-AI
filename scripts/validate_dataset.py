"""
Dataset Validation Suite
========================
Quality gate: validates every collected JSON trajectory before it enters training.

Run:
    python scripts/validate_dataset.py

Output:
    Console summary report + data/raw/custom_hindi/validation_report.json
"""

import os
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "custom_hindi"))


def _is_finite(value: Any) -> bool:
    """Returns True if value is a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_sample(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validates a single sample JSON file.

    Returns:
        (is_valid, list_of_rejection_reasons)
    """
    reasons = []

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

    # 3. Per-stroke validation
    all_points: List[Dict] = []
    last_t = None
    for stroke_idx, stroke in enumerate(strokes):
        if not isinstance(stroke, list) or len(stroke) == 0:
            reasons.append(f"Stroke {stroke_idx}: empty or malformed")
            continue

        for pt_idx, pt in enumerate(stroke):
            label = f"stroke[{stroke_idx}] pt[{pt_idx}]"

            # 4. Finite coordinates
            x = pt.get("x")
            y = pt.get("y")
            if not _is_finite(x):
                reasons.append(f"{label}: non-finite x={x}")
            if not _is_finite(y):
                reasons.append(f"{label}: non-finite y={y}")

            # 5. Timestamp monotonicity
            t = pt.get("t")
            if t is not None:
                if not _is_finite(t):
                    reasons.append(f"{label}: non-finite timestamp t={t}")
                elif last_t is not None and t < last_t:
                    reasons.append(f"{label}: timestamp not monotonic (t={t} < previous t={last_t})")
                last_t = t

            all_points.append(pt)

    # 6. Minimum point count
    if len(all_points) < 10:
        reasons.append(f"Too few points: {len(all_points)} (minimum 10 required)")

    # 7. Statistics block checks
    stats = data.get("statistics", {})
    duration = stats.get("duration_ms", 0)
    path_len = stats.get("path_length_px", 0)

    if duration == 0:
        reasons.append("Zero duration: sample may be corrupted")

    if duration > 30_000:
        reasons.append(f"Suspiciously long duration: {duration}ms (threshold: 30,000ms)")

    if path_len < 10:
        reasons.append(f"Path length too short: {path_len:.1f}px (minimum 10px)")

    # 8. Required metadata fields
    if not data.get("word"):
        reasons.append("Missing 'word' field")

    # 9. Reasonable bounding box (coordinates should not all be identical)
    if all_points:
        xs = [pt.get("x", 0) for pt in all_points if _is_finite(pt.get("x", None))]
        ys = [pt.get("y", 0) for pt in all_points if _is_finite(pt.get("y", None))]
        if xs and (max(xs) - min(xs)) < 1.0:
            reasons.append(f"Degenerate X range: max-min={max(xs)-min(xs):.2f}px (all points on same vertical line)")
        if ys and (max(ys) - min(ys)) < 1.0:
            reasons.append(f"Degenerate Y range: max-min={max(ys)-min(ys):.2f}px (all points on same horizontal line)")

    return len(reasons) == 0, reasons


def run_validation() -> None:
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    writers = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not writers:
        print("No writer directories found.")
        return

    total = 0
    valid_count = 0
    rejected: List[Dict[str, Any]] = []
    rejection_reasons: Dict[str, int] = {}

    for writer in sorted(writers):
        writer_dir = os.path.join(DATA_DIR, writer)
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
                    rel_path = os.path.relpath(filepath, DATA_DIR)
                    rejected.append({"file": rel_path, "reasons": reasons})
                    for r in reasons:
                        # Bucket reason by prefix for summary
                        key = r.split(":")[0].strip()
                        rejection_reasons[key] = rejection_reasons.get(key, 0) + 1

    # ── Console Report ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"  Total samples scanned : {total}")
    print(f"  Valid                 : {valid_count}")
    print(f"  Rejected              : {len(rejected)}")
    if total > 0:
        print(f"  Pass rate             : {valid_count / total * 100:.1f}%")
    print()

    if rejection_reasons:
        print("  Rejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    else:
        print("  All samples passed validation  [OK]")

    if rejected:
        print()
        print("  Failed samples (first 20):")
        for entry in rejected[:20]:
            print(f"    {entry['file']}")
            for r in entry["reasons"]:
                print(f"      • {r}")
        if len(rejected) > 20:
            print(f"    ... and {len(rejected) - 20} more (see validation_report.json)")

    print("=" * 60)

    # ── JSON Report ─────────────────────────────────────────────
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_directory": DATA_DIR,
        "total_samples": total,
        "valid_samples": valid_count,
        "rejected_samples": len(rejected),
        "pass_rate": round(valid_count / total * 100, 2) if total > 0 else 0.0,
        "rejection_summary": rejection_reasons,
        "rejected_files": rejected
    }

    report_path = os.path.join(DATA_DIR, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    run_validation()
