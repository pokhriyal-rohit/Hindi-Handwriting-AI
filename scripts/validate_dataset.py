"""
scripts/validate_dataset.py
============================
CLI wrapper around src.datasets.validation.

Scans every collected trajectory JSON under data/raw/custom_hindi/,
applies 9 quality checks per sample, and writes a full report to
data/raw/custom_hindi/validation_report.json.

Run:
    python scripts/validate_dataset.py [--data-dir PATH]
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

# Allow running as a top-level script without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.validation import validate_canonical_dataset

DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "canonical")
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset Validation Suite")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory of canonical dataset (default: data/canonical)"
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    report = validate_canonical_dataset(data_dir)

    total    = report["total_samples"]
    valid    = report["valid_samples"]
    rejected = report["rejected_samples"]

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"  Total samples scanned : {total}")
    print(f"  Valid                 : {valid}")
    print(f"  Rejected              : {rejected}")
    if total > 0:
        print(f"  Pass rate             : {report['pass_rate']:.1f}%")
    print()

    if "error" in report:
        print(f"  ERROR: {report['error']}")
    elif rejected > 0:
        print("  Rejection reasons:")
        for reason, count in sorted(report["rejection_summary"].items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
        print()
        print("  Failed samples (first 20):")
        for entry in report["rejected_files"][:20]:
            print(f"    {entry['file']}")
            for r in entry["reasons"]:
                print(f"      * {r}")
        if len(report["rejected_files"]) > 20:
            print(f"    ... and {len(report['rejected_files']) - 20} more (see validation_report.json)")
    else:
        print("  All samples passed validation. [OK]")

    print("=" * 60)

    # ── JSON report ───────────────────────────────────────────────────────────
    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_directory": data_dir,
        **report,
    }
    report_path = os.path.join(data_dir, "validation_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\n  Full report saved to: {report_path}")
    except OSError as e:
        print(f"\n  Warning: Could not save report: {e}")

    # Exit code 1 if any rejections (useful in CI pipelines)
    return 1 if rejected > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
