import json
import csv
import os
from typing import Dict, Any, List
from src.evaluation.config import EvaluationConfig

def generate_json_report(config: EvaluationConfig, summary_results: Dict[str, Any], filepath: str):
    """Generates a machine-readable JSON evaluation report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    report = {
        "metadata": config.dict(),
        "results": summary_results
    }
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

def generate_markdown_report(config: EvaluationConfig, summary_results: Dict[str, Any], filepath: str):
    """Generates a human-readable Markdown evaluation report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("# Handwriting AI Evaluation Report\n\n")
        f.write("## 1. System & Configuration\n")
        f.write(f"- **Evaluation Version:** {config.evaluation_version}\n")
        f.write(f"- **Dataset Version:** {config.dataset_version}\n")
        f.write(f"- **Renderer Version:** {config.renderer_version}\n")
        f.write(f"- **Model Version:** {config.model_version}\n\n")
        
        f.write("## 2. Statistical Summary\n")
        f.write("| Metric | Mean | Std/Other |\n")
        f.write("|---|---|---|\n")
        
        # Simple rendering for demo
        for key, val in summary_results.items():
            if "mean" in key:
                base = key.replace("_mean", "")
                std = summary_results.get(f"{base}_std", "-")
                f.write(f"| {base} | {val:.4f} | {std} |\n")
                
        f.write("\n## 3. Metric Versions\n")
        for m, v in config.metric_versions.items():
            f.write(f"- {m}: v{v}\n")

def generate_csv_summary(config: EvaluationConfig, summary_results: Dict[str, Any], filepath: str):
    """Appends to a continuous CSV tracking file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    write_header = not os.path.exists(filepath)
    
    # Flatten config + metrics
    row = config.dict()
    row.update(summary_results)
    
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
