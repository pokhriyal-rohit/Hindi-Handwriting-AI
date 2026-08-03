import os
import json
from pathlib import Path

def format_bytes(size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def get_experiment_details(exp_dir):
    exp_dir = Path(exp_dir)
    details = {
        "exp": exp_dir.name,
        "model": "Unknown",
        "epochs": "?",
        "cer": "-",
        "wer": "-",
        "dtw": "-",
        "samples": "-",
        "ckpt_size": "-",
        "notes": ""
    }
    
    # Determine Model
    if "ocr" in exp_dir.name.lower():
        details["model"] = "CRNN"
    elif "lstm" in exp_dir.name.lower():
        details["model"] = "LSTM"
    elif "transformer" in exp_dir.name.lower():
        details["model"] = "Transformer"
        
    # Read metrics
    eval_json_paths = [
        exp_dir / "evaluation_validation.json",
        exp_dir / "metrics" / "evaluation_validation.json",
        exp_dir / "evaluation_test.json",
        exp_dir / "metrics.json"
    ]
    
    metrics_found = False
    for path in eval_json_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    metrics = json.load(f)
                
                # Check for standard OCR / Trajectory keys
                if "cer" in metrics:
                    details["cer"] = f"{metrics['cer']:.4f}"
                if "wer" in metrics:
                    details["wer"] = f"{metrics['wer']:.4f}"
                if "dtw" in metrics:
                    details["dtw"] = f"{metrics['dtw']:.4f}"
                if "num_samples" in metrics:
                    details["samples"] = str(metrics["num_samples"])
                elif "total_samples" in metrics:
                    details["samples"] = str(metrics["total_samples"])
                    
                metrics_found = True
                break
            except Exception as e:
                details["notes"] = f"Error reading metrics: {e}"
                
    # Look at config.yaml if it exists to extract Epochs
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                
            if "epochs" in config:
                details["epochs"] = str(config["epochs"])
            elif "training" in config and "epochs" in config["training"]:
                details["epochs"] = str(config["training"]["epochs"])
                
        except Exception:
            pass
            
    # Quick check for checkpoints and their sizes
    ckpts = list(exp_dir.glob("*.pt"))
    if not ckpts:
        ckpts = list(exp_dir.glob("checkpoints/*.pt"))
    
    if ckpts:
        # Find the best checkpoint size (or latest)
        best_ckpt = next((c for c in ckpts if "best" in c.name), ckpts[0])
        details["ckpt_size"] = format_bytes(os.path.getsize(best_ckpt))
    
    if not metrics_found and not ckpts:
        details["notes"] = "No metrics/ckpts found (Empty?)"
    elif not metrics_found:
        details["notes"] = "Trained but not evaluated"
        
    return details

def main():
    experiments = []
    base_dirs = [Path("experiments/OCR"), Path("experiments")]
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
            
        for exp_dir in base_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != "OCR":
                # Avoid duplicating OCR if scanned in the second pass
                if base_dir.name == "experiments" and (base_dir / "OCR" / exp_dir.name).exists():
                    continue
                
                experiments.append(get_experiment_details(exp_dir))
            
    # Sort by experiment name (chronological assuming timestamp format)
    experiments.sort(key=lambda x: x["exp"])
    
    if not experiments:
        print("No experiments found in 'experiments/' or 'experiments/OCR/'.")
        return

    # Generate Markdown Table
    headers = ["Exp", "Model", "Epochs", "CER", "WER", "DTW", "Samples", "Ckpt Size", "Notes"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---'] * len(headers))}|")
    
    for exp in experiments:
        row = [
            exp['exp'],
            exp['model'],
            exp['epochs'],
            exp['cer'],
            exp['wer'],
            exp['dtw'],
            exp['samples'],
            exp['ckpt_size'],
            exp['notes']
        ]
        print(f"| {' | '.join(row)} |")

if __name__ == "__main__":
    main()
