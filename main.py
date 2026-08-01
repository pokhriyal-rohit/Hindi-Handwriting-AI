import argparse
import sys
import os
import json
import torch
import shutil
from datetime import datetime, timezone

from src.utils.config import load_colab_config
from src.utils.environment import capture_environment, save_environment, get_git_commit, compute_dataset_hash

def cmd_setup(args):
    print("=== Environment Verification ===")
    print(f"Python: {sys.version.split(' ')[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    cfg = load_colab_config()
    canonical_root = "data/canonical"
    if not os.path.exists(canonical_root):
        print(f"ERROR: Canonical dataset not found at {canonical_root}.")
        print("Run scripts/build_canonical_dataset.py first.")
        sys.exit(1)
        
    hash_val = compute_dataset_hash(canonical_root)
    print(f"Dataset Hash (SHA-256): {hash_val[:16]}...")
    print("Setup verified successfully.")

def cmd_info(args):
    cfg = load_colab_config()
    canonical_root = "data/canonical"
    env = capture_environment(cfg, canonical_root, time.time() if 'time' in globals() else 0)
    
    print("=== Repository Info ===")
    print(f"Git Commit: {env['git_commit']}")
    print(f"Dataset Hash: {env['dataset_hash']}")
    
    # Calculate Writers dynamically
    writers = set()
    for mode in ["online", "offline"]:
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(canonical_root, mode, split)
            if os.path.exists(split_dir):
                for w in os.listdir(split_dir):
                    if os.path.isdir(os.path.join(split_dir, w)) and w.startswith("writer_"):
                        writers.add(w)
    print(f"Total Unique Writers: {len(writers)}")
    
    # Parse validation report for samples
    val_report_path = os.path.join(canonical_root, "validation_report.json")
    if os.path.exists(val_report_path):
        with open(val_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
            
        online_samples = sum(s.get("total_samples", 0) for s in report.get("online", {}).values())
        offline_samples = sum(s.get("total_samples", 0) for s in report.get("offline", {}).values())
        
        print(f"Total Canonical Samples (Online): {online_samples}")
        print(f"Total Canonical Samples (Offline): {offline_samples}")
    else:
        print("Dataset sample counts not found. Run 'python main.py validate-dataset' first.")
        
    print(f"Python: {env['python_version'].split(' ')[0]}")
    print(f"Torch Version: {env['torch_version']}")
    print(f"CUDA: {'Available' if env['cuda_available'] else 'None'} ({env['cuda_version']})")
    print(f"GPU: {env.get('gpu_name', 'None')}")
    print(f"CPU Cores: {env['cpu_count']}")
    print(f"RAM (GB): {env['ram_gb']}")
    print(f"Config Hash: {hash(str(cfg))}")

def cmd_train(args):
    # Dynamic imports to speed up CLI
    from src.training.train import train_model
    import time
    
    print("Loading config...")
    cfg = load_colab_config()
    t_cfg = cfg.get("training", {})
    m_cfg = cfg.get("model", {})
    
    # Generate Experiment ID
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    exp_num = len(existing_runs) + 1
    arch_name = m_cfg.get("architecture", "baseline_lstm")
    exp_id = f"{date_str}_{exp_num:03d}_{arch_name}"
    
    print(f"Starting Training: {exp_id}")
    
    # Run training using the refactored train_model (we will update it to accept exp_id and config)
    # Passing epochs from config, override with CLI if needed
    epochs = t_cfg.get("epochs", 100)
    batch_size = t_cfg.get("batch_size", 32)
    
    train_model(epochs=epochs, exp_id=exp_id, batch_size=batch_size, resume_checkpoint=None)

def cmd_resume(args):
    from src.training.train import train_model
    cfg = load_colab_config()
    t_cfg = cfg.get("training", {})
    
    exp_id = args.exp_id
    if not exp_id:
        print("ERROR: --exp_id is required for resume.")
        sys.exit(1)
        
    ckpt_path = os.path.join("experiments", exp_id, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        # Fallback to the old checkpoint format if latest doesn't exist
        print(f"ERROR: No latest.pt found for {exp_id} at {ckpt_path}.")
        sys.exit(1)
        
    train_model(epochs=t_cfg.get("epochs", 100), exp_id=exp_id, batch_size=t_cfg.get("batch_size", 32), resume_checkpoint=ckpt_path)

def cmd_evaluate(args):
    from src.evaluation.evaluate import run_evaluation
    mode = "ocr" if args.command == "evaluate-ocr" else "auto"
    run_evaluation(args.exp_id, args.split, mode=mode)

def cmd_preview(args):
    from src.evaluation.preview import run_preview
    mode = "ocr" if args.command == "preview-ocr" else "auto"
    run_preview(args.exp_id, args.num_samples, mode=mode)

def cmd_benchmark(args):
    print("Benchmark not yet implemented.")

def cmd_validate_dataset(args):
    from scripts.validate_dataset import main as validate_main
    sys.argv = ["scripts/validate_dataset.py", "--data-dir", "data/canonical"]
    validate_main()

def cmd_ingest_offline(args):
    from scripts.build_offline_dataset import main
    main()

def cmd_train_ocr(args):
    from src.training.train_ocr import train_ocr_model
    from src.utils.config import load_yaml
    import time
    
    print("Loading OCR config...")
    cfg_path = os.path.join("configs", "ocr.yaml")
    if os.path.exists(cfg_path):
        cfg = load_yaml(cfg_path)
    else:
        cfg = {}
    
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ocr_exp_dir = os.path.join("experiments", "OCR")
    os.makedirs(ocr_exp_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(ocr_exp_dir) if os.path.isdir(os.path.join(ocr_exp_dir, d))]
    exp_num = len(existing_runs) + 1
    exp_id = f"{date_str}_{exp_num:03d}_ocr"
    augment = getattr(args, "augment", False)
    train_ocr_model(config=cfg, exp_id=exp_id, augment=augment)

def cmd_recognize(args):
    from src.inference.recognize import run_recognition
    
    if not args.image:
        print("Please provide --image path.")
        sys.exit(1)
        
    if not args.exp_dir:
        print("Please provide --exp_dir path to the trained OCR experiment.")
        sys.exit(1)
        
    run_recognition(args.image, args.exp_dir)

def cmd_validate_model(args):
    print("Model validation not yet implemented.")

def main():
    parser = argparse.ArgumentParser(description="Hindi Handwriting AI Platform CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    parser_setup = subparsers.add_parser("setup", help="Verify environment")
    parser_info = subparsers.add_parser("info", help="Print repository info")
    
    parser_train = subparsers.add_parser("train", help="Train trajectory generator")
    
    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained trajectory model")
    parser_eval.add_argument("--exp_id", type=str, required=True, help="Experiment ID to evaluate")
    parser_eval.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate on (default: validation)")

    # evaluate-ocr
    parser_eval_ocr = subparsers.add_parser("evaluate-ocr", help="Evaluate a trained OCR model")
    parser_eval_ocr.add_argument("--exp_id", type=str, required=True, help="Experiment ID to evaluate")
    parser_eval_ocr.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate on (default: validation)")

    # preview
    parser_preview = subparsers.add_parser("preview", help="Generate previews from a trained trajectory model")
    parser_preview.add_argument("--exp_id", type=str, required=True, help="Experiment ID to preview")
    parser_preview.add_argument("--num_samples", type=int, default=5, help="Number of samples to preview (default: 5)")

    # preview-ocr
    parser_preview_ocr = subparsers.add_parser("preview-ocr", help="Generate previews from a trained OCR model")
    parser_preview_ocr.add_argument("--exp_id", type=str, required=True, help="Experiment ID to preview")
    parser_preview_ocr.add_argument("--num_samples", type=int, default=5, help="Number of samples to preview (default: 5)")

    parser_resume = subparsers.add_parser("resume", help="Resume training")
    parser_resume.add_argument("--exp_id", type=str, required=True)
    
    parser_train_ocr = subparsers.add_parser("train-ocr", help="Train OCR CRNN model")
    parser_train_ocr.add_argument("--augment", action="store_true", help="Enable data augmentation for training")
    
    parser_ingest = subparsers.add_parser("ingest-offline", help="Ingest offline images")
    
    parser_recognize = subparsers.add_parser("recognize", help="Recognize text from image")
    parser_recognize.add_argument("--image", type=str, required=True)
    parser_recognize.add_argument("--exp_dir", type=str, required=True, help="Path to OCR experiment directory")
    
    parser_val_data = subparsers.add_parser("validate-dataset", help="Validate canonical data")
    
    parser_vmodel = subparsers.add_parser("validate-model", help="Validate model checkpoints and inference")
    parser_vmodel.set_defaults(func=cmd_validate_model)
    
    args = parser.parse_args()
    
    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command in ["evaluate", "evaluate-ocr"]:
        cmd_evaluate(args)
    elif args.command in ["preview", "preview-ocr"]:
        cmd_preview(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "train-ocr":
        cmd_train_ocr(args)
    elif args.command == "ingest-offline":
        cmd_ingest_offline(args)
    elif args.command == "recognize":
        cmd_recognize(args)
    elif args.command == "validate-dataset":
        cmd_validate_dataset(args)
    elif args.command == "validate-model":
        cmd_validate_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
