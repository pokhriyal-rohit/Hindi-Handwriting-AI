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
    data_dir = cfg["dataset"].get("online", {}).get("train", "data/canonical/online/train")
    if not os.path.exists(data_dir):
        print(f"ERROR: Canonical dataset not found at {data_dir}.")
        print("Run scripts/build_canonical_dataset.py first.")
        sys.exit(1)
        
    hash_val = compute_dataset_hash(data_dir)
    print(f"Dataset Hash (SHA-256): {hash_val[:16]}...")
    print("Setup verified successfully.")

def cmd_info(args):
    cfg = load_colab_config()
    data_dir = cfg["dataset"].get("online", {}).get("train", "data/canonical/online/train")
    env = capture_environment(cfg, data_dir, time.time() if 'time' in globals() else 0)
    
    print("=== Repository Info ===")
    print(f"Git Commit: {env['git_commit']}")
    print(f"Dataset Hash: {env['dataset_hash']}")
    
    try:
        manifest_path = os.path.join(data_dir, "..", "..", "manifests", "statistics.json")
        with open(manifest_path, "r") as f:
            stats = json.load(f)
            # The writers count is in writers.yaml, so let's load that
            writers_path = os.path.join(data_dir, "..", "..", "manifests", "writers.yaml")
            import yaml
            with open(writers_path, "r") as wf:
                writers = yaml.safe_load(wf)
                print(f"Writers: {len(writers)}")
                
            print(f"Total Valid Samples: {stats.get('valid_samples', 'unknown')}")
    except Exception:
        print("Dataset statistics not found.")
        
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
        
    ckpt_path = os.path.join("runs", exp_id, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        # Fallback to the old checkpoint format if latest doesn't exist
        print(f"ERROR: No latest.pt found for {exp_id} at {ckpt_path}.")
        sys.exit(1)
        
    train_model(epochs=t_cfg.get("epochs", 100), exp_id=exp_id, batch_size=t_cfg.get("batch_size", 32), resume_checkpoint=ckpt_path)

def cmd_evaluate(args):
    print("Evaluation not yet implemented.")

def cmd_preview(args):
    print("Preview not yet implemented.")

def cmd_benchmark(args):
    print("Benchmark not yet implemented.")

def cmd_validate_dataset(args):
    from scripts.validate_dataset import validate_directory
    cfg = load_colab_config()
    data_dir = cfg["dataset"].get("online", {}).get("train", "data/canonical/online/train")
    if os.path.exists(data_dir):
        validate_directory(data_dir, stop_on_first_error=False)
    else:
        print(f"Directory {data_dir} not found.")

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
    
    train_ocr_model(config=cfg, exp_id=exp_id)

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
    
    parser_resume = subparsers.add_parser("resume", help="Resume training")
    parser_resume.add_argument("--exp_id", type=str, required=True)
    
    parser_train_ocr = subparsers.add_parser("train-ocr", help="Train OCR CRNN model")
    
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
