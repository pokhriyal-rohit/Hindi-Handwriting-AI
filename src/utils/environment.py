import os
import sys
import platform
import hashlib
import json
import time
import subprocess
from datetime import datetime, timezone
import torch
import psutil

def hash_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dataset_hash(dataset_dir: str) -> str:
    """
    Computes a comprehensive SHA-256 hash of the canonical dataset.
    Hashes dataset_manifest.yaml, statistics.json, splits.yaml, and every sample.
    """
    hasher = hashlib.sha256()
    
    # Files that define the dataset structure and stats
    manifests_dir = os.path.join(dataset_dir, "..", "manifests")
    critical_files = [
        os.path.join(manifests_dir, "dataset_manifest.yaml"),
        os.path.join(manifests_dir, "statistics.json"),
        os.path.join(manifests_dir, "splits.yaml"),
        os.path.join(manifests_dir, "writers.yaml")
    ]
    
    for cf in critical_files:
        if os.path.exists(cf):
            hasher.update(f"{os.path.basename(cf)}:".encode('utf-8'))
            hasher.update(hash_file(cf).encode('utf-8'))
            
    # Hash every sample deterministically
    all_json_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json'):
                all_json_files.append(os.path.join(root, file))
                
    all_json_files.sort()
    for jf in all_json_files:
        rel_path = os.path.relpath(jf, dataset_dir)
        hasher.update(f"{rel_path}:".encode('utf-8'))
        hasher.update(hash_file(jf).encode('utf-8'))
        
    return hasher.hexdigest()

def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"

def capture_environment(config_dict: dict, dataset_dir: str, start_time: float) -> dict:
    dataset_hash = compute_dataset_hash(dataset_dir)
    
    env_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "git_commit": get_git_commit(),
        "dataset_hash": dataset_hash,
        "hostname": platform.node(),
        "os": platform.platform(),
        "training_start_time": start_time,
        "config": config_dict
    }
    return env_info

def save_environment(env_info: dict, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    with open(os.path.join(run_dir, "dataset_hash.txt"), "w", encoding="utf-8") as f:
        f.write(env_info["dataset_hash"])
