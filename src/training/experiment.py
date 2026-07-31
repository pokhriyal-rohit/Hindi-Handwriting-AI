import os
import json
import yaml
from datetime import datetime

class ExperimentTracker:
    def __init__(self, base_dir: str = "experiments", exp_id: str = None):
        if exp_id is None:
            # Auto-increment exp ID
            os.makedirs(base_dir, exist_ok=True)
            existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
            exp_num = len(existing) + 1
            exp_id = f"exp_{exp_num:03d}"
            
        self.exp_dir = os.path.join(base_dir, exp_id)
        
        self.dirs = {
            "predictions": os.path.join(self.exp_dir, "predictions"),
            "overlays": os.path.join(self.exp_dir, "overlays"),
            "logs": os.path.join(self.exp_dir, "logs")
        }
        
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        self.metrics_file = os.path.join(self.exp_dir, "metrics.json")
        self.metrics = []
        
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                self.metrics = json.load(f)

    def save_config(self, config: dict):
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    def log_epoch(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        entry.update(metrics)
        self.metrics.append(entry)
        
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_path(self, folder: str, filename: str) -> str:
        """Helper to get path for predictions/overlays"""
        return os.path.join(self.dirs[folder], filename)
        
    def get_checkpoint_path(self, epoch: int = None) -> str:
        if epoch is None:
            return os.path.join(self.exp_dir, "checkpoint.pt")
        return os.path.join(self.exp_dir, f"checkpoint_{epoch}.pt")
