import os
import yaml
from typing import Dict, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file and returns it as a dictionary."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_colab_config(config_dir: str = "configs") -> Dict[str, Any]:
    """
    Loads and merges independent configuration files.
    """
    dataset_cfg = load_yaml(os.path.join(config_dir, "dataset.yaml"))
    model_cfg = load_yaml(os.path.join(config_dir, "model.yaml"))
    training_cfg = load_yaml(os.path.join(config_dir, "training.yaml"))
    evaluation_cfg = load_yaml(os.path.join(config_dir, "evaluation.yaml"))
    
    cfg = {
        "dataset": dataset_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "evaluation": evaluation_cfg
    }
    return cfg
