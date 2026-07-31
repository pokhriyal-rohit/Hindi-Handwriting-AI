import platform
import sys
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"

def get_runtime_metadata() -> Dict[str, Any]:
    """
    Captures the exact environmental fingerprint of the pipeline execution.
    Crucial for reproducibility and debugging.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        
        # System Info
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split(" ")[0],
        "cpu": platform.processor(),
        
        # GPU could be populated dynamically via torch.cuda.get_device_name(0) if available
        "gpu": "Unknown (PyTorch not loaded)",
        
        # Versioning (would be tied to setup.py or __version__ variables)
        "inference_version": "1.0.0",
        "dataset_version": "1.0.0",
        "renderer_version": "1.0.0",
        "evaluation_version": "1.0.0",
    }
