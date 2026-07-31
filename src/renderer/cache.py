import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig

class RendererCache:
    """
    Caches renderer outputs to disk to prevent redundant rasterization/generation.
    Hashes the TrajectorySample content and the RenderingConfig.
    """
    def __init__(self, cache_dir: str = ".cache/renderer"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_hash(self, sample: TrajectorySample, config: RenderingConfig) -> str:
        # Hash text, writer, and all stroke coordinates
        data_str = f"{sample.text}_{sample.writer_id}_"
        for stroke in sample.strokes:
            for pt in stroke.points:
                data_str += f"{pt.x},{pt.y},{pt.pen_state},{pt.pressure};"
                
        # Hash config
        config_str = config.model_dump_json()
        
        hasher = hashlib.md5()
        hasher.update(data_str.encode("utf-8"))
        hasher.update(config_str.encode("utf-8"))
        
        return hasher.hexdigest()
        
    def get_cached_path(self, sample: TrajectorySample, config: RenderingConfig, ext: str) -> Optional[Path]:
        file_hash = self._compute_hash(sample, config)
        path = self.cache_dir / f"{file_hash}.{ext}"
        if path.exists():
            return path
        return None
        
    def get_cache_write_path(self, sample: TrajectorySample, config: RenderingConfig, ext: str) -> Path:
        file_hash = self._compute_hash(sample, config)
        return self.cache_dir / f"{file_hash}.{ext}"
