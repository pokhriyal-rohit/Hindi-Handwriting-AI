import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.renderer.exceptions import CacheError

class RendererCache:
    """
    Multi-level cache for the rendering engine.
    Level 1: Trajectory cache (Not implemented yet - for pre-layout)
    Level 2: SVG cache
    Level 3: Raster cache (PNG/PDF)
    Level 4: Animation cache (MP4/GIF)
    """
    def __init__(self, cache_dir: str = ".cache/renderer"):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise CacheError(f"Failed to create cache directory {self.cache_dir}: {e}")
            
    def _compute_hash(self, sample: TrajectorySample, config: RenderingConfig) -> str:
        # Hash base geometry + text + styling
        data_str = f"{sample.text}_{sample.writer_id}_"
        for stroke in sample.strokes:
            for pt in stroke.points:
                data_str += f"{pt.x},{pt.y},{pt.pen_state},{pt.pressure};"
                
        # Hash versioning and configuration
        config_str = (
            f"rend_{config.renderer_version}_lay_{config.layout_model}_{config.layout_version}_"
            f"exp_{config.exporter_version}_"
            f"smooth_{config.smoothing}_press_{config.pressure_model}_ink_{config.ink_model}_"
            f"sw_{config.base_stroke_width}_c_{config.export.stroke_color}_bg_{config.export.background_color}_"
            f"dim_{config.export.canvas_width}x{config.export.canvas_height}_"
            f"plugins_{json.dumps(config.plugin_versions)}"
        )
        
        hasher = hashlib.md5()
        hasher.update(data_str.encode("utf-8"))
        hasher.update(config_str.encode("utf-8"))
        
        return hasher.hexdigest()
        
    def get_cached_path(self, sample: TrajectorySample, config: RenderingConfig, ext: str) -> Optional[Path]:
        file_hash = self._compute_hash(sample, config)
        path = self.cache_dir / f"{file_hash}.{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path
        return None
        
    def get_cache_write_path(self, sample: TrajectorySample, config: RenderingConfig, ext: str) -> Path:
        file_hash = self._compute_hash(sample, config)
        return self.cache_dir / f"{file_hash}.{ext}"
        
    def serve_from_cache(self, sample: TrajectorySample, config: RenderingConfig, ext: str, destination: str) -> bool:
        """If cache hit, copies to destination and returns True."""
        try:
            cached_path = self.get_cached_path(sample, config, ext)
            if cached_path:
                shutil.copy2(cached_path, destination)
                return True
            return False
        except Exception as e:
            raise CacheError(f"Cache retrieval failed: {e}")
            
    def save_to_cache(self, sample: TrajectorySample, config: RenderingConfig, ext: str, source: str) -> None:
        """Saves generated output to cache."""
        try:
            cache_path = self.get_cache_write_path(sample, config, ext)
            shutil.copy2(source, cache_path)
        except Exception as e:
            raise CacheError(f"Cache write failed: {e}")
