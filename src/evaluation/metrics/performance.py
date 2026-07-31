import time
import os
import tempfile
import psutil
from typing import Dict, Any, List
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.evaluation.metrics.base import BaseMetric
from src.renderer.config import RenderingConfig
from src.renderer.pipeline import RenderingEngine

def _measure_rendering_format(trajectory: TrajectorySample, format: str) -> Dict[str, Any]:
    """Helper to benchmark the RenderingEngine independently for a specific format."""
    config = RenderingConfig()
    engine = RenderingEngine(config)
    
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, f"out.{format}")
        try:
            engine.render(trajectory, out_path, format=format)
        except Exception:
            return {f"{format}_time": float('inf'), f"{format}_memory_mb": 0.0}
            
    t_end = time.perf_counter()
    
    # We can't perfectly isolate peak memory per render without OS hooks, 
    # but we can record current process memory for the proxy
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    
    return {
        f"{format}_time": float(t_end - t_start),
        f"{format}_memory_mb": float(mem_mb)
    }

@Registry.register_metric("render_svg")
class SVGGenerationTimeMetric(BaseMetric):
    @classmethod
    def name(cls) -> str: return "render_svg"
    @classmethod
    def version(cls) -> str: return "1.0.0"
    @classmethod
    def description(cls) -> str: return "SVG rendering time and memory usage."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        return _measure_rendering_format(prediction, "svg")

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as np
        times = [r["svg_time"] for r in results if r["svg_time"] != float('inf')]
        if not times: return {}
        return {"svg_time_mean": float(np.mean(times))}

@Registry.register_metric("inference_latency")
class InferenceLatencyMetric(BaseMetric):
    @classmethod
    def name(cls) -> str: return "inference_latency"
    @classmethod
    def version(cls) -> str: return "1.0.0"
    @classmethod
    def description(cls) -> str: return "Model inference latency extracted from prediction metadata."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        lat = prediction.extensions.get("inference_latency_ms", 0.0)
        return {"inference_latency_ms": float(lat)}

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as np
        lats = [r["inference_latency_ms"] for r in results if r["inference_latency_ms"] > 0]
        if not lats: return {}
        return {
            "inference_latency_mean": float(np.mean(lats)),
            "inference_samples_per_second": 1000.0 / float(np.mean(lats))
        }

@Registry.register_metric("system_memory")
class SystemMemoryUsageMetric(BaseMetric):
    @classmethod
    def name(cls) -> str: return "system_memory"
    @classmethod
    def version(cls) -> str: return "1.0.0"
    @classmethod
    def description(cls) -> str: return "GPU and CPU memory usage tracked via metadata."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        return {
            "gpu_memory_mb": float(prediction.extensions.get("gpu_memory_mb", 0.0)),
            "cpu_memory_mb": float(prediction.extensions.get("cpu_memory_mb", 0.0)),
            "checkpoint_size_mb": float(prediction.extensions.get("checkpoint_size_mb", 0.0))
        }

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as np
        cpu = [r["cpu_memory_mb"] for r in results]
        gpu = [r["gpu_memory_mb"] for r in results]
        if not cpu: return {}
        return {
            "cpu_memory_mean": float(np.mean(cpu)),
            "gpu_memory_mean": float(np.mean(gpu)),
            "checkpoint_size_mb": float(results[0].get("checkpoint_size_mb", 0.0))
        }
