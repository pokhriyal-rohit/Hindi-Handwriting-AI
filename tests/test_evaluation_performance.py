from src.registry import Registry
from src.evaluation.metrics.performance import SVGGenerationTimeMetric, InferenceLatencyMetric, SystemMemoryUsageMetric
from tests.test_renderer import dummy_trajectory

def test_svg_generation_metric():
    metric = Registry.get_metric("render_svg")()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    res = metric.evaluate(traj1, traj2)
    assert "svg_time" in res
    assert res["svg_time"] >= 0.0
    
def test_inference_latency_metric():
    metric = Registry.get_metric("inference_latency")()
    traj1 = dummy_trajectory()
    traj1.extensions["inference_latency_ms"] = 45.5
    traj2 = dummy_trajectory()
    
    res = metric.evaluate(traj1, traj2)
    assert res["inference_latency_ms"] == 45.5
    
    summary = metric.summarize([res])
    assert "inference_samples_per_second" in summary
    assert summary["inference_samples_per_second"] > 0
    
def test_system_memory_metric():
    metric = Registry.get_metric("system_memory")()
    traj1 = dummy_trajectory()
    traj1.extensions["gpu_memory_mb"] = 1200.5
    traj1.extensions["cpu_memory_mb"] = 500.0
    traj2 = dummy_trajectory()
    
    res = metric.evaluate(traj1, traj2)
    assert res["gpu_memory_mb"] == 1200.5
    assert res["cpu_memory_mb"] == 500.0
