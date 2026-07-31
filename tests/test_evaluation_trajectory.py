from src.registry import Registry
from src.evaluation.metrics.trajectory import DTWMetric, FrechetMetric, StrokeCountDifferenceMetric, EndpointErrorMetric
from tests.test_renderer import dummy_trajectory

def test_stroke_count_metric():
    metric = Registry.get_metric("stroke_count")()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    # Same trajectories
    res = metric.evaluate(traj1, traj2)
    assert res["stroke_difference"] == 0
    
    # Modify traj2
    traj2.strokes.pop()
    res2 = metric.evaluate(traj1, traj2)
    assert res2["stroke_difference"] == 1
    
    summary = metric.summarize([res, res2])
    assert summary["stroke_difference_mean"] == 0.5
    assert summary["stroke_difference_max"] == 1.0

def test_endpoint_error_metric():
    metric = Registry.get_metric("endpoint_error")()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    res = metric.evaluate(traj1, traj2)
    assert res["endpoint_error"] == 0.0
    
    # Move endpoint
    traj2.strokes[-1].points[-1].x += 3.0
    traj2.strokes[-1].points[-1].y += 4.0
    
    res2 = metric.evaluate(traj1, traj2)
    assert res2["endpoint_error"] == 5.0 # 3-4-5 triangle

def test_dtw_frechet_metrics():
    dtw = Registry.get_metric("dtw")()
    frechet = Registry.get_metric("frechet")()
    
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    try:
        dtw_res = dtw.evaluate(traj1, traj2)
        assert dtw_res["dtw_distance"] == 0.0
        
        f_res = frechet.evaluate(traj1, traj2)
        assert f_res["frechet_distance"] == 0.0
    except ImportError as e:
        print(f"Skipping scientific metric evaluation test due to missing packages: {e}")
