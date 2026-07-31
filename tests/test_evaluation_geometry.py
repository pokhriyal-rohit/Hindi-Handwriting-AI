from src.registry import Registry
from src.evaluation.metrics.geometry import PathLengthDifferenceMetric, BoundingBoxDifferenceMetric, SmoothnessScoreMetric
from tests.test_renderer import dummy_trajectory
import copy

def test_path_length_metric():
    metric = Registry.get_metric("path_length")()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    # Same
    res1 = metric.evaluate(traj1, traj2)
    assert res1["length_difference"] == 0.0
    
    # Modify length
    traj2.strokes[0].points[-1].x += 10.0
    res2 = metric.evaluate(traj1, traj2)
    assert res2["length_difference"] > 0.0
    
def test_bounding_box_metric():
    metric = Registry.get_metric("bounding_box")()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    res1 = metric.evaluate(traj1, traj2)
    assert res1["width_difference"] == 0.0
    assert res1["height_difference"] == 0.0
    
    # Shift bounding box size
    for pt in traj2.strokes[0].points:
        pt.x *= 2
        
    res2 = metric.evaluate(traj1, traj2)
    assert res2["width_difference"] > 0.0

def test_smoothness_metric():
    metric = Registry.get_metric("smoothness")()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    try:
        res1 = metric.evaluate(traj1, traj2)
        assert res1["smoothness_difference"] == 0.0
    except ImportError as e:
        print(f"Skipping smoothness test due to missing packages: {e}")
