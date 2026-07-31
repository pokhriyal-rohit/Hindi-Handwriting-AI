import pytest
import numpy as np
from src.datasets.structures import Point, Stroke, TrajectorySample, DatasetMetadata
from src.datasets.continuous import ModularCoordinateRepresentation

def test_continuous_delta_representation():
    # Create mock data
    pt1 = Point(x=0.0, y=0.0, pen_state=1)
    pt2 = Point(x=10.0, y=0.0, pen_state=0)
    pt3 = Point(x=10.0, y=10.0, pen_state=1)
    pt4 = Point(x=20.0, y=10.0, pen_state=0) # Lift
    
    stroke1 = Stroke(stroke_id=0, points=[pt1, pt2])
    stroke2 = Stroke(stroke_id=1, points=[pt3, pt4])
    
    metadata = DatasetMetadata(
        dataset_name="test",
        dataset_version="1.0",
        is_synthetic=False
    )
    
    traj = TrajectorySample(
        sample_id="test_1",
        writer_id="w1",
        script="devanagari",
        language="hi",
        text="test",
        strokes=[stroke1, stroke2],
        metadata=metadata
    )
    
    # Test un-normalized
    rep = ModularCoordinateRepresentation(scaler_name="identity")
    encoded = rep.encode(traj)
    
    assert encoded.shape == (4, 3)
    # First point delta is [0, 0]
    assert np.allclose(encoded[0], [0.0, 0.0, 1.0])
    # pt2 - pt1 = [10, 0] with pen=0
    assert np.allclose(encoded[1], [10.0, 0.0, 0.0])
    # pt3 - pt2 = [0, 10]
    assert np.allclose(encoded[2], [0.0, 10.0, 1.0])
    # pt4 - pt3 = [10, 0]
    assert np.allclose(encoded[3], [10.0, 0.0, 0.0])
    
    # Test decoding back
    decoded_traj = rep.decode(encoded, start_pos=(0.0, 0.0))
    assert decoded_traj is not None
    assert len(decoded_traj.strokes) == 2
    assert decoded_traj.strokes[0].points[1].x == 10.0
    
    # Test normalization fit
    rep_norm = ModularCoordinateRepresentation(scaler_name="standard")
    rep_norm.fit([traj])
    
    stats = rep_norm.statistics()
    assert stats["scaler"] == "StandardScaler"
    # dx = [0, 10, 0, 10] => mean dx = 20/4 = 5.0
    assert np.isclose(stats["mean"][0], 5.0)
    # dy = [0, 0, 10, 0] => mean dy = 10/4 = 2.5
    assert np.isclose(stats["mean"][1], 2.5)

def test_robust_edge_cases():
    rep = ModularCoordinateRepresentation(scaler_name="standard")
    
    # 1. Empty Trajectory
    empty_metadata = DatasetMetadata(dataset_name="test", dataset_version="1.0", is_synthetic=False)
    empty_traj = TrajectorySample(sample_id="empty", writer_id="w1", script="devanagari", language="hi", text="", strokes=[], metadata=empty_metadata)
    assert rep.encode(empty_traj).size == 0
    assert rep.decode(np.array([])) is None
    
    # 2. Empty Strokes & Single Point Strokes
    pt1 = Point(x=0.0, y=0.0, pen_state=0)
    stroke_single = Stroke(stroke_id=0, points=[pt1])
    stroke_empty = Stroke(stroke_id=1, points=[])
    single_traj = TrajectorySample(sample_id="single", writer_id="w1", script="devanagari", language="hi", text="", strokes=[stroke_single, stroke_empty], metadata=empty_metadata)
    rep.fit([single_traj])
    encoded_single = rep.encode(single_traj)
    assert encoded_single.shape == (1, 3) # only 1 valid point
    
    # 3. Extreme Coordinates & Thousands of pen lifts
    pts = []
    for i in range(2000):
        # alternate huge positive and negative coordinates, and alternate pen lifts
        pen = i % 2
        pts.append(Point(x=1e6 if pen else -1e6, y=-1e6 if pen else 1e6, pen_state=pen))
    
    # Pack them into a single massive stroke (even if technically pen lifts mean stroke ends, we just test encoding robustness)
    massive_stroke = Stroke(stroke_id=0, points=pts)
    massive_traj = TrajectorySample(sample_id="massive", writer_id="w1", script="devanagari", language="hi", text="", strokes=[massive_stroke], metadata=empty_metadata)
    
    # Should not crash during fit or encode
    rep.fit([massive_traj])
    encoded_massive = rep.encode(massive_traj)
    assert encoded_massive.shape == (2000, 3)
    
    decoded_massive = rep.decode(encoded_massive)
    assert len(decoded_massive.strokes) > 0 # successfully parsed back into multiple strokes due to pen=0

