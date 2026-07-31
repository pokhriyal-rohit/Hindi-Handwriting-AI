import os
import tempfile
from src.datasets.structures import TrajectorySample, Stroke, Point, DatasetMetadata
from src.renderer.config import RenderingConfig
from src.renderer.pipeline import RenderingEngine
from src.renderer.exceptions import InvalidTrajectoryError
def dummy_trajectory():
    return TrajectorySample(
        sample_id="test_001",
        writer_id="w1",
        script="devanagari",
        language="hi",
        text="test",
        strokes=[
            Stroke(stroke_id=0, points=[
                Point(x=10.0, y=10.0, pen_state=1),
                Point(x=20.0, y=20.0, pen_state=1),
                Point(x=30.0, y=10.0, pen_state=1)
            ]),
            Stroke(stroke_id=1, points=[
                Point(x=50.0, y=50.0, pen_state=1),
                Point(x=60.0, y=60.0, pen_state=1)
            ])
        ],
        metadata=DatasetMetadata(
            dataset_name="test",
            dataset_version="1.0",
            is_synthetic=True
        )
    )

def test_rendering_engine_initialization():
    config = RenderingConfig()
    engine = RenderingEngine(config)
    assert engine.config.smoothing == "moving_average"

def test_svg_export(dummy_trajectory):
    config = RenderingConfig(smoothing=None, pressure_model="constant", ink_model="constant")
    engine = RenderingEngine(config)
    
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        tmp_path = f.name
        
    try:
        engine.render(dummy_trajectory, tmp_path, format="svg")
        assert os.path.exists(tmp_path)
        with open(tmp_path, "r") as svg_file:
            content = svg_file.read()
            assert "<svg" in content
            assert "<path" in content
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_layout_scaling(dummy_trajectory):
    config = RenderingConfig(smoothing=None)
    config.export.canvas_width = 1000
    config.export.canvas_height = 1000
    config.export.margin = 100
    
    engine = RenderingEngine(config)
    # _apply_layout is internal, we can test it directly
    layout_sample = engine._apply_layout(dummy_trajectory)
    
    # Check bounding box is within margins
    pts = layout_sample.to_array()
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    
    assert min(xs) >= 100
    assert min(ys) >= 100
    assert max(xs) <= 900
    assert max(ys) <= 900

def test_invalid_trajectory_exception():
    config = RenderingConfig()
    engine = RenderingEngine(config)
    
    # Passing an empty list instead of a TrajectorySample should raise InvalidTrajectoryError
    try:
        engine.render([], "output.svg")
        assert False, "Should have raised InvalidTrajectoryError"
    except InvalidTrajectoryError:
        pass
        
    # Passing a TrajectorySample with no strokes should also raise
    empty_sample = dummy_trajectory()
    empty_sample.strokes = []
    
    try:
        engine.render(empty_sample, "output.svg")
        assert False, "Should have raised InvalidTrajectoryError"
    except InvalidTrajectoryError:
        pass
