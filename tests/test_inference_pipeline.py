from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

def test_inference_pipeline():
    config = InferenceConfig(model_name="dummy_predictor")
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    result = pipeline.generate("AB")
    
    # Assertions
    assert result.input_text == "AB"
    assert result.normalized_text == "AB"
    assert result.metadata["raw_tokens"] == [65, 66] # ASCII for A, B
    
    # Dummy Predictor returns 2 points per token, pen up at very end. 
    # Tokens = 2 -> 4 raw outputs.
    assert len(result.metadata["raw_outputs"]) == 4
    
    # Reconstructed trajectory
    trajectory = result.trajectory
    assert trajectory.text == "AB"
    
    # 4 points, the last one has pen_state=0, so it will flush 1 stroke of 4 points
    assert len(trajectory.strokes) == 1
    assert len(trajectory.strokes[0].points) == 4
    
    # Check absolute coordinates (dx=1, dy=0)
    assert trajectory.strokes[0].points[0].x == 1.0
    assert trajectory.strokes[0].points[3].x == 4.0
    
    session.shutdown()
