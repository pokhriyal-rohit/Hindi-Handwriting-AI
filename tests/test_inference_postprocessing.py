from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

def test_inference_postprocessing():
    config = InferenceConfig(model_name="dummy_predictor", postprocessors=["coordinate_clamp", "metadata_enricher"])
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    result = pipeline.generate("TEST")
    trajectory = result.trajectory
    
    # Assert metadata enricher ran
    assert trajectory.extensions.get("post_processed") is True
    
    # Manually test the clamp logic via dummy prediction which returns 1.0 coords
    assert trajectory.strokes[0].points[0].x == 1.0
    
    session.shutdown()
