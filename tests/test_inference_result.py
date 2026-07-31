from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

def test_inference_result_serialization():
    config = InferenceConfig(model_name="dummy_predictor")
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    result = pipeline.generate("SERIALIZE")
    
    # Test JSON dump
    data_dict = result.dict()
    assert data_dict["input_text"] == "SERIALIZE"
    assert "trajectory" in data_dict
    assert "timing" in data_dict
    assert "total_ms" in data_dict["timing"]
    
    session.shutdown()
