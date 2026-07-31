import os
import tempfile
from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

def _get_pipeline() -> InferencePipeline:
    config = InferenceConfig(model_name="deterministic_hindi", enable_cache=True)
    session = InferenceSession(config)
    return InferencePipeline(session)

def test_e2e_single_char():
    pipeline = _get_pipeline()
    res = pipeline.generate("क")
    assert res.input_text == "क"
    assert len(res.trajectory.strokes) > 0
    pipeline.session.shutdown()

def test_e2e_word():
    pipeline = _get_pipeline()
    res = pipeline.generate("नमस्ते")
    assert res.input_text == "नमस्ते"
    assert len(res.trajectory.strokes) > 0
    pipeline.session.shutdown()

def test_e2e_empty_input():
    pipeline = _get_pipeline()
    # Assuming empty string returns an empty trajectory gracefully or raises error.
    # Our pipeline will just ord("") which is []
    res = pipeline.generate("")
    assert res.input_text == ""
    assert len(res.trajectory.strokes) == 0
    pipeline.session.shutdown()

def test_e2e_cache_hit():
    pipeline = _get_pipeline()
    res1 = pipeline.generate("CACHE_HIT_TEST")
    assert res1.cache_statistics.get("trajectory_hit") is False
    
    res2 = pipeline.generate("CACHE_HIT_TEST")
    assert res2.cache_statistics.get("trajectory_hit") is True
    
    pipeline.session.shutdown()

def run_integration_scenarios():
    test_e2e_single_char()
    test_e2e_word()
    test_e2e_empty_input()
    test_e2e_cache_hit()

if __name__ == "__main__":
    run_integration_scenarios()
