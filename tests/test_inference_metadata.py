from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

def test_inference_metadata():
    config = InferenceConfig(model_name="deterministic_hindi")
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    result = pipeline.generate("METADATA")
    
    runtime = result.metadata.get("runtime")
    assert runtime is not None
    assert "git_commit" in runtime
    assert "timestamp" in runtime
    assert "python_version" in runtime
    assert "os" in runtime
    
    session.shutdown()
