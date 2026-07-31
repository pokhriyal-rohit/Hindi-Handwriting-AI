from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession

def test_session_initialization():
    config = InferenceConfig(device="cpu", random_seed=99)
    session = InferenceSession(config)
    
    assert session.config.device == "cpu"
    assert session.config.random_seed == 99
    assert session.renderer is not None
    
    session.warmup()
    session.shutdown()
    assert session.predictor is None
    assert session.renderer is None
