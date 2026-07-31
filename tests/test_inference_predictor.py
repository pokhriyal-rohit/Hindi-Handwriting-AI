from src.registry import Registry
from src.inference.config import InferenceConfig
from src.inference.predictor.base import DummyPredictor

def test_dummy_predictor():
    config = InferenceConfig()
    predictor_cls = Registry.get_model("dummy_predictor")
    predictor = predictor_cls(config)
    
    # Must fail if not loaded
    try:
        predictor.predict([1, 2])
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass
        
    predictor.load_model()
    predictor.warmup()
    
    # Test predict
    out = predictor.predict([1, 2, 3])
    assert len(out) == 6
    assert out[0] == [1.0, 0.0, 1.0]
    
    # Test shutdown
    predictor.shutdown()
    try:
        predictor.predict([1])
        assert False, "Should raise RuntimeError after shutdown"
    except RuntimeError:
        pass

def test_deterministic_predictor():
    import src.inference.predictor.deterministic
    config = InferenceConfig()
    predictor_cls = Registry.get_model("deterministic_hindi")
    predictor = predictor_cls(config)
    predictor.load_model()
    
    # Text: "नमस्ते" -> tokens via ord
    tokens = [ord(c) for c in "नमस्ते"]
    out1 = predictor.predict(tokens)
    out2 = predictor.predict(tokens)
    
    # Must be perfectly reproducible
    assert out1 == out2
    assert len(out1) > 0
