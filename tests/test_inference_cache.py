from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

def test_inference_cache():
    config = InferenceConfig(model_name="dummy_predictor", enable_cache=True)
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    # First run (Miss)
    res1 = pipeline.generate("CACHE_ME")
    assert res1.cache_statistics.get("trajectory_hit") is False
    
    # Modify cache directly to prove we are hitting it
    cached_traj = session.cache.get_trajectory("CACHE_ME")
    cached_traj.text = "TAMPERED"
    session.cache.set_trajectory("CACHE_ME", cached_traj)
    
    # Second run (Hit)
    res2 = pipeline.generate("CACHE_ME")
    assert res2.cache_statistics.get("trajectory_hit") is True
    assert res2.trajectory.text == "TAMPERED" # Proves it didn't run reconstruction
    
    # Ensure deepcopy protection (mutating res2 shouldn't alter cache)
    res2.trajectory.text = "CORRUPTED"
    res3 = pipeline.generate("CACHE_ME")
    assert res3.trajectory.text == "TAMPERED"
    
    session.shutdown()
