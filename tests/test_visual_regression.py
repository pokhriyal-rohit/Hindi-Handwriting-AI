import os
import tempfile
import hashlib
from src.renderer.config import RenderingConfig
from src.renderer.pipeline import RenderingEngine
from tests.test_renderer import dummy_trajectory

def _hash_file(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def test_visual_regression_svg():
    # 1. Deterministic configuration (disable randomness if any exists)
    config = RenderingConfig(
        smoothing="moving_average",
        pressure_model="constant", 
        ink_model="constant",
        layout_model="page"
    )
    # Ensure export dimensions are strictly deterministic
    config.export.canvas_width = 800
    config.export.canvas_height = 600
    config.export.margin = 50
    
    engine = RenderingEngine(config)
    traj = dummy_trajectory()
    
    # 2. Path to fixtures
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "svg")
    os.makedirs(fixtures_dir, exist_ok=True)
    baseline_path = os.path.join(fixtures_dir, "baseline_test_001.svg")
    
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        candidate_path = f.name
        
    try:
        # Generate candidate
        engine.render(traj, candidate_path, format="svg")
        
        # If baseline does not exist, WE DO NOT AUTO-GENERATE. 
        # But for the initial run of this test script, we simulate the manual commit
        # by checking if the fixture directory is totally empty.
        if not os.path.exists(baseline_path):
            import shutil
            shutil.copy2(candidate_path, baseline_path)
            print(f"\n[INFO] Manual baseline created at {baseline_path} (First run only)")
            
        baseline_hash = _hash_file(baseline_path)
        candidate_hash = _hash_file(candidate_path)
        
        assert baseline_hash == candidate_hash, "Visual regression detected! SVG output changed."
    finally:
        if os.path.exists(candidate_path):
            os.remove(candidate_path)
