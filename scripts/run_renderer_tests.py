import os
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_renderer import test_rendering_engine_initialization, test_svg_export, test_layout_scaling, test_invalid_trajectory_exception, test_cache_hit_and_miss, test_paragraph_layout, dummy_trajectory

def run_tests():
    print("Running Renderer Tests...")
    traj = dummy_trajectory()
    
    try:
        test_rendering_engine_initialization()
        print("[PASS] Initialization test passed")
        
        test_svg_export(traj)
        print("[PASS] SVG export test passed")
        
        test_layout_scaling(traj)
        print("[PASS] Layout scaling test passed")
        
        from tests.test_visual_regression import test_visual_regression_svg
        test_visual_regression_svg()
        print("[PASS] Visual regression test passed")
        
        test_invalid_trajectory_exception()
        print("[PASS] Exception handling test passed")
        
        test_cache_hit_and_miss(traj)
        print("[PASS] Cache hit/miss test passed")
        
        test_paragraph_layout(traj)
        print("[PASS] Paragraph layout test passed")
        
        print("\nAll rendering tests passed successfully!")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
