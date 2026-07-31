import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_inference_tests():
    print("Running Inference Framework Tests...")
    try:
        from tests.test_inference_session import test_session_initialization
        test_session_initialization()
        print("[PASS] InferenceSession initialized successfully")
        
        from tests.test_inference_predictor import test_dummy_predictor
        test_dummy_predictor()
        print("[PASS] Predictor contract test passed")
        
        from tests.test_inference_pipeline import test_inference_pipeline
        test_inference_pipeline()
        print("[PASS] Inference Pipeline execution test passed")
        
        from tests.test_inference_postprocessing import test_inference_postprocessing
        test_inference_postprocessing()
        print("[PASS] Post Processing test passed")
        
        from tests.test_inference_result import test_inference_result_serialization
        test_inference_result_serialization()
        print("[PASS] Inference Result serialization test passed")
        
        from tests.test_inference_cache import test_inference_cache
        test_inference_cache()
        print("[PASS] Inference Cache test passed")
        
        print("\nAll inference tests passed successfully!")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_inference_tests()
