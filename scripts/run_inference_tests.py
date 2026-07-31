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
        
        print("\nAll inference tests passed successfully!")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_inference_tests()
