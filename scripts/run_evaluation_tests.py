import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_evaluation import test_metric_registry, test_base_metric_validation

def run_evaluation_tests():
    print("Running Evaluation Framework Tests...")
    
    try:
        test_metric_registry()
        print("[PASS] Metric registry test passed")
        
        test_base_metric_validation()
        print("[PASS] BaseMetric validation test passed")
        
        print("\nAll evaluation tests passed successfully!")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation_tests()
