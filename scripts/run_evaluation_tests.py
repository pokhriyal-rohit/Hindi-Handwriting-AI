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
        
        from tests.test_evaluation_trajectory import test_stroke_count_metric, test_endpoint_error_metric, test_dtw_frechet_metrics
        test_stroke_count_metric()
        print("[PASS] Stroke Count metric test passed")
        
        test_endpoint_error_metric()
        print("[PASS] Endpoint Error metric test passed")
        
        test_dtw_frechet_metrics()
        print("[PASS] DTW/Frechet gracefully tested")
        
        from tests.test_evaluation_geometry import test_path_length_metric, test_bounding_box_metric, test_smoothness_metric
        test_path_length_metric()
        print("[PASS] Path Length metric test passed")
        
        test_bounding_box_metric()
        print("[PASS] Bounding Box metric test passed")
        
        test_smoothness_metric()
        print("[PASS] Smoothness gracefully tested")
        
        from tests.test_evaluation_performance import test_svg_generation_metric, test_inference_latency_metric, test_system_memory_metric
        test_svg_generation_metric()
        print("[PASS] SVG Generation metric test passed")
        
        test_inference_latency_metric()
        print("[PASS] Inference Latency metric test passed")
        
        test_system_memory_metric()
        print("[PASS] System Memory metric test passed")
        
        print("\nAll evaluation tests passed successfully!")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation_tests()
